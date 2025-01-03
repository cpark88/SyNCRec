#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/7/14
# @Author  : Chung Park and Taesan Kim and Hyungjun Yoon and Junui Hong
# @Desc    : modules in model

import numpy as np
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal



def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states



class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)


    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class LayerNoFFN(nn.Module):
    def __init__(self, args):
        super(LayerNoFFN, self).__init__()
        self.attention = SelfAttention(args)
        # self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        # intermediate_output = self.intermediate(attention_output)
        return attention_output

class EncoderNoFFN(nn.Module):
    def __init__(self, args):
        super(EncoderNoFFN, self).__init__()
        layer = LayerNoFFN(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
    

    
class NoisyGating(nn.Module):
    def __init__(self, args):
        super(NoisyGating, self).__init__()
        self.w_gate = nn.Linear(args.hidden_size*args.max_seq_length, args.expert_num)
        self.w_noise = nn.Linear(args.hidden_size*args.max_seq_length, args.expert_num)
        self.noise_epsilon=1e-3
        self.softplus = nn.Softplus()
        self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, input_tensor):

        clean_logits = self.w_gate(input_tensor)
        raw_noise_stddev = self.w_noise(input_tensor)
        noise_stddev = ((self.softplus(raw_noise_stddev) + self.noise_epsilon))
        noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev) # B x expert_num
        softmax_output = self.softmax(noisy_logits)

        return softmax_output



class PatitionedGating(nn.Module):
    def __init__(self, args):
        super(PatitionedGating, self).__init__()
        self.cross_expert_ratio=args.cross_expert_ratio
        self.expert_num=args.expert_num
        self.cross_expert_num=int(args.expert_num*args.cross_expert_ratio)
        self.single_expert_num=args.expert_num-self.cross_expert_num

        self.w_gate_total = nn.Linear(args.hidden_size*args.max_seq_length, self.expert_num)
        
        self.w_gate_cross = nn.Linear(args.hidden_size*args.max_seq_length, self.cross_expert_num)
        self.w_gate_single = nn.Linear(args.hidden_size*args.max_seq_length, self.single_expert_num)
        self.w_noise_cross = nn.Linear(args.hidden_size*args.max_seq_length, self.cross_expert_num)
        self.w_noise_single = nn.Linear(args.hidden_size*args.max_seq_length, self.single_expert_num)

        self.softmax = nn.Softmax(dim=1)
        self.noise_epsilon=1e-3
        self.softplus = nn.Softplus()
        self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        self.hidden_size=args.hidden_size
        self.max_seq_length=args.max_seq_length
    
    def forward(self, input_tensor, cross_yn, basic_gating):
        raw_noise_stddev_cross = self.w_noise_cross(input_tensor)
        noise_stddev_cross = ((self.softplus(raw_noise_stddev_cross) + self.noise_epsilon))
        raw_noise_stddev_single = self.w_noise_single(input_tensor)
        noise_stddev_single = ((self.softplus(raw_noise_stddev_single) + self.noise_epsilon))

        
        logits_single = self.w_gate_single(input_tensor)
        logits_single = logits_single + ( torch.randn_like(logits_single) * noise_stddev_single)# B x expert_num
        
        softmax_single = self.softmax(logits_single) # B x single_expert_num
        
        logits_cross = self.w_gate_cross(input_tensor) 
        logits_cross = logits_cross + ( torch.randn_like(logits_cross) * noise_stddev_cross)# B x expert_num
        
        softmax_cross = self.softmax(logits_cross) # B x cross_expert_num

        if basic_gating=='y': # basig gating
            softmax_total =  self.softmax(self.w_gate_total(input_tensor))
        else:#basic_gating=='n'
            if cross_yn=='y':
                softmax_total = torch.cat([softmax_single*(1-self.cross_expert_ratio), softmax_cross*self.cross_expert_ratio], dim=1)
            else: #cross_yn=='n':
                softmax_total = torch.cat([softmax_single*(self.cross_expert_ratio), softmax_cross*(1-self.cross_expert_ratio)], dim=1)
        

        return softmax_total
