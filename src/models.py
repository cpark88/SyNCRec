# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 10:57
# @Author  : Hui Wang

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, LayerNorm, Intermediate, NoisyGating, PatitionedGating, EncoderNoFFN

import itertools
from itertools import combinations
import math
from torch.distributions.normal import Normal



class CausalModel(nn.Module):
    def __init__(self, args):
        super(CausalModel, self).__init__()
        self.type_embeddings = nn.Embedding(args.type_size, args.hidden_size, padding_idx=0)        
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
      
        self.prediction_layer = Intermediate(args)

        
        ###
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.noisy_gating = NoisyGating(args)
        self.patitioned_gating = PatitionedGating(args)
        self.ffn_layer = Intermediate(args)
        self.att_layer = EncoderNoFFN(args)
        ###


        
        self.args = args
        self.device = torch.device("cuda:"+str(self.args.local_rank))

        # sequence modeling 
        self.mlm_output = nn.Linear(args.hidden_size, args.item_size)
        self.mip_norm = nn.Linear(args.hidden_size, args.hidden_size)

        if self.args.loss_type == 'negative':
            self.criterion = nn.BCELoss(reduction='none')
        else:
            self.criterion = nn.NLLLoss(ignore_index=0)
        self.apply(self.init_weights)



        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

                
        self.expert_num = args.expert_num
        self.task_num = args.task_num



        if self.args.expert_layer=='ffn':
            # ffn 만 expert인 버전
            self.expert_=torch.nn.ModuleList([self.ffn_layer for i in range(self.expert_num)])
        elif self.args.expert_layer=='transformer':
            # transformer 전체가 expert 인 버전     
            self.expert_=torch.nn.ModuleList([self.item_encoder for i in range(self.expert_num)])
        else:
            pass
        
        
        self.tower_ = torch.nn.ModuleList([self.prediction_layer for i in range(self.task_num)])


        
        if self.args.cross_detach=='y' or self.args.single_detach=='y':
            self.basic_gating='n'
        else:
            self.basic_gating='y'
            
        # noise with partitioned gating 
        self.gate_ = torch.nn.ModuleList([self.patitioned_gating.to(self.device) for i in range(self.task_num)])        
        self.mip_norm = nn.Linear(self.args.hidden_size, self.args.hidden_size)


        #shaply update
        self.alpha = nn.Linear(1,1,bias=False)
        self.beta = nn.Linear(1,1,bias=False)
        self.shaply_values_update = torch.tensor([1/(self.args.task_num -1) for i in range(self.args.task_num -1)])


        
    def add_position_embedding(self, item_seq, type_seq):
        seq_length = item_seq.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        
        type_embeddings = self.type_embeddings(type_seq)
        item_embeddings = self.item_embeddings(item_seq)
        
        position_embeddings = self.position_embeddings(position_ids)
        

        sequence_emb = item_embeddings + type_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb


    # MIP sample neg items
    def masked_item_prediction(self, sequence_output, target_item):
        '''
        :param sequence_output: [B L H]
        :param target_item: [B L H]
        :return: scores [B*L]
        '''
        sequence_output = self.mip_norm(sequence_output.view([-1,self.args.hidden_size])) # [B*L H]
        target_item = target_item.view([-1,self.args.hidden_size]) # [B*L H]
        score = torch.mul(sequence_output, target_item) # [B*L H]
        return torch.sigmoid(torch.sum(score, -1)) # [B*L]

    
    def pretrain_seq(self, item_input, item_pos, item_neg, test_neg , item_answer, type_input, type_pos):
        ############## 1. causality mask ##############
        attention_mask = (item_input > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        
        subsequent_mask = subsequent_mask.to(self.device)       
            
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        ############## 2. single-only ##############
        single_domain_loss_list=[] # 각 domain loss를 담아두는 리스트
        loss_contrastive_single = 0
        domain_list=[5,6,7,8,9]
        
        gate_value_list=[]
        
        cross_expert_num=int(self.expert_num*self.args.cross_expert_ratio)
        single_expert_num=self.expert_num-cross_expert_num
        sequence_encoder_output_single=torch.zeros([item_input.shape[0],self.args.max_seq_length,self.args.hidden_size]).to(self.device)  
        sequence_encoder_output_single_mim=torch.zeros([item_input.shape[0],self.args.max_seq_length,self.args.hidden_size]).to(self.device)  

        
        for domain_index in domain_list:
            #embedding
            sequence_emb_single = self.add_position_embedding((item_input*(type_input==domain_index)), (type_input*(type_input==domain_index))) # B T D


            # encoder
            if self.args.expert_layer=='ffn':
                encoded_layers = self.att_layer(sequence_emb_single,
                                                  extended_attention_mask,
                                                  output_all_encoded_layers=True)
                sequence_enc_single = encoded_layers[-1] # [B L H] # 마지막 레이어
                gate_value = self.gate_[domain_index-len(domain_list)](sequence_emb_single.view([sequence_emb_single.size(0),-1]), cross_yn='n', basic_gating=self.basic_gating).unsqueeze(1) # B x 1 x expert_num
                gate_value_list.append(torch.mean(gate_value,dim=0)[0])
    
    
                fea_1_single = torch.stack([self.expert_[i](sequence_enc_single).view(sequence_enc_single.size(0),-1) for i in range(single_expert_num)], dim = 1) # expert_num B T*D
                fea_2_single = torch.stack([self.expert_[i+single_expert_num](sequence_enc_single).view(sequence_enc_single.size(0),-1) for i in range(cross_expert_num)], dim = 1) # expert_num B T*D
                if self.args.single_detach=='y':
                    fea_2_single = fea_2_single.detach()
                else:
                    pass    
                fea_single=torch.cat([fea_1_single, fea_2_single],dim=1)            
            
            
            elif self.args.expert_layer=='transformer':
                gate_value = self.gate_[domain_index-len(domain_list)](sequence_emb_single.view([sequence_emb_single.size(0),-1]), cross_yn='n', basic_gating=self.basic_gating).unsqueeze(1) # B x 1 x expert_num
                gate_value_list.append(torch.mean(gate_value,dim=0)[0])
    

                fea_1_single = torch.stack([self.expert_[i](sequence_emb_single, extended_attention_mask,output_all_encoded_layers=True)[-1].view(sequence_emb_single.size(0),-1) for i in range(single_expert_num)], dim = 1) # expert_num B T*D
                fea_2_single = torch.stack([self.expert_[i+single_expert_num](sequence_emb_single, extended_attention_mask,output_all_encoded_layers=True)[-1].view(sequence_emb_single.size(0),-1) for i in range(cross_expert_num)], dim = 1) # expert_num B T*D
                if self.args.single_detach=='y':
                    fea_2_single = fea_2_single.detach()
                else:
                    pass    
                fea_single=torch.cat([fea_1_single, fea_2_single],dim=1)          

            else:
                pass



            # gating으로 취합
            task_fea_single = torch.bmm(gate_value, fea_single).squeeze(1).view(-1,self.args.max_seq_length,self.args.hidden_size)
            sequence_output_single_tmp = self.tower_[domain_index-len(domain_list)](task_fea_single)
            sequence_encoder_output_single += sequence_output_single_tmp*((type_input==domain_index).unsqueeze(-1))


            # single-only loss
            # lrb를 위한 single domain loss
            single_domain_loss=self.cross_entropy(sequence_output_single_tmp, item_pos*(type_pos==domain_index), item_neg*(type_pos==domain_index), type_input)
            single_domain_loss_list.append(single_domain_loss) # 각 도메인 loss
            
            
            # 실제 학습 위한 single domain loss
            # single_domain_loss=self.cross_entropy(sequence_output_single_tmp, item_pos*(type_pos==domain_index), item_neg*(type_pos==domain_index), type_input*(type_pos==domain_index)) # 각 도메인 별 모수 개수로 나눠서 모수가 적어도 학습이 많이 되게 함.  
            single_domain_loss=self.cross_entropy_single(sequence_output_single_tmp, item_pos*(type_pos==domain_index), item_neg*(type_pos==domain_index), type_input*(type_pos==domain_index))  # 나누지 않아 기존과 같음 (amazon)
            # single_domain_loss=self.cross_entropy_log_inverse(sequence_output_single_tmp, item_pos*(type_pos==domain_index), item_neg*(type_pos==domain_index), type_input*(type_pos==domain_index), type_pos)  # batch 내 특정 도메인 데이터 수/전체 데이터수 의 log 스케일로 가중치 (skt)
            loss_contrastive_single += single_domain_loss * (1/len(domain_list))
            
            
            
            
            
        ############## 3. loss re-balancing part ##############
        if self.args.lrb=='y':
            # 2-2. cross-domain loss        
            cross_domain_loss_list=[] # cross에서의 각 도메인 loss 담아주는 list
            loss_contrastive_cross = 0
            sequence_emb_cross = self.add_position_embedding(item_input, type_input)


            if self.args.expert_layer=='ffn':
                encoded_layers = self.att_layer(sequence_emb_cross,
                                                  extended_attention_mask,
                                                  output_all_encoded_layers=True)
                sequence_enc_cross = encoded_layers[-1] # [B L H] # 마지막 레이어


                gate_value = self.gate_[domain_index-len(domain_list)](sequence_emb_cross.view([sequence_emb_cross.size(0),-1]), cross_yn='y', basic_gating=self.basic_gating).unsqueeze(1) # B x 1 x expert_num
                gate_value_list.append(torch.mean(gate_value,dim=0)[0])
    
    
                fea_1_cross = torch.stack([self.expert_[i](sequence_enc_cross).view(sequence_enc_cross.size(0),-1) for i in range(single_expert_num)], dim = 1) # expert_num B T*D
                fea_2_cross = torch.stack([self.expert_[i+single_expert_num](sequence_enc_cross).view(sequence_enc_cross.size(0),-1) for i in range(cross_expert_num)], dim = 1) # expert_num B T*D
                if self.args.single_detach=='y':
                    fea_1_cross = fea_1_cross.detach()
                else:
                    pass    
                fea_cross=torch.cat([fea_1_cross, fea_2_cross],dim=1)     


            
            elif self.args.expert_layer=='transformer':
                
                gate_value = self.gate_[-1](sequence_emb_cross.view([sequence_emb_cross.size(0),-1]), cross_yn='y', basic_gating=self.basic_gating).unsqueeze(1)
                gate_value_list.append(torch.mean(gate_value,dim=0)[0])            
                
                fea_1_cross = torch.stack([self.expert_[i](sequence_emb_cross, extended_attention_mask,output_all_encoded_layers=True)[-1].view(sequence_emb_cross.size(0),-1) for i in range(single_expert_num)], dim = 1)
                # print('++++',fea_1_cross[1,:,:])
                fea_2_cross = torch.stack([self.expert_[i+single_expert_num](sequence_emb_cross, extended_attention_mask,output_all_encoded_layers=True)[-1].view(sequence_emb_cross.size(0),-1) for i in range(cross_expert_num)], dim = 1) # B x expert_num x (T*hidden_size)
    
                if self.args.cross_detach=='y':
                    fea_1_cross = fea_1_cross.detach()
                else:
                    pass
                fea_cross=torch.cat([fea_1_cross, fea_2_cross],dim=1)

            else:
                pass
            
            
            
            #학습용
            sequence_encoder_output_cross = torch.bmm(gate_value, fea_cross).squeeze(1).view(-1,self.args.max_seq_length,self.args.hidden_size).to(self.device) # B x T x hidden_size
            sequence_encoder_output_cross = self.tower_[-1](sequence_encoder_output_cross) # B x T x hidden_size


 
            #option 1
            # loss_contrastive_cross_total = self.cross_entropy(sequence_encoder_output_cross, item_pos, item_neg, type_input)
            loss_contrastive_cross_total = torch.tensor(0)
            #option 2 (lrb 위한)
            for domain_index in domain_list:
                cross_domain_loss=self.cross_entropy(sequence_encoder_output_cross, item_pos*(type_pos==domain_index), item_neg*(type_pos==domain_index), type_input*(type_pos==domain_index))   
                # cross_domain_loss=self.cross_entropy_log_inverse(sequence_encoder_output_cross, item_pos*(type_pos==domain_index), item_neg*(type_pos==domain_index), type_input*(type_pos==domain_index), type_pos)   
                cross_domain_loss_list.append(cross_domain_loss)  # --> [x, x, x,x, x]          
            
            # 2-3. loss re-balancing
            diff=[k[0]-k[1] for k in zip(single_domain_loss_list,cross_domain_loss_list)]# cross의 loss가 더 크다면 더 작은 가중치로 loss를 밸런싱
            
            self.shaply_values_update = self.alpha(self.shaply_values_update.unsqueeze(1).to(self.device)) + self.beta(torch.tensor([i for i in diff]).unsqueeze(1).to(self.device))
            self.shaply_values_update = self.shaply_values_update.squeeze(1)
            
            # shapley_softmax=self.softmax_with_temperature(preds=torch.tensor([i for i in diff]),temperature=self.args.temperature) #선택지2
            shapley_softmax=self.softmax_with_temperature(preds=torch.tensor([i for i in self.shaply_values_update]),temperature=self.args.temperature) #선택지2
            shapley_values_softmax={}
            for k,j in zip(domain_list,shapley_softmax):# loss에 가중치 부여하는 버전에서 추가 
                shapley_values_softmax[k]=j
            # print(shapley_values_softmax)


            # 최종 loss
            for domain_index in domain_list:
                # cross_domain_loss=self.cross_entropy(sequence_encoder_output_cross, item_pos*(type_pos==domain_index), item_neg*(type_pos==domain_index), type_input*(type_pos==domain_index)) # 각 도메인 별 모수 개수로 나눠서 모수가 적어도 학습이 많이 되게 함.  
                cross_domain_loss=self.cross_entropy_single(sequence_encoder_output_cross, item_pos*(type_pos==domain_index), item_neg*(type_pos==domain_index), type_input*(type_pos==domain_index))  # 나누지 않아 기존과 같음 (amazon)
                # cross_domain_loss=self.cross_entropy_log_inverse(sequence_encoder_output_cross, item_pos*(type_pos==domain_index), item_neg*(type_pos==domain_index), type_input*(type_pos==domain_index), type_pos)  # batch 내 특정 도메인 데이터 수/전체 데이터수 의 log 스케일로 가중치 (skt)
                loss_contrastive_cross += cross_domain_loss * shapley_values_softmax[domain_index]
            # loss_contrastive_cross = loss_contrastive_cross/(item_pos>0).sum()





        

        else: #args.lrb='n'
            # 2-2. cross-domain loss        
            cross_domain_loss_list=[] # cross에서의 각 도메인 loss 담아주는 list
            loss_contrastive_cross = 0
            sequence_output_shared = self.add_position_embedding(item_input, type_input)
        
            gate_value = self.gate_[-1](sequence_output_shared.view([sequence_output_shared.size(0),-1]), cross_yn='y', basic_gating=self.basic_gating).unsqueeze(1)
            
            fea_1_cross = torch.stack([self.expert_[i](sequence_output_shared, extended_attention_mask,output_all_encoded_layers=True)[-1].view(sequence_output_shared.size(0),-1) for i in range(single_expert_num)], dim = 1)
            fea_2_cross = torch.stack([self.expert_[i+single_expert_num](sequence_output_shared, extended_attention_mask,output_all_encoded_layers=True)[-1].view(sequence_output_shared.size(0),-1) for i in range(cross_expert_num)], dim = 1)
            
            fea_1_cross = fea_1_cross.detach()
            fea_cross=torch.cat([fea_1_cross, fea_2_cross],dim=1)     
            
            

            sequence_encoder_output_cross = torch.bmm(gate_value, fea_cross).squeeze(1).view(-1,self.args.max_seq_length,self.args.hidden_size).to(self.device)
            sequence_encoder_output_cross = self.tower_[-1](sequence_encoder_output_cross)
    


    
            # loss_contrastive_cross_total = self.cross_entropy(sequence_encoder_output_cross, item_pos, item_neg, type_input)
            loss_contrastive_cross_total=self.cross_entropy_log_inverse(sequence_encoder_output_cross, item_pos, item_neg, type_input, type_pos)  # batch 내 특정 도메인 데이터 수/전체 데이터수 의 log 스케일로 가중치 (skt)
            
            loss_contrastive_cross = torch.tensor(0)
            loss_contrastive_single = torch.tensor(0)





        
        # 3. MIP
        if self.args.mip=='y':
            # pos_item_embs = self.item_embeddings(item_input)
            neg_item_embs = self.item_embeddings(item_input[[torch.randperm(item_input.size()[0])]])# 실제 정답인 item_pos를 batch 내에서 shuffling (item_neg도 할수 있으나 loss가 떨어지지 않았음)
            pos_score = self.masked_item_prediction(sequence_encoder_output_cross, sequence_encoder_output_single) # B*T 개 나옴
            neg_score = self.masked_item_prediction(sequence_encoder_output_cross, neg_item_embs)
            mip_distance = torch.sigmoid(pos_score - neg_score)
            mip_loss = self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))
            # mip_mask = (masked_item_sequence == self.args.mask_id).float()
            # mip_loss = torch.sum(mip_loss * mip_mask.flatten())
            mip_loss = mip_loss*((item_pos>0).flatten())
            mip_loss = torch.mean(mip_loss)

        else: 
            mip_loss = torch.tensor(0)


        
        # 4. Final objectives
        loss=(loss_contrastive_single/1000 + loss_contrastive_cross/1000 + loss_contrastive_cross_total + mip_loss).to(self.device) 
            
            

        # print(gate_value_list)
        return loss, loss_contrastive_single/1000, loss_contrastive_cross/1000, mip_loss
    
    
    def get_last_emb(self, item_input,type_input, item_pos, item_neg, type_pos, cuda_yn='y'):
        cross_expert_num=int(self.expert_num*0.8)
        single_expert_num=self.expert_num-cross_expert_num
        
        
        # for sequence modeling (decoder)
        attention_mask = (item_input > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        
        # if self.args.cuda_condition:
        if cuda_yn=='y':
            subsequent_mask = subsequent_mask.to(self.device)
        else:
            subsequent_mask = subsequent_mask.to('cpu')
            
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


        sequence_emb = self.add_position_embedding(item_input, type_input)



        if self.args.expert_layer=='ffn':
            encoded_layers = self.att_layer(sequence_emb,
                                              extended_attention_mask,
                                              output_all_encoded_layers=True)
            sequence_output_cross = encoded_layers[-1] # [B L H] # 마지막 레이어


            gate_value = self.gate_[-1](sequence_emb.view([sequence_emb.size(0),-1]), cross_yn='y',basic_gating=self.basic_gating).unsqueeze(1) # B x 1 x expert_num
            fea = torch.stack([self.expert_[i](sequence_output_cross).view(sequence_emb.size(0),-1) for i in range(self.expert_num)], dim = 1) # # B x expert_num x (T*hidden_size)      

        elif self.args.expert_layer=='transformer':
            gate_value = self.gate_[-1](sequence_emb.view([sequence_emb.size(0),-1]), cross_yn='y',basic_gating=self.basic_gating).unsqueeze(1) # B x 1 x expert_num
            fea = torch.stack([self.expert_[i](sequence_emb, extended_attention_mask, output_all_encoded_layers=True)[-1].view(sequence_emb.size(0),-1) for i in range(self.expert_num)], dim = 1) # # B x expert_num x (T*hidden_size)

        else:
            pass

       
        sequence_output = torch.bmm(gate_value, fea).squeeze(1).view(-1,self.args.max_seq_length,self.args.hidden_size)
        sequence_output = self.tower_[-1](sequence_output)        


        eos_output = sequence_output[:,-1,]
        
        
        return eos_output
    
    
    def cross_entropy(self, seq_out, pos_ids, neg_ids,type_input):
        # [batch seq_len hidden_size]
        pos_emb = self.item_embeddings(pos_ids)
        neg_emb = self.item_embeddings(neg_ids)
        

            
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / (torch.sum(istarget)+1e-24)#sum(istarget)
        

        return loss
   
    def cross_entropy_single(self, seq_out, pos_ids, neg_ids,type_input):
        # [batch seq_len hidden_size]
        pos_emb = self.item_embeddings(pos_ids)
        neg_emb = self.item_embeddings(neg_ids)
        

            
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        )# / (torch.sum(istarget)+1e-24)#sum(istarget)
        

        return loss 


    def cross_entropy_log_inverse(self, seq_out, pos_ids, neg_ids, type_input, type_pos):
        # [batch seq_len hidden_size]
        pos_emb = self.item_embeddings(pos_ids)
        neg_emb = self.item_embeddings(neg_ids)



        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) * ( -torch.log( (torch.sum(istarget)+1e-24) / ((type_pos>0).sum()+(torch.sum(istarget)*1)) ) ) #( -torch.log( (torch.sum(istarget)+1e-24) / ((type_pos>0).sum()+1e-24)) )
        

        return loss 


    
    
    def get_acc(self, seq_out, target_pos):
        # test_item_emb = self.embedding_layer.item_embeddings.weight # [C H]
        test_item_emb = self.item_embeddings.weight # [C H]        
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1)) # [B L C]
        rating_pred = torch.argsort(-rating_pred)[:, :, 0] # [B L]
        rating_pred = rating_pred.view(target_pos.size(0) * self.args.max_seq_length) # [B*L]
        target = target_pos.view(target_pos.size(0) * self.args.max_seq_length) # [B*L]
        istarget = (target_pos > 1).view(target_pos.size(0) * self.args.max_seq_length)
        acc = torch.sum((rating_pred==target).float()*istarget) / torch.sum(istarget)
        return acc
    

        
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            

    
    def softmax_with_temperature(self, preds,temperature):
        ex = torch.exp(preds/temperature)
        return ex / torch.sum(ex, axis=0)




