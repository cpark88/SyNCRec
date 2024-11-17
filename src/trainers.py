#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/7/14
# @Author  : Chung Park
# @Desc    : trainer

import numpy as np
import tqdm
import random
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
import torch.nn.functional as F
from utils import get_metric
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import socket
from torch.cuda.amp import autocast, GradScaler
import gc
from torch.distributed.optim import ZeroRedundancyOptimizer

class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.mlm_output = nn.Linear(args.hidden_size, args.item_size-1)
        self.local_rank = args.local_rank
        self.device = torch.device("cuda:"+str(self.local_rank))
        self.model = model.to(self.device)
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
        if self.args.loss_type == 'negative':
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.NLLLoss()

        ###ddp part
        self.local_rank=self.args.local_rank
        with_cuda=True
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:"+str(self.local_rank))
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for RecGPT" % torch.cuda.device_count())
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        ###
        
        ###amp part
        self.scaler = GradScaler(enabled=False,growth_interval=100)
        ###

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return ([HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix))

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_negatve_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.module.item_embeddings(pos_ids)
        neg_emb = self.model.module.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / (torch.sum(istarget) + 1e-24)

        return loss
    
    def cross_entropy(self, seq_out, pos_ids):
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        # only mlm_loss        
        sequence_output = self.mlm_output(seq_emb) # [batch*seq_len class_num]
        
        labels = pos_ids.view(-1, 1) # [batch*seq_len class_num]
        
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float().view(-1, 1) # [batch*seq_len, 1]
        labels = (labels * istarget).long()
        loss = self.criterion(nn.LogSoftmax(dim=-1)(sequence_output), labels.view(-1, ))
        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        # test_item_emb = self.model.module.embedding_layer.item_embeddings(test_neg_sample)
        test_item_emb = self.model.module.item_embeddings(test_neg_sample)        
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        # test_item_emb = self.model.module.embedding_layer.item_embeddings.weight
        test_item_emb = self.model.module.item_embeddings.weight        
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred
            
class PretrainTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(PretrainTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def pretrain(self, epoch, dataloader, train=True):
        str_code = "train" if train else "test"
        
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                          desc="Recommendation EP_%s:%d" % (str_code, epoch),
                          total=len(dataloader),
                          bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            
            loss_contrastive_single_avg = 0.0
            loss_contrastive_cross_avg = 0.0
            mip_loss_avg = 0.0
            loss_avg = 0.0
            expert_disentangled_loss_avg = 0.0
            load_loss_total_avg =0.0
            
            print("Device",self.device)
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                with autocast(enabled=False):
                    batch = tuple(t.to(self.device) for t in batch)
    
                    item_input, item_pos, item_neg, test_neg , item_answer, type_input, type_pos = batch
                    loss, loss_contrastive_single, loss_contrastive_cross, mip_loss = self.model.module.pretrain_seq(item_input, item_pos, item_neg, test_neg, item_answer, type_input, type_pos) 

                self.optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), max_norm=1)
                self.optim.step()
                loss_contrastive_single_avg += loss_contrastive_single.detach().item()
                loss_contrastive_cross_avg += loss_contrastive_cross.detach().item()
                mip_loss_avg += mip_loss.detach().item()
                loss_avg += loss.detach().item()


            post_fix = {
                "epoch": epoch,
                "loss_avg": '{:.4f}'.format(loss_avg/len(rec_data_iter)),
                "loss_contrastive_single_avg": '{:.4f}'.format(loss_contrastive_single_avg/len(rec_data_iter)),
                "loss_contrastive_cross_avg": '{:.4f}'.format(loss_contrastive_cross_avg/len(rec_data_iter)),
                "mip_loss_avg": '{:.4f}'.format(mip_loss_avg/len(rec_data_iter)),  
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')


        else:        
            self.model.eval()
            
            pred_list = None
            type_pos_list=[]
            with torch.no_grad():
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)#cpu self.device


                    item_input, item_pos, item_neg, test_neg , item_answer, type_input, type_pos = batch

                    recommend_output = self.model.to(self.device).module.get_last_emb(item_input, type_input, item_pos, item_neg, type_pos, cuda_yn='y')


                    test_neg_items = torch.cat((item_answer, test_neg), -1)

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    elif i!=0 and type_pos[:,-1].shape[0]==self.args.batch_size:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                    else:
                        pass
                    
                    if type_pos[:,-1].shape[0]==self.args.batch_size: 
                        type_pos_list.append(type_pos[:,-1].cpu().detach().numpy().copy())

                type_pos_final=np.concatenate(np.array(type_pos_list))

                return self.get_sample_scores(epoch, pred_list), self.get_sample_scores(epoch, pred_list[type_pos_final==5]) , self.get_sample_scores(epoch, pred_list[type_pos_final==6]) , self.get_sample_scores(epoch, pred_list[type_pos_final==7]) , self.get_sample_scores(epoch, pred_list[type_pos_final==8]) , self.get_sample_scores(epoch, pred_list[type_pos_final==9])