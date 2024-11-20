#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/7/14
# @Author  : Chung Park and Taesan Kim and Hyungjun Yoon and Junui Hong
# @Desc    : run model training


import pickle
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import os
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import gc

from datasets import CausalDataset
from trainers import PretrainTrainer
from models import CausalModel
from utils import check_path, set_seed, EarlyStopping
from dataset.vocab import WordVocab


def main_worker():
    parser = argparse.ArgumentParser()
    
    # define baseline or skt
    parser.add_argument("--loss_type", default="negative", type=str)
    # data directory
    parser.add_argument('--output_dir', default='output/', type=str)

    # model args
    parser.add_argument("--model_name", default='Pretrain', type=str)

    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=128, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=128, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)


    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    parser.add_argument("--data_name", default='skt', type=str, help="data_name")

    parser.add_argument('--world_size', default=8, type=int,help='number of nodes for distributed training')
    parser.add_argument('--local_rank', default=None, type=int,help='node rank for distributed training')
    parser.add_argument('--ngpus_per_node', default=None, type=int,help='node rank for distributed training')
    
    parser.add_argument('--strd_ym', default=None, type=str,help='node rank for distributed training')

    parser.add_argument('--expert_num', default=None, type=int,help='the number of expers')
    parser.add_argument('--task_num', default=None, type=int,help='the number of tasks or domains+1')
    parser.add_argument('--temperature', default=1, type=float,help='softmax temperature')


    parser.add_argument('--lrb', default='y', type=str, help='lrb or not')
    parser.add_argument('--mip', default='y', type=str, help='mip or not')


    parser.add_argument('--cross_expert_ratio', default=0.8, type=float, help='number of cross experts')
    parser.add_argument('--cross_detach', default='n', type=str, help='detach or not in expert')
    parser.add_argument('--single_detach', default='n', type=str, help='detach or not in expert')
    parser.add_argument('--expert_layer', default='transformer', type=str, help='which layer to be expert (transformer, ffn)')

    parser.add_argument('--except_type', default=[100], type=list,help='except type index list') #13: adot    
    # lrb==y mip==y : ours / n, n : SASRec
    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    dist_url='env://'
    args.global_rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Use GPU: {} for training".format(args.local_rank))
    torch.cuda.set_device(args.local_rank)    

    dist.init_process_group(backend='nccl', init_method=dist_url, world_size=args.world_size, rank=args.global_rank)
    dist.barrier()

    print("Data Loading Start!")
    with open("data/"+args.strd_ym+"/pretrain/"+args.data_name+"_list_dataset_"+args.strd_ym+".pkl", "rb") as fp:   # Unpickling
        user_seq = pickle.load(fp)
    fp.close()
    print("Data Loading Complete!")
    columns=['type','item']

    voca=[]
    len_voca=[]
    for i in columns:
        output_voca_path = "pretrained_model/voca/"+args.data_name+"_voca_"+i+"_"+args.strd_ym+".ep"
        voca_tmp = WordVocab.load_vocab(output_voca_path)
        voca.append(voca_tmp)
        len_voca.append(len(voca_tmp))

    max_type = len_voca[0]       
    max_item = len_voca[1]

    args.type_size = max_type
    args.item_size = max_item

    # save model args
    try:
        args_str = f'{args.model_name}-{args.data_name}-strd_ym_{args.strd_ym}'
        args.output_path = args.output_dir + args.data_name + '/'
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        args.log_file = os.path.join(args.output_path, args_str + '.txt')
        print(args)
        with open(args.log_file, 'a') as f:
            f.write(str(args) + '\n')
        # save model
        checkpoint = args_str + '.pt'
        args.checkpoint_path = os.path.join(args.output_path, checkpoint)
    except:
        pass
 
    torch.cuda.empty_cache()
    # model loading
    model = CausalModel(args=args) #seq_to_profile
    trainer = PretrainTrainer(model, None, None,
                  None, args)
    

    # test dataset loading
    test_dataset = CausalDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=4, persistent_workers=True)

    evaluation_performance=0
    for epoch in range(args.epochs):
        gc.collect()
        train_dataset = CausalDataset(args, user_seq, data_type='train') # for not windowing
        train_sampler = DistributedSampler(
            train_dataset,
            # num_replicas=8,
            rank=args.local_rank,
            shuffle=True,
        )
        train_dataloader = DataLoader(train_dataset, 
                               batch_size=args.batch_size, #32
                               num_workers=4,sampler=train_sampler, shuffle=False, pin_memory=True)
        train_sampler.set_epoch(epoch)

        trainer.pretrain(epoch, train_dataloader, train=True)
        dist.barrier()

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            test_dataset = CausalDataset(args, user_seq, data_type='test')
            test_sampler = SequentialSampler(test_dataset)
            test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=30, persistent_workers=True)
            (scores, result_info),(scores_5, result_info_5),(scores_6, result_info_6),(scores_7, result_info_7),(scores_8, result_info_8),(scores_9, result_info_9) = trainer.pretrain(epoch, test_dataloader, train=False)
            scores_list = ['HIT_1', 'NDCG_1', 'HIT_5', 'NDCG_5', 'HIT_10', 'NDCG_10', 'MRR']
            if  evaluation_performance <= scores[-4]:
                print('---------------Evaluation Result-------------------')
                print('Previous',scores_list[-4] ,':',evaluation_performance,'|| Current',scores_list[-4], ':',scores[-4])
                print('Epoch',epoch,':Evaluation Performance is less than previous epoch! Update.')
                evaluation_performance=scores[-4]
                trainer.save(args.checkpoint_path)            
                print('---------------------------------------------------')
            else:
                print('---------------Evaluation Result-------------------')
                print('Previous',scores_list[-4] ,':',evaluation_performance,'|| Current',scores_list[-4], ':',scores[-4])
                print('Epoch',epoch,':Evaluation Performance is more than previous epoch! No update.')
                print('---------------------------------------------------')

        dist.barrier()

    dist.barrier()
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:

        print('---------------Sample 100 results-------------------')
        (scores, result_info),(scores_5, result_info_5),(scores_6, result_info_6),(scores_7, result_info_7),(scores_8, result_info_8),(scores_9, result_info_9) = trainer.pretrain(epoch, test_dataloader, train=False)

        print(args_str)
        print(result_info)
        
        print('5:',result_info_5)
        print('6:',result_info_6)
        print('7:',result_info_7)
        print('8:',result_info_8)
        print('9:',result_info_9)
        with open(args.log_file, 'a') as f:
            f.write(args_str + '\n')
            f.write(result_info + '\n')

main_worker()
