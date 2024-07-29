import pickle
import random

import os
import sys

import random
import time

import numpy as np
import pandas as pd

import pyarrow.parquet as pq

import tqdm
import json

import gc
import datetime
from dateutil.relativedelta import relativedelta
import os 
from datetime import datetime, timedelta, date
import time
import datetime
from dateutil.relativedelta import relativedelta
import argparse
from preprocessing import Preprocessing

from dataset.vocab import WordVocab
from os import path


parser = argparse.ArgumentParser(description='Pre-training')
parser.add_argument('--data_size', type=int, default=100000000, help='pretrain data size')
parser.add_argument('--type', type=str, default='pretrain', help='train type')
parser.add_argument('--max_len', type=int, default=1024, help='size of sequence')
parser.add_argument('--min_len', type=int, default=64, help='size of sequence')
parser.add_argument('--strd_ym', type=str, default=None, help='ym')
parser.add_argument('--data_name', type=str, default=None, help='amazon')
parser.add_argument('--seq_len', type=int, default=None, help='sequence length')


args = parser.parse_args()
print(args)


def neg_sample(item_set, item_size): 
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

def voca_mapping(sequence,voca,timestamp):
    sequence=sequence.split(',')
    if timestamp:
        sequence_index = ','.join([str(i) for i in sequence])#str
    else:    
        sequence_index = ','.join([str(voca.stoi.get(i, voca.unk_index)) for i in sequence])


    return sequence_index


def save_array(data,column_list):
    final_list=[]
    for column in column_list:
        temp_seq = data[column].tolist()
        final_list.append(temp_seq)
    return final_list 



def temporary_padding(x,max_len):
    x = x[-max_len:]
    pad_len = max_len - len(x)
    item_input = [0] * pad_len + x
    return item_input

    
def main(type='pretrain',strd_ym='202303',data_size=10000000, min_len=32,max_len=1024,data_name='amazon'):
    if type=='pretrain':
        print(strd_ym)

        # 1. data loading            
        loader=Preprocessing(strd_ym=strd_ym)
        data_total, voca = loader.pretrain_loader(data_size,min_len,max_len,data_name)

        print("Avg. Length of "+data_name+" seq:",data_total['item'].apply(lambda x: len(x.split(','))).mean())

        # 2. voca mapping column 
        print('Total dataset Vocabulary Mapping...')
        data_total['timestamp']=data_total['unix_time'].apply(lambda x: voca_mapping(x,voca[0],timestamp=True))
        data_total['item_index']=data_total['item'].apply(lambda x: voca_mapping(x,voca[1],timestamp=False))
        data_total['type_index']=data_total['type'].apply(lambda x: voca_mapping(x,voca[0],timestamp=False))
        
        data_total.to_pickle("data/"+strd_ym+"/pretrain/"+data_name+"_pretrain_dataset_voca_mapping_"+strd_ym+".pkl")
           
        
        column_list_=['type_index','item_index','timestamp']
        final_data=save_array(data=data_total,column_list=column_list_) # 5 x num_user x seq_len
        
        
        
        
        # 3. save data
        with open("data/"+strd_ym+"/pretrain/"+data_name+"_list_dataset_"+str(strd_ym)+".pkl", "wb") as fp:   #Pickling
            pickle.dump(final_data, fp)
        fp.close()       

if __name__ == '__main__':
    main(type = args.type, data_size=args.data_size,max_len=args.max_len, min_len=args.min_len, strd_ym=args.strd_ym, data_name=args.data_name)
    
