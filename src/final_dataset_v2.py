import pickle
import random

import os
import sys

import random
import time

import numpy as np
import pandas as pd
#import seaborn as sns
# from skinet.utils.hive_tools import py_hive_connection, dbGetQuery, load_from_hive, dbSendUpdate, load_to_hive

########
import pyarrow.parquet as pq
from pyhive import hive

from skt.mls import *
from skt.ye import hive_execute, get_hdfs_conn, hive_to_pandas, pandas_to_parquet, slack_send, get_spark, get_hdfs_conn

from skt.gcp import get_bigquery_client, bq_insert_overwrite, get_max_part, bq_to_df, bq_to_pandas, pandas_to_bq_table, load_query_result_to_table, PROJECT_ID
import tqdm
########



import json

import gc
# import tqdm
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
# torch.cuda.empty_cache()
# conn = py_hive_connection()



########
parser = argparse.ArgumentParser(description='Pre-training')

parser.add_argument('--data_size', type=int, default=100000000, help='pretrain data size')
parser.add_argument('--type', type=str, default='pretrain', help='train type')
parser.add_argument('--max_len', type=int, default=1024, help='size of sequence')
parser.add_argument('--min_len', type=int, default=64, help='size of sequence')
parser.add_argument('--strd_ym', type=str, default=None, help='ym')
parser.add_argument('--data_name', type=str, default=None, help='skt')
parser.add_argument('--seq_len', type=int, default=None, help='sequence length')
# parser.add_argument('--data_type', type=str, default=None, help='data type')



args = parser.parse_args()
print(args)

# param = data_parameters[args.type]

# today = datetime.date.today() #매월 초
# strd_ym=today.strftime("%Y%m")
# bf_ym=(today-relativedelta(months=1)).strftime("%Y%m") #지난달
#########

def neg_sample(item_set, item_size): 
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

def voca_mapping(sequence,voca,timestamp):
    sequence=sequence.replace("수능(1,2학년)","수능(1_2학년)").replace("War Hammer 40,000","War Hammer 40_000") # word내 콤마 있으면 제외..
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

    
def main(type='pretrain',strd_ym='202303',data_size=10000000, min_len=32,max_len=1024,data_name='skt'):
    if type=='pretrain':
        print(strd_ym)


        #######
        # 한 user 내에서 마지막 두개만 남겨두는 data split --> 논문 세팅 o
        # 1. hive load 원본 테이블 로딩

            
        loader=Preprocessing(strd_ym=strd_ym)
        data_total, voca = loader.pretrain_loader(data_size,min_len,max_len,data_name)

        print("Avg. Length of "+data_name+" seq:",data_total['item'].apply(lambda x: len(x.split(','))).mean())

        
        # 2. voca mapping column 추가테이블

  
        print('Total dataset Vocabulary Mapping...')
        data_total['timestamp']=data_total['unix_time'].apply(lambda x: voca_mapping(x,voca[0],timestamp=True))
        data_total['item_index']=data_total['item'].apply(lambda x: voca_mapping(x,voca[1],timestamp=False))#여기서 itme=기타 면 cat1 대체 할 수 있을듯 
        # data_total['cat1_index']=data_total['cat1'].apply(lambda x: voca_mapping(x,voca[2],timestamp=False))
        # data_total['cat2_index']=data_total['cat2'].apply(lambda x: voca_mapping(x,voca[1],timestamp=False))
        data_total['type_index']=data_total['type'].apply(lambda x: voca_mapping(x,voca[0],timestamp=False))
        
        # -----여기에서 feature sture 정보를 가공 -------#
        # data_total['feature_store']=data_total.drop(columns=['unix_time','item','cat1','cat2','type','timestamp','item_index','cat1_index','cat2_index','type_index','svc_mgmt_num']) #컬럼은 더 확인해봐야함...ym?
        # data_total['feature_store'].apply(lambda x: x ~~~~)#전처리
        # data_total['feature_store']--> list의 텍스트 형태로 만들기--> '32,12,0.11,0.32' 이런식으로 콤마를 기준으로 feature를 나눈다.
        # 이후 아래 save_array 함수에 column feature_store를 추가하여 적용
        

        
        ###############################################
        
        
        data_total.to_pickle("data/"+strd_ym+"/pretrain/"+data_name+"_pretrain_dataset_voca_mapping_"+strd_ym+".pkl")
           
            
        
        
        # column_list_=['type_index','cat2_index','cat1_index','item_index','timestamp']
        column_list_=['type_index','item_index','timestamp']
        final_data=save_array(data=data_total,column_list=column_list_) # 5 x num_user x seq_len
        
        
        
        
        # 3. 전체 데이터셋 저장
        with open(data_name+"_list_dataset_"+str(strd_ym)+".pkl", "wb") as fp:   #Pickling
            pickle.dump(final_data, fp)
        # fp.close()
        
        # # 4. chunk 데이터셋 저장
        # num_gpus=8
        # total_len=len(final_data[0])
        # chunk_len=len(final_data[0])//num_gpus
        
        # chunk_index=[]
        # for index in range(num_gpus):
        #     index_list=[index*chunk_len,(index+1)*chunk_len]
        #     chunk_index.append(index_list)        
        
        # for k in tqdm.tqdm(range(num_gpus)):

        #     type_seq=final_data[0][chunk_index[k][0]:chunk_index[k][1]]
        #     cat2_seq=final_data[1][chunk_index[k][0]:chunk_index[k][1]]
        #     cat1_seq=final_data[2][chunk_index[k][0]:chunk_index[k][1]]
        #     item_seq=final_data[3][chunk_index[k][0]:chunk_index[k][1]]
        #     timestamp_seq=final_data[4][chunk_index[k][0]:chunk_index[k][1]]
        #     chunk_list=[type_seq,cat2_seq,cat1_seq,item_seq,timestamp_seq]

        #     with open(data_name+"_list_dataset_"+str(k)+"_"+str(strd_ym)+".pkl", "wb") as fp:   #Pickling
        #         pickle.dump(chunk_list, fp)
            
        
        
        

if __name__ == '__main__':
    main(type = args.type, data_size=args.data_size,max_len=args.max_len, min_len=args.min_len, strd_ym=args.strd_ym, data_name=args.data_name)
    
