import pickle
import random
import os
import sys
import random
import time
import numpy as np
import pandas as pd
import json
from dataset.vocab import WordVocab
import gc
import datetime
from dateutil.relativedelta import relativedelta
import argparse
from tqdm import tqdm

import pyarrow.parquet as pq
from pyhive import hive


from os import path

class Preprocessing:
    def __init__(self, strd_ym):

        self.strd_ym=strd_ym #202212 add
        
    def pretrain_get_data(self, data_size=20000000,min_len=32,max_len=1024, data_name='amazon'):# 64/1024
        print("Data Loading Start with",data_name)
        
        root_path='data/'+self.strd_ym+'/pretrain'
        if not os.path.exists(root_path):
            os.makedirs(root_path)        

        data_path=root_path+"/"+data_name+"_pretrain_dataset_"+self.strd_ym+".pkl" 
        if not path.exists(data_path):
            print('No file!')
            qry=f"""SELECT  svc_mgmt_num
            ,       array_to_string(array_agg(item),",","") as item
            ,       array_to_string(array_agg(type),",","") as type
            ,       array_to_string(array_agg(unix_time),",","") as unix_time
            ,       array_length(array_agg(item)) as list_len


            FROM
                (
                SELECT  svc_mgmt_num
                ,       regexp_replace(regexp_replace(regexp_replace(item,' ',''),'"',''),',','') as item
                ,       regexp_replace(regexp_replace(regexp_replace(type,' ',''),'"',''),',','') as type
                ,       unix_time
                FROM    database.svc_amazon_log_20231201
                ORDER BY svc_mgmt_num, cast(unix_time as int) asc
                ) as a
            GROUP BY svc_mgmt_num
            order by rand()
            """

            data=bq_to_pandas(qry)
            data.to_pickle(data_path)
        else:
            print('File already exists!')
            data=pd.read_pickle(data_path)

        return data

    
    
    def build_voca(self,voca_raw):
        voca = []
        for i in voca_raw:
            line = i.split(",")
            voca.extend(line)

        print('Voca_len: ',len(set(voca)))
        return voca, len(set(voca))


    def make_voca(self,data_name='amazon'):
        # conn = py_hive_connection()
        root_path='pretrained_model/voca'
        if not os.path.exists(root_path):
            os.makedirs(root_path)   

        data_path=root_path+"/"+data_name+"_voca_"+'item'+"_"+self.strd_ym+".ep" #폴더 자동생성하도록 차후 수정해야함

        if not path.exists(data_path):
            print('No file!')

            columns=['type','item']
            voca_set=[]
            for i in columns:
                print('voca dataset loading...')

                qry=f'''select regexp_replace(regexp_replace({i},' ',''),'"','') as {i} , count(*) as cnt from database.svc_amazon_log_20231201 where {i} is not null group by regexp_replace(regexp_replace({i},' ',''),'"','')'''
                data_raw=bq_to_pandas(qry)
                data_raw=data_raw.dropna().reset_index(drop=True)
                print('voca dataset loading complete!')
                len_voca = data_raw.shape[0]
                min_freq=0        

                print('Vocabularay Building...')
                voca = WordVocab(data_raw,min_freq=min_freq)

                output_voca_path = root_path+"/"+data_name+"_voca_"+i+"_"+self.strd_ym+".ep"
                voca.save_vocab(output_voca_path)

                voca = WordVocab.load_vocab(output_voca_path)
                print("최종 voca: ", len(voca))
                print("원래 voca: ", len_voca)
                voca_set.append(voca)
        else:
            print('Vocab already exist!')
            columns=['type','item']

            voca_set=[]
            len_voca=[]
            for i in columns:
                output_voca_path = root_path+"/"+data_name+"_voca_"+i+"_"+self.strd_ym+".ep"
                voca_tmp = WordVocab.load_vocab(output_voca_path)
                voca_set.append(voca_tmp)
                len_voca.append(len(voca_tmp))
            

        return voca_set     
        

    
    def pretrain_loader(self,data_size,min_len,max_len,data_name):

        data = self.pretrain_get_data(data_size,min_len,max_len,data_name)
        print(data['item'].head())

        voca = self.make_voca(data_name)
    
        return data, voca