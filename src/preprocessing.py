import pickle
import random
import os
import sys
import random
import time
import numpy as np
import pandas as pd
# from skinet.utils.hive_tools import py_hive_connection, dbGetQuery, load_from_hive, dbSendUpdate, load_to_hive
import json
from dataset.vocab import WordVocab
# import torch
import gc
import datetime
from dateutil.relativedelta import relativedelta
import argparse
from tqdm import tqdm
# conn = py_hive_connection()


########

import pyarrow.parquet as pq
from pyhive import hive

from skt.mls import *
from skt.ye import hive_execute, get_hdfs_conn, hive_to_pandas, pandas_to_parquet, slack_send, get_spark, get_hdfs_conn

from skt.gcp import get_bigquery_client, bq_insert_overwrite, get_max_part, bq_to_df, bq_to_pandas, pandas_to_bq_table, load_query_result_to_table, PROJECT_ID


########


from os import path

class Preprocessing:
    def __init__(self, strd_ym):

        self.strd_ym=strd_ym #202212 add
        
    def pretrain_get_data(self, data_size=20000000,min_len=32,max_len=1024, data_name='skt'):# 64/1024
        print("Data Loading Start with",data_name)
        
        if data_name=='skt':
            # conn = py_hive_connection()
            bq = get_bigquery_client()
            #root_path가 없으면 만들고 그 안에 data 적재 
            root_path='data/'+self.strd_ym+'/pretrain' 
            if not os.path.exists(root_path):
                os.makedirs(root_path)        
            data_path=root_path+"/"+data_name+"_pretrain_dataset_"+self.strd_ym+".pkl" 
            

            if not path.exists(data_path):
                print('No file!')
                
                #서번 앞 3자리를 잘라 여러번에 걸처 데이터 로딩 (터짐 방지), 전체 데이터 양을 입력하면 그에 맞게 각 파트들의 데이터 양도 조정됨.
                sub_token_length=1#3

                qry=f"select * from (select  substr(svc_mgmt_num,1,{sub_token_length}) as token, count(*) as cnt from x1112020.svc_{data_name}_log_list_temp_v4  group by substr(svc_mgmt_num,1,{sub_token_length})) as a order by cnt"
                token=bq_to_pandas(qry)
                
                

                token['cnt_int']=(token['cnt']*(data_size/token['cnt'].sum())).astype(int) #data_size에 비례해서 조정
                print(token)

                data=pd.DataFrame()
                # token=token.head(2)
                for i in tqdm(token.values):


                    print('data_size_token:',i[0],i[1],i[2])
                    data_tmp = bq_to_pandas(f"select * from x1112020.svc_{data_name}_log_list_temp_v4 where list_len>={min_len} and list_len<={max_len} and substr(svc_mgmt_num,1,{sub_token_length}) in ("+ "'"+f"{i[0]}" + "'"+f") limit {i[2]}")
                    
                    
                    
                    ###수정중 
                    
                    
                    
                    print('Data shape of token',i,':',data_tmp.shape)

                    data=data.append(data_tmp)
                    
                data=data.sample(frac=1) #shuffling
                data=data.reset_index(drop=True)
                data.to_pickle(data_path)

                print('Data shape',data.shape)

                print("Data Loading Complete!")

            else:
                print('File already exists!')
                data=pd.read_pickle(data_path)
                
        else:#amazon
            # conn = py_hive_connection()
            bq = get_bigquery_client()

            root_path='data/'+self.strd_ym+'/pretrain'
            if not os.path.exists(root_path):
                os.makedirs(root_path)        
            
            data_path=root_path+"/"+data_name+"_pretrain_dataset_"+self.strd_ym+".pkl" 
            if not path.exists(data_path):
                print('No file!')
                # data = load_from_hive(conn,f"select svc_mgmt_num, item, cat1, cat2, type, unix_time from default.svc_amazon_log_list_temp") #무조건 load_from_hive (리스트 내 괄호가 제거되어 나옴)
                
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
                    FROM    x1112020.svc_amazon_log_20231201
                    ORDER BY svc_mgmt_num, cast(unix_time as int) asc
                    ) as a
                GROUP BY svc_mgmt_num
                order by rand()
                """

                data=bq_to_pandas(qry)
                # data=data.sample(frac=1) #shuffling
                # data=data.reset_index(drop=True)
                
                
                
                
                
                data.to_pickle(data_path)
            else:
                print('File already exists!')
                data=pd.read_pickle(data_path)

        return data

    
    
    def build_voca(self,voca_raw):
        voca = []
        for i in voca_raw:
            # i = i.replace("[","").replace("]","").replace('"',"").replace(' ',"").replace("'","") #[ ] 가 이제 없음
            line = i.split(",")
            voca.extend(line)

        print('Voca_len: ',len(set(voca)))
        return voca, len(set(voca))


    def make_voca(self,data_name='skt'):
        # conn = py_hive_connection()
        bq = get_bigquery_client()
        root_path='pretrained_model/voca'
        if not os.path.exists(root_path):
            os.makedirs(root_path)   
            
            
        if data_name=='skt':    
            data_path=root_path+"/"+data_name+"_voca_"+'item'+"_"+self.strd_ym+".ep" #폴더 자동생성하도록 차후 수정해야함

            if not path.exists(data_path):
                print('No file!')
                
                
                columns=['type','cat2','cat1','item']
                voca_set=[]
                for i in columns:
                    print('voca dataset loading...')
                    # data_raw=dbGetQuery(conn,f"select {i}, count(*) as cnt from di_mno_offer.big_model_features where {i} not in ('','#',' ','#|#') and {i} is not null group by {i}")



                #item이 기타 일시 cat1으로 대체 

                    data_raw=bq_to_pandas(f"select if({i}='기타',cat1,{i}) as {i}, count(*) as cnt from x1112020.svc_skt_log_temp_v4 where {i} not in ('','#',' ','#|#') and {i} is not null and type not in ('prod', 'prod2') group by if({i}='기타',cat1,{i})")
                # #######


                    data_raw=data_raw.dropna().reset_index(drop=True)
                    print('voca dataset loading complete!')
                    # voca_ = list(data_raw[i])
                    len_voca = data_raw.shape[0]
                    # voca_freq = pd.DataFrame(voca_,columns=[i]).groupby(i)[i].count()
            #             min_freq = voca_freq.quantile(q=[0.01,0.05,0.1,0.2,0.3,0.5,0.55,0.6,0.75,0.99,1])[0.2]
                    min_freq=0        

                    print('Vocabularay Building...')
                    voca = WordVocab(data_raw,min_freq=min_freq)

                    output_voca_path = root_path+"/"+data_name+"_voca_"+i+"_"+self.strd_ym+".ep"#param['voca_path']
                    voca.save_vocab(output_voca_path)

                    voca = WordVocab.load_vocab(output_voca_path)
                    print("최종 voca: ", len(voca))
                    print("원래 voca: ", len_voca)
                    voca_set.append(voca)
            else:
                print('Vocab already exist!')
                columns=['type','cat2','cat1','item']

                voca_set=[]
                len_voca=[]
                for i in columns:
                    output_voca_path =root_path+"/"+data_name+"_voca_"+i+"_"+self.strd_ym+".ep"
                    voca_tmp = WordVocab.load_vocab(output_voca_path)
                    voca_set.append(voca_tmp)
                    len_voca.append(len(voca_tmp))
                    
                    
        else: # amanzon case
            data_path=root_path+"/"+data_name+"_voca_"+'item'+"_"+self.strd_ym+".ep" #폴더 자동생성하도록 차후 수정해야함

            if not path.exists(data_path):
                print('No file!')

                # columns=['type','cat2','cat1','item']
                columns=['type','item']
                voca_set=[]
                for i in columns:
                    print('voca dataset loading...')
                    # data_raw=load_from_hive(conn,f'''select regexp_replace(regexp_replace({i},' ',''),'"','') as {i} , count(*) as cnt from default.svc_amazon_log_temp where {i} is not null group by regexp_replace(regexp_replace({i},' ',''),'"','')''') #리스트 형태에서는 띄어쓰기가 무시되고 붙지만,여기서는 그대로 반영됨
                    
                    qry=f'''select regexp_replace(regexp_replace({i},' ',''),'"','') as {i} , count(*) as cnt from x1112020.svc_amazon_log_20231201 where {i} is not null group by regexp_replace(regexp_replace({i},' ',''),'"','')'''
                    data_raw=bq_to_pandas(qry)
                    
                    

                    data_raw=data_raw.dropna().reset_index(drop=True)
                    print('voca dataset loading complete!')
                    # voca_ = list(data_raw[i])
                    len_voca = data_raw.shape[0]
                    # voca_freq = pd.DataFrame(voca_,columns=[i]).groupby(i)[i].count()
            #             min_freq = voca_freq.quantile(q=[0.01,0.05,0.1,0.2,0.3,0.5,0.55,0.6,0.75,0.99,1])[0.2]
                    min_freq=0        

                    print('Vocabularay Building...')
                    voca = WordVocab(data_raw,min_freq=min_freq)

                    output_voca_path = root_path+"/"+data_name+"_voca_"+i+"_"+self.strd_ym+".ep"#param['voca_path']
                    voca.save_vocab(output_voca_path)

                    voca = WordVocab.load_vocab(output_voca_path)
                    print("최종 voca: ", len(voca))
                    print("원래 voca: ", len_voca)
                    voca_set.append(voca)
            else:
                print('Vocab already exist!')
                # columns=['type','cat2','cat1','item']
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