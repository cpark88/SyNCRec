import pickle
import random
import os
import sys
import random
import time
import numpy as np
import pandas as pd
import json
import gc
import datetime
from dateutil.relativedelta import relativedelta
import argparse
from tqdm import tqdm
from os import path

from dataset.vocab import WordVocab

class Preprocessing:
    def __init__(self, strd_ym):

        self.strd_ym=strd_ym 
        
    def pretrain_get_data(self, data_size=20000000,min_len=32,max_len=1024, data_name='amazon'):# 64/1024
        print("Data Loading Start with",data_name)
        
        root_path='data/'+self.strd_ym+'/pretrain'
        if not os.path.exists(root_path):
            os.makedirs(root_path)        

        data_path=root_path+"/"+data_name+"_pretrain_dataset_"+self.strd_ym+".pkl" 
        if not path.exists(data_path):
            amazon_seq = pd.read_table('dataset/raw/amazon_seq.txt')
            amazon_seq=amazon_seq.drop(columns='Unnamed: 0')
            amazon_seq.to_pickle(data_path)
        else:
            print('File already exists!')
            amazon_seq=pd.read_pickle(data_path)

        return amazon_seq

    
    
    def build_voca(self,voca_raw):
        voca = []
        for i in voca_raw:
            line = i.split(",")
            voca.extend(line)

        print('Voca_len: ',len(set(voca)))
        return voca, len(set(voca))


    def make_voca(self,data_name='amazon'):
        root_path='pretrained_model/voca'
        if not os.path.exists(root_path):
            os.makedirs(root_path)   

        data_path=root_path+"/"+data_name+"_voca_"+'item'+"_"+self.strd_ym+".ep" 

        if not path.exists(data_path):
            print('No file!')

            columns=['type','item']
            voca_set=[]
            for i in columns:
                print('voca dataset loading...')

                voca = pd.read_table(f'dataset/raw/voca_{i}.txt')
                voca = voca.drop(columns='Unnamed: 0')
                min_freq=0        

                print('Vocabularay Building...')
                voca = WordVocab(voca, min_freq=min_freq)

                output_voca_path = root_path+"/"+data_name+"_voca_"+i+"_"+self.strd_ym+".ep"
                voca.save_vocab(output_voca_path)
                voca = WordVocab.load_vocab(output_voca_path)
                voca_set.append(voca)
        else:
            print('Vocab already exist!')
            columns=['type','item']

            voca_set=[]
            for i in columns:
                output_voca_path = root_path+"/"+data_name+"_voca_"+i+"_"+self.strd_ym+".ep"
                voca = WordVocab.load_vocab(output_voca_path)
                voca_set.append(voca)

        return voca_set
        

    
    def pretrain_loader(self,data_size,min_len,max_len,data_name):

        data = self.pretrain_get_data(data_size,min_len,max_len,data_name)
        print(data['item'].head())

        voca = self.make_voca(data_name)
    
        return data, voca