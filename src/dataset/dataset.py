import gzip
import os
import tqdm
import torch
import random
from torch.utils.data import Dataset
from random import choice
import numpy as np
import pandas as pd

################################################################################################
#1. Pretraining 
################################################################################################

def neg_sample(item_set, item_size):  
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


# unidirectional version -- array version 202304
class BERTDatasetPretrain(Dataset):
    def __init__(self, user_seq,vocab):#utf-8
        self.user_seq=user_seq # 5 x num user x 2 x seq_len (데이터 타입수, 유저수, 인풋/아웃풋, seq 길이)
        self.voca=vocab
        
        
    def __len__(self):
        return self.user_seq.shape[1]

    def __getitem__(self, index):
        
        sequence = self.user_seq[:,index,:,:] # 5 x num_user x 2 x seq_len -->5 x 2 x seq_len --> type, cat2, cat1, item, timestamp 순서
        
        
        speical_tokens=np.array([0,1,2,3,4])
        bert_neg_1 = []
        item_set = set(np.append(sequence[1][0],speical_tokens)) #cat1 & input 
        for item in sequence[1][0]:
            bert_neg_1.append(neg_sample(item_set, len(self.voca[1]))) 
            
        bert_neg_2 = []
        item_set = set(np.append(sequence[2][0],speical_tokens)) #cat1 & input 
        for item in sequence[2][0]:
            bert_neg_2.append(neg_sample(item_set, len(self.voca[2])))
            
        bert_neg_3 = []
        item_set = set(np.append(sequence[3][0],speical_tokens)) #cat1 & input 
        for item in sequence[3][0]:
            bert_neg_3.append(neg_sample(item_set, len(self.voca[3]))) 
            
        bert_test_neg_3 = []
        item_set =set(np.append(sequence[3][0],speical_tokens)) #cat1 & input 
        for item in range(100):
            bert_test_neg_3.append(neg_sample(item_set, len(self.voca[3]))) 
        
        
        
        

        output = {
                  "bert_input_0": sequence[0][0], # type정보
                  "bert_label_0": sequence[0][1],
                  "bert_input_1": sequence[1][0],
                  "bert_label_1": sequence[1][1],
                  "bert_input_2": sequence[2][0],
                  "bert_label_2": sequence[2][1],
                  "bert_input_3": sequence[3][0],
                  "bert_label_3": sequence[3][1],#type은 맞추지 않을 것이므로 제외

                  "bert_temporal_enc":sequence[4][0],
                  
                  # "bert_neg_0" : neg_0,#원래 시퀀스로 대체
                  "bert_neg_1" : bert_neg_1,
                  "bert_neg_2" : bert_neg_2,
                  "bert_neg_3" : bert_neg_3,
                  
                  
                  # "bert_test_neg_0" : test_neg_0,
                  # "bert_test_neg_1" : test_neg_1,
                  # "bert_test_neg_2" : test_neg_2,
            
                  "bert_test_neg_3" : bert_test_neg_3,
            
                  # "bert_test_neg_4" : test_neg_4,
            
                  # "bert_input_0_shuffle" : bert_input_0_shuffle, #contrastive version엔 필수
                  # "bert_input_1_shuffle" : bert_input_1_shuffle, #contrastive version엔 필수
                  # "bert_input_2_shuffle" : bert_input_2_shuffle, #contrastive version엔 필수
                  # "bert_input_3_shuffle" : bert_input_3_shuffle, #contrastive version엔 필수                
            
            
                  
                  
                  
                  
                 }

        # return {key: torch.tensor(value,dtype=torch.int32) for key, value in output.items()}
        return {key: torch.tensor(value) for key, value in output.items()}

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# # unidirectional version
# class BERTDatasetPretrain(Dataset):
#     def __init__(self, text, vocab, seq_len, encoding="cp949", corpus_lines=None):#utf-8
#         self.vocab_0 = vocab[0]
#         self.vocab_1 = vocab[1]
#         self.vocab_2 = vocab[2]
#         self.vocab_3 = vocab[3]
        
#         self.seq_len = seq_len #max_len
        
#         self.corpus_lines = corpus_lines
#         self.encoding = encoding
        
        
#         self.text = text
        
    
# #         self.category0 = [line for line in tqdm.tqdm(self.text['type'], desc="Loading Dataset", total=corpus_lines)]
# #         self.category1 = [line for line in tqdm.tqdm(self.text['cat2'], desc="Loading Dataset", total=corpus_lines)]
# #         self.category2 = [line for line in tqdm.tqdm(self.text['cat1'], desc="Loading Dataset", total=corpus_lines)]
# #         self.category3 = [line for line in tqdm.tqdm(self.text['item'], desc="Loading Dataset", total=corpus_lines)]
# #         self.unix_time = [line for line in tqdm.tqdm(self.text['unix_time'], desc="Loading Dataset", total=corpus_lines)] #202212 add

        
    
#         self.category0 = [line for line in tqdm.tqdm(text['type'], desc="Loading Dataset", total=corpus_lines)]
#         self.category1 = [line for line in tqdm.tqdm(text['cat2'], desc="Loading Dataset", total=corpus_lines)]
#         self.category2 = [line for line in tqdm.tqdm(text['cat1'], desc="Loading Dataset", total=corpus_lines)]
#         self.category3 = [line for line in tqdm.tqdm(text['item'], desc="Loading Dataset", total=corpus_lines)]
#         self.unix_time = [line for line in tqdm.tqdm(text['unix_time'], desc="Loading Dataset", total=corpus_lines)] #202212 add
        
        
# #         #11st
# #         self.category0 = [line for line in tqdm.tqdm(self.text['action_type'], desc="Loading Dataset", total=corpus_lines)]
# #         self.category1 = [line for line in tqdm.tqdm(self.text['category1'], desc="Loading Dataset", total=corpus_lines)]
# #         self.category2 = [line for line in tqdm.tqdm(self.text['category2'], desc="Loading Dataset", total=corpus_lines)]
# #         self.category3 = [line for line in tqdm.tqdm(self.text['category3'], desc="Loading Dataset", total=corpus_lines)]
# #         # self.category4 = [line for line in tqdm.tqdm(self.text['category4'], desc="Loading Dataset", total=corpus_lines)]

# #         self.unix_time = [line for line in tqdm.tqdm(self.text['unix_time'], desc="Loading Dataset", total=corpus_lines)] #202212 add
        
        
        

        
        

#         self.corpus_lines = len(self.category1) #lines
#         # self.corpus_lines = len(category1) #lines
        
#     def __len__(self):
#         return self.corpus_lines

#     def __getitem__(self, item):
 
#         t0, t1, t2, t3, temporal = self.random_sent(item)


#         # t0_random, t0_label, t1_random, t1_label, t2_random, t2_label, t3_random, t3_label,  temporal ,neg_1,neg_2,neg_3, tokens_0_shuffle,tokens_1_shuffle,tokens_2_shuffle,tokens_3_shuffle, test_neg_3 = self.random_word([t0, t1, t2, t3,temporal])

#         t0, t0_label, t1, t1_label, t2, t2_label, t3, t3_label,  temporal ,neg_1,neg_2,neg_3, test_neg_3 = self.random_word([t0, t1, t2, t3,temporal])

        
#         # t0_shuffle = tokens_0_shuffle
#         # bert_input_0_shuffle = (t0_shuffle)[-self.seq_len:]#[:self.seq_len]
#         # t1_shuffle = tokens_1_shuffle
#         # bert_input_1_shuffle = (t1_shuffle)[-self.seq_len:]#[:self.seq_len]        
#         # t2_shuffle = tokens_2_shuffle
#         # bert_input_2_shuffle = (t2_shuffle)[-self.seq_len:]#[:self.seq_len]        
#         # t3_shuffle = tokens_3_shuffle
#         # bert_input_3_shuffle = (t3_shuffle)[-self.seq_len:]#[:self.seq_len]        
        
#         # neg_0 = (neg_0)[-self.seq_len:] # max_len limit  (neg_0)[:self.seq_len]은 이전버전, 현재는 가장 최근 seq를 가지고 오도록함.
#         neg_1 = (neg_1)[-self.seq_len:] # max_len limit 
#         neg_2 = (neg_2)[-self.seq_len:] # max_len limit 
#         neg_3 = (neg_3)[-self.seq_len:] # max_len limit 

       
#         # t0 = t0_random #+ [self.vocab_0.eos_index]
#         # t0_label = t0_label# + [self.vocab_0.eos_index]
#         bert_input_0 = (t0)[-self.seq_len:] # max_len limit 
#         bert_label_0 = (t0_label)[-self.seq_len:]
         
#         # t1 = t1_random# + [self.vocab_1.eos_index]
#         # t1_label =  t1_label# + [self.vocab_1.pad_index]
#         bert_input_1 = (t1)[-self.seq_len:]
#         bert_label_1 = (t1_label)[-self.seq_len:]
        
#         # t2 = t2_random #+ [self.vocab_2.eos_index]
#         # t2_label = t2_label# + [self.vocab_2.eos_index]
#         bert_input_2 = (t2)[-self.seq_len:]
#         bert_label_2 = (t2_label)[-self.seq_len:]
        
#         # t3 = t3_random #+ [self.vocab_3.eos_index]
#         # t3_label = t3_label #+ [self.vocab_3.eos_index]
#         bert_input_3 = (t3)[-self.seq_len:]
#         bert_label_3 = (t3_label)[-self.seq_len:]
        
#         temporal_enc = temporal #+ [self.vocab_1.eos_index]
#         temporal_enc = (temporal_enc)[-self.seq_len:]

#         padding_0 = [self.vocab_0.pad_index for _ in range(self.seq_len - len(bert_input_0))]
#         padding_1 = [self.vocab_1.pad_index for _ in range(self.seq_len - len(bert_input_1))]
#         padding_2 = [self.vocab_2.pad_index for _ in range(self.seq_len - len(bert_input_2))]
#         padding_3 = [self.vocab_3.pad_index for _ in range(self.seq_len - len(bert_input_3))]
#         padding_4 = [self.vocab_1.pad_index for _ in range(self.seq_len - len(temporal_enc))]
        
#         # padding을 앞으로 배치 
#         # bert_input_0_shuffle=padding_0+bert_input_0_shuffle
#         # bert_input_1_shuffle=padding_1+bert_input_1_shuffle
#         # bert_input_2_shuffle=padding_2+bert_input_2_shuffle
#         # bert_input_3_shuffle=padding_3+bert_input_3_shuffle
        
        
        
#         # neg_0=padding_0+neg_0
#         neg_1=padding_1+neg_1
#         neg_2=padding_2+neg_2
#         neg_3=padding_3+neg_3
        
        
        
#         bert_input_0=padding_0+bert_input_0
#         bert_label_0=padding_0+bert_label_0
        
#         bert_input_1=padding_1+bert_input_1
#         bert_label_1=padding_1+bert_label_1
        
#         bert_input_2=padding_2+bert_input_2
#         bert_label_2=padding_2+bert_label_2
        
#         bert_input_3=padding_3+bert_input_3
#         bert_label_3=padding_3+bert_label_3
    
        
#         temporal_enc=padding_4+temporal_enc

        

#         output = {
#                   "bert_input_0": bert_input_0, # type정보
#                   "bert_label_0": bert_label_0,
#                   "bert_input_1": bert_input_1,
#                   "bert_label_1": bert_label_1,
#                   "bert_input_2": bert_input_2,
#                   "bert_label_2": bert_label_2,
#                   "bert_input_3": bert_input_3,
#                   "bert_label_3": bert_label_3,#type은 맞추지 않을 것이므로 제외

#                   "bert_temporal_enc":temporal_enc,
                  
#                   # "bert_neg_0" : neg_0,#원래 시퀀스로 대체
#                   "bert_neg_1" : neg_1,
#                   "bert_neg_2" : neg_2,
#                   "bert_neg_3" : neg_3,
                  
                  
#                   # "bert_test_neg_0" : test_neg_0,
#                   # "bert_test_neg_1" : test_neg_1,
#                   # "bert_test_neg_2" : test_neg_2,
            
#                   "bert_test_neg_3" : test_neg_3,
            
#                   # "bert_test_neg_4" : test_neg_4,
            
#                   # "bert_input_0_shuffle" : bert_input_0_shuffle, #contrastive version엔 필수
#                   # "bert_input_1_shuffle" : bert_input_1_shuffle, #contrastive version엔 필수
#                   # "bert_input_2_shuffle" : bert_input_2_shuffle, #contrastive version엔 필수
#                   # "bert_input_3_shuffle" : bert_input_3_shuffle, #contrastive version엔 필수                
            
            
                  
                  
                  
                  
#                  }

#         # return {key: torch.tensor(value,dtype=torch.int32) for key, value in output.items()}
#         return {key: torch.tensor(value) for key, value in output.items()}

    
    


#     def random_word(self, sentence):
        
#         bf_ym='202301'   

#         tokens_0=sentence[0].split(',') # input data --> 정수  
#         # index=np.where([i!='btv' and i!='tmap' for i in tokens_0]) # cdr, btv, tmap, 11st, tmbr, xdr #6:btv, 7: cdr, 8:tmbr, 9:tmap, 10:11st

#         # tokens_0=np.array(tokens_0)[index].tolist()
        
        
        
#         tokens_1=sentence[1].split(',')
#         # tokens_1=np.array(tokens_1)[index].tolist()
        
#         tokens_2=sentence[2].split(',')
#         # tokens_2=np.array(tokens_2)[index].tolist()
        
#         tokens_3=sentence[3].split(',')
#         # tokens_3=np.array(tokens_3)[index].tolist()
        
#         temporal=sentence[4].split(',') # 202212 add
#         # temporal=np.array(temporal)[index].tolist()
        
#         temporal=[int(i) for i in temporal] # 202212 add
    
    
    
        
#         output_label_0 = []
#         output_label_1 = []
#         output_label_2 = []
#         output_label_3 = []


#         for i, token in enumerate(zip(tokens_0,tokens_1,tokens_2,tokens_3)):
            
#             # item=='기타' 인 경우 cat1으로 대체 
#             if token[3]=='기타':
#                 tokens_3[i]=token[2]
                
            
            
#             tokens_0[i] = self.vocab_0.stoi.get(token[0], self.vocab_0.unk_index)
#             tokens_1[i] = self.vocab_1.stoi.get(token[1], self.vocab_1.unk_index) # item명을 정수 index로 변환하고, 없으면 Unk_index로 치환해라.
#             tokens_2[i] = self.vocab_2.stoi.get(token[2], self.vocab_2.unk_index)
#             tokens_3[i] = self.vocab_3.stoi.get(token[3], self.vocab_3.unk_index)

            
            
            
#         tokens_0_input=tokens_0[:-1] # 마지막 하나는 제외
#         tokens_1_input=tokens_1[:-1] # 마지막 하나는 제외
#         tokens_2_input=tokens_2[:-1] # 마지막 하나는 제외
#         tokens_3_input=tokens_3[:-1] # 마지막 하나는 제외
#         temporal=temporal[:-1]
        

#         output_label_0=tokens_0[1:] # 한 item씩 shifted
#         output_label_1=tokens_1[1:] # 한 item씩 shifted
#         output_label_2=tokens_2[1:] # 한 item씩 shifted
#         output_label_3=tokens_3[1:] # 한 item씩 shifted
        
        

       
        
#         # train용 negative
#         # neg_0 = list(np.zeros(shape=np.shape(tokens_0_input)))
#         neg_1 = list(np.zeros(shape=np.shape(tokens_1_input)))
#         neg_2 = list(np.zeros(shape=np.shape(tokens_2_input)))
#         neg_3 = list(np.zeros(shape=np.shape(tokens_3_input)))

        
#         token_1_diff=set(range(5,len(self.vocab_1))).difference(tokens_1_input)
#         token_2_diff=set(range(5,len(self.vocab_2))).difference(tokens_2_input)
#         token_3_diff=set(range(5,len(self.vocab_3))).difference(tokens_3_input)
        
        
#         for k, token in enumerate(zip(tokens_1_input)):
#             neg_1[k] = choice([i for i in token_1_diff])
#             neg_2[k] = choice([i for i in token_2_diff])
#             neg_3[k] = choice([i for i in token_3_diff])
            
 
            
# #         # test용 100개 negative
#         test_neg_3 = list(np.zeros(shape=(100,)))
        
#         for k in range(100):
#             test_neg_3[k] = choice([i for i in token_3_diff])      
      
#         temporal=[0 for i in temporal]

#         return tokens_0_input,output_label_0,tokens_1_input,output_label_1,tokens_2_input,output_label_2,tokens_3_input,output_label_3,temporal,neg_1,neg_2,neg_3, test_neg_3
        

        
        
#     def random_sent(self, index):
#         t0,t1,t2,t3,u = self.get_corpus_line(index)
#         return t0,t1,t2,t3,u


#     def get_corpus_line(self, item):

#         return self.category0[item],self.category1[item],self.category2[item], self.category3[item], self.unix_time[item]
        


################################################################################################
#2. Finetuning
################################################################################################




class BERTDatasetFinetune(Dataset):
    def __init__(self, text, vocab, seq_len, encoding="utf-8", corpus_lines=None):
        self.vocab_0 = vocab[0]
        self.vocab_1 = vocab[1]
        self.vocab_2 = vocab[2]
        self.vocab_3 = vocab[3]
        self.vocab_4 = vocab[4]
        
  
        self.seq_len = seq_len #max_len
        
        self.corpus_lines = corpus_lines
        self.encoding = encoding
        self.text = text
        
    
        self.category0 = [line for line in tqdm.tqdm(self.text['action_type'], desc="Loading Dataset", total=corpus_lines)]
        self.category1 = [line for line in tqdm.tqdm(self.text['category1'], desc="Loading Dataset", total=corpus_lines)]
        self.category2 = [line for line in tqdm.tqdm(self.text['category2'], desc="Loading Dataset", total=corpus_lines)]
        self.category3 = [line for line in tqdm.tqdm(self.text['category3'], desc="Loading Dataset", total=corpus_lines)]
        self.category4 = [line for line in tqdm.tqdm(self.text['category4'], desc="Loading Dataset", total=corpus_lines)]

        self.unix_time = [line for line in tqdm.tqdm(self.text['unix_time'], desc="Loading Dataset", total=corpus_lines)] #202212 add
        
        

        self.corpus_lines = len(self.category1) #lines       
        
    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
 
        t0, t1, t2, t3, t4,temporal = self.random_sent(item)

        t0_random, t0_label, t1_random, t1_label, t2_random, t2_label, t3_random, t3_label, t4_random, t4_label, temporal,neg_0,neg_1,neg_2,neg_3,neg_4,test_neg_0,test_neg_1,test_neg_2,test_neg_3,test_neg_4 = self.random_word([t0, t1, t2, t3, t4,temporal])

        
        
        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        # t0 = [self.vocab_0.sos_index] + t0_random #+ [self.vocab_0.eos_index]
        # t0_label = [self.vocab_0.sos_index] + t0_label# + [self.vocab_0.eos_index]
        
        
        # neg_0 = neg_0 #+ [self.vocab_0.eos_index]
        neg_0 = (neg_0)[-self.seq_len:] # max_len limit  (neg_0)[:self.seq_len]은 이전버전, 현재는 가장 최근 seq를 가지고 오도록함.
        neg_1 = (neg_1)[-self.seq_len:] # max_len limit 
        neg_2 = (neg_2)[-self.seq_len:] # max_len limit 
        neg_3 = (neg_3)[-self.seq_len:] # max_len limit 
        neg_4 = (neg_4)[-self.seq_len:] # max_len limit 
       
        t0 = t0_random #+ [self.vocab_0.eos_index]
        t0_label = t0_label# + [self.vocab_0.eos_index]
        bert_input_0 = (t0)[-self.seq_len:] # max_len limit 
        bert_label_0 = (t0_label)[-self.seq_len:]
         
        t1 = t1_random# + [self.vocab_1.eos_index]
        t1_label =  t1_label# + [self.vocab_1.pad_index]
        bert_input_1 = (t1)[-self.seq_len:]
        bert_label_1 = (t1_label)[-self.seq_len:]
        
        t2 = t2_random #+ [self.vocab_2.eos_index]
        t2_label = t2_label# + [self.vocab_2.eos_index]
        bert_input_2 = (t2)[-self.seq_len:]
        bert_label_2 = (t2_label)[-self.seq_len:]
        
        t3 = t3_random #+ [self.vocab_3.eos_index]
        t3_label = t3_label #+ [self.vocab_3.eos_index]
        bert_input_3 = (t3)[-self.seq_len:]
        bert_label_3 = (t3_label)[-self.seq_len:]

        t4 = t4_random# + [self.vocab_4.eos_index]
        t4_label =  t4_label #+ [self.vocab_4.eos_index]
        bert_input_4 = (t4)[-self.seq_len:]
        bert_label_4 = (t4_label)[-self.seq_len:]
        
        temporal_enc = temporal #+ [self.vocab_1.eos_index]
        temporal_enc = (temporal_enc)[-self.seq_len:]

        padding_0 = [self.vocab_0.pad_index for _ in range(self.seq_len - len(bert_input_0))]
        padding_1 = [self.vocab_1.pad_index for _ in range(self.seq_len - len(bert_input_1))]
        padding_2 = [self.vocab_2.pad_index for _ in range(self.seq_len - len(bert_input_2))]
        padding_3 = [self.vocab_3.pad_index for _ in range(self.seq_len - len(bert_input_3))]
        padding_4 = [self.vocab_4.pad_index for _ in range(self.seq_len - len(bert_input_4))]
        padding_5 = [self.vocab_1.pad_index for _ in range(self.seq_len - len(temporal_enc))]
        
        # padding을 앞으로 배치 
        neg_0=padding_0+neg_0
        neg_1=padding_1+neg_1
        neg_2=padding_2+neg_2
        neg_3=padding_3+neg_3
        neg_4=padding_4+neg_4
        
        
        
        bert_input_0=padding_0+bert_input_0
        bert_label_0=padding_0+bert_label_0
        
        bert_input_1=padding_1+bert_input_1
        bert_label_1=padding_1+bert_label_1
        
        bert_input_2=padding_2+bert_input_2
        bert_label_2=padding_2+bert_label_2
        
        bert_input_3=padding_3+bert_input_3
        bert_label_3=padding_3+bert_label_3
        
        bert_input_4=padding_4+bert_input_4
        bert_label_4=padding_4+bert_label_4
        
        temporal_enc=padding_5+temporal_enc

        

        output = {
                  # "bert_input_0": bert_input_0,
                  # "bert_label_0": bert_label_0,
                  "bert_input_1": bert_input_1,
                  "bert_label_1": bert_label_1,
                  "bert_input_2": bert_input_2,
                  "bert_label_2": bert_label_2,
                  "bert_input_3": bert_input_3,
                  "bert_label_3": bert_label_3,
                  # "bert_input_4": bert_input_4,
                  # "bert_label_4": bert_label_4,

                  "bert_temporal_enc":temporal_enc,
                  
                  # "bert_neg_0" : neg_0,
                  "bert_neg_1" : neg_1,
                  "bert_neg_2" : neg_2,
                  "bert_neg_3" : neg_3,
                  # "bert_neg_4" : neg_4,
                  
                  
                  # "bert_test_neg_0" : test_neg_0,
                  # "bert_test_neg_1" : test_neg_1,
                  # "bert_test_neg_2" : test_neg_2,
                  "bert_test_neg_3" : test_neg_3,
                  # "bert_test_neg_4" : test_neg_4,
                  
                  
                  
                  
                 }

        return {key: torch.tensor(value) for key, value in output.items()}

    
    


    def random_word(self, sentence):
        
        bf_ym='202301'
        # category_meta = pd.read_pickle("~/myfiles/t1112020_gvol/2023/sequential_recommendation/gt_reference/pretrained_model/voca/category_meta"+bf_ym+".pkl")        
        tokens_0=sentence[0].split(',') # input data --> 정수  
        tokens_1=sentence[1].split(',')  
        tokens_2=sentence[2].split(',') 
        tokens_3=sentence[3].split(',') 
        tokens_4=sentence[4].split(',') 
        
        temporal=sentence[5].split(',') # 202212 add
        temporal=[int(i) for i in temporal] # 202212 add
        

        
        
        output_label_0 = []
        output_label_1 = []
        output_label_2 = []
        output_label_3 = []
        output_label_4 = []


        for i, token in enumerate(zip(tokens_0,tokens_1,tokens_2,tokens_3,tokens_4)):
            
            
            tokens_0[i] = self.vocab_0.stoi.get(token[0], self.vocab_0.unk_index)
            tokens_1[i] = self.vocab_1.stoi.get(token[1], self.vocab_1.unk_index) # item명을 정수 index로 변환하고, 없으면 Unk_index로 치환해라.
            tokens_2[i] = self.vocab_2.stoi.get(token[2], self.vocab_2.unk_index)
            tokens_3[i] = self.vocab_3.stoi.get(token[3], self.vocab_3.unk_index)
            tokens_4[i] = self.vocab_4.stoi.get(token[4], self.vocab_4.unk_index)
            

        tokens_0_input=tokens_0[:-1] # 마지막 하나는 제외
        tokens_1_input=tokens_1[:-1] # 마지막 하나는 제외
        tokens_2_input=tokens_2[:-1] # 마지막 하나는 제외
        tokens_3_input=tokens_3[:-1] # 마지막 하나는 제외
        tokens_4_input=tokens_4[:-1] # 마지막 하나는 제외
        temporal=temporal[:-1]
        

        output_label_0=tokens_0[1:] # 한 item씩 shifted
        output_label_1=tokens_1[1:] # 한 item씩 shifted
        output_label_2=tokens_2[1:] # 한 item씩 shifted
        output_label_3=tokens_3[1:] # 한 item씩 shifted
        output_label_4=tokens_4[1:] # 한 item씩 shifted
        
        
        # test용 100개 negative
        test_neg_0 = list(np.zeros(shape=(100,)))
        test_neg_1 = list(np.zeros(shape=(100,)))
        test_neg_2 = list(np.zeros(shape=(100,)))
        test_neg_3 = list(np.zeros(shape=(100,)))
        test_neg_4 = list(np.zeros(shape=(100,)))
        
        for k in range(100):
            test_neg_0[k] = 0#choice([i for i in range(5,len(self.vocab_0)) if i not in tokens_0_input])
            test_neg_1[k] = choice([i for i in range(5,len(self.vocab_1)) if i not in tokens_1_input])
            test_neg_2[k] = choice([i for i in range(5,len(self.vocab_2)) if i not in tokens_2_input])
            test_neg_3[k] = choice([i for i in range(5,len(self.vocab_3)) if i not in tokens_3_input])
            test_neg_4[k] = 0#choice([i for i in range(5,len(self.vocab_4)) if i not in tokens_4_input])        
        
        # train용 negative
        neg_0 = list(np.zeros(shape=np.shape(tokens_0_input)))
        neg_1 = list(np.zeros(shape=np.shape(tokens_1_input)))
        neg_2 = list(np.zeros(shape=np.shape(tokens_2_input)))
        neg_3 = list(np.zeros(shape=np.shape(tokens_3_input)))
        neg_4 = list(np.zeros(shape=np.shape(tokens_4_input)))
        
        
        
        for k, token in enumerate(zip(tokens_1_input)):
            neg_0[k] = 0#choice([i for i in range(5,len(self.vocab_0)) if i not in tokens_0_input])
            neg_1[k] = choice([i for i in range(5,len(self.vocab_1)) if i not in tokens_1_input])
            neg_2[k] = choice([i for i in range(5,len(self.vocab_2)) if i not in tokens_2_input])
            neg_3[k] = choice([i for i in range(5,len(self.vocab_3)) if i not in tokens_3_input])
            neg_4[k] = 0#choice([i for i in range(5,len(self.vocab_4)) if i not in tokens_4_input])
        
        
#         # hierarhical negative sampling
#         for i, token in enumerate(zip(tokens_0_input)):
#             neg_0[i] = choice([i for i in range(0,len(self.vocab_0)) if i not in tokens_0_input])
            
            
#             neg_1[i] = choice([i for i in range(0,len(self.vocab_1)) if i not in tokens_1_input])
            
            
#             neg_2[i] = choice([i for i in range(0,len(self.vocab_2)) if i not in tokens_2_input])
            
            
#             hier2=list(set(category_meta[category_meta['category2'].isin(tokens_2_input)]['category3']))+random.sample([j for j in range(len(self.vocab_3))],5) # 나와 같은 카테고리의 아이템을 negative로 추출
#             # hier2=list(set(category_meta[~category_meta['category2'].isin(tokens_2_input)]['category3'])) # 나와 다른 카테고리의 아이템을 negative로 추출
#             neg_3[i] = choice([i for i in range(0,len(self.vocab_3)) if (i not in tokens_3_input) and (i in hier2)])
            
            
#             neg_4[i] = choice([i for i in range(0,len(self.vocab_4)) if i not in tokens_4_input])
        
        
        

        return tokens_0_input,output_label_0,tokens_1_input,output_label_1,tokens_2_input,output_label_2,tokens_3_input,output_label_3,tokens_4_input,output_label_4,temporal,neg_0,neg_1,neg_2,neg_3,neg_4,test_neg_0,test_neg_1,test_neg_2,test_neg_3,test_neg_4
        

        
        
    def random_sent(self, index):
        t0,t1,t2,t3,t4,u = self.get_corpus_line(index)
        return t0,t1,t2,t3,t4,u


    def get_corpus_line(self, item):

        return self.category0[item],self.category1[item],self.category2[item], self.category3[item], self.category4[item], self.unix_time[item]

    
    
################################################################################################
#3. Inference
################################################################################################


class BERTDatasetFinetuneInf(Dataset):
    def __init__(self, text, vocab, seq_len, encoding="utf-8", corpus_lines=None):
        self.vocab = vocab
        self.seq_len = seq_len
        
        self.corpus_lines = corpus_lines
        self.encoding = encoding
        self.text = text
        
        self.lines = [line for line in tqdm.tqdm(self.text['gid'], desc="Loading Dataset", total=corpus_lines)]

        self.time = [line for line in tqdm.tqdm(self.text['time'], desc="Loading Dataset")]
        self.sex_cd = [line for line in tqdm.tqdm(self.text['sex_cd'], desc="Loading Dataset")]
        self.age_cd = [line for line in tqdm.tqdm(self.text['age_cd'], desc="Loading Dataset")]
        self.status = [line for line in tqdm.tqdm(self.text['status'], desc="Loading Dataset")]
        self.day_cd = [line for line in tqdm.tqdm(self.text['day_cd'], desc="Loading Dataset")]
        self.score = [line for line in tqdm.tqdm(self.text['score'], desc="Loading Dataset")]
        
        
        #self.binary_label = [line for line in tqdm.tqdm(self.text['binary_label'], desc="Loading Dataset")]

        self.corpus_lines = len(self.lines) 


    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        time = self.get_time(item)
        
        sex_cd = self.get_sex_cd(item)
        age_cd = self.get_age_cd(item)
        status = self.get_status(item)
        day_cd = self.get_day_cd(item)
        score = self.get_score(item)
        
        #binary_label = self.get_binary_label(item)
        t1 = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]

        bert_input = (t1)[:self.seq_len]
        bert_label = (t1_label)[:self.seq_len]
 
        bert_time = [0] + [int(i) for i in time.split(',')] +[0]
        bert_time = (bert_time)[:self.seq_len]
        
        bert_status = [0] + [int(i) for i in status.split(',')] + [0]
        bert_status = (bert_status)[:self.seq_len]
        
        bert_sex_cd = ([int(sex_cd) for _ in range(len(t1))])[:self.seq_len]
        bert_age_cd = ([int(age_cd) for _ in range(len(t1))])[:self.seq_len]
        bert_day_cd = ([int(day_cd) for _ in range(len(t1))])[:self.seq_len]
        bert_score = ([int(score) for _ in range(len(t1))])[:self.seq_len]


        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        #padding_2 = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_time))]
        #sec_padding = [(sec_list[-1] + i + 1) %(24*60*60) for i range(self.seq_len - len(sec_list))]
        bert_input.extend(padding)
        bert_label.extend(padding)
        bert_time.extend(padding)
        #bert_time.extend(padding_2)
        bert_sex_cd.extend(padding)
        bert_age_cd.extend(padding)
        bert_status.extend(padding)
        bert_day_cd.extend(padding)
        bert_score.extend(padding)
       
        #bert_binary_label.extend(padding)
        output = {"bert_input": bert_input,
                  #"bert_label": bert_label,
                  #"bert_binary_label": binary_label,
                  "bert_sex_cd" : bert_sex_cd,
                  "bert_day_cd" : bert_day_cd,
                  "bert_score" : bert_score,
                  "bert_age_cd" : bert_age_cd
                  #"bert_status" : bert_status
                  #"bert_time" : bert_time
                 }

        return {key: torch.tensor(value) for key, value in output.items()}

    
    
    def get_time(self, item):
        return self.time[item]
    
    def get_age_cd(self, item):
        return self.age_cd[item]
    
    def get_sex_cd(self, item):
        return self.sex_cd[item] 
    
    def get_status(self, item):
        return self.status[item]

    def get_day_cd(self, item):
        return self.day_cd[item]
    
    def get_score(self, item):
        return self.score[item]
    
#     def get_binary_label(self, item):
#         return self.binary_label[item]

    def random_word(self, sentence):
        tokens = sentence.split(',') # input data 형태에 따라 다르게 
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob > 1: #masked LM 없도록
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        t1 = self.get_corpus_line(index)
        return t1


    def get_corpus_line(self, item):
        return self.lines[item]

    
    
    
################################################################################################
#4. End2End  
################################################################################################

class BERTDatasetE2E(Dataset):
    def __init__(self, text, vocab, seq_len, encoding="utf-8", corpus_lines=None):
        self.vocab = vocab
        self.seq_len = seq_len
        
        self.corpus_lines = corpus_lines
        self.encoding = encoding
        self.text = text
        
        self.lines = [line for line in tqdm.tqdm(self.text['gid'], desc="Loading Dataset", total=corpus_lines)]

        self.time = [line for line in tqdm.tqdm(self.text['time'], desc="Loading Dataset")]
        self.sex_cd = [line for line in tqdm.tqdm(self.text['sex_cd'], desc="Loading Dataset")]
        self.age_cd = [line for line in tqdm.tqdm(self.text['age_cd'], desc="Loading Dataset")]
        self.status = [line for line in tqdm.tqdm(self.text['status'], desc="Loading Dataset")]
        self.day_cd = [line for line in tqdm.tqdm(self.text['day_cd'], desc="Loading Dataset")]
        self.score = [line for line in tqdm.tqdm(self.text['score'], desc="Loading Dataset")]
        
        
        self.binary_label = [line for line in tqdm.tqdm(self.text['binary_label'], desc="Loading Dataset")]

        self.corpus_lines = len(self.lines) 


    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        time = self.get_time(item)
        
        sex_cd = self.get_sex_cd(item)
        age_cd = self.get_age_cd(item)
        status = self.get_status(item)
        day_cd = self.get_day_cd(item)
        score = self.get_score(item)
        
        binary_label = self.get_binary_label(item)
        t1 = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]

        bert_input = (t1)[:self.seq_len]
        bert_label = (t1_label)[:self.seq_len]
 
        bert_time = [0] + [int(i) for i in time.split(',')] +[0]
        bert_time = (bert_time)[:self.seq_len]
        
        bert_status = [0] + [int(i) for i in status.split(',')] + [0]
        bert_status = (bert_status)[:self.seq_len]
        
        bert_sex_cd = ([int(sex_cd) for _ in range(len(t1))])[:self.seq_len]
        bert_age_cd = ([int(age_cd) for _ in range(len(t1))])[:self.seq_len]
        bert_day_cd = ([int(day_cd) for _ in range(len(t1))])[:self.seq_len]
        bert_score = ([int(score) for _ in range(len(t1))])[:self.seq_len]


        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        #padding_2 = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_time))]
        #sec_padding = [(sec_list[-1] + i + 1) %(24*60*60) for i range(self.seq_len - len(sec_list))]
        bert_input.extend(padding)
        bert_label.extend(padding)
        bert_time.extend(padding)
        #bert_time.extend(padding_2)
        bert_sex_cd.extend(padding)
        bert_age_cd.extend(padding)
        bert_status.extend(padding)
        bert_day_cd.extend(padding)
        bert_score.extend(padding)
       
        #bert_binary_label.extend(padding)
        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "bert_binary_label": binary_label,
                  "bert_sex_cd" : bert_sex_cd,
                  "bert_day_cd" : bert_day_cd,
                  "bert_score" : bert_score,
                  "bert_age_cd" : bert_age_cd
                  #"bert_status" : bert_status
                  #"bert_time" : bert_time
                 }

        return {key: torch.tensor(value) for key, value in output.items()}

    
    
    def get_time(self, item):
        return self.time[item]
    
    def get_age_cd(self, item):
        return self.age_cd[item]
    
    def get_sex_cd(self, item):
        return self.sex_cd[item] 
    
    def get_status(self, item):
        return self.status[item]

    def get_day_cd(self, item):
        return self.day_cd[item]
    
    def get_score(self, item):
        return self.score[item]
    
    def get_binary_label(self, item):
        return self.binary_label[item]

    def random_word(self, sentence):
        tokens = sentence.split(',') # input data 형태에 따라 다르게 
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        t1 = self.get_corpus_line(index)
        return t1


    def get_corpus_line(self, item):
        return self.lines[item]