import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


from transformers import *
import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import numpy as np
import re
import pickle
import time
import pandas as pd
from pathlib import Path
import random
# from torch.utils.tensorboard import SummaryWriter

# load datasets
# df = pd.read_csv('./data/hackathon_train.csv', encoding='cp949', index_col=0)
df = pd.read_excel('/home/kaleb/Qualcom_comp/data_phase2/train_data.xlsx', index_col=0)

df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
df['Age_float'] = df['Age']

df = pd.get_dummies(df, columns=['Age'], prefix='cat')

# print(df)

# load pretrained model
model_name = 'kykim/bert-kor-base'
# model_name = 'monologg/kobigbird-bert-base'
# model_name = 'beomi/kcbert-base'
# model_name = 'snunlp/KR-BERT-char16424'
# model_name = 'skt/kobert-base-v1'
# model_name = 'klue/bert-base'
# model_name = 'klue/roberta-base'
# model_name = 'skt/kogpt2-base-v2'
def get_model(model_name):
    # * Model          | Tokenizer          | Pretrained weights shortcut
    # MODEL=(DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased')
    # tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer = KoBERTTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    n_hl = model.config.num_hidden_layers
    # embed_dim = model.config.embedding_size
    embed_dim = 768
    return model, tokenizer, n_hl, embed_dim

model, tokenizer, n_hl, embed_dim = get_model(model_name)

def convert_mbti_to_label(mbti: str):
    """
    :param mbti: string. length=4
    :return:
    """
    stand = 'ISTJ'  # [0, 0, 0, 0]
    result = []
    for i in range(4):
        if stand[i] == mbti[i]:
            result.append(0)
        else:
            result.append(1)

    return result

# now reload the first data, and concatenate it with the second data
df_merged = pd.read_csv('./data/hackathon_train.csv', encoding='cp949', index_col=0)
# print(df_merged)
df_merged['Age_float'] = df_merged['Age']
df_merged = pd.get_dummies(df_merged, columns=['Age'], prefix='cat')
df_merged['Short_Answer'] = df_merged['Answer'].apply(lambda x : x.split('>')[0][x.index('<')+1:])
df_merged['Long_Answer'] = df_merged['Answer'].apply(lambda x: x.replace(f"<{(x.split('>')[0])[x.index('<')+1:]}>", ''))
df_merged = df_merged.drop(['Answer'], axis=1)

# print(df_merged)

df = pd.concat([df, df_merged], axis=0)
# print(df)

# sort them according to the "User_ID"
# print(df['User_ID'].unique())
df = df.sort_values(by=['User_ID'])

# now concatenate the "Long_Answer" of k similar user_ids


# print(df['User_ID'].unique())

# a train-test split

trade_off = 204

train_df = df[df['User_ID'] <= trade_off]
test_df = df[df['User_ID'] > trade_off]

prop_train, prop_test = len(train_df) / len(df), len(test_df) / len(df)

print(f'prop train = {prop_train}, prop test = {prop_test}')


train_tensor = tokenizer(train_df['Long_Answer'].to_list(), max_length=model.config.max_position_embeddings, return_tensors='pt', padding=True)
test_tensor = tokenizer(test_df['Long_Answer'].to_list(), max_length=model.config.max_position_embeddings, return_tensors='pt', padding=True)

print('there is no problem with the tokenizer')

class MyMapDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        data = {k:v[idx] for k,v in self.data.items()}
        return data
    
train_dataset = MyMapDataset(train_tensor)
test_dataset = MyMapDataset(test_tensor)

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False)

def forward(model, dl, device=0):
    pooled = []
    # hidden = []
    model.cuda(device)
    model.eval()
    for data in dl:
        data = {k:v.cuda(device) for k,v in data.items()}
        with torch.no_grad():
            output = model(**data, output_hidden_states=False)
            # input_ids = data['input_ids']
            # output = model(**data, return_dict=True)
        p = output.pooler_output
        # p, h = output.last_hidden_state[:,0,:], output.last_hidden_state # only [CLS] token embedding
        pooled.append(p) # pooler output
        # hidden.append(h[-1][:,0,:]) # only [CLS] token embedding 
        # hidden.append(h) # all token embedding
    return torch.cat(pooled)#, torch.cat(hidden)

train_result = forward(model, train_dl, device=0)
test_result = forward(model, test_dl, device=1)

torch.save(train_result, f'{model_name.replace("/", "_")}_train_phase2_merge_sorted.pt')
torch.save(test_result, f'{model_name.replace("/", "_")}_test_phase2_merge_sorted.pt')

print("saved successfully")