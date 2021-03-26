# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 20:32:28 2021

@author: 86176
"""

import pandas as pd
import numpy as np
import jieba
from tqdm import tqdm
import re
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from collections import Counter
import pickle
import random
import os
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
vector_size = 150
def Train_divide():
    f_train = open(r".\data\train_new.csv",'w',encoding="UTF-8")
    f_valid = open(r".\data\valid_new.csv",'w',encoding="UTF-8")
    with open(r".\data\train.csv","r",encoding="UTF-8") as fp:
        head=fp.readline()
        f_train.write(head)
        f_valid.write(head)
        for line in fp:
            if random.random()>0.1:
                f_train.write(line)
            else:
                f_valid.write(line)
    f_train.close()
    f_valid.close()
def data_deal(comments):
    #数据清洗
    stop_words = pd.read_table('./tmp/stop.txt',header=None)[0].tolist
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    sentences = comments.map(lambda x:pattern.sub("",x))
    #jieba分词、去停用词
    sentences = sentences.map(lambda x: [word for word in list(jieba.cut(x,cut_all=False)) if word.strip() not in stop_words()])
    return sentences
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    for i in range(0, num_examples, batch_size):
        end = min(i+batch_size,num_examples)
        yield  features.iloc[i:end], labels.iloc[i:end]
def Get_word(text,words):
    for word in text:
        words.add(word)
class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d,self).__init__()
    def forward(self,x):
        return F.max_pool1d(x,kernel_size=x.shape[2])
class TextCNN(nn.Module):
    def __init__(self, embed_matrix, embedding_dim, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_channels = num_channels
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embed_matrix)
        self.embedding.weight.requires_grad = True
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()
        for o, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=embedding_dim,
                                       out_channels=o,
                                       kernel_size=k))
            
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(sum(num_channels), 1)
    def forward(self,inputs):
        embeddings=self.embedding(inputs)
        embeddings=embeddings.permute(0,2,1)
        h = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs],dim=1)
        output = self.linear(self.dropout(h))
        output = torch.sigmoid(output).squeeze()
        return output
                
def train(net,train_path,valid_path,batch_size,optimizer,device,num_epochs,word2id):
    net.train()
    train_data = pd.read_csv(train_path,sep='\t',encoding='utf-8')
    valid_data = pd.read_csv(valid_path,sep='\t',encoding='utf-8')
    train_comments = data_deal(train_data['comment'])
    valid_comments = data_deal(valid_data['comment'])
    
    loss=torch.nn.BCELoss()
    optimizer=optimizer
    for epoch in range(num_epochs):
        train_iter = data_iter(batch_size=batch_size,features=train_comments,labels=train_data['label'])
        valid_iter = data_iter(batch_size=batch_size,features=valid_comments,labels=valid_data['label'])
        for train_x,label in train_iter:
            train_x = sent2id(train_x,word2id)
            label = list(label)
            label = torch.from_numpy(np.array(label, dtype=np.float32)).to(device)
            train_x = torch.from_numpy(train_x).to(device)
            train_y_pred = net(train_x)
            l=loss(train_y_pred,label)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        y_preds=[]
        y_sum=[]
        for valid_x,label in valid_iter:
            valid_x = sent2id(valid_x,word2id)
            net.eval()
            valid_x = torch.from_numpy(valid_x).to(device)
            with torch.no_grad():
                y_pred=net(valid_x)
            y_pred = torch.round(y_pred).cpu().numpy().tolist()
            y_preds += y_pred
            y_sum+=list(label)
            f1 = f1_score(y_sum, y_preds)
        print("Validation Results - Epoch: {}  F1: {:.2f}".format(epoch + 1, f1))
def sent2id(sents,word2id):
    sent_len=[len(sent) for sent in sents]
    max_len = max(sent_len)
    id_sents=[]
    for sent in sents:
        sent_id = []
        for word in sent:
            if word in word2id.keys():
                sent_id.append(word2id[word])
            else:
                sent_id.append(word2id["unk"])
        while(len(sent_id)<max_len):
            sent_id.append(word2id["pad"])
        id_sents.append(sent_id)
    return np.array(id_sents, dtype=np.int64)
if __name__ == '__main__':
    if not os.path.exists('data/train_new.csv'):
        Train_divide()
    train_new_path = r".\data\train_new.csv"
    test_new_path = r".\data\test_new.csv"
    valid_new_path = r".\data\valid_new.csv"
    train_path = r".\data\train.csv"
    train_data = pd.read_csv(train_path,sep='\t',encoding='utf-8')
    test_data = pd.read_csv(test_new_path,sep=',',encoding='utf-8')

    train_comments = data_deal(train_data['comment'])
    test_comments = data_deal(test_data['comment'])
    
    model_path = r'.\model\w2v.model'
    if not os.path.exists(model_path):
        comments = pd.concat([train_comments,test_comments],axis=0,ignore_index=True)
        model = Word2Vec(size=vector_size, 
                         window=2, 
                         min_count=1, 
                         workers=5,
                         sg=0, 
                         iter=20) 
        model.build_vocab(comments)      
        model.train(comments, total_examples=model.corpus_count, epochs=model.iter)   
        model.save("model/w2v.model")
        model = Word2Vec.load(model_path)
        model.wv.save_word2vec_format('./model/embed.txt')
    model = KeyedVectors.load_word2vec_format("./model/embed.txt")  #加载词袋
    id2word = model.index2word
    train_iter = data_iter(batch_size=16,features=train_comments,labels=train_data['label'])
    test_iter = data_iter(batch_size=16,features=test_comments,labels=test_data['id'])
    words = set()   #存储词汇
    for comment,_ in train_iter:    #_表示此处不使用
        comment.apply(Get_word,args=(words,))
    for comment,_ in test_iter:
        comment.apply(Get_word,args=(words,))
    set_id2word=set(id2word)
    out_vocab = list(words-set_id2word)
    out_vocab.insert(0,'pad')
    out_vocab.insert(1,'unk')
    word2id = dict(zip(out_vocab+id2word, np.arange(len(id2word)+len(out_vocab)).tolist()))
    embed_matrix = model.vectors
    embedding_dim = embed_matrix.shape[1]
    embed_matrix = torch.from_numpy(np.vstack((np.zeros((len(out_vocab), embedding_dim)), embed_matrix)))
    vocab_size = embed_matrix.shape[0]
    net = TextCNN(embed_matrix=embed_matrix, 
                  embedding_dim=embedding_dim, 
                  kernel_sizes=[2,3,4],
                  num_channels=[80,80,80])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-3)
    
    train(net=net,
          train_path=train_new_path,
          valid_path=valid_new_path,
          batch_size=16,
          optimizer=optimizer,
          device=device,
          num_epochs=20,
          word2id=word2id)
    
    # 预测
    y_preds = []
    net.eval()
    with torch.no_grad():
        test_new_iter = data_iter(batch_size=16,features=test_comments,labels=test_data['id'])
        
        for test_x, y in test_new_iter:
            test_x = sent2id(test_x, word2id)
            test_x = torch.from_numpy(test_x).to(device)
            y_pred = net(test_x)
            y_pred = torch.round(y_pred).cpu().numpy().tolist()
            y_preds += y_pred
    for i in range(len(y_preds)):
        y_preds[i] = int(y_preds[i])
    result = pd.read_csv("data/sample.csv")
    result['label'] = y_preds
    result[['id', 'label']].to_csv('./result/result_TextCNN_1.csv', index=None)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    