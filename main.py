from const import Config
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import json
import model
from torch.autograd import Variable
import utils

torch.manual_seed(1)
np.random.seed(1)

config=Config()
parse=argparse.ArgumentParser()

parse.add_argument('--mode',default='train',help='train/test')
parse.add_argument('--cuda',default=False)
args=parse.parse_args()

mode=args.mode
use_cuda=args.cuda
if use_cuda:
    torch.cuda.manual_seed(1)

model=model.EntModel(config,use_cuda)
if use_cuda:
    model.cuda()
if mode=="test":
    print("loading model")
    state_dict=torch.load(open(config.model_file),'rb')
    model.load_state_dict(state_dict)

def load_data(data_file):
    all_data=[]
    with open(config.train_data_file,'r',encoding='utf-8') as f:
        for line in f:
            data=[]
            sentence=json.loads(line)
            data.append(sentence["sentence"])
            data.append(sentence["tags"])
            all_data.append(data)
    f.close()
    return all_data



def get_batch(batch):
    batch=sorted(batch,key=lambda x:len(x[0]),reverse=True)
    X_len=[len(s[0]) for s in batch]
    max_batch_sent_len=max(X_len)
    X=[]
    Y=[]
    for s in batch:
        X.append(s[0]+[config.vocab["<unk>"]]*(max_batch_sent_len-len(s[0])))
        Y.append(s[1]+[0]*(max_batch_sent_len-len(s[1])))
    sentence_tensor=utils.convert_long_variable(X,use_cuda)
    tags_tensor=utils.convert_long_tensor(Y,use_cuda)
    length_tensor=utils.convert_long_tensor(X_len,use_cuda)
    return sentence_tensor,tags_tensor,length_tensor


def train_step(train_data,optimizer):
    optimizer.zero_grad()
    model.train()
    count=0
    total_loss=0
    for j in range(0, len(train_data), config.batch_size):
        print("run bactch : % d" % j)
        batch = train_data[j:j + config.batch_size]
        sentence_tensor, tags_tensor,length_tensor=get_batch(batch)
        loss=model.get_loss(sentence_tensor,tags_tensor,length_tensor)
        loss.backward()
        optimizer.step()
        print("minibatch : %d , loss : %.5f " % (j,loss.item()))
        total_loss+=loss.item()
        count+=1
    print("-------------------------------------------------------------")
    print("avg loss : %.5f"%(total_loss/count))
    print("-------------------------------------------------------------")

def dev_step(dev_data):
    print("test the model")
    model.eval()
    for j in range(0,len(dev_data),config.batch_size):
        batch = train_data[j:j + config.batch_size]
        sentence_tensor, tags_tensor, length_tensor = get_batch(batch)
        _,paths=model(sentence_tensor,length_tensor)



def train(train_data):
    optimizer=optim.Adam(model.parameters())
    print("start train")
    for i in range(config.epoch_size):
        print("train epoch : %d" % i)
        train_step(train_data,optimizer)


if __name__=='__main__':
    train_data=load_data(config.train_data_file)
    train(train_data)