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
parse.add_argument('--device',default="3")

args=parse.parse_args()

mode=args.mode
use_cuda=args.cuda
device_id=args.device
if use_cuda:
    torch.cuda.manual_seed(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id

model=model.EntModel(config,use_cuda)
if use_cuda:
    model.cuda()
if mode=="test":
    print("loading model")
    state_dict=torch.load(open(config.model),'rb')
    model.load_state_dict(state_dict)

def load_data(data_file):
    all_data=[]
    with open(data_file,'r',encoding='utf-8') as f:
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

def train_step(train_data,optimizer,dev_data):
    model.train()
    count=0
    total_loss=0
    for j in range(0, len(train_data), config.batch_size):
        optimizer.zero_grad()
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
    f1=dev_step(dev_data)
    return f1

def dev_step(dev_data):
    print("test the model")
    model.eval()
    predict_entity=[]
    gold_entity=[]
    origin=0.
    found=0.
    correct=0.
    for j in range(0,len(dev_data),config.batch_size):
        batch = dev_data[j:j + config.batch_size]
        sentence_tensor, tags_tensor, length_tensor = get_batch(batch)
        _,paths=model(sentence_tensor,length_tensor)
        for sentence,tags ,path,length in zip(sentence_tensor,tags_tensor,paths,length_tensor):
            sentence=sentence[:length]
            tags=tags[:length]
            sentence=list(sentence.cpu().numpy())
            tags=list(tags.cpu().numpy())
            predict_entity=utils.calculate(sentence,path,config.id2words,config.id2tags)
            gold_entity=utils.calculate(sentence,tags,config.id2words,config.id2tags)
            origin+=len(gold_entity)
            found+=len(predict_entity)
            for p_tag in predict_entity:
                if p_tag in gold_entity:
                    correct+=1

    p=0. if found==0 else (correct/found)
    r=0. if origin==0 else (correct/origin)
    f1=0. if p+r==0 else (2*p*r)/(p+r)
    print("precision : %.3f , recall : %.3f , f1 : %.3f " % (p,r,f1))
    return f1


def train(train_data,dev_data):
    optimizer=optim.Adam(model.parameters())
    print("start train")
    best_f1=0
    for i in range(config.epoch_size):
        print("train epoch : %d" % i)
        f1=train_step(train_data,optimizer,dev_data)
        if f1>best_f1:
            best_f1=f1
            print("-----------------------------------------")
            print("best f1 : %.3f " % best_f1)
            print("-----------------------------------------")
            print("save model....")
            best_model_file=os.path.join(config.model_file,"epoch-%d_f1-%.3f.pt"%(i,best_f1))
            torch.save(model.state_dict(),best_model_file)


if __name__=='__main__':
    train_data=load_data(config.train_data_file)
    dev_data=load_data(config.test_data_ffile)
    train(train_data,dev_data)