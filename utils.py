from torch.autograd import Variable
import torch
from const import Config
import json
config=Config()

def convert_long_tensor(var,use_cuda):
    var=torch.LongTensor(var)
    if use_cuda:
        var=var.cuda(async=True)
    return var

def convert_long_variable(var,use_cuda):
    return Variable(convert_long_tensor(var,use_cuda))


def calculate(x,y,id2word,id2tag):
    '''

    {"sentence": ["陈", "明", "亮", "又", "哭", "又", "闹", "，", "但", "仍", "无", "济", "于", "事", "。"],
    "tags": ["B-PER", "I-PER", "I-PER", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}

    '''
    entity=[]
    res=[]
    for j in range(len(x)):
        if x[j]==0 or y[j]==0:
            continue
        if j+1<len(x) and id2tag[y[j]][0]=='B' and id2tag[y[j+1]][0]=='I':
            entity=[id2word[x[j]]+'/'+id2tag[y[j]]]
        elif j+1<len(x) and id2tag[y[j]][0]=='I' and len(entity)!=0 and id2tag[y[j+1]][0]=='I' and entity[-1].split('/')[1][1:]==id2tag[y[j]][1:]:
            entity.append(id2word[x[j]]+'/'+id2tag[y[j]])
        elif j+1<len(x) and id2tag[y[j]][0]=='I' and len(entity)!=0 and id2tag[y[j+1]][0]=='O' and entity[-1].split('/')[1][1:]==id2tag[y[j]][1:]:
            entity.append(id2word[x[j]]+'/'+id2tag[y[j]])
            res.append(entity)
            entity=[]
        elif j+1<len(x) and id2tag[y[j]][0]=='B' and (id2tag[y[j+1]][0]=='O' or id2tag[y[j+1]][0]=='B'):
            res.append([id2word[x[j]]+'/'+id2tag[y[j]]])
            entity=[]
        else:
            entity=[]
    return res
