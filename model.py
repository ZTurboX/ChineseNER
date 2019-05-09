import torch.nn as nn
import const
from torch.nn.parameter import Parameter
import numpy as np
import torch
import utils

class EntModel(nn.Module):
    def __init__(self,config:const.Config,use_cuda=False):
        super(EntModel,self).__init__()
        self.embedding_size=config.embedding_size
        self.hidden_size=config.hidden_size
        self.words_number=config.words_number
        self.batch_size=config.batch_size
        self.vocab=config.vocab
        self.word_embedding=nn.Embedding(self.words_number,self.embedding_size)
        self.load_pretrained_embedding(config)
        self.dropout=nn.Dropout(config.dropout)
        self.use_cuda = use_cuda
        self.rnn=nn.LSTM(self.embedding_size,self.hidden_size//2,num_layers=1,bidirectional=True,batch_first=True)

        self.tags_size=config.tags_number
        self.tag2id=config.tags
        self.hidden2tag=nn.Linear(self.hidden_size,self.tags_size)

        self.transitions=nn.Parameter(torch.randn(self.tags_size,self.tags_size))
        self.transitions.data[self.tag2id["START"],:]=-1000.
        self.transitions.data[:,self.tag2id["STOP"]]=-1000.


    def load_pretrained_embedding(self,config:const.Config):
        words_vectors={}
        for line in open(config.vector_file,encoding='utf-8').readlines():
            items=line.strip().split()
            words_vectors[items[0]]=[float(x) for x in items[1:]]
        embeddding_matrix=np.asarray(np.random.normal(0,0.9,(self.words_number,300)),dtype='float32')

        for word in self.vocab:
            if word in words_vectors:
                embeddding_matrix[self.vocab[word]]=words_vectors[word]
        self.word_embedding.weight=nn.Parameter(torch.tensor(embeddding_matrix))

    def run_rnn(self,sentence_tensor,length_tensor):
        #embed=self.word_embedding(sentence_tensor).view(self.batch_size,sentence_tensor.shape[1],self.embedding_size)
        embed = self.word_embedding(sentence_tensor)
        embed=self.dropout(embed)
        embed_pack=nn.utils.rnn.pack_padded_sequence(embed,length_tensor,batch_first=True)
        lstm_output,_=self.rnn(embed_pack)
        lstm_output,_=nn.utils.rnn.pad_packed_sequence(lstm_output,batch_first=True)
        #lstm_output=lstm_output.view(self.batch_size,-1,self.hidden_size)
        output=self.hidden2tag(lstm_output)
        return output

    def get_gold_score(self,logits,tags):
        score=torch.zeros(1)
        if self.use_cuda:
            score=score.cuda()
        tags_tensor=torch.cat([utils.convert_long_tensor([self.tag2id["START"]],self.use_cuda),tags])
        for i,feat in enumerate(logits):
            score=score+self.transitions[tags_tensor[i+1],tags_tensor[i]]+feat[tags_tensor[i+1]]
        score=score+self.transitions[self.tag2id["STOP"],tags_tensor[-1]]
        return score

    def log_sum_exp(self,input):
        _,idx=torch.max(input,1)
        max_score=input[0,idx.item()]
        max_score_broadcast=max_score.view(1,-1).expand(1,input.size()[1])
        res=max_score+torch.log(torch.sum(torch.exp(input-max_score_broadcast)))
        return res

    def get_forward_score(self,lstm_output):
        init_score=torch.full((1,self.tags_size),0.)
        if self.use_cuda:
            init_score=init_score.cuda()
        init_score[0][self.tag2id["START"]]=0.
        forward_var=init_score
        for feat in lstm_output:
            score_t=[]
            for next_tag in range(self.tags_size):
                emit_score=feat[next_tag].view(1,-1).expand(1,self.tags_size)
                transition_score=self.transitions[next_tag].view(1,-1)
                next_tag_var=forward_var+transition_score+emit_score
                score_t.append(self.log_sum_exp(next_tag_var).view(1))
            forward_var=torch.cat(score_t).view(1,-1)
        terminal_var=forward_var+self.transitions[self.tag2id["STOP"]]
        forward_score=self.log_sum_exp(terminal_var)
        return forward_score


    def get_loss(self,sentence_tensor,tags_tensor,length_tensor):
        lstm_output=self.run_rnn(sentence_tensor,length_tensor)
        gold_score=torch.zeros(1)
        forward_score=torch.zeros(1)
        if self.use_cuda:
            gold_score=gold_score.cuda()
            forward_score=forward_score.cuda()
        for logits,tag,length in zip(lstm_output,tags_tensor,length_tensor):
            logits=logits[:length]
            tag=tag[:length]
            gold_score+=self.get_gold_score(logits,tag)
            forward_score+=self.get_forward_score(logits)
        return forward_score-gold_score

    def viterbi_decode(self,lstm_output):
        backpointers=[]
        init_vvars=torch.full((1,self.tags_size),0.)
        init_vvars=init_vvars
        if self.use_cuda:
            init_vvars=init_vvars.cuda()
        init_vvars[0][self.tag2id["START"]]=0
        forward_var=init_vvars
        for feat in lstm_output:
            bptrs_t=[]
            viterbivars_t=[]
            for next_tag in range(self.tags_size):
                next_tag_var=forward_var+self.transitions[next_tag]
                _,idx=torch.max(next_tag_var,1)
                best_tag_id=idx.item()
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var=(torch.cat(viterbivars_t)+feat).view(1,-1)
            backpointers.append(bptrs_t)
        terminal_var=forward_var+self.transitions[self.tag2id["STOP"]]
        _,idx=torch.max(terminal_var,1)
        best_tag_id=idx.item()
        path_score=terminal_var[0][best_tag_id]

        best_path=[best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id=bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        #start=best_path.pop()
        best_path.reverse()
        return path_score,best_path

    def forward(self,sentence_tensor,length_tensor):
        lstm_output=self.run_rnn(sentence_tensor,length_tensor)
        scores=[]
        paths=[]
        for logits,length in zip(lstm_output,length_tensor):
            logit=logits[:length]
            score,path=self.viterbi_decode(logit)
            scores.append(score)
            paths.append(path)
        return scores,paths


