import json
import os

class Config:
    def __init__(self):
        home="./"
        data_home=os.path.join(home,"data")

        self.train_data_file=os.path.join(data_home,"train.json")
        self.test_data_ffile=os.path.join(data_home,"test.json")
        self.vector_file=os.path.join(data_home,"vectors-300.txt")
        self.vocab_file=os.path.join(data_home,"vocab.json")
        self.tags_file=os.path.join(data_home,"tags.json")

        self.vocab=json.load(open(self.vocab_file,'r',encoding='utf-8'))
        self.tags=json.load(open(self.tags_file,'r',encoding='utf-8'))

        self.words_number=len(self.vocab)
        self.tags_number=len(self.tags)

        self.embedding_size=300
        self.hidden_size=200
        self.batch_size=1
        self.epoch_size=100
        self.dropout=0.5

        self.id2words={j:i for i,j in self.vocab.items()}
        self.id2tags={j:i for i,j in self.tags.items()}

        self.model=os.path.join(data_home,'model','model.pt')
        self.model_file=os.path.join(data_home,'output')


