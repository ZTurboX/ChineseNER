import json

def split_char(raw_data_file,split_char_file):
    all_sentence=[]
    with open(raw_data_file,'r',encoding='gbk') as f:
        for line in f.readlines():
            sentence = []
            s = line.strip().split()
            for item in s:
                char = []
                tag=[]
                if item.find('/') != -1:
                    char = list(item[:item.find('/')])
                    sentence.extend(char)

                else:
                    char = list(item)
                    sentence.extend(char)

            all_sentence.append(sentence)
    f.close()

    with open(split_char_file,'a',encoding='utf-8') as fs:
        for sentence in all_sentence:
            s=' '.join(sentence)
            fs.writelines(s)
            fs.writelines('\n')
    fs.close()


def convert_data(raw_data_file,data_file):
    all_sentence = []
    with open(raw_data_file, 'r', encoding='gbk') as f:
        for line in f.readlines():
            text={}
            sentence = []
            tags=[]
            s = line.strip().split()
            for item in s:
                char = []
                tag=[]
                if item.find('/') != -1:
                    char = list(item[:item.find('/')])
                    sentence.extend(char)
                    if item[item.find('/') + 1:] == 'ns':
                        tag.append('B-LOC')
                        tag.extend(['I-LOC'] * (len(item[:item.find('/')]) - 1))
                    if item[item.find('/') + 1:] == 'nr':
                        tag.append('B-PER')
                        tag.extend(['I-PER'] * (len(item[:item.find('/')]) - 1))
                    if item[item.find('/') + 1:] == 'nt':
                        tag.append('B-ORG')
                        tag.extend(['I-ORG'] * (len(item[:item.find('/')]) - 1))
                    tags.extend(tag)
                else:
                    char = list(item)
                    sentence.extend(char)
                    tags.extend(['O'] * len(item))
                text["sentence"] = sentence
                text["tags"] = tags
            all_sentence.append(text)
    f.close()

    with open(data_file,'a',encoding='utf-8') as fs:
        for sentence in all_sentence:
            json.dump(sentence,fs,ensure_ascii=False)
            fs.writelines('\n')
    fs.close()




def get_vocab(data_file,vocab_file,tag_file):
    vocab=["<unk>"]
    label=[]
    with open(data_file,'r',encoding='utf-8') as f:
        for line in f:
            data=json.loads(line)
            word=data["sentence"]
            tags=data["tags"]
            for w in word:
                if w not in vocab:
                    vocab.append(w)
            for t in tags:
                if t not in label:
                    label.append(t)
    f.close()
    word2id={j:i for i,j in enumerate(vocab)}
    tags2id={j:i for i,j in enumerate(label)}

    with open(vocab_file,'w',encoding='utf-8') as fs:
        fs.write(json.dumps(word2id,ensure_ascii=False,indent=4))
    fs.close()
    with open(tag_file,'w',encoding='utf-8') as ft:
        ft.write(json.dumps(tags2id,ensure_ascii=False,indent=4))
    ft.close()

def data2id(data_file,data2id_file):
    with open("./data/vocab.json",'r',encoding='utf-8') as f:
        vocab=json.load(f)
    f.close()
    with open("./data/tags.json",'r',encoding='utf-8') as fs:
        tags=json.load(fs)
    fs.close()

    data2id=[]
    with open(data_file,'r',encoding='utf-8') as fw:
        for line in fw:
            data=json.loads(line)
            word=[vocab[w] if w in vocab else vocab["<unk>"] for w in data["sentence"]]
            label=[tags[t] for t in data["tags"]]
            data2id.append({"sentence":word,"tags":label})
    fw.close()

    with open(data2id_file,'a',encoding='utf-8') as ft:
        for item in data2id:
            json.dump(item,ft,ensure_ascii=False)
            ft.write("\n")
    ft.close()

def get_test_data(raw_test_file,labeled_test_file,test_file):
    all_sentence=[]
    with open(raw_test_file,'r',encoding='gbk') as f:
        for line in f.readlines():
            line=line.strip()
            all_sentence.append(line)
    f.close()

    convert_data=[]
    with open(labeled_test_file,'r',encoding='gbk') as fs:
        sentence_length=[len(sentence) for sentence in all_sentence]
        slen=sentence_length[0]
        words=[]
        tags=[]
        count=0
        i=1
        for line in fs.readlines():
            if count!=slen:
                count += 1
                w, t = line.strip().split(" ")
                words.append(w)
                if t == "0":
                    t = "O"
                tags.append(t)
            else:
                convert_data.append({"sentence":words,"tags":tags})
                words=[]
                tags=[]
                count=0
                slen=sentence_length[i]
                i+=1
                w, t = line.strip().split(" ")
                words.append(w)
                if t == "0":
                    t = "O"
                tags.append(t)
                count+=1
        convert_data.append({"sentence": words, "tags": tags})
    fs.close()

    with open(test_file,'a',encoding='utf-8') as fw:
        for sentence in convert_data:
            json.dump(sentence,fw,ensure_ascii=False)
            fw.write("\n")
    fw.close()






if __name__=='__main__':
    raw_data_file='./raw_data/raw_train.txt'
    split_train_char_file='./raw_data/split_train_char.txt'
    train_data_file='./raw_data/train.json'
    #split_char(raw_data_file,split_train_char_file)
    #convert_data(raw_data_file,train_data_file)
    vocab_file="./data/vocab.json"
    tag_file="./data/tags.json"
    #get_vocab(train_data_file,vocab_file,tag_file)
    train_file="./data/train/train.json"
    #data2id(train_data_file,train_file)
    raw_test_file="./raw_data/raw_test.flat"
    labeled_test_file="./raw_data/test_labeled.txt"
    test_file="./raw_data/test.json"
    new_test_file="./data/test/test.json"
    #data2id(test_file,new_test_file)
    #get_test_data(raw_test_file,labeled_test_file,test_file)







