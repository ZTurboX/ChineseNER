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


if __name__=='__main__':
    raw_data_file='./raw_data/train.txt'
    split_train_char_file='./raw_data/split_train_char.txt'
    train_data_file='./raw_data/train.json'
    #split_char(raw_data_file,split_train_char_file)
    #convert_data(raw_data_file,train_data_file)







