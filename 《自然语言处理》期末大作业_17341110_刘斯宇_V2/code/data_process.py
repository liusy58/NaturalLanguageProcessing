import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import string
import jieba
#import plotly.graph_objs as go
pun = "。，、？【 】 %“”：；（）％《》‘’{} ⑦()、%^>℃：.”“^-——=&#@￥· 0123456789–…．\"#$%&'()*+,-./:;<=>@[\]^_`{|}~?"
enpun = '[,.!\']'
#判断是否是汉字
def is_chinese(uchar):
    if uchar >= u'\u4E00' and uchar <= u'\u9FA5':
        return True
    else:
        return False


#判断是否是数字
def is_digit(uchar):
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


#判断是否是英文字母
def is_alpha(uchar):
    if (uchar >= u'\u0041' and uchar <= u'\u005A') or (uchar >= u'\u0061'
                                                       and uchar <= u'\u007A'):
        return True
    else:
        return False


def div_chinese(filename):
    f = open('ch.txt','w')
    res = []
    wordlist = []
    word2fre={}
    wordlist.append('<pad>')
    lines = open(filename, 'r')
    for line in lines:
        line = ' '.join(jieba.cut(line, cut_all=False))
        line = line.strip()
        line = line.split(' ')
        for index, words in enumerate(line):
            flag = True
            for word in words:
                if word not in pun and not is_chinese(word) and not is_digit(
                        word) and not is_alpha(word):
                    flag = False
                    break
            if not flag:
                line[index] = 'UNK'
                #print(line)
        line.insert(0, '<BOS>')
        line.append('<EOS>')
        res.append(line)
    for lines in res:
        for word in lines:
            if word not in wordlist:
                wordlist.append(word)
                word2fre[word]=1
            else:
                word2fre[word]+=1
    word2fre = sorted(word2fre.items(), key=lambda item: item[1], reverse=True)
    ch2index={}
    cnt=0
    for item in word2fre[:8000]:
        ch2index[item[0]]=cnt
        cnt+=1
        f.write(item[0])
        f.write('\n') 
    
#############
##NOTE:生成train.cn文件
##############
    f=open('train.cn','w')

    for lines in res:
        for (index,word) in enumerate(lines):
            if word not in ch2index.keys():
                f.write(str(ch2index['UNK']))
            else:
                f.write(str(ch2index[word]))
            if index==len(lines)-1:
                f.write('\n')
            else:
                f.write(' ')



def div_English(filename):
    f = open('en.txt','w')
    res = []
    word_list = []
    word_list.append('<pad>')
    word2fre={}
    lines = open(filename, 'r')
    for line in lines:
        line = line.strip('\n')
        line = line[:-2] + ' ' + line[-1]
        line_list = line.split(' ')
        line_list.insert(0, '<BOS>')
        line_list.append('<EOS>')
        res.append(line_list)
    for line in res:
        for word in line:
            if len(word)==0:
                continue
            if len(word)>1 and word[-1] in enpun:
                word=word[:-1]
            if word not in word_list:
                word_list.append(word)
                word2fre[word]=1
            else:
                word2fre[word]+=1

    word2fre = sorted(word2fre.items(), key=lambda item: item[1], reverse=True)
    en2index={}
    cnt=0
    for item in word2fre[:4000]:
        en2index[item[0]]=cnt
        cnt+=1
        f.write(item[0])
        f.write('\n')     
    en2index['UNK'] = len(en2index)
    f=open('train.en','w')

    lines = open(filename, 'r')

    for lines in res:
        for (index,word) in enumerate(lines):
            if len(word)==0:
                continue
            if len(word)>1 and word[-1] in enpun:
                word=word[:-1]
            if word not in en2index.keys():
                f.write(str(en2index['UNK']))
            else:
                f.write(str(en2index[word]))
            if index==len(lines)-1:
                f.write('\n')
            else:
                f.write(' ')
            
##方便起见将所有的souce和target合起来一起处理。
div_chinese('train_source.txt')
div_English('train_target.txt')

