# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:12:07 2016
2017/02/13
@author: yuching

"""

va_file_root = '/Users/yuching/Desktop/test_data/va_eng.csv'
SentiWordNet_root = '/Users/yuching/Desktop/test_data/SentiWordNet_dictionary.csv'
input_file_root='/Users/yuching/Desktop/test_data/id_rating_2.csv'
output_file_root='/Users/yuching/Desktop/result/0710_ALL_rating_output(1).csv'

data_row = 0
documents = []
output = []

op_user = []
op_ove = []
op_help = []
op_days = []
op_raU = []
op_sen = []
op_len = []
op_senti_giw_pos = []
op_senti_giw_neg = []
op_senti_giw_tot = []
op_senti_sennet_pos = []
op_senti_sennet_neg = []
op_senti_sennet_tot = []
op_pos_ADJ = []
op_pos_ADP = []
op_pos_ADV = []
op_pos_CONJ = []
op_pos_DET = []
op_pos_NOUN = []
op_pos_NUM = []
op_pos_PRT = []
op_pos_PRON = []
op_pos_VERB = []
op_pos_punctuation = []


class output_data(object):
    def __init__(self,_ID, _overall, _helpful,_helpavg,_days, _sen, _len,
                 _senti_giw_pos, _senti_giw_neg, _senti_giw_tot,
                 _senti_sennet_pos, _senti_sennet_neg, _senti_sennet_tot,
                 _pos_ADJ,_pos_ADP,_pos_ADV,_pos_CONJ,_pos_DET,_pos_NOUN,
                 _pos_NUM,_pos_PRT,_pos_PRON,_pos_VERB,_op_pos_punctuation):
        self._ID = _ID
        self._overall = _overall
        self._helpful = _helpful
        self._helpavg = _helpavg
        self._days = _days
        self._sen = _sen
        self._len = _len
        self._senti_giw_pos = _senti_giw_pos
        self._senti_giw_neg = _senti_giw_neg
        self._senti_giw_tot = _senti_giw_tot
        self._senti_sennet_pos = _senti_sennet_pos
        self._senti_sennet_neg = _senti_sennet_neg
        self._senti_sennet_tot = _senti_sennet_tot
        self._pos_ADJ = _pos_ADJ
        self._pos_ADP = _pos_ADP
        self._pos_ADV = _pos_ADV
        self._pos_CONJ = _pos_CONJ
        self._pos_DET = _pos_DET
        self._pos_NOUN = _pos_NOUN
        self._pos_NUM = _pos_NUM
        self._pos_PRT = _pos_PRT
        self._pos_PRON = _pos_PRON
        self._pos_VERB = _pos_VERB
        self._op_pos_punctuation = _op_pos_punctuation
class va_data(object):
    def __init__(self, word, value):
        self.word = word
        self.value = value

from collections import OrderedDict
import nltk
import json
#from gensim import corpora, models, similarities
import logging

# 0 for negtive , 1 for postive
va_dic = {}
with open(va_file_root, 'r') as fva:
    for line in fva:
        temp = line.split(',')
        va_dic.setdefault(temp[0], int(temp[1]))
fva.close()

# postive negtive word
sentiwn_dic = {}
with open(SentiWordNet_root, 'r') as fsentiwn:
    for line in fsentiwn:
        temp = line.split(' ')
        sentiwn_dic.setdefault(temp[2][0:len(temp[2])-1], [temp[0],temp[1]])
fsentiwn.close()

data_objects = []
with open(input_file_root, 'r') as f:
    for line in f:
        data_row += 1
        temp=line.split('::::')
        op_user.append(temp[0])
        op_ove.append(temp[1])
        op_help.append(temp[2])
        op_days.append(temp[3])
        documents.append(temp[4])
        op_raU.append(temp[5])
        # print ("read:", data_row)
f.close()

count_rau=0
op_raUscore=[]
for i in range(0, data_row, 1):
    # print op_raU[i][0:len(op_raU[i])-1]
    temp=str(op_raU[i][0:len(op_raU[i])-1]).split(":::")
    count_rau += 1
    total=0
    if(len(temp) >= 2):
        for j in range(0, len(temp)-1, 1):
            temp2=str(temp[j]).split(":")
            total+=int(temp2[1])
        op_raUscore.append(round(total / float(len(temp) - 1),2))
        #print "rauser:", count_rau, total, len(temp) - 1, round(total / float(len(temp) - 1),2)
    else:
        op_raUscore.append(0)
        #print "rauser:", count_rau, total, len(temp) - 1

#斷句

data_row = 0
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

for document in documents:
    sen_texts=sent_tokenizer.tokenize(document)
    op_sen.append(len(sen_texts))
    data_row += 1
    print ("op_sen.append:", len(sen_texts),data_row)

#斷詞
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import brown

data_row=0
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
train_sents = brown.tagged_sents(tagset='universal')
unigram_tagger = nltk.UnigramTagger(train_sents)
for document in documents:
    texts_tokenized = []
    data_row+=1
    count_english_punc = 0
    for word in word_tokenize(document.decode('utf-8')):
        if word not in english_punctuations:
            texts_tokenized.append(word.lower())
        else:
            count_english_punc+=1
    op_len.append(len(texts_tokenized))
    print ("op_len.append:", len(texts_tokenized), data_row)

    # tokens = nltk.word_tokenize(
    #     "Tagged corpora use many different conventions for tagging words. To help us get started, we will be looking at a simplified tagset.")
    text_len=len(texts_tokenized)
    tagged = unigram_tagger.tag(texts_tokenized)
    tag_fd = nltk.FreqDist(tag for (word, tag) in tagged)

    if(text_len>0):
        op_pos_ADJ.append(round(float(tag_fd["ADJ"])/text_len,2))
        op_pos_ADP.append(round(float(tag_fd["ADP"])/text_len,2))
        op_pos_ADV.append(round(float(tag_fd["ADV"])/text_len,2))
        op_pos_CONJ.append(round(float(tag_fd["CONJ"])/text_len,2))
        op_pos_DET.append(round(float(tag_fd["DET"])/text_len,2))
        op_pos_NOUN.append(round(float(tag_fd["NOUN"])/text_len,2))
        op_pos_NUM.append(round(float(tag_fd["NUM"])/text_len,2))
        op_pos_PRT.append(round(float(tag_fd["PRT"])/text_len,2))
        op_pos_PRON.append(round(float(tag_fd["PRON"])/text_len,2))
        op_pos_VERB.append(round(float(tag_fd["VERB"])/text_len,2))
        op_pos_punctuation.append(round((float(tag_fd["."])+float(count_english_punc)) / text_len, 2))
    else:
        op_pos_ADJ.append(float(0))
        op_pos_ADP.append(float(0))
        op_pos_ADV.append(float(0))
        op_pos_CONJ.append(float(0))
        op_pos_DET.append(float(0))
        op_pos_NOUN.append(float(0))
        op_pos_NUM.append(float(0))
        op_pos_PRT.append(float(0))
        op_pos_PRON.append(float(0))
        op_pos_VERB.append(float(0))
        op_pos_punctuation.append(0)
        # print round(float(tag_fd["ADJ"])/text_len,2), \
        #     round(float(tag_fd["ADP"])/text_len,2), \
        #     round(float(tag_fd["ADV"])/text_len,2), \
        #     round(float(tag_fd["CONJ"])/text_len,2), \
        #     round(float(tag_fd["DET"])/text_len,2), \
        #     round(float(tag_fd["NOUN"])/text_len,2), \
        #     round(float(tag_fd["NUM"])/text_len,2), \
        #     round(float(tag_fd["PRT"])/text_len,2),\
        #     round(float(tag_fd["PRON"])/text_len,2), \
        #     round(float(tag_fd["VERB"])/text_len,2)
    senti_val_pos = float(0)
    senti_val_neg = float(0)
    senti_sw_pos = 0
    senti_sw_neg = 0
    for txt_tk in texts_tokenized:
        if txt_tk in va_dic:
            if va_dic[txt_tk] == 0:
                senti_val_neg += 1
            elif va_dic[txt_tk] == 1:
                senti_val_pos += 1
        if txt_tk in sentiwn_dic:
            senti_sw_pos += float(sentiwn_dic[txt_tk][0])
            senti_sw_neg += float(sentiwn_dic[txt_tk][1])

    if(len(texts_tokenized)>0):
        senti_val_pos = round(senti_val_pos/len(texts_tokenized), 5)
        senti_val_neg = round(senti_val_neg/len(texts_tokenized), 5)
        senti_sw_pos = round(senti_sw_pos / len(texts_tokenized), 5)
        senti_sw_neg = round(senti_sw_neg / len(texts_tokenized), 5)
    else:
        senti_val_pos = 0
        senti_val_neg = 0
        senti_sw_pos = 0
        senti_sw_neg = 0
    op_senti_giw_pos.append(senti_val_pos)
    op_senti_giw_neg.append(senti_val_neg)
    op_senti_giw_tot.append(round(senti_val_pos - senti_val_neg, 5))
    # print ("op_senti_giw:", senti_val_pos,senti_val_neg)
    op_senti_sennet_pos.append(senti_sw_pos)
    op_senti_sennet_neg.append(senti_sw_neg)
    op_senti_sennet_tot.append(round(senti_sw_pos - senti_sw_neg,5))
    # print ("op_senti_sennet:", senti_sw_pos, senti_sw_neg)

for i in range(0, data_row, 1):
    output.append(output_data(op_user[i],op_ove[i], op_help[i],op_raUscore[i],op_days[i],op_sen[i], op_len[i],
                              op_senti_giw_pos[i], op_senti_giw_neg[i],op_senti_giw_tot[i],
                              op_senti_sennet_pos[i], op_senti_sennet_neg[i],op_senti_sennet_tot[i],
                              op_pos_ADJ[i],op_pos_ADP[i],op_pos_ADV[i],op_pos_CONJ[i],op_pos_DET[i],
                              op_pos_NOUN[i],op_pos_NUM[i],op_pos_PRT[i],op_pos_PRON[i],op_pos_VERB[i],op_pos_punctuation[i]))
data_row = 0

filew = open(output_file_root, 'w')
filew.write("ID,Overall,Days,Sentence,Words,"
            "GIW_pos,GIW_neg,GIW_add,SWN_pos,SWN_neg,SWN_add,"
            "pos_ADJ,pos_ADP,pos_ADV,pos_CONJ,pos_DET,"
            "pos_NOUN,pos_NUM,pos_PRT,pos_PRON,pos_VERB,pos_punctuation,"
            "Helpful,HelpfulAvg\n")
for i in output:
    filew.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" %
                (i._ID, i._overall , i._days , i._sen , i._len ,
                 i._senti_giw_pos,i._senti_giw_neg,i._senti_giw_tot,
                 i._senti_sennet_pos, i._senti_sennet_neg, i._senti_sennet_tot,
                 i._pos_ADJ, i._pos_ADP, i._pos_ADV, i._pos_CONJ, i._pos_DET,
                 i._pos_NOUN, i._pos_NUM, i._pos_PRT, i._pos_PRON, i._pos_VERB,i._op_pos_punctuation,
                 i._helpful, i._helpavg))
    data_row += 1
    print("write:", data_row)
filew.close()


