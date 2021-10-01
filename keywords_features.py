#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:30:40 2019

@author: rahulshukla
"""

import nltk
import pandas as pd
import numpy as np
import csv
from numpy import linalg as LA

# FEATURES OF KEYWORDS

key_feature_list=[]
row_count_train=0

def token(rows):
    tokenized=nltk.word_tokenize(rows)
    for word in tokenized:
        if word != ',':
            if word not in key_feature_list:
            
                key_feature_list.append(word)
            

with open('Dataset_train_2.csv','r') as ds:
    
    read=csv.reader(ds) 
    for row in read:
        row_count_train = row_count_train + 1
        token(row[3])

key_features=np.array(key_feature_list)
no_of_key_features=key_features.size



#making training document vector for title attribute

doc_key_vectors_train = np.zeros((row_count_train,no_of_key_features))

r = -1

def dvec(frow):
    tokenized=nltk.word_tokenize(frow)
    doc_key_feature_list_train=[]
    
    for word in tokenized:
        if word != ',':
            doc_key_feature_list_train.append(word)
            
    for i in range(no_of_key_features):
        for w in doc_key_feature_list_train:
            if w == key_features[i]:
                doc_key_vectors_train[r][i] = doc_key_vectors_train[r][i]+1
 
#making testing document vector               

with open('Dataset_train_2.csv','r') as dstr:
    read_train=csv.reader(dstr)
    for row in read_train:
        r = r+1
        print(r)
        dvec(row[3])

row_count_test = 0        

with open('Dataset_test_2.csv','r') as dste:
    read_test=csv.reader(dste)
    for row in read_test:
        row_count_test = row_count_test+1
 

doc_key_vectors_test = np.zeros((row_count_test,no_of_key_features))

r1 = -1

def dvec2(frow):
    tokenized=nltk.word_tokenize(frow)
    doc_key_feature_list_test=[]
    
    for word in tokenized:
        if word != ',':
            doc_key_feature_list_test.append(word)
            
    for i in range(no_of_key_features):
        for w in doc_key_feature_list_test:
            if w == key_features[i]:
                doc_key_vectors_test[r1][i] = doc_key_vectors_test[r1][i]+1
                

with open('Dataset_test_2.csv','r') as dste:
    read_test=csv.reader(dste)
    for row in read_test:
        r1 = r1+1
        print(r1)
        dvec2(row[3])