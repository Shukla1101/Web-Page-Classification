import nltk
import pandas as pd
import numpy as np
import csv
from numpy import linalg as LA

# FEATURES OF KEYWORDS

H1_feature_list=[]
row_count_train=0

def token(rows):
    tokenized=nltk.word_tokenize(rows)
    for word in tokenized:
        if word != ',':
            if word not in H1_feature_list:
                H1_feature_list.append(word)
            

with open('Dataset_train_2.csv','r') as ds:
    
    read=csv.reader(ds) 
    for row in read:
        row_count_train = row_count_train + 1
        print()
        token(row[4])

H1_features=np.array(H1_feature_list)
no_of_H1_features=H1_features.size



#making training document vector for title attribute

doc_H1_vectors_train = np.zeros((row_count_train,no_of_H1_features))

r = -1

def dvec(frow):
    tokenized=nltk.word_tokenize(frow)
    doc_H1_feature_list_train=[]
    
    for word in tokenized:
        if word != ',':
            doc_H1_feature_list_train.append(word)
            
    for i in range(no_of_H1_features):
        for w in doc_H1_feature_list_train:
            if w == H1_features[i]:
                doc_H1_vectors_train[r][i] = doc_H1_vectors_train[r][i]+1
 
#making testing document vector               

with open('Dataset_train_2.csv','r') as dstr:
    read_train=csv.reader(dstr)
    for row in read_train:
        r = r+1
        print(r)
        dvec(row[4])

row_count_test = 0        

with open('Dataset_test_2.csv','r') as dste:
    read_test=csv.reader(dste)
    for row in read_test:
        row_count_test = row_count_test+1
 

doc_H1_vectors_test = np.zeros((row_count_test,no_of_H1_features))

r1 = -1

def dvec2(frow):
    tokenized=nltk.word_tokenize(frow)
    doc_H1_feature_list_test=[]
    
    for word in tokenized:
        if word != ',':
            doc_H1_feature_list_test.append(word)
            
    for i in range(no_of_H1_features):
        for w in doc_H1_feature_list_test:
            if w == H1_features[i]:
                doc_H1_vectors_test[r1][i] = doc_H1_vectors_test[r1][i]+1
                

with open('Dataset_test_2.csv','r') as dste:
    read_test=csv.reader(dste)
    for row in read_test:
        r1 = r1+1
        dvec2(row[4])