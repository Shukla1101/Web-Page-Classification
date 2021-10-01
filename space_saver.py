#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:16:46 2020

@author: rahulshukla
"""


import nltk
import pandas as pd
import numpy as np
import csv
from numpy import linalg as LA
 


no_of_title_features = 11898
ntf = no_of_title_features 
nmf = no_of_meta_features 
nkf = no_of_key_features 
nH1f = no_of_H1_features 
nH2f = no_of_H2_features 
nH3f = no_of_H3_features 


rctr = row_count_train 
rcte = row_count_test 

total_features = np.concatenate((title_features,meta_features))
total_features = np.concatenate((total_features,key_features))
total_features = np.concatenate((total_features,H1_features))
total_features = np.concatenate((total_features,H2_features))
total_features = np.concatenate((total_features,H3_features))

total_no_of_features = ntf + nmf + nkf + nH1f + nH2f + nH3f 

document_vectors_train = np.concatenate((doc_title_vectors_train ,doc_meta_vectors_train),1)
document_vectors_train = np.concatenate((document_vectors_train,doc_key_vectors_train),1)
document_vectors_train = np.concatenate((document_vectors_train,doc_H1_vectors_train),1)
document_vectors_train = np.concatenate((document_vectors_train,doc_H2_vectors_train),1)
document_vectors_train = np.concatenate((document_vectors_train,doc_H3_vectors_train),1)

document_vectors_test = np.concatenate((doc_title_vectors_test,doc_meta_vectors_test),1)
document_vectors_test = np.concatenate((document_vectors_test,doc_key_vectors_test),1)
document_vectors_test = np.concatenate((document_vectors_test,doc_H1_vectors_test),1)
document_vectors_test = np.concatenate((document_vectors_test,doc_H2_vectors_test),1)
document_vectors_test = np.concatenate((document_vectors_test,doc_H3_vectors_test),1)



#normalizing training document vectors 
normalized_doc_vectors_train = np.zeros((rctr,total_no_of_features)) 

      
for i in range(1,rctr):
    norm_factor = LA.norm(document_vectors_train[i])
    print (i)
    normalized_doc_vectors_train[i] = document_vectors_train[i]/norm_factor


actual_and_predicted_train = np.full((rctr,2),-1)
with open('Dataset_train_2.csv','r') as ds:
    
    read=csv.reader(ds)   
    i=0
    
    for row in read:
        if row[14] == 'Arts':
            actual_and_predicted_train[i][0] = 0
        if row[14] == 'Business':
            actual_and_predicted_train[i][0] = 1
        if row[14] == 'Science':
            actual_and_predicted_train[i][0] = 2
        if row[14] == 'Sports':
            actual_and_predicted_train[i][0] = 3
        
        i = i + 1
        
normalized_doc_vectors_train = np.concatenate((normalized_doc_vectors_train,actual_and_predicted_train),1) 
  
#normalizing testing document vectors 
normalized_doc_vectors_test = np.zeros((rcte,total_no_of_features)) 

      
for i in range(rcte):
    norm_factor = LA.norm(document_vectors_test[i])
    
    normalized_doc_vectors_test[i] = document_vectors_test[i]/norm_factor


actual_and_predicted_test = np.full((rcte,2),-1)
with open('Dataset_test_2.csv','r') as ds:
    
    read=csv.reader(ds)
    
    i=0
    
    for row in read:
        if row[14] == 'Arts':
            actual_and_predicted_test[i][0] = 0
        if row[14] == 'Business':
            actual_and_predicted_test[i][0] = 1
        if row[14] == 'Science':
            actual_and_predicted_test[i][0] = 2
        if row[14] == 'Sports':
            actual_and_predicted_test[i][0] = 3
        
        i = i + 1
        
normalized_doc_vectors_test = np.concatenate((normalized_doc_vectors_test,actual_and_predicted_test),1)   
      
    
    
    
    
    
    
    
    
     