# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:30:55 2020

@author: HP
"""
## import no_of_features and no_of_documents_test,no_of_documents_train from whereveer available
import math
import  numpy 

no_of_features = total_no_of_features
no_of_documents_test = row_count_test
no_of_documents_train = row_count_train

class feature:
    def __init__(self,index,value):
        self.index=index
        self.value=value
def isNaN(num):
    return num != num

def make_listofnonzero_features(document_vector,no_of_documents,no_of_features):
    final_list=[]
    actual_class=[]
    for x in range(no_of_documents):
        print(x)
        temp=[]
        for y in range(no_of_features):
            if isNaN(document_vector[x][y])!=True and document_vector[x][y]!=0 :
                o1=feature(y,document_vector[x][y])
                temp.append(o1)
        if temp!=[]:                      ## for removing NaN lines
            final_list.append(temp)
            actual_class.append(document_vector[x][no_of_features])
    return final_list,actual_class





document_list_test,actual_class_test=make_listofnonzero_features(normalized_doc_vectors_test,no_of_documents_test,no_of_features)
document_list_train,actual_class_train=make_listofnonzero_features(normalized_doc_vectors_train,no_of_documents_train,no_of_features)
