
import nltk
import pandas as pd
import numpy as np
import csv
from numpy import linalg as LA

tf =  title_features 
mf =  meta_features 
kf =  key_features 
H1f =  H1_features 
H2f =  H2_features 
H3f =  H3_features 
#from Anchor_features import Anchor_features as af
#from Para_features import Para_features as pf
#from Strong_features import Strong_features as sf
#from Bold_features import Bold_features as bf
#from List_features import List_features as lf
#from Italic_features import Italic_features as If
#from Emphasis_features import Emphasis_features as ef

tvtr =  doc_title_vectors_train 
mvtr =  doc_meta_vectors_train 
kvtr =  doc_key_vectors_train 
H1vtr =  doc_H1_vectors_train 
H2vtr =  doc_H2_vectors_train 
H3vtr =  doc_H3_vectors_train 
#from Anchor_features import doc_Anchor_vectors_train as avtr
#from Para_features import doc_Para_vectors_train as pvtr
#from Strong_features import doc_Strong_vectors_train as svtr
#from Bold_features import doc_Bold_vectors_train as bvtr
#from List_features import doc_List_vectors_train as lvtr
#from Italic_features import doc_Italic_vectors_train as ivtr
#from Emphasis_features import doc_Emphasis_vectors_train as evtr

tvte = doc_title_vectors_test 
mvte = doc_meta_vectors_test 
kvte = doc_key_vectors_test 
H1vte = doc_H1_vectors_test 
H2vte = doc_H2_vectors_test 
H3vte = doc_H3_vectors_test 
#from Anchor_features import doc_Anchor_vectors_test as avte
#from Para_features import doc_Para_vectors_test as pvte
#from Strong_features import doc_Strong_vectors_test as svte
#from Bold_features import doc_Bold_vectors_test as bvte
#from List_features import doc_List_vectors_test as lvte
#from Italic_features import doc_Italic_vectors_test as ivte
#from Emphasis_features import doc_Emphasis_vectors_test as evte

ntf = no_of_title_features 
nmf = no_of_meta_features 
nkf = no_of_key_features 
nH1f = no_of_H1_features 
nH2f = no_of_H2_features 
nH3f = no_of_H3_features 
#from Anchor_features import no_of_Anchor_features as naf
#from Para_features import no_of_Para_features as npf
#from Strong_features import no_of_Strong_features as nsf
#from Bold_features import no_of_Bold_features as nbf
#from List_features import no_of_List_features as nlf
#from Italic_features import no_of_Italic_features as nif
#from Emphasis_features import no_of_Emphasis_features as nef

rctr = row_count_train 
rcte = row_count_test 

temp_features = np.concatenate((tf,mf))
temp_features2 = np.concatenate((temp_features,kf))
temp_features3 = np.concatenate((temp_features2,H1f))
temp_features4 = np.concatenate((temp_features3,H2f))
#temp_features5 = np.concatenate((temp_features4,H3f))
#temp_features6 = np.concatenate((temp_features5,af))
#temp_features7 = np.concatenate((temp_features6,pf))
#temp_features8 = np.concatenate((temp_features7,sf))
#temp_features9 = np.concatenate((temp_features8,bf))
#temp_features10 = np.concatenate((temp_features9,lf))
#temp_features11 = np.concatenate((temp_features10,If))
total_features = np.concatenate((temp_features4,H3f))

total_no_of_features = ntf + nmf + nkf + nH1f + nH2f + nH3f #+ naf + npf + nsf + nbf + nlf + nif + nef

temp_doc_vector_train = np.concatenate((tvtr,mvtr),1)
temp_doc_vector_train2 = np.concatenate((temp_doc_vector_train,kvtr),1)
temp_doc_vector_train3 = np.concatenate((temp_doc_vector_train2,H1vtr),1)
temp_doc_vector_train4 = np.concatenate((temp_doc_vector_train3,H2vtr),1)
#temp_doc_vector_train5 = np.concatenate((temp_doc_vector_train4,H3vtr),1)
#temp_doc_vector_train6 = np.concatenate((temp_doc_vector_train5,avtr),1)
#temp_doc_vector_train7 = np.concatenate((temp_doc_vector_train6,pvtr),1)
#temp_doc_vector_train8 = np.concatenate((temp_doc_vector_train7,svtr),1)
#temp_doc_vector_train9 = np.concatenate((temp_doc_vector_train8,bvtr),1)
#temp_doc_vector_train10 = np.concatenate((temp_doc_vector_train9,lvtr),1)
#temp_doc_vector_train11 = np.concatenate((temp_doc_vector_train10,ivtr),1)
document_vectors_train = np.concatenate((temp_doc_vector_train4,H3vtr),1)

temp_doc_vector_test = np.concatenate((tvte,mvte),1)
temp_doc_vector_test2 = np.concatenate((temp_doc_vector_test,kvte),1)
temp_doc_vector_test3 = np.concatenate((temp_doc_vector_test2,H1vte),1)
temp_doc_vector_test4 = np.concatenate((temp_doc_vector_test3,H2vte),1)
#temp_doc_vector_test5 = np.concatenate((temp_doc_vector_train4,H3vte),1)
#temp_doc_vector_test6 = np.concatenate((temp_doc_vector_train5,avte),1)
#temp_doc_vector_test7 = np.concatenate((temp_doc_vector_train6,pvte),1)
#temp_doc_vector_test8 = np.concatenate((temp_doc_vector_train7,svte),1)
#temp_doc_vector_test9 = np.concatenate((temp_doc_vector_train8,bvte),1)
#temp_doc_vector_test10 = np.concatenate((temp_doc_vector_train9,lvte),1)
#temp_doc_vector_test11 = np.concatenate((temp_doc_vector_train10,ivte),1)
document_vectors_test = np.concatenate((temp_doc_vector_test4,H3vte),1)

#normalizing training document vectors 
normalized_doc_vectors_train = np.zeros((rctr,total_no_of_features)) 

      
for i in range(rctr):
    norm_factor = LA.norm(document_vectors_train[i])
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
            actual_and_predicted_train[i][0] = 0
        if row[14] == 'Business':
            actual_and_predicted_train[i][0] = 1
        if row[14] == 'Science':
            actual_and_predicted_train[i][0] = 2
        if row[14] == 'Sports':
            actual_and_predicted_train[i][0] = 3
        
        i = i + 1
        
normalized_doc_vectors_test = np.concatenate((normalized_doc_vectors_test,actual_and_predicted_test),1)   
    
    
    
    
    
    
    
    
    
    
    