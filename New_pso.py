##imports
import random
import numpy as np 
from array import *
import nltk
import pandas as pd
import csv
from numpy import linalg as LA
import scipy
import random
import numpy as np 
from array import *
from sklearn import metrics



def cosine_pred_class(particle_position,single_document_list):
    cosine_value=[]
    for cno in range (0,n_categories):
        sum_of_values=0
        sum_of_square_part=0
        sum_of_square_doc=0
        for ind_val_object in single_document_list:
            val_particle=particle_position[ind_val_object.index+ n_features * cno]
            val_document=ind_val_object.value
            sum_of_values=sum_of_values + val_particle*val_document
            sum_of_square_part=sum_of_square_part+val_particle**2
            sum_of_square_doc=sum_of_square_doc+val_document**2
            
        if(sum_of_values==0):
            cosine_value.append(sum_of_values)
        else:
            cosine_value.append(sum_of_values/((sum_of_square_part**0.5)*(sum_of_square_doc**0.5)))
            
    return(cosine_value.index(max(cosine_value)))
    
    

def fitness_function(particle_position,document_list,actual_class):
    predicted_class=[]
    for i in range(len(document_list)):
        predicted_class.append(cosine_pred_class(particle_position,document_list[i]))
    metric_dict=metrics.classification_report(actual_class, predicted_class, digits=4,output_dict=True)
    x=metric_dict['weighted avg']
    return(x['f1-score'])
    

    
    

###
## import no_of features as n_features from the file from where it need to be imported
## first run make_sparse_list file for making document_list_train as well as test and actual_clas_test and train
W = 0.5
c1 = 0.8
c2 = 0.9
target = 1

n_features = total_no_of_features

n_categories = int(input("inform no. of categories: "))
n_iterations = int(input("Inform the number of iterations: "))
target_error = float(input("Inform the target error: "))
n_particles = int(input("Inform the number of particles: "))

actual_class_train = [round(x) for x in actual_class_train]
actual_class_test = [round(x) for x in actual_class_test]


particle_position_vector = np.random.rand(n_particles, n_categories * n_features)
pbest_position = particle_position_vector
pbest_fitness_value = np.array([0 for _ in range(n_particles)])
gbest_fitness_value = 0
gbest_position = np.array([0 for _ in range(n_features*n_categories)])
velocity_vector = np.zeros((n_particles, n_categories * n_features))

iteration = 0

while iteration < n_iterations:
    for i in range(n_particles):
        fitness_cadidate = fitness_function(particle_position_vector[i],document_list_train,actual_class_train)
        
        
        
        if(pbest_fitness_value[i] < fitness_cadidate):
            pbest_fitness_value[i] = fitness_cadidate
            pbest_position[i] = particle_position_vector[i]

        if(gbest_fitness_value < fitness_cadidate):
            gbest_fitness_value = fitness_cadidate
            gbest_position = particle_position_vector[i]

    if(abs(gbest_fitness_value - target) < target_error):
        break
    
    for i in range(n_particles):
        new_velocity = (W*velocity_vector[i]) + (c1*random.random()) * (pbest_position[i] - particle_position_vector[i]) + (c2*random.random()) * (gbest_position-particle_position_vector[i])
        new_position = new_velocity + particle_position_vector[i]
        particle_position_vector[i] = new_position

    iteration = iteration + 1
    print(gbest_fitness_value)

print("Final f1 value with the best particle is ",fitness_function(gbest_position,document_list_test, actual_class_test ))