

def cosine_pred_class(particle_position,single_document_list):
    cosine_value=[]
    for cno in range (0,n_categories):
        sum_of_val=0
        sum_of_square_part=0
        sum_of_square_doc=0
        for ind_val_object in single_document_list:
            val_particle=particle_position[ind_val_object.index+no_of_features*cno]
            val_document=ind_val_object.value
            sum_of_val=sum_of_val+val_particle*val_document
            sum_of_square_part=sum_of_square_part+val_particle**2
            sum_of_square_doc=sum_of_square_doc+val_document**2
            
        if(sum_of_values==0):
            cosine_value.append(sum_of_values)
        else:
            consine_value.append(sum_of_values/((sum_of_squares_part**0.5)*(sum_of_squares_doc**0.5)))
            
    return(cosine_value.index(max(cosine_values)))
    
    

def fitness_function(particle_position,document_list,actual_class):
    predicted_class=[]
    for i in range(len(document_list)):
        predicted_class.append(cosine_pred_class(particle_position,document_list[i]))
    metric_dict=metrics.classification_report(actual_class, predicted_class, digits=4,output_dict=True)
    x=metric_dict['weighted avg']
    return(x['f1-score'])
    

    
    
    
    