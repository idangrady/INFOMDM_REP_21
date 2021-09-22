#Helper functions 

#Libraries
import numpy as np
from collections import Counter 

#   Functions: 
    
    
# function that return the Gimi impurity
def gimi_indx(e): 
    """
    we assume all the values are 0 1
    """
    if len(e)==0: return 0#             return if the tree is empty
    else:
        prob = np.sum(e)/len(e)#        find probability     
        return(prob * (1-prob))#        return gini index entropy
        

#Funtion that return the information gain 
def information_gain(y,X_colm,split_tresh):
    # Calculate the parent Entropy
    parent_ent = gimi_indx(y)
    
    # Calculate the weighted childs Entropy
    left_child ,right_child  = _split(X_colm,split_tresh)
    
    if(len(left_child) ==0 or len(right_child)==0): return 0 # check if we there is reason to continue. If its 0 ==> Nothing changed from the parent node

    # weight_child
    weight_left = len(left_child) / len(y)
    weight_right = len(right_child) / len(y)

    output_information_gain= parent_ent - (weight_left *gimi_indx(left_child) +weight_right * (gimi_indx(right_child))) # subctracting from the parent the weighted Gimi fo each child 
    return(output_information_gain)

#helper function for the information gain
def _split(x_colm, split_trash):
    left_indx = np.argwhere(x_colm <=split_trash )
    right_indx = np.argwhere(x_colm>split_trash)
    return(left_indx, right_indx)

#function that returns the best split in a current option 
def best_split(X,y,nmin, minleaf,nfeat):
    """
    we assume that all the data is numeric
    """
    length_node = len(X)
    # find all possible splits
    x_sort = np.array(np.sort(np.unique(X)))
    possible_splits = (x_sort[0:len(x_sort)-1] + x_sort[1:]) /2
    best_parms = {'b_split': np.inf, 'b_trashold': None}
    step_ =1/length_node

    
    # itterating throught all of the possible trashholds to find the best split
    for idx_tr, curr_trash in enumerate(possible_splits):
        group_l = y[X<= curr_trash]
        group_r = y[X> curr_trash]
        
        # checking the constraint. If 
        if (len(group_l) <minleaf 
            or len(group_r)< minleaf
            or group_l.shape[1]<nfeat
            or group_r[1]<nfeat):
            pass
        
            
        # find gini
        ent_l = (len(group_l)*step_)*(gimi_indx(group_l))
        ent_r = (len(group_r)*step_)* gimi_indx(group_r)
        result_ent = ent_l+ ent_r
        
        if(result_ent <best_parms['b_split'] ): # updating the best parms if needed
            best_parms['b_split'] =result_ent ; best_parms['b_trashold'] = curr_trash
    return(best_parms['b_split'],best_parms['b_trashold'])


#function that return the most common value in the array ==> Would be valueable for the predict function ==> Where we need to determine based on the amount of visibale outcomes our prediction
def get_majority_in_class(y):
    counter = Counter(y)
    most_comm_label,freq_most_common = counter.most_common()[0] # ger the most commot attribute in array plus amount of apperence
    return (most_comm_label,freq_most_common)