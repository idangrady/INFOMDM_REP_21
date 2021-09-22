#Try something

import numpy as np
from  helper_funcs import gimi_indx, information_gain, _split, best_split, get_majority_in_class

class Node:
    #initiate a class of node with the necessary parms
    #init
    def __init__(self,data= None, feature = None, threhold = None, left = None, right= None,value = None): # Only leaf would have a value!!
        self.data = data
        self.feature = feature
        self.threhold =threhold
        self.left = left
        self.right = right
        self.value = value
        
        

credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)

d,n = credit_data.shape

y = credit_data[:,-1]
x = credit_data[:,:-1]


def check_if_possible(X,feature,nmin):
    parent_data = X.data
    row_d, feat_n = parent_data.shape
    # Checking if we dont violate 
    if(nmin<row_d):
        return None
    else: True
    
    
    
# =============================================================================
#     # Checking wherever I can split the 
##    parent_data= parent_data[np.argsort(parent_data[:, feature])]

#     left_node_x = parent_data[:feature,:]
#     right_node_x =parent_data[feature +1:,:]
#     
#     if (left_node_x.shape[1] <= nfeat
#         or right_node_x.shape[1] <= nfeat):
#         return None
# 
# =============================================================================
    
    


def tree_grow(x, y, nmin, minleaf,nfeat):
    parent_data =x.data
    d,n = parent_data.shape
    best_parms = {'feature': None,'b_split': np.inf, 'b_trashold': None}
    
    if check_if_possible(x,y,nmin):
        for col_feat in range(n-2):        
                best_splt_x_1, best_trash_x_1 = best_split(parent_data[:,col_feat], y,2,2,4)
                if (check_if_possible(x,y,best_splt_x_1,col_feat,nmin,nfeat)):     
                    best_parms['feature'] = col_feat
                    best_parms['b_split'] = best_splt_x_1
                    best_parms['b_trashold'] = best_trash_x_1
                    
    if (best_parms['b_trashold'] == None):
        most_comm_label,freq_most_common = get_majority_in_class(y)
        x.value = most_comm_label
        return x
    parent_data = parent_data= parent_data[np.argsort(parent_data[:, best_parms['feature']])]       # we sort the data check the y
    left_node_x = parent_data[:best_parms['feature'],:]
    right_node_x =parent_data[:best_parms['feature']+1,:]
    
    # check if we need to flip the +1 
    left_node = Node(data=right_node_x, feature =None, threhold=best_parms['b_trashold'])
    right_node = Node(data=right_node_x, feature =None, threhold=best_parms['b_trashold'])
    
    
    #Creating both children nodes and assign to them current value ==> needed to be deleted in case of splitting in the future. 
    left_node = Node(data =left_node_x ,feature = best_parms['feature'],threhold = best_parms['b_trashold'])
    right_node = Node(data =right_node_x ,feature = best_parms['feature'],threhold = best_parms['b_trashold'])
    
    x.left = left_node
    x.right = right_node
    
    # index at the top
    tree_grow(left_node,left_node.data[:,-1],nmin,minleaf,nfeat)
    tree_grow(right_node,right_node.data[:,-1],nmin,minleaf,nfeat)
     







data_ =Node(data = x)
tree_grow(data_,y,2,2,2)



def assign_childern(curr_node, feature, trashhold):
    left_node = curr_node[curr_node[:,feature]<= trashhold]
    right_node = curr_node[curr_node[:,feature]> trashhold]
    
    return(left_node,right_node)