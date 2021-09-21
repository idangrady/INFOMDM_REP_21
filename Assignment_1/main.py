import numpy as np
from  helper_funcs import gimi_indx, information_gain, _split, best_split, get_majority_in_class

class Node:
    #initiate a class of node with the necessary parms
    #init
    def __init__(self,data = None,feature = None, threhold = None, left = None, right= None,value = None): # Only leaf would have a value!!
        self.data = data
        self.feature = feature
        self.threhold =threhold
        self.left = left
        self.right = right
        self.value = value
    
    # check if the node is a lead node
    def is_leaf(self):
        return self.value is not None 
    
    
def tree_grow( x, y, nmin, minleaf,nfeat):
    """
    nmin:  if a node contains fewer cases than nmin
    minleaf :a split that creates a node with fewer than minleaf observations is not acceptable
    nfeat : Number of features to consider on every split
    """
    # initiate as parms ==> get the best from every node
    best_parms = {'feature': None, 'val': None, 'gini_indx':np.inf}
    d,n = x.shape
    

    pass

    # Check that the current node is following the constraints. 


def tree_pred(x,tr):
    # Predict
    pass


def get_data():
    return(np.genfromtxt(r'C:\Users\admin\Documents\GitHub\INFOMDM_REP_21\Assignment_1\credit.txt', delimiter=',', skip_header=True))

def create_node(cur_node,split_feat,trashhold):
    # take the parent node and return two nodes : Right and Left based on the Treshhold
    pass
    
    
credit_data= get_data()
print(best_split(credit_data[:,3], credit_data[:,5]))

