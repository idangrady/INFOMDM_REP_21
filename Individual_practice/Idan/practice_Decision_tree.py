import numpy as np
import pandas as pd
from collections import Counter 
# Check wherver we could use this library to the Assignment ==> Basically it gets an array/list and order based on most frequenst appearing
#Similar to Hot_One Function or bag of words


def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return - np.sum([p * np.log2(p) for p in ps if p>0])
    

class Node:
    def __init__(self,feature = None, threhold = None, left = None, right= None, value = None):
        self.feature = feature
        self.threhold =threhold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None #check why ==> I did not understand this
    
class Decision_Tree():
    def __innit__(self,min_sample_split = 2, max_depth = 100, n_feats = None):
        
        #N_Feats randomly selects our decision
        
        #Innit
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        
    def fit(self,x,y):
        d,n = x.shape
        self.n_feats = n if not self.n_feats else min(n,self.n_feats)
         
        #Growing the tree
        
        #Stopping criteria: 
            

        
        
    def grow_tree(self,x,y,depth = 0):
        d,n = x.shape
        n_label = len(np.unique(y)) # Getting all unique different labels
        if(depth <= self.max_depth
           or n_label == 1
           or n <self.min_sample_split):
            leaf_value = self.most_common_l(y)
            return(Node(value = leaf_value))
        # Identifying the root nod with dunction grow_Tree
        self.root = self.grow_tree(x,y)
        #Cecking who apears most frequenst
        
        features_idxs = np.random.choice(n,self.n_feats,replace = False)
        best_feature, best_tresh = self.best_criteria_(self,x,y,features_idxs)
        
    def best_criteria_(self,x,y,features):
        best_gain=-1
        best_split_inx,best_split_thresh = None,None
        
        # it is a greedy algo
        # We should itteate through all of them and check wherever the entropy is the smallest.
        for feat_idx in features: 
            #Select Col vector of x
            col_sample = x[:,feat_idx]
            # check throgh all possible Threshhol ==> Greedy Algo
            threshholds = np.unique(col_sample) # all unique possible treshholds in the Col_Smaple 
            
            # Find the best information gain of the possible split
            for thresh in threshholds:
                current_gain =self.information_gain(y,col_sample,thresh)
                if current_gain >best_gain:
                    best_gain =current_gain
                    best_split_inx = feat_idx
                    best_split_thresh = thresh
            
            
        
        
        
    def information_gain(self,y,x_col,split_thresh):
        #Parten Entropy
        parent_enttropy = entropy(y)
        #Generate_split
        left_idx,right_idx = self.split(x_col,split_thresh)
        #Wieghtend avg child Entropy
        #Claculating Chile Entropy
        num_features = len(y)
        
        chid_ent =0
        
        return(parent_enttropy-chid_ent)
        
        
    
    def _split(self,x_col,split_thresh):
        left_values = np.argewhere(x_col <= split_thresh).flatten()
        right_values =  np.argewhere(x_col > split_thresh).flatten()
        return (left_values,right_values)
    
    
    def most_common_l(self,y):
        counter = Counter(y)
        most_comm_label,freq_most_common = counter.most_common()[0] # ger the most commot attribute in array plus amount of apperence
        return (most_comm_label,freq_most_common)
                
    def tree_pred(self,X):
        #travers our tree
        pass