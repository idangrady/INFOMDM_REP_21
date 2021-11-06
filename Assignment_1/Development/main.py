import numpy as np
from  helper_funcs import gimi_indx, information_gain, _split, best_split, get_majority_in_class, check_if_possible
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree
class Node:
    #initiate a class of node with the necessary parms
    #init
    def __init__(self,data = None,feature = None, threhold = None, left = None, right= None,value = None): # Only leaf would have a value!!
        self.data = data # including the y
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
    parent_data =x.data
    d,n = parent_data.shape
    best_parms = {'feature': None,'b_split': np.inf, 'b_trashold': None}
    
    if (check_if_possible(x,nmin)==True):
        for col_feat in range(n-2):        
            if col_feat ==2:
                print("2")
            best_splt_x_1, best_trash_x_1 = best_split(parent_data[:,col_feat], y,nmin,minleaf,nfeat)
            if(best_splt_x_1 <best_parms['b_split'] ):
                best_parms['feature'] = col_feat
                best_parms['b_split'] = best_splt_x_1
                best_parms['b_trashold'] = best_trash_x_1
      
    if (best_parms['b_split'] == np.inf):
        most_comm_label,freq_most_common = get_majority_in_class(x.data[:,-1])
        x.value = most_comm_label
        return x

    parent_data = parent_data= parent_data[np.argsort(parent_data[:, best_parms['feature']])]       # we sort the data check the y
    left_node_x = parent_data[parent_data[:,best_parms['feature']] <= best_parms['b_trashold']]
    right_node_x =parent_data[parent_data[:,best_parms['feature']] > best_parms['b_trashold']]
    if left_node_x.shape[0] ==2:
        print("hear")

    


    # check if we need to flip the +1 
    left_node = Node(data=right_node_x, feature =None, threhold=best_parms['b_trashold'])
    right_node = Node(data=right_node_x, feature =None, threhold=best_parms['b_trashold'])
    
    #Creating both children nodes and assign to them current value ==> needed to be deleted in case of splitting in the future. 
    left_node = Node(data =left_node_x ,feature = best_parms['feature'],threhold = best_parms['b_trashold'])
    right_node = Node(data =right_node_x ,feature = best_parms['feature'],threhold = best_parms['b_trashold'])
    
    x.left = left_node
    x.right = right_node
    
    print(parent_data)
    print(left_node.data)
    print(right_node.data)
    print(best_parms)
    
    # index at the top   
    tree_grow(left_node,left_node.data[:,-1],nmin,minleaf,nfeat)
    tree_grow(right_node,right_node.data[:,-1],nmin,minleaf,nfeat)

    # Check that the current node is following the constraints. 


def tree_pred(x,tr):
    # Predict
    pass


def get_data():
    #return(np.genfromtxt(r'/Users/Marc/Documents/UU/Master - Computing Science/2021-2022/INFOMDM_REP_21/INFOMDM_REP_21/Assignment_1/credit.txt', delimiter=',', skip_header=True))
    return(np.genfromtxt(r'C:\Users\admin\Documents\GitHub\INFOMDM_REP_21\Assignment_1\credit.txt', delimiter=',', skip_header=True))

def create_node(cur_node,split_feat,trashhold):
    # take the parent node and return two nodes : Right and Left based on the Treshhold
    pass
    
    
credit_data= get_data()


data_ =Node(data = credit_data)
our_tree_data=tree_grow(data_,data_.data[:,-1],3,3,2)

clf = DecisionTreeClassifier(criterion = 'gini', min_samples_leaf=3,min_samples_split=3) # Train Decision Tree Classifer clf = clf.fit(X_train,y_train)
precg = clf.fit(data_.data[:,:-1], data_.data[:,-1])

#check = clf(criterion = 'gini', min_samples_leaf=3,min_samples_split=3)


sklearn.tree.plot_tree(precg, filled=True)
print("ss")
#print(best_split(credit_data[:,3], credit_data[:,5]))