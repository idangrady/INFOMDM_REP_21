# Marc de Fluiter (5928087)
# Idan Grady (7304447)
# Pascal Verkade (6045057)

import numpy as np

def tree_grow(x, y, nmin, minleaf, nfeat):
    """Grows a classification tree on the input data under the given constraints
    
    Params:
        x (np.array(2-D) (numeric data)):    Data matrix on which we want to grow the classification tree
        y (np.array(1-D) (0/1 data)):        Vector with class labels corresponding to the data matrix
        nmin (type):                         Value for the nmin constraint, the number of observations below which a node has to become a leaf node
        minleaf (int):                       Value for the minleaf constraint, the minimum number of observations leaf nodes have to contain after splitting the data
        nfeat (type):                        Value for the nfeat constraint, the number of features to consider at each split
    
    return: 
        Classification_Tree:    The classification tree obtained from the input data under the given constraints
    """

    # Instantiate the tree and add the amount of rows for both class labels
    tree = Classification_Tree()
    rows_class_0 = sum(y == 0)
    rows_class_1 = sum(y == 1)
    tree.rows_0 = rows_class_0
    tree.rows_1 = rows_class_1

    # Check if node is pure
    if rows_class_0 == 0 or rows_class_1 == 0:

        # If node is pure return node as leaf with the correct classification
        tree.class_label = 0 if rows_class_0 >= rows_class_1 else 1
        return tree

    # Get amount of rows and columns
    nrows, ncolumns = x.shape

    # Check nmin constraint
    if nmin is not None and nrows < nmin:

        # With too few rows return node as leaf with the correct classification
        tree.class_label = 0 if rows_class_0 >= rows_class_1 else 1
        return tree

    # Determine columns to be considered for splitting
    if nfeat is None or nfeat >= ncolumns:
        # When nfeat is not given or bigger than amount of columns consider all columns
        column_numbers = range(ncolumns)
    else:
        # When nfeat is given we consider nfeat random columns
        column_numbers = np.random.choice(ncolumns, nfeat, replace = False)

    # Initialize information about the best split
    best_split_info = {'Feature': None, 'Threshold': None, 'Impurity': np.inf}

    # Find best split over the given features/column numbers
    for feature in column_numbers:

        # Determine the best split for the current feature
        feature_threshold, feature_impurity = best_split(x, y, feature, minleaf)

        # If the found impurity is lower than the impurity of the current best split, we keep this new split
        if feature_impurity < best_split_info['Impurity']:
            best_split_info = {'Feature': feature, 'Threshold': feature_threshold, 'Impurity': feature_impurity}


    # Check if a split is found under the given constraints
    if best_split_info['Feature'] is None:

        # When no split is found return node as leaf with the correct classification
        tree.class_label = 0 if rows_class_0 >= rows_class_1 else 1
        return tree

    # Retrieve information about the best split
    best_split_feature   = best_split_info['Feature']
    best_split_threshold = best_split_info['Threshold']

    # Add split information to the tree
    tree.feature   = best_split_feature
    tree.threshold = best_split_threshold


    # Determine the indices of the split in the data and class labels according to the best split
    left_node_column_indices  = x[:, best_split_feature] <= best_split_threshold
    right_node_column_indices = x[:, best_split_feature] >  best_split_threshold

    # Grow the left and right children trees and add them to the parent node
    tree.left_tree  = tree_grow(x[left_node_column_indices],  y[left_node_column_indices],  nmin, minleaf, nfeat)
    tree.right_tree = tree_grow(x[right_node_column_indices], y[right_node_column_indices], nmin, minleaf, nfeat)

    # Return the full tree
    return tree


def tree_pred(x, tr):
    """Predicts the class labels for a data matrix with a single classification tree
    
    Params:
        x (np.array(2-D)):           Data matrix for which we want to predict the class labels
        tr (Classification_Tree):    The classification tree object used for classification
    
    return: 
        [int]:    A list containing the class label prediction for each data entry 
    """
    
    # Return the class label prediction for each matrix entry
    return [tree_pred_entry(entry, tr) for entry in x]


def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    """Grows a list of classification trees on m bootstramp samples of the input data under the given constraints
    
    Params:
        x (np.array(2-D) (numeric data)):    Data matrix on which we want to grow the classification tree
        y (np.array(1-D) (0/1 data)):        Vector with class labels corresponding to the data matrix
        nmin (int):                          Value for the nmin constraint, the number of observations below which a node has to become a leaf node
        minleaf (int):                       Value for the minleaf constraint, the minimum number of observations leaf nodes have to contain after splitting the data
        nfeat (int):                         Value for the nfeat constraint, the number of features to consider at each split
        m (int):                             The number of bootstramp samples we want to take
    
    return: 
        [Classification_Tree]:    A list of classification trees obtained from m bootstrap samples of the input data under the given constraints
    """

    # Draw m bootstrap samples
    bootstrap_samples = [bootstrap_sample(x, y) for _ in range(m)]

    # Execute the tree_grow procedure on every sample
    return [tree_grow(sample, y_, nmin, minleaf, nfeat) for sample,y_ in bootstrap_samples]


def tree_pred_b(x, tr_list):
    """Predicts the class labels for a data matrix with a list of classification trees using majority voting
    
    Params:
        x (np.array(2-D) (numeric data)):    Data matrix for which we want to predict the class labels
        tr_list ([Classification_Tree]):     A list containing classification tree objects which are used for classification
    
    return: 
        np.array(1-D) (0/1 data):    A vector which contains the class label prediction for each data entry 
    """

    predict_ = sum(
        np.array([float(i) for i in tree_pred(x, entry)]) for entry in tr_list
    )

    predict_ /= len(tr_list)
    predict_ = np.where(predict_>=0.5,1,0)
    return(predict_)


def best_split(x, y, feature, minleaf):
    """Determines the best possible split in the input data on the given feature
    
    Params:
        x (np.array(2-D) (numeric data)):    Data matrix for which we want to determine the best split
        y (np.array(1-D) (0/1 data)):        Vector with class labels corresponding to the data matrix
        feature (int):                       Column number of the feature on which we determine the best split
        minleaf (int):                       Value for the minleaf constraint, the minimum number of observations the leaf nodes have to contain after splitting

    return: 
        (float, float):    A pair of the threshold value of the best split found together with the impurity of this split
    """

    # Retrieve the specific column and its length
    column = x[:, feature]
    length_column = len(column)

    # Enumerate all possible split points
    sorted_column = np.array(np.sort(np.unique(column)))
    possible_splitpoints = (sorted_column[:-1] + sorted_column[1 :]) / 2

    # Find the best split point
    best_split_info = {'Threshold': None, 'Impurity': np.inf}

    for threshold in possible_splitpoints:

        # Split the class labels on the threshold value
        left_data  = y[column <= threshold]
        right_data = y[column >  threshold]

        # Check minleaf constraint
        if minleaf is not None and (len(left_data) < minleaf or len(right_data) < minleaf):
            # When one part has too few rows we do not consider this split point
            continue

        # Determine the gini impurity of this split
        left_impurity  = len(left_data)  / length_column * gini_impurity(left_data)
        right_impurity = len(right_data) / length_column * gini_impurity(right_data)
        split_impurity = left_impurity + right_impurity

        # When the impurity is lower then the current best split we use this new split
        if split_impurity < best_split_info['Impurity']: 
            best_split_info = {'Threshold': threshold, 'Impurity': split_impurity}

    # Return the best split information
    return (best_split_info['Threshold'], best_split_info['Impurity'])


def gini_impurity(labels):
    """Determines the gini impurity of the given class labels
    
    Params:
        labels (np.array(1-D) (numeric data)):    The class labels from which we want to calculate the gini impurity
    
    return: 
        float:    The gini impurity of the data
    """

    if len(labels) == 0:
        # Return 0 if the data is empty
        return 0

    prob = np.sum(labels) / len(labels)
    return prob * (1 - prob) 


def tree_pred_entry(x_entry, tr):
    """Predicts the class label for a data entry with a single classification tree
    
    Params:
        x_entry (np.array(1-D) (numeric data)):     Vector containing the values of a data entry
        tr (Classification_Tree):                   The classification tree object used for classification
    
    return: 
        int:    The class label prediction
    """

    # Return the class label if it is a leaf node
    if tr.class_label is not None:
        return tr.class_label

    # Otherwise drop down the tree from the current split node
    if x_entry[tr.feature] <= tr.threshold:
        # For a value below (or equal to) the threshold we drop down the left tree
        return tree_pred_entry(x_entry, tr.left_tree)
    else:
        # For a value above the threshold we drop down the right tree
        return tree_pred_entry(x_entry, tr.right_tree)


def bootstrap_sample(x, y):
    """Draws a bootstramp sample from a dataset consisting of a data matrix and a vector with class labels
    
    Params:
        x (np.array(2-D) (numeric data)):    Data matrix from which we want to draw a bootstrap sample
        y (np.array(1-D) (0/1 data)):        Vector with class labels corresponding to the data matrix 
    
    return: 
        (np.array(2-D) (numeric data), np.array(1-D) (0/1 data)):    A pair of the sampled matrix data together with the class labels which are sampled accordingly
    """

    # Determine the row count of the data
    nrows, _ = x.shape

    # Draw this amount of row numbers randomly, with replacement
    row_numbers = np.random.choice(nrows, size = nrows, replace = True)

    # Return the bootstrap sample based on the random list of row numbers
    return ([x[row_numbers, :], y[row_numbers]])



class Classification_Tree:
    """ The Classification Tree object is used to represent a classification tree consisting of data entries, parameters for use in classifiers and subtrees
    
    Params:
        rows_0 (int):                       The amount of rows with class label 0
        rows_1 (int):                       The amount of rows with class label 1
        feature (int):                      The Column number of the feature on which we determined the split of this (sub)tree / root node
        threshold (float):                  The threshold value of the best split of this (sub)tree / root node
        left_tree (Classification_Tree):    Left subtree of the root node of our this/current classification tree (only in split nodes)
        right_tree (Classification_Tree):   Right subtree of the root node of this/current classification tree (only in split nodes)
        class_label (int):                  Label classified by majority class  (only in leaf node)
    """
    # Classification tree node class which stores the relevant information for a node in the classification tree
    
    # Initializer for a classification tree node object
    def __init__(self, rows_0 = None, rows_1 = None, feature = None, threshold = None, left_tree = None, right_tree = None, class_label = None):

        # The amount of rows with class label 0 and 1
        self.rows_0 = rows_0
        self.rows_1 = rows_1

        # The feature and the threshold value of the split, only for split nodes
        self.feature = feature
        self.threshold = threshold

        # Information of the trees starting from both children nodes, only for split nodes
        self.left_tree = left_tree
        self.right_tree = right_tree
        
        # The majority class label prediction, only in leaf node
        self.class_label = class_label