import numpy as np
from collections import Counter 

class Classification_Tree:
    
    # Initializer for a tree object
    def __init__(self, rows_0 = None, rows_1 = None, feature = None, threshold = None, left_tree = None, right_tree = None, class_label = None):

        self.rows_0 = rows_0
        self.rows_1 = rows_1

        # A leaf node does not split the data into sub-trees
        self.feature = feature
        self.threshold = threshold

        self.left_tree = left_tree
        self.right_tree = right_tree
        
        # Only a leaf node contains a class label prediction
        self.class_label = class_label


# Function that grows the classification tree
def tree_grow(x, y, nmin, minleaf, nfeat):
    """
    x:       matrix with data
    y:       vector with class labels
    nmin:    if a node contains fewer cases than nmin
    minleaf: a split that creates a node with fewer than minleaf observations is not acceptable
    nfeat:   Number of features to consider on every split
    """
    
    # Instantiate the tree and add the amount of rows for both class labels
    tree = Classification_Tree()
    rows_class_0 = sum(y == 0)
    rows_class_1 = sum(y == 1)
    tree.rows_0 = rows_class_0
    tree.rows_1 = rows_class_1


    # Check if node is pure
    if rows_class_0 == 0 or rows_class_1 == 0:

        # If node is pure return node as leaf with classification
        if rows_class_0 >= rows_class_1:
            tree.class_label = 0
        else:
            tree.class_label = 1

        return tree


    # Get amount of rows and columns
    nrows, ncolumns = x.shape

    # Check nmin constraint
    if nmin is not None and nrows < nmin:

        # With too few rows return node as leaf with classification
        if rows_class_0 >= rows_class_1:
            tree.class_label = 0
        else:
            tree.class_label = 1

        return tree


    # Determine columns to be considered for splitting
    if nfeat is None or nfeat >= ncolumns:
        # When nfeat is not given or bigger than amount of columns consider all columns
        column_numbers = range(ncolumns)
    else:
        # When nfeat is given we consider nfeat random columns
        column_numbers = np.random.choice(ncolumns, nfeat, replace = False)
    
    # Find best split over given column numbers
    best_split_info = {'Feature': None, 'Threshold': None, 'Impurity': np.inf}

    for feature in column_numbers:

        # Determine the best split for the current feature
        feature_threshold, feature_impurity = best_split(x, y, feature, minleaf)

        # If impurity is lower than current best split, use the new split
        if feature_impurity < best_split_info['Impurity']:
            best_split_info = {'Feature': feature, 'Threshold': feature_threshold, 'Impurity': feature_impurity}
    

    # Check if a split is found under the given constraints
    if best_split_info['Feature'] is None:

        # With no split found return node as leaf with classification
        if rows_class_0 >= rows_class_1:
            tree.class_label = 0
        else:
            tree.class_label = 1

        return tree


    # Get information about the best split
    best_split_feature   = best_split_info['Feature']
    best_split_threshold = best_split_info['Threshold']

    # Add split information to the tree
    tree.feature   = best_split_feature
    tree.threshold = best_split_threshold


    # Determine the indices of the split in the data and class labels according to the best split
    left_node_column_indices  = x[:, best_split_feature] <= best_split_threshold
    right_node_column_indices = x[:, best_split_feature] >  best_split_threshold

    # Grow the left and right trees and add them to the parent node
    tree.left_tree  = tree_grow(x[left_node_column_indices],  y[left_node_column_indices],  nmin, minleaf, nfeat)
    tree.right_tree = tree_grow(x[right_node_column_indices], y[right_node_column_indices], nmin, minleaf, nfeat)


    # Return the full tree
    return tree


# Use classification tree to predict class labels for matrix data
def tree_pred(x, tr):
    
    # Return the class label prediction for each matrix entry
    return [tree_pred_entry(entry, tr) for entry in x]


# Tree_grow_b procedure executes tree_grow on multiple bootstrap samples
def tree_grow_b(x, y, nmin, minleaf, nfeat, m):

    # Draw m bootstrap samples
    bootstrap_samples = [bootstrap_sample(x) for _ in range(0, m)]

    # Execute the tree_grow procedure on every sample
    return [tree_grow(sample, y, nmin, minleaf, nfeat) for sample in bootstrap_samples]


# Use a list of classification trees to predict the class labels for matrix data (with majority voting)
def tree_pred_b(x, tr_list):

    # Return the class label prediction for each matrix entry
    return [tree_pred_b_entry(entry, tr_list) for entry in x]



# Function that returns the best split for a feature
def best_split(x, y, feature, minleaf):
    """
    we assume that all the data is numeric
    """
    # Get column
    column = x[:, feature]
    length_column = len(column)

    # Enumerate all possible split points
    sorted_column = np.array(np.sort(np.unique(column)))
    possible_splitpoints = (sorted_column[0 : -1] + sorted_column[1 :]) / 2

    # Find the best split point
    best_split_info = {'Threshold': None, 'Impurity': np.inf}
    
    for threshold in possible_splitpoints:

        # Split data on threshold
        left_data  = y[column <= threshold]
        right_data = y[column >  threshold]
        
        # Check minleaf constraint
        if minleaf is not None and (len(left_data) < minleaf or len(right_data) < minleaf):
            # When one part has too few rows we do not consider this split point
            continue
        
        # Determine the gini impurity
        left_impurity  = len(left_data)  / length_column * gini_impurity(left_data)
        right_impurity = len(right_data) / length_column * gini_impurity(right_data)
        impurity = left_impurity + right_impurity
        
        # When the impurity is lower then the current best split use the new split
        if impurity < best_split_info['Impurity']: 
            best_split_info = {'Threshold': threshold, 'Impurity': impurity}

    # Return the best split information
    return (best_split_info['Threshold'], best_split_info['Impurity'])

# Function that returns the Gini impurity
def gini_impurity(data): 
    """
    we assume all the values are 0 1
    """
    if len(data) == 0: 
        # Return 0 if the data is empty
        return 0
    else:
        # Otherwise return the gini impurity
        prob = np.sum(data) / len(data)    
        return prob * (1 - prob) 


# Drawing a bootstrap sample from a dataset
def bootstrap_sample(x):

    # Get data size
    nrows, _ = x.shape

    # Draw this amount of row numbers
    row_numbers = np.random.choice(nrows, size = nrows, replace = True)

    # Return the bootstrap sample based on the random list of row numbers
    return x[row_numbers, :]


# Use classification tree to predict class labels for a single data entry
def tree_pred_entry(x_entry, tr):

    # Return prediction value of leaf node
    if tr.class_label is not None:
        return tr.class_label

    # Otherwise drop down tree from current node
    if x_entry[tr.feature] <= tr.threshold:
        return tree_pred_entry(x_entry, tr.left_tree)
    else:
        return tree_pred_entry(x_entry, tr.right_tree)

# Use classification tree to predict class labels for a single data entry
def tree_pred_b_entry(x_entry, tr_list):

    # Make a prediction for every classification tree
    predictions_list = [tree_pred_entry(x_entry, tr) for tr in tr_list]

    # Return class label which occurs the most
    predictions = Counter(predictions_list)
    most_frequent_prediction, _ = predictions.most_common()[0]
    return most_frequent_prediction
    

# Read data from file
def get_data():
    #return np.genfromtxt(r'/Users/Marc/Documents/UU/Master - Computing Science/2021-2022/INFOMDM_REP_21/INFOMDM_REP_21/Assignment_1/credit.txt', delimiter = ',', skip_header = True)
    return np.genfromtxt(r'C:\Users\admin\Documents\GitHub\INFOMDM_REP_21\Assignment_1\credit.txt', delimiter = ',', skip_header = True)


# Main function
def main():

    # Get data from file
    credit_data = get_data()

    # Split data and class labels
    x = credit_data[:, : - 1]
    y = credit_data[:, -1]

    # Grow classification tree
    classification_tree = tree_grow(x, y, 3, 3, 4)

    classification_tree = tree_grow(x, y, None, None, None) # <- No constraints
    classification_tree = tree_grow(x, y, 2, 1, None)       # <- Practically the same


    print_tree(classification_tree)

    # Compare labels with predictions
    print(y)
    print(tree_pred(x, classification_tree))

    # Try _b functions
    classification_tree_list = tree_grow_b(x, y, 3, 3, 4, 10)
    print(tree_pred_b(x, classification_tree_list))


# Temp print tree function
def print_tree(tr, n = 0):
    print(("\t" * n) + str(tr.rows_0) + "|" + str(tr.rows_1), end = "")

    if tr.class_label is not None:
        print(" Label: " + str(tr.class_label))
        return

    print("\tcol_" + str(tr.feature) + " = " + str(tr.threshold))
    print_tree(tr.left_tree,  n + 1)
    print_tree(tr.right_tree, n + 1)


if __name__ == "__main__":
    main()