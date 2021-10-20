"""
Assignment _2
"""
#Imports

#Algorithms
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Analysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error 
from sklearn.feature_selection import chi2
#import  sklearn.metrics.precision_recall_fscore_support as matrix_recall_precision
import itertools
import os


# =============================================================================
# 
# def automated_concat_file(path):
#     filelist = []
# 
#     filenames = ['file1.txt', 'file2.txt', ...]
#     with open('path/to/output/file', 'w') as outfile:
#         for line in itertools.chain.from_iterable(itertools.imap(open, filnames)):
#             outfile.write(line)
# 
# =============================================================================


reviews = {}
complete_list = []
true_list = []
fake_list = []


def try_(path):
    filelist = []

    for root, dirs, files in os.walk(path):
    	for file in files:
            #append the file name to the list
            filelist.append(os.path.join(root,file))
            with open(os.path.join(root, file), 'r') as f:
                text = f.read()
                complete_list.append(text)
                if 'truth' in root:
                    reviews[text] = 1
                    true_list.append(text)
                else:
                    reviews[text] = 0
                    fake_list.append(text)
    return(filelist)


print(try_("Data/negative_polarity"))
print(len(true_list), len(fake_list))
print(len(reviews))

