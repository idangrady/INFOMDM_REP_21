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



def automated_concat_file(files):
    filelist = []
    
    for file in files:
        with open('result.txt', 'w') as result:
            result.write(str(file)+'\n')
            
    return result

   
def try_(path):
    filelist = []
    txt_= []
    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root,file))
            txt_.append(automated_concat_file(filelist))
        filelist = []
    return(txt_)



def try_2(path):
    reviews = {}
    testlist = []
    filelist = []
    arr = np.zeros((800,1)) #.reshape((([800], [1])))
    idx =0
    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root,file))
            #append the file name to the list
            filelist.append(os.path.join(root,file))
            with open(os.path.join(root, file), 'r') as f:
                text = f.read()
                testlist.append(text)
                if 'truth' in root:
                    reviews[text] = 1
                    arr[idx] = 1
                    idx+=1
                else:
                    reviews[text] = 0
                    arr[idx] = 0
                    idx+=1
    
    
    print(arr)
    print(np.sum(reviews==1))
    print(len(filelist))
    print(len(reviews))
    #print(len(filelist) ==len(reviews))
    return(len(filelist) ==len(reviews))





#print(try_("Data"))

print(try_2("Data/negative_polarity"))
#txt_.append(automated_concat_file(filelist))
