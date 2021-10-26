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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error 
from sklearn.feature_selection import chi2
#import  sklearn.metrics.precision_recall_fscore_support as matrix_recall_precision
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
from sklearn import utils
import itertools
import os
import random as rd
from sklearn import tree
from sklearn.model_selection import KFold
import pandas as pd
import statistics







#containing words
reviews = {}
complete_list = []
true_list = []
fake_list = []


def get_data(path):
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



#shuffle the list randomly
def shuffle_list(list_):
    size_list = len(list_)
    foo = rd.SystemRandom()
    for loc, sub_list in reversed(list(enumerate(list_))):
        current_loc = loc
        destination_loc = foo.randint(0,current_loc)
        
        desination_item = list_[destination_loc]
        
        c = sub_list
        
        #perform_shuffle
        list_[current_loc] = desination_item
        list_[destination_loc]  = c
        
    return(list_)
        
        
def print_tree(model):
    tree.plot_tree(model)



def get_score(model,x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)




def train_folds(classifier ,data_concat,  fold, n_fold=5):
    kfold = fold(n_splits=n_fold)
    
    scores= []
    
    for train_idx, test_idx in kfold.split(data_concat):


        #convert to Range so we can slice the array
        start_idx_train, end_idx_train = train_idx[0], train_idx[-1]
        start_idx_test, end_idx_test = test_idx[0], test_idx[-1]
        
        print(f" Test start  {start_idx_test}  end:{end_idx_test}")
        
    #slicing the array
        X_train_fold = data_concat[start_idx_train :end_idx_train , 1:]
        X_test_fold = data_concat[start_idx_test: end_idx_test,1:]
        
        y_train_fold =( data_concat[start_idx_train:end_idx_train,0]).astype('int')
        y_test_fold = (data_concat[start_idx_test: end_idx_test,0]).astype('int')
        
        y_train_fold, y_test_fold = np.expand_dims(y_train_fold, axis  =1), np.expand_dims(y_test_fold, axis =1)

        print(X_train_fold.shape, X_test_fold.shape, y_train_fold.shape, y_test_fold.shape)
        print()
        get_score_ = get_score(classifier,X_train_fold, X_test_fold, y_train_fold, y_test_fold )
        scores.append(get_score_)
    
    
    #Return
    return statistics.mean(scores)



get_data("Data/negative_polarity")

#shuffle data

labels_list = ((list(reviews.items())))
shuffle_list = shuffle_list(labels_list)

#lables after shuffle
Labels = (np.array(shuffle_list)[:,1])
Labels = np.expand_dims(Labels, axis= 1)


data_ = list(np.array(shuffle_list)[:,0])

#Text Cleaning

#vectorizers
C_tvectorizer = CountVectorizer(min_df = 3, max_df = 0.8 ) # ngram_range = (1,2,3)
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)

#vctorizling
vectorized_data  = C_tvectorizer.fit_transform(data_).toarray()
b_gram = bigram_vectorizer.fit_transform(data_).toarray()

# Tf–idf term weighting 
#maing the data to Tf-IDF values
transformer = TfidfTransformer( smooth_idf=False)
tfidf = transformer.fit_transform(vectorized_data).toarray()



print(tfidf.shape)
print(Labels.shape)

df = pd.DataFrame(Labels)
df_2 = pd.DataFrame(tfidf)

concat_df = pd.concat([df,df_2], axis= 1)

data_nump_conc = np.array(concat_df)
data_nump_conc = data_nump_conc.astype('float32')
# divide to X_train, X_test, y_train, y_test

y_check = np.expand_dims(data_nump_conc[:,0], axis = 1)

print(train_folds(MultinomialNB(), data_nump_conc,KFold,5))

input= input("Continue? ")

