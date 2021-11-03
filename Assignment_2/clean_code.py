# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 22:36:34 2021

@author: admin
"""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Analysis
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error
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
from matplotlib import pyplot






#containing words
reviews = {}
complete_list = []
true_list = []
fake_list = []
df_ = pd.DataFrame(columns = ["Model", "Accuracy","Train Accuracy", "Precision", "Recall", "F1 Score","Folds","type","Best Parameters"])

def get_data(path):
    filelist = []
    xx= (os.walk(path))
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

def plot_importance_features(model):
    importance = model.best_estimator_.coef_
    importance= np.squeeze(importance, axis = 0)
    
    pyplot.figure(figsize=(8, 6))
    ind = list(np.argpartition(importance, -10)[-10:])
    values =list(importance[ind])
    
    pyplot.bar ( values, ind)
    pyplot.show()

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

def model_cv(model):
    name_model =str(model)[:-2]
    if name_model =="MultinomialNB":
        return (name_model, {'alpha': [0.01, 0.1, 0.25, 0.5, 1, 2.5, 10]})
    elif name_model =="DecisionTreeClassifier":
        return (name_model, {'max_depth': [None, 2, 4, 8, 16], 'min_samples_split': [2, 4, 8, 16]})
    elif name_model =="LogisticRegression":
        return( name_model, {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 0.25, 0.5, 1, 2.5, 10]})
    else:
        return (name_model, {'n_estimators': [10, 50, 100, 500, 1000], 'max_depth': [None, 2, 4, 8,16,30], 'min_samples_split': [2, 4, 8,16,30]}) # add for the real analysis also 500 and 1000
    

# =============================================================================
def append_data_to_df(data, df):
    list_ob = list(data)
    length = len(df)
    df.loc[length] = list_ob
    return  df


def train_folds(classifier ,data_concat,  fold, type_,n_fold=5, print_plots = False):

    scores= []
    train_accu = []
    precisions = []
    recalls = []
    fscores = []
    
    x_train, x_test, y_train, y_test = train_test_split(data_concat[:,1:], data_concat[:,0], test_size=0.8, random_state= 40)
    
    name, params = model_cv(classifier)
    if name.lower() == "logisticregression":
        tuned_params = GridSearchCV(LogisticRegression(solver='liblinear'), params ,  n_jobs=-1)
    else: 
        tuned_params = GridSearchCV(classifier, params, cv=n_fold, n_jobs=-1)
    tuned_params.fit(x_train, y_train)
    y_pred = tuned_params.predict(x_test)
    
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average = 'macro')
  
    scores.append(tuned_params.score(x_test, y_test))
    train_accu.append(tuned_params.score(x_train, y_train))
    precisions.append(precision)
    recalls.append(recall)
    fscores.append(fscore)
    
    if print_plots:
        plot_importance_features(tuned_params)
    return (str(classifier)[:-2], statistics.mean(scores),statistics.mean(train_accu), statistics.mean(precisions), statistics.mean(recalls), statistics.mean(fscores),n_fold, type_,tuned_params.best_params_), tuned_params




get_data('Data/negative_polarity')
#get_data("Data/negative_polarity")

#shuffle data

labels_list = ((list(reviews.items())))
shuffle_list = shuffle_list(labels_list)

#lables after shuffle
Labels = (np.array(shuffle_list)[:,1])
Labels = np.expand_dims(Labels, axis= 1)


X_all = np.array(shuffle_list)[:,0]



data_ = list(np.array(shuffle_list)[:,0])

#Text Cleaning

#vectorizers
C_tvectorizer = CountVectorizer(min_df = 2, max_df = 0.7) # ngram_range = (1,2,3)
bigram_vectorizer = CountVectorizer(ngram_range = (2, 2), min_df = 2,max_df = 0.7) # token_pattern = r'\b\w+\b',

#vctorizling
vectorized_data  = C_tvectorizer.fit_transform(data_).toarray()
b_gram = bigram_vectorizer.fit_transform(data_).toarray()

# Tf–idf term weighting 
#maing the data to Tf-IDF values
transformer = TfidfTransformer( smooth_idf=False)
single_tfidf = transformer.fit_transform(vectorized_data).toarray()

#Bigram transformer
b_gram_transformer = TfidfTransformer( smooth_idf=False)
b_gram_tfidf = b_gram_transformer.fit_transform(b_gram).toarray()

save=True
idx = 0
list_of_vectoresed_word = [single_tfidf, b_gram_tfidf ] #C_tvectorizer bigram_vectorizer



for tfidf in list_of_vectoresed_word:
    type_ = "Unigram"
    if (idx-(len(list_of_vectoresed_word)/2)>=0):
        type_ ="Bigram"
    idx +=1
    df = pd.DataFrame(Labels)
    df_2 = pd.DataFrame(tfidf)

    concat_df = pd.concat([df,df_2], axis= 1)
    data_nump_conc = np.array(concat_df)
    data_nump_conc = data_nump_conc.astype('float32')

    data_split_training =  concat_df.iloc[round(len(df)/5):,:]
    data_split_testing = concat_df.iloc[:round(len(df)/5)+1:,:]

    data_training = np.array(data_split_training)
    data_training = data_training.astype('float32')
    data_testing = np.array(data_split_testing)
    data_testing = data_testing.astype('float32')

    for k in [5,10]:
        for model in [ MultinomialNB(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]:

            # (Accuracy, Precision, Recall, F-score)
            result, model =  train_folds(model, data_training, KFold,type_, k,print_plots = False)
            df= append_data_to_df(result,df_)
    
            x_testing = data_testing[:, 1:]
            y_testing = data_testing[:,0]
            y_testing_pred = model.predict(x_testing)
            
            acc = model.score(x_testing, y_testing)
            results = precision_recall_fscore_support(y_testing, y_testing_pred, average = 'macro')
            print('model: ', model)
            print(' acc: ',  acc, ' precision: ', results[0], ' recall: ', results[1], ' fscore: ', results[2])
    print(df)
    
if save:
    df.to_csv('Result.csv', index=False)
