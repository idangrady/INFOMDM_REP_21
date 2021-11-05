
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
df_ = pd.DataFrame(columns = ["Model","Train Accuracy",  "Accuracy", "Precision", "Recall", "F1 Score","Folds","type","Best Parameters"])
dftesting = pd.DataFrame(columns = ["Model", "Accuracy","Precision", "Recall", "F1 Score","Best Parameters"])

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
        return (name_model, {'n_estimators': [10, 50, 100], 'max_depth': [None, 2, 4, 8,16,30], 'min_samples_split': [2, 4, 8,16,30]}) # add for the real analysis also 500 and 1000
    

# =============================================================================
def append_data_to_df(data, df):
    list_ob = list(data)
    length = len(df)
    df.loc[length] = list_ob
    return  df


def train_folds(classifier ,x_train, x_test, y_train, y_test,  fold, type_,n_fold=5, print_plots = False):

    scores= []
    train_accu = []
    precisions = []
    recalls = []
    fscores = []
    
    
    
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
    return (str(classifier)[:-2], statistics.mean(scores),statistics.mean(train_accu), statistics.mean(precisions), statistics.mean(recalls), statistics.mean(fscores),n_fold, type_,tuned_params.best_params_), tuned_params, str(classifier)[:-2]

def ncnemars_test(y,model_1,model_2):
    
    output_matrix = np.zeros((2,2)) #creating the output matrix
    
    #check when model are inacuurate and place it in the right loc in the matrix
    output_matrix[1][0] = np.sum(np.where(((y+model_1 >1) | (y+model_1 == 0) ) & (y+model_2 ==1),1,0))  #model 1 TN or TP, modele 2 FN or FP
    output_matrix[0][1] = np.sum(np.where(((y+model_2 >1) | (y+model_2 == 0) ) & (y+model_1 ==1),1,0))  #model 2 TN or TP, modele 1 FN or FP
    
    #check when model are are the same in both ==> their correct output and mistakes
    output_matrix[0][0] = np.sum(np.where((y+model_2 ==1) & (y+model_1 ==1),1,0)) #both models incorrect (FN or FP)
    output_matrix[1][1] =np.sum(np.where(y+ model_1+model_2>2 ,1,0))  + (np.sum(np.where(y+ model_1+model_2==0 ,1,0))) #both models correct (TP or TN)
    
    #check if we can divide
    if (output_matrix[1][0] - output_matrix[0][1] != 0):
        p_value = ((abs(output_matrix[1][0]) - (output_matrix[0][1]))-1)**2 / (output_matrix[1][0] + output_matrix[0][1])
    # if not
    else:
        p_value  ="can not divided by 0"
        
    return(output_matrix,p_value)


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
bigram_vectorizer = CountVectorizer(ngram_range = (1, 2), min_df = 2,max_df = 0.7) # token_pattern = r'\b\w+\b',

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

print_features=False
save=False
idx = 0
list_of_vectoresed_word = [(single_tfidf, C_tvectorizer), (b_gram_tfidf, bigram_vectorizer) ] #C_tvectorizer bigram_vectorizer

best_models ={}
model_names = [ "MultinomialNB", "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier"]
y_nm =0
for tfidf, vectorizer in list_of_vectoresed_word:
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
    



    x_train, x_test, y_train, y_test = train_test_split(data_training[:,1:], data_training[:,0], test_size=0.8, random_state= 40)
    
    for k in [5,10]:
        for model in [ MultinomialNB(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]:

            # (Accuracy, Precision, Recall, F-score)
            result, model, name =  train_folds(model, x_train, x_test, y_train, y_test, KFold,type_, k,print_plots = False)
            
            
            x_testing = data_testing[:, 1:]
            y_testing = data_testing[:,0]
            y_testing_pred = model.predict(x_testing)
            
            acc = model.score(x_testing, y_testing)
            train_score = model.score(data_training[:,1:], data_training[:,0])
            precision, recall, fscore, _ = precision_recall_fscore_support(y_testing, y_testing_pred, average = 'macro')


            
            test_result = (name,train_score, acc,  precision,  recall, fscore,k,type_, model.best_params_)

            
            df_= append_data_to_df(test_result,df_)
            
            if (k==10 and type_=="Bigram"):
                if name in best_models:
                    if best_models[name][1] < acc:
                        best_models[name] =[model, acc, y_testing_pred]
                else:
                    best_models[name] =[model, acc, y_testing_pred]
                    y_nm =y_testing
                
            
            print('model: ', model)
            print(' acc: ',  acc, ' precision: ', precision, ' recall: ', recall, ' fscore: ', fscore)
    

    #region Top 5 features
    
    full_data = np.array(concat_df).astype('float32')

# =============================================================================
#     vocabulary_mapping = vectorizer.vocabulary_ 
#     reverse_vocabulary_mapping = {}
#     for k, v in vocabulary_mapping.items():
#         reverse_vocabulary_mapping[v] = k
# 
#     top_N = 5
#     
#     def subtract_log_probs(array):
#         return array[1] - array[0]
#     
#     model = MultinomialNB()
#     model.fit(full_data[:, 1:], full_data[:, 0])
# 
#     feature_log_probabilities = model.feature_log_prob_
#     probabilities = subtract_log_probs(feature_log_probabilities)
# 
#     max_indices = (-probabilities).argsort()[:top_N]
#     min_indices = probabilities.argsort()[:top_N]
# 
#     if print_features:
#         for i in range(0, top_N):
#             print("Feature: " + reverse_vocabulary_mapping[min_indices[i]] + ", Score: " + str(probabilities[min_indices[i]]))
#         for i in range(0, top_N):
#             print("Feature: " + reverse_vocabulary_mapping[max_indices[i]] + ", Score: " + str(probabilities[max_indices[i]]))
#         
# =============================================================================
    #endregion

significant = []
# compute the significant
for model1 in model_names:
    for model2 in model_names:
        if str(model1) ==str(model2):
            continue
        else:
            #get the list
            model_1_list = best_models[model1]
            model_2_list = best_models[model2]
            
            model_1_model, model_1_acc,model_1_pred = model_1_list
            model_2_model, model_2_acc,model_2_pred = model_2_list
            
            
            _, _p_val = ncnemars_test(y_nm, model_1_pred,model_2_pred )
            significant.append((model1,model2,_p_val))
            

print(significant)
if save:
    df_.to_csv('Result_combined_uni_Bi.csv', index=False)
    #dftesting.to_csv('ResultTesting.csv', index=False)

significant.to_csv('Result_combined_uni_Bi.csv', index=False)