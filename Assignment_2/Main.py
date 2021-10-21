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
import itertools
import os
import random as rd
from sklearn import tree




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
    tree.plot_tree(preidct)



get_data("Data/negative_polarity")

#shuffle data

labels_list = ((list(reviews.items())))
shuffle_list = shuffle_list(labels_list)

#lables after shuffle
Labels = (np.array(shuffle_list)[:,1])

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

# divide to X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(tfidf, Labels, test_size=0.25, random_state=42)



#DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=0)
preidct = model.fit(X_train, y_train)