import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def data_1():
    data ={
    'age' : ['youth', 'youth', 'middle_age', 'senior', 'senior', 'senior','middle_age', 'youth', 'youth', 'senior', 'youth', 'middle_age','middle_age', 'senior'],
    'income' : ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium','low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student' : ['no','no','no','no','yes','yes','yes','no','yes','yes','yes','no','yes','no'],
    'credit_rate' : ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair','excellent', 'excellent', 'fair', 'excellent'],
    'default' : ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes','yes', 'yes', 'yes', 'no']
}
    
    data_list = list(data.values())
    return (np.array(data.keys()),(np.array(data_list)))


def get_data():
    return(np.genfromtxt(r'C:\Users\admin\Documents\GitHub\INFOMDM_REP_21\Assignment_1\credit.txt', delimiter=',', skip_header=True))

def gini_im(arr):
    total_length = len(arr)
    prob =np.bincount(arr)/ total_length
    return np.prod(prob)
 
def impurity(arr):
    if len(arr) == 0:
        return 0
    p = np.sum(arr) / len(arr)
    return p * (1 - p)
    
def check_accuracy(x,y, indx, threshhold):
    output_arr = np.zeros(x.shape)
    y_con = y[:,:]> threshhold
    x_con = x[:,:]> threshhold
    output = y_con == x_con
    return(output[:,0])


def best_split(x,y):
    x_length = len(x)

    income_sorted = np.array(np.sort(np.unique(x)))
    income_splitpoints = (
        income_sorted[: len(income_sorted) - 1]
        + income_sorted[1 : len(income_sorted)]
    ) / 2

    best_imp_split = np.inf
    threshhold = 0
    step_si = 1/x_length

    for split_idx, split in enumerate(income_splitpoints):
        ff = x[split_idx]

        label_1 = y[x<=split]
        lebel_2 = y[x>split]

        #calculate impurity
        impurity_l_1 = impurity(label_1)
        impurity_l_2 = impurity(lebel_2)

        result_imp = impurity_l_1+impurity_l_2
        if result_imp <best_imp_split:
            best_imp_split=result_imp
            threshhold =split

    return best_imp_split,threshhold
            
        

if __name__ == "__main__":
# =============================================================================
#     
#     print(credit_data[0])
#     print(credit_data[:,3])
#     print(credit_data[4,0])
#     print(np.sort(np.unique(credit_data[:,3])))
#     print(np.sum(credit_data[:,5]))
#     print(credit_data.sum(axis=0))
#     print(credit_data[credit_data[:,0] > 27]) # good for the splitting
#     print(np.arange(0, 10))
#     print(np.arange(0, 10)[credit_data[:,0] > 27])
#     
#     x = np.random.choice(np.arange(0,10),size = 5,replace=False)
#     data = get_data()
#     print(data[x])
#     test = np.delete(credit_data, x, axis=0)
#     print("Test")
#     print(test)
# =============================================================================

#Practice_1
# =============================================================================
#     array=np.array([1,0,1,1,1,0,0,1,1,0,1])
#     array_perfect_split = np.array([1,1,1,1,0,0,0,0])
#     
#     features,items = data_1()
#     print(items.shape)
#     
#     
#     array2 = [1,2,3,4]
#     print(gini_im(array_perfect_split))
# 
#     
# =============================================================================
# =============================================================================
#     unique_val = np.unique(array)
#     plus_ = array[array==0]
#     length = np.bincount(array)
#     print(np.argmax(length))
# =============================================================================
    #print(length)
    
    credit_data = np.array(get_data())
    print((credit_data[:,3]))
    print(credit_data[:,5])
    print(best_split(credit_data[:,3], credit_data[:,5]))
    

        
