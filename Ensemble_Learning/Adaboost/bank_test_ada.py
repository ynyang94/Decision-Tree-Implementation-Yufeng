#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 22:45:36 2023

@author: ivanyang
"""

import WeightedDT as WDT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
########################### load data here
column_name = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
               'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

types = {'age': int, 'job': str, 'marital': str,
         'education': str, 'default': str, 'balance': int,
         'housing': str, 'loan': str, 'contact': str, 'day':int, 'month': str, 'duration': int, 'campaign': int,
         'pdays': int, 'previous': int, 'poutcome': str, 'y': str}

train_data = pd.read_csv('train.csv',names = column_name, dtype = types, header = None )
test_data = pd.read_csv('test.csv',names = column_name, dtype = types, header = None )
train_size = train_data.shape[0]
test_size = test_data.shape[0]
# categorical features are fine to use directly, now start dealing with numeric features.
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
for nfeature in numeric_features:
    median_train = train_data[nfeature].median()
    train_data[nfeature] = train_data[nfeature].apply(lambda x:1 if x > median_train else 0)
    train_data[nfeature] = train_data[nfeature].astype(str)
    median_test = test_data[nfeature].median()
    test_data[nfeature] = test_data[nfeature].apply(lambda x:1 if x > median_test else 0)
    test_data[nfeature] = test_data[nfeature].astype(str)
    
#print(train_data['age'].dtype)
#print(train_data['loan'].dtype)
#print(test_data['age'].dtype)
#print(test_data['loan'].dtype)

# input features here without replacement of unknow features.
features = {'age':['0','1'],
            'job':['admin.','unknown','unemployed','management','housemaid','entrepreneur','student','blue-collar','self-employed','retired','technician','services'],# unknown includex
            'marital':['married','divorced','single'],
            'education':['unknown','secondary','primary','tertiary'],#includes unknown.
            'default': ['yes','no'],
            'balance': ['0','1'],
            'housing': ['yes', 'no'],
            'loan': ['yes', 'no'],
            'contact': ['unknown','telephone','cellular'],#unkown includes
            'day': ['0','1'],
            'month':['jan','feb','mar','apr', 'may','jun', 'jul', 'aug', 'sep', 'oct','nov','dec'],
            'duration':['0','1'],
            'campaign': ['0','1'],
            'pdays':['0','1'],
            'previous':['0','1'],
            'poutcome':['unknown','other','failure','success']}#unknow include

label = {'y':['yes','no']}

####################################Test starts here.

T = 100 #500

alphas = np.zeros((T,))
# Initialize all weights to be a uniform distribution.
weights = np.array([1/train_size for x in range(train_size)])

# Initialize vector for recording training error per tree at each time.
train_err = np.zeros((T,))
test_err = np.zeros((T,))
# Initialize combined train and test error for weighted DT.
train_err_T = np.zeros((T,))
test_err_T = np.zeros((T,))

# 
train_py = np.zeros((train_size,))
test_py = np.zeros((test_size,))

for t in range(T):
    # Call Weighted DT.
    dt_generator = WDT.WeightedID3(metric_selection= 'entropy', max_depth = 2)
    decision_tree = dt_generator.generate_decision_tree(train_data, features, label, weights)
    
    # Construct training error
    # compute predicted label.
    train_data['py']= dt_generator.classify(decision_tree, train_data)
    # Construct a vector recording the wrong predicted label.
    label_match = train_data.apply(lambda row: 0 if row['y'] == row['py'] else 1, axis=1) 
    err = label_match.sum() / train_size
    # train error for each tree at each time.
    train_err[t] = err
    
    # calculate weighted error.
    # if label is matached, yi*h(i) = 1, otherwise it is -1
    label_match = train_data.apply(lambda row: 1 if row['y'] == row['py'] else -1, axis=1) 
    label_match = np.array(label_match.tolist())
    w = weights[ label_match == -1]
    err = np.sum(w)
    
    # Update alpha.
    alpha = 0.5 * np.log((1 - err) / err)
    alphas[t] = alpha

    # update weights.
    weights = np.exp(label_match * -alpha) * weights
    total = np.sum(weights)
    weights = weights / total
    
    # test error
    test_data['py']= dt_generator.classify(decision_tree, test_data)
    label_match = test_data.apply(lambda row: 0 if row['y'] == row['py'] else 1, axis=1) 
    # Test error for each tree at each time. 
    test_err[t] = label_match.sum() / test_size

    # construct final hypothesis.
    # train
    py = np.array(train_data['py'].tolist())
    py[py == 'yes'] = 1
    py[py == 'no'] = -1
    py = py.astype(int)
    # Compute combined predictor.
    train_py = train_py + alpha * py
    py = py.astype(str)
    py[train_py > 0] = 'yes'
    py[train_py <=0] = 'no'
    train_data['py'] = pd.Series(py)
    # Cumulative error.
    label_match = train_data.apply(lambda row: 0 if row['y'] == row['py'] else 1, axis=1) 
    err = label_match.sum() / train_size
    train_err_T[t] = err

    # test
    py = np.array(test_data['py'].tolist())
    py[py == 'yes'] = 1
    py[py == 'no'] = -1
    py = py.astype(int)
    # Compute Combined predictor.
    test_py = test_py + py * alpha
    py = py.astype(str)
    py[test_py > 0] = 'yes'
    py[test_py <=0] = 'no'
    test_data['py'] = pd.Series(py)
    # Cumulative error. 
    label_match = test_data.apply(lambda row: 0 if row['y'] == row['py'] else 1, axis=1) 
    err = label_match.sum() / test_size
    test_err_T[t] = err



fig , (ax1, ax2) = plt.subplots(2,1)

ax1.plot(train_err, 'g',linewidth = 3)
ax1.plot(test_err, 'r', linewidth = 2) 
 
ax1.legend(['train', 'test'])
ax1.set_xlabel('iteration', fontsize=10)
ax1.set_ylabel('Error', fontsize=10)


ax2.plot(train_err_T, 'g', linewidth = 3)
ax2.plot(test_err_T, 'r', linewidth = 2)  
ax2.legend(['train', 'test'])
ax2.set_xlabel('iteration', fontsize=10)
ax2.set_ylabel('Error', fontsize=10)
fig.suptitle(' Each Tree predictor vs. Combined Predictor')
fig.savefig('adaboost.png')   

