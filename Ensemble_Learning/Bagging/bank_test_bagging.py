#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 09:57:42 2023

@author: ivanyang
"""
import DecisionTree as DT
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
    
print(train_data['age'].dtype)
print(train_data['loan'].dtype)
print(test_data['age'].dtype)
print(test_data['loan'].dtype)

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
T = 100 # 100,500
depth = 10

train_err = np.zeros((T,))
test_err = np.zeros((T,))

train_py = np.zeros((train_size,))
test_py = np.zeros((test_size,))

metric_choice = 'entropy'
sample_size = round(train_size * 0.6)

for t in range(T):
    sample = np.random.choice(train_size,size = sample_size,replace = True )
    sampled_train_data = train_data.iloc[sample]
    dt_generator = DT.ID3(metric_selection='entropy', max_depth= depth)
    decision_tree = dt_generator.generate_decision_tree(sampled_train_data, features, label)
    # train part.
    py = dt_generator.classify(decision_tree, train_data)
    py = np.array(py.tolist())
    py[py == 'yes'] = 1
    py[py == 'no'] = -1
    py = py.astype(int)
    train_py = train_py + py
    
    py = py.astype(str)
    py[train_py > 0 ] = 'yes'
    py[train_py <= 0] = 'no'
    train_data['py'] = pd.Series(py)
    
    mismatch = train_data.apply(lambda row: 0 if row['y'] == row['py'] else 1,axis =1 ).sum()
    err = mismatch / train_size
    train_err[t] = err
    
    # test part.
    py = dt_generator.classify(decision_tree, test_data)
    py = np.array(py.tolist())
    py[py == 'yes'] = 1
    py[py == 'no'] = -1
    py = py.astype(int)
    test_py = test_py + py
    
    py = py.astype(str)
    py[train_py > 0 ] = 'yes'
    py[train_py <= 0] = 'no'
    test_data['py'] = pd.Series(py)
    
    mismatch = test_data.apply(lambda row: 0 if row['y'] == row['py'] else 1,axis =1 ).sum()
    err = mismatch / train_size
    test_err[t] = err
    #print('train error', train_err[t], 'test error', test_err[t])

# plot the data
fig = plt.figure()

plt.plot(train_err, color='tab:blue')
plt.plot(test_err, color='tab:orange')
plt.xlabel('iteration')
plt.ylabel('error')
plt.legend(['train', 'test'])
fig.suptitle('training and test error for bagging methods')
fig.savefig('bagging.png')





