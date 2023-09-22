#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 09:57:42 2023

@author: ivanyang
"""
import DecisionTree as DT
import pandas as pd
import numpy as np
################# loading data here
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

######### categorical features are fine to use directly, now start dealing with numeric features.
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

features_include_unknown = ['job','education', 'contact', 'poutcome']

for f in features_include_unknown:
    # get the rank of feature values.
    rank = train_data[f].value_counts(sort = True).index.tolist()
    if rank[0] == 'unknown':
        # replace it by the second frequent feature value
        replace_feature = rank[1]
    else:
        replace_feature = rank[0]
    train_data[f] = train_data[f].apply(lambda x: replace_feature if x == 'unknown' else x)

################################ Test starts here
max_depth = 16

train_acc = np.zeros((16,3))
test_acc = np.zeros((16,3))


metric_options = ['entropy', 'major_error', 'gini_index']


for i in range(max_depth):
    iter_indx = 0
    for metric_selection in metric_options:
        # create an ID3 tree with specific metric.
        dt_generator = DT.ID3(metric_selection=metric_selection, max_depth=i+1)
        # construct decision tree
        decision_tree = dt_generator.generate_decision_tree(train_data, features, label)
        # train acc
        # predict
        train_data['plabel']= dt_generator.classify(decision_tree, train_data)
        correct_train = 0
    
        for indx in range(train_size):
            if train_data['plabel'][indx] == train_data['y'][indx]:
                correct_train += 1
        train_acc[i][iter_indx] = correct_train/train_size
        # test acc
        # predict
        test_data['plabel']= dt_generator.classify(decision_tree, test_data)
        correct_test = 0
    
        for indx in range(test_size):
            if test_data['plabel'][indx] == test_data['y'][indx]:
                correct_test += 1
        test_acc[i][iter_indx] = correct_test/test_size
        iter_indx += 1

print('train accuracy using different metrics are: \n', train_acc)
print('test accuracy using different metrics are:\n', test_acc)

print('average training accuracy is:\n', np.mean(train_acc,axis = 0))
print('average testing accuracy is:\n', np.mean(test_acc,axis = 0))






