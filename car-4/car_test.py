#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:22:52 2023

@author: ivanyang
"""
# Test case for car data. The code following is all written by me.

import DecisionTree as DT
import pandas as pd
import numpy as np

################################ specify the attributes
column_name = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
types = {'buying': str, 'maint': str, 'doors': str, 'persons': str, 'lug_boot': str, 'safety': str, 'label': str}

# feature and label are two dictonary.
# feature's key is set to be 6 features, values are feture values.
features = {'buying': ['vhigh', 'high', 'med', 'low'], 'maint':  ['vhigh', 'high', 'med', 'low'], 
            'doors':  ['2', '3', '4', '5more'], 'persons': ['2', '4', 'more'], 
            'lug_boot': ['small', 'med', 'big'],  'safety':  ['low', 'med', 'high'] }

label = {'label': ['unacc', 'acc', 'good', 'vgood']}
################################# load train and test data
train_data =  pd.read_csv('train.csv', names=column_name, dtype=types, header = None)
# return the number of rows.
train_size = train_data.shape[0]

test_data =  pd.read_csv('test.csv', names=column_name, dtype=types, header = None)
test_size = test_data.shape[0]

################################# specify the maximal depth.
max_depth = 6

train_acc = np.zeros((3,1))
test_acc = np.zeros((3,1))


metric_options = ['entropy', 'major_error', 'gini_index']
iter_indx = 0
for metric_selection in metric_options:
    # create an ID3 tree with specific metric.
    dt_generator = DT.ID3(metric_selection=metric_selection, max_depth=max_depth)
    # construct decision tree
    decision_tree = dt_generator.generate_decision_tree(train_data, features, label)
    # train acc
    # predict
    train_data['plabel']= dt_generator.classify(decision_tree, train_data)
    correct_train = 0
    
    for indx in range(train_size):
        if train_data['plabel'][indx] == train_data['label'][indx]:
            correct_train += 1
    train_acc[iter_indx][0] = correct_train/train_size
    # test acc
    # predict
    test_data['plabel']= dt_generator.classify(decision_tree, test_data)
    correct_test = 0
    
    for indx in range(test_size):
        if test_data['plabel'][indx] == test_data['label'][indx]:
            correct_test += 1
    test_acc[iter_indx][0] = correct_test/test_size
    iter_indx += 1

print('train accuracy using different metrics are: \n', train_acc)
print('test accuracy using different metrics are:\n', test_acc)