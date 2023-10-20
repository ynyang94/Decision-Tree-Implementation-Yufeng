#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:57:28 2023

@author: ivanyang
"""

import pandas as pd
import numpy as np
import LinearRegression

# load train data
train_data = pd.read_csv('train.csv',header = None)
raw_data = train_data.values

m = raw_data.shape[0]
n = raw_data.shape[1]
# Get train input and output, reshape data by our requirements.
X_train = np.copy(raw_data)
X_train[:,n-1] = 1

y_train = raw_data[:,n-1]
y_train = np.reshape(y_train,(m,1))

# load test data
test_data = pd.read_csv('test.csv',header = None)
raw_data_test = test_data.values

m_test = raw_data_test.shape[0]
n_test = raw_data_test.shape[1]
# reshape test data by our requirements.
X_test = np.copy(raw_data_test)
X_test[:,n-1] = 1

y_test = raw_data_test[:,n_test-1]
y_test = np.reshape(y_test, (m_test,1))

# Test starts here.
LMS = LinearRegression.LinearRegression()
# set max_iter = 5000
LMS.max_iter = 5000
# GD
LMS.set_method('gd')
w_gd = LMS.optimizer(X_train, y_train)
print('w learned by GD:\n',w_gd)
test_val_gd = LMS.obj_value(X_test, w_gd, y_test)
print('test loss is:',test_val_gd)

# SGD
LMS.set_method('sgd')
# set a more conservative learning rate
LMS.lr = 0.005
w_sgd = LMS.optimizer(X_train, y_train)
print('w learned by SGD:\n',w_sgd)
test_val_sgd = LMS.obj_value(X_test, w_sgd, y_test)
print('test loss is:',test_val_sgd)

# Compute optimal solution.
LMS.set_method('optimum')
w_opt = LMS.optimizer(X_train, y_train)
print('w optimum is:\n',w_opt)
test_val_opt = LMS.obj_value(X_test, w_opt, y_test)
print('test loss is:',test_val_opt)
