#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 22:49:23 2023

@author: ivanyang
"""

import pandas as pd
import numpy as np
import NeuralNetwork
import math
import matplotlib.pyplot as plt
# load train data
load_train_data = pd.read_csv('train.csv')
train_data = load_train_data.values
# add bias term/data processing
# change the label with value 0 to -1
m = train_data.shape[0]
n = train_data.shape[1]
X_train = np.copy(train_data)
X_train[:,n-1] = 1
y_train = train_data[:,n-1]
y_train[y_train == 0] = -1

# load test data.
load_test_data = pd.read_csv('test.csv')
test_data = load_test_data.values
# add bias term
m1 = test_data.shape[0]
n1 = test_data.shape[1]
X_test = np.copy(test_data)
X_test[:,n1-1] = 1
# change the label value with 0 to -1.
y_test = test_data[:,n1-1]
y_test[y_test == 0] = -1

## Test process here.
gamma_set = np.array([0.01, 0.1, 0.5, 1, 2, 5, 10, 100])
in_dim = X_train.shape[1]
out_dim = 1

hidden_size_list = [5, 10, 25, 50, 100]

for width in hidden_size_list:
    print('hidden size is:', width)
    width_list = [in_dim, width, width, out_dim]
    NN= NeuralNetwork.NeuralNetwork(width_list)
    
    NN.mini_batch_SGD(X_train.reshape([-1, in_dim]), y_train.reshape([-1,1]))
    pred = NN.predict(X_train)
    
    pred = np.sign(pred)
    train_err = np.sum(np.abs(pred - np.reshape(y_train,(-1,1)))) / (2*y_train.shape[0])
    
    pred = NN.predict(X_test)
    
    pred = np.sign(pred)
    test_err = np.sum(np.abs(pred - np.reshape(y_test,(-1,1)))) / (2*y_test.shape[0])
    print('train_error: ', train_err, ' test_error: ', test_err)

