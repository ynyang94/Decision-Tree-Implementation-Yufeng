#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 15:23:30 2023

@author: ivanyang
"""

import pandas as pd
import numpy as np
import perceptron
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
## test data verification.
percep = perceptron.perceptron()
# define training epoch (10) and lr = 0.005.
percep.set_lr(0.005)
percep.set_T(10)

# standard perceptron
w_standard = percep.standard_perceptron(X_train, y_train)
pred_y_standard = percep.standard_pred_val(X_test, y_test, X_train, y_train)
err = np.sum(np.abs(y_test - pred_y_standard))/(2*y_test.shape[0])

## voted perceptron
W_vote, C_vote = percep.voted_perceptron(X_train, y_train)
pred_y_voted = percep.voted_pred_val(X_test, y_test, X_train, y_train)
err1 = np.sum(np.abs(y_test - pred_y_voted))/(2*y_test.shape[0])

## averange perceptron
a = percep.average_perceptron(X_train, y_train)
pred_y_avg = percep.average_pred_val(X_test, y_test, X_train, y_train)
err2 = np.sum(np.abs(y_test - pred_y_avg))/(2*y_test.shape[0])
result = np.dot(np.transpose(C_vote),W_vote)

# print the results. 
print('standard: %s, voted: %s, average: %s', err, err1,err2)
print('w learned by standard perceptron:%s',w_standard)
print("W is : %s \n",W_vote)
print("C is: %s \n", C_vote)
print('w learned by average perceptron: %s', a)
print('W^TC is',result)


