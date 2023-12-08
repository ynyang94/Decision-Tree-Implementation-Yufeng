#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 21:01:41 2023

@author: ivanyang
"""
import pandas as pd
import numpy as np
import logistic
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


LR = logistic.logistic_regression()
v_list = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
for v in v_list:
    LR.set_v(v)
    print('variance:', v)
    
    w= LR.MAP(X_train, y_train)


    pred = np.matmul(X_train, w)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1
    train_err = np.sum(np.abs(pred - np.reshape(y_train,(-1,1)))) / (2* y_train.shape[0])

    pred = np.matmul(X_test, w)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1

    test_err = np.sum(np.abs(pred - np.reshape(y_test,(-1,1)))) / (2 * y_test.shape[0])
    print('MAP train_error: ', train_err, 'MAP test_error: ', test_err)

w0 = LR.MAP(X_train, y_train)


pred = np.matmul(X_train, w0)
pred[pred > 0] = 1
pred[pred <= 0] = -1
train_err = np.sum(np.abs(pred - np.reshape(y_train,(-1,1)))) / (2* y_train.shape[0])

pred = np.matmul(X_test, w0)
pred[pred > 0] = 1
pred[pred <= 0] = -1

test_err = np.sum(np.abs(pred - np.reshape(y_test,(-1,1)))) / (2 * y_test.shape[0])
print('MLE train_error: ', train_err, 'MLE test_error: ', test_err)
