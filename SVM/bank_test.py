#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:12:54 2023

@author: ivanyang
"""

import pandas as pd
import numpy as np
import SupportVectorMachine
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

##### Test starts here.
# Test C
C_set = np.array([100, 500, 700])
C_set = C_set / 873
# Test gamma.
gamma_set = np.array([0.01,0.1, 0.5, 1, 5,100])
svm = SupportVectorMachine.SVM()
# set epoch
svm.epoch = 100
# a is the hyper-parameter in learning rate.
svm.a = 0.1
#svm.gamma = 0.5
svm.lr = 0.1
# count
c=0
old_indx = 0
for C in C_set:
    print('C is', C)
    svm.set_C(C)
    # Linear SVM primal.
    w_primal = svm.sgd_primal(X_train, y_train)
    print('w primal ', w_primal)
    pred_train = np.dot(X_train,w_primal)
    pred_train[pred_train > 0 ] = 1
    pred_train[pred_train <= 0] = -1
    train_err = np.sum(np.abs(pred_train - y_train)) / (2 *y_train.shape[0])

    pred_test = np.dot(X_test,w_primal)
    pred_test[pred_test > 0] = 1
    pred_test[pred_test <= 0] = -1

    test_err = np.sum(np.abs(pred_test - y_test)) / (2*y_test.shape[0])
    print('linear SVM primal train_error:', train_err, ' test_error: ', test_err)
    
    

    # Linear SVM dual 
    w_dual = svm.dual_svm(X_train, y_train)
    print('w dual: ', w_dual)

    pred_train = np.dot(X_train,w_dual)
    pred_train[pred_train > 0 ] = 1
    pred_train[pred_train <= 0] = -1
    train_err = np.sum(np.abs(pred_train - y_train)) / (2 *y_train.shape[0])

    pred_test = np.dot(X_test,w_dual)
    pred_test[pred_test > 0] = 1
    pred_test[pred_test <= 0] = -1

    test_err = np.sum(np.abs(pred_test - y_test)) / (2*y_test.shape[0])
    print('linear SVM dual train_error: ', train_err, ' test_error: ', test_err)
    

    # Kernel SVM dual
    for gamma in gamma_set:
        print('gamma:', gamma)
        svm.set_gamma(gamma)
        alpha = svm.kernel_svm_train(X_train, y_train)
        indx = np.where(alpha > 0)[0]
        # Compute active set length.
        print('support vector length:', len(indx))
        # train 
        y = svm.kernel_prediction(alpha, X_train, y_train, X_train)
        train_err = np.sum(np.abs(y - y_train)) / (2*y_train.shape[0])
    
        # test
        y_hat = svm.kernel_prediction(alpha, X_train, y_train, X_test)
        test_err = np.sum(np.abs(y_hat - y_test)) / (2*y_test.shape[0])
        print('nonlinear SVM train_error: ', train_err, ' test_error: ', test_err)
        # Compute # of intersect value
        if c > 0:
            intersect = len(np.intersect1d(indx, old_indx))
            print('number of intersect value:', intersect)
        c = c + 1
        old_indx = indx
        
        ## Nonlinear Perceptron part
        err_vec = svm.kernel_perceptron(X_train, y_train)
        y = svm.kernel_perceptron_predict(err_vec,X_train,y_train,X_train)
        train_err =  np.sum(np.abs(y - y_train)) / (2*y_train.shape[0])
        # test
        y_hat = svm.kernel_perceptron_predict(err_vec,X_train,y_train,X_test)
        test_err = np.sum(np.abs(y_hat - y_test)) / (2*y_test.shape[0])
        print('nonlinear perceptron train_error: ', train_err, ' test_error: ', test_err)

# svm.set_epoch(200)
# err_count = svm.kernel_perceptron(X_train, y_train)
# percep_predict = svm.kernel_perceptron_predict(err_count,X_train,y_train,X_test)