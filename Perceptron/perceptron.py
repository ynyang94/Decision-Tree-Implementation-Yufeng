#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 20:58:52 2023

@author: ivanyang
"""

import numpy as np
import pandas as pd
import math

class perceptron:
    def __init__(self):
        self.T = 10
        self.lr = 0.01
    # functions for adjust lr and times.
    def set_T(self,times):
        self.T = times
    
    def set_lr(self,learningrate):
        self.lr = learningrate
    
    # implementation of standard perceptron
    def standard_perceptron(self, X, y):
        m = X.shape[0]
        n = X.shape[1]
        # initialization.
        w = np.zeros((n,))
        indx = np.arange(m)
        for t in range(self.T):
            # shuffle the data.
            np.random.shuffle(indx)
            X = X[indx,:]
            y = y[indx]
            for i in range(m):
                pred_y = np.dot(X[i,:],w)
                if y[i]*pred_y <=0:
                    # update the weight predictors.
                    w = w + (self.lr*y[i])*np.transpose(X[i,:])
        return w
    
    # Compute predicted labels.
    def standard_pred_val(self,X_test,y_test,X,y):
        w = self.standard_perceptron(X, y)
        pred_y = np.sign(np.dot(X_test,w))
        pred_y[pred_y == 0] = -1
        return pred_y
    # implement voted perceptron.
    def voted_perceptron(self,X,y):
        m = X.shape[0]
        n = X.shape[1]
        w = np.zeros((n,))
        indx = np.arange(m)
        C = np.array([])
        W = np.array([])
        c = 0
        for t in range(self.T):
            np.random.shuffle(indx)
            X = X[indx,:]
            y = y[indx]
            for i in range(m):
                pred_y = np.dot(X[i,:],w)
                if y[i]*pred_y <= 0:
                    # recording W and counts.
                    W = np.append(W,w)
                    C = np.append(C,c)
                    # update w.
                    w = w + (self.lr*y[i])*np.transpose(X[i,:])
                    c = 1
                else:
                    c = c+1
        row = C.shape[0]
        W = np.reshape(W,(row,-1))
        return W,C
    #compute predicted labels.
    def voted_pred_val(self,X_test,y_test,X,y):
        W,C = self.voted_perceptron(X, y)
        n = X_test.shape[0]
        #k = C.shape[0]
        pred_y = np.zeros((n,))
        for i in range(n):
            term1 = np.sign(np.dot(W,np.transpose(X_test[i,:])))
            term1[term1 == 0] = -1
            val = np.multiply(C,term1)
            pred_y[i] = np.sign(np.sum(val))
        pred_y[pred_y == 0] = -1
        return pred_y
    
    # implement average perceptron.
    def average_perceptron(self,X,y):
        m = X.shape[0]
        n = X.shape[1]
        w = np.zeros((n,))
        indx = np.arange(m)
        
        a = np.zeros((n,))
        for t in range(self.T):
            np.random.shuffle(indx)
            X = X[indx,:]
            y = y[indx]
            for i in range(m):
                pred_y = np.dot(X[i,:],w)
                if y[i]*pred_y <= 0:
                    # update w
                    w = w + (self.lr*y[i])*np.transpose(X[i,:])
                a = a+w
        return a
    
    # predict labels by average perceptron.
    def average_pred_val(self,X_test,y_test,X,y):
        a = self.average_perceptron(X, y)
        pred_y = np.sign(np.dot(X_test,a))
        pred_y[pred_y == 0] = -1
        return pred_y
    
    
    
    