#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:55:57 2023

@author: ivanyang
"""

import numpy as np
import math
import scipy

class logistic_regression:
    
    def __init__(self):
        # gamma0
        self.lr = 0.8
        self.epoch = 100
        self.v = 0.1
        self.d = 5
    
    def set_lr(self,lr):
        self.lr = lr
    
    def set_epoch(self,epoch):
        self.epoch = epoch
    
    def set_v(self,v):
        self.v = v
        
    def set_d(self,d):
        self.d = d
    
    def MAP(self, X,y):
        m,n = X.shape
        w = np.zeros((n,1))
        indx = np.arange(m)
        for t in range(self.epoch):
            np.random.shuffle(indx)
            X = X[indx,:]
            y = y[indx]
            i = np.random.randint(0,m,size=1)
            term0 = y[i]*np.transpose(X[i,:])
            term0 = term0.reshape((-1,1))
            term2 =y[i]*np.dot(X[i,:],w)
            #print(term2.shape)
            #term2 = np.array(term2,dtype = np.float128)
            term1 = - m* (scipy.special.expit(-term2))*term0
            grad = w/(self.v)+term1
            lr = self.lr / (1 + self.lr / self.d * t)
            w = w - lr*grad
        return w
    
    def ML(self,X,y):
        m,n = X.shape
        w = np.zeros((n,1))
        indx = np.arange(m)
        for t in range(self.epoch):
            np.random.shuffle(indx)
            X = X[indx,:]
            y = y[indx]
            i = np.random.randint(0,m,size=1)
            term0 = y[i]*np.transpose(X[i,:])
            term0 = term0.reshape((-1,1))
            term2 =y[i]*np.dot(X[i,:],w)
            term1 = - m* (scipy.special.expit(-term2))*term0
            grad = term1
            lr = self.lr / (1 + (self.lr / self.d) * t)
            w = w - lr*grad
        return w
    