#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 21:07:49 2023

@author: ivanyang
"""

import numpy as np
X=np.array([[0.5,-1,0.3,1],
           [-1,-2,-2,1],
           [1.5,0.2,-2.5,1]])
y = np.array([1,-1,1])
lr_set = [0.01, 0.005, 0.0025]
def MAP(X,y,lr,w):
    m,n = X.shape
    
    indx = np.arange(m)
    np.random.shuffle(indx)
    X = X[indx,:]
    y = y[indx]
    i = np.random.randint(0,m,size=1)
    #print(i)
    term0 = y[i]*X[i,:]
    term2 = np.sum(w*X[i,:])
    term1 = - m* (1/(1+np.exp(y[i])*term2))*term0
    grad = w+term1
    return grad
w = np.zeros((4,))
for lr in lr_set:
    grad = MAP(X,y,lr,w)
    w = w - lr*grad
    print(grad)
    #print(w)