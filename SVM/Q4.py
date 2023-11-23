#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:19:09 2023

@author: ivanyang
"""
import numpy as np
X=np.array([[0.5,-1,0.3,1],
           [-1,-2,-2,1],
           [1.5,0.2,-2.5,1]])
y = np.array([1,-1,1])

def sgd_primal(w,X,y,lr):
    # X refers to the input matrix in R^{mxn}, y refers to vector R^m.
    m = X.shape[0]
    n = X.shape[1]
    # initialize w
    indx = np.arange(m)
    np.random.shuffle(indx)
    X = X[indx,:]
    y = y[indx]
    for i in range(m):
        term1 = y[i]*np.dot(X[i,:],w)
        sub_grad = np.copy(w)
        sub_grad[n-1] = 0 
        if term1 < 1:
            sub_grad = sub_grad - m*y[i]*np.transpose(X[i,:])
        w = w - lr*sub_grad
                
    return w


w = np.zeros((4,))

w1 = sgd_primal(w, X, y, 0.01)

print(w1)

w2 = sgd_primal(w1, X, y, 0.005)
print(w2)

w3 = sgd_primal(w2, X, y, 0.0025)
print(w3)