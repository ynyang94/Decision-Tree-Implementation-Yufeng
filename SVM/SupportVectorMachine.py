#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:36:09 2023

@author: ivanyang
"""
# In this document, we define a class : SVM and put all attributes(functions)
# in this class.
import numpy as np
import pandas as pd
import scipy.optimize as opt

class SVM:
    def __init__(self):
        self.C = 1
        # lr refers to gamma_t in svm primal sgd
        self.lr = 0.01
        self.epoch = 100
        # gamma is used for gaussian kernel
        self.gamma = 0.1
        self.a = 1
        
    def set_C(self,C):
        self.C = C
    
    def set_lr(self,lr):
        self.lr = lr
    
    def set_epoch(self,epoch):
        self.epoch = epoch
    
    def set_gamma(self,gamma):
        self.gamma = gamma
    
    def set_d(self,a):
        self.a = a
    
    def sgd_primal(self,X,y):
        # X refers to the input matrix in R^{mxn}, y refers to vector R^m.
        m = X.shape[0]
        n = X.shape[1]
        # initialize w
        w = np.zeros((n,))
        indx = np.arange(m)
        for t in range(self.epoch):
            np.random.shuffle(indx)
            X = X[indx,:]
            y = y[indx]
            for i in range(m):
                term1 = y[i]*np.dot(X[i,:],w)
                sub_grad = np.copy(w)
                sub_grad[n-1] = 0 
                # Compute subgradient here
                if term1 < 1:
                    sub_grad = sub_grad - self.C*m*y[i]*np.transpose(X[i,:])
                lr = self.lr/(1+(self.lr/self.a)*t)
                w = w - lr*sub_grad 
        return w
    
    def obj_primal(self,w,X,y):
        # Compute primal objective value.
        m = w.shape[0]
        obj_val1 = 0.5*np.linalg.norm(w[0:m-1],2)**2
        obj_val2 = self.C*np.sum(np.maximum(0,1-np.multiply(y,np.dot(X,w))))
        return obj_val1+obj_val2
    
    def obj_dual(self,X,y,alpha):
        # To compute objective value
        # Remember to delete the last column in X, which is all 1.
        m = X.shape[0]
        n = X.shape[1]
        X = X[:,0:n-1]
        alpha = np.reshape(alpha,(-1,1))
        q = alpha.shape[0]
        alpha = np.reshape(alpha,(q,))
        vector1 = np.multiply(alpha,y)
        vector2 = np.multiply(vector1.reshape((-1,1)),X)
        vector2 = np.reshape(vector2, (-1,1))
        obj_val = 0.5*np.dot(np.transpose(vector2),vector2)-np.sum(alpha)
        return obj_val
    
    def constraint(self,alpha,y):
        # Equality constraint.
        alpha = np.reshape(alpha,(-1,1))
        q = alpha.shape[0]
        alpha = np.reshape(alpha,(q,))
        val = np.dot(np.transpose(alpha),y)
        return val
    
    def dual_svm(self,X,y):
        m = X.shape[0]
        n = X.shape[1]
        bound = [(0,self.C)]*m
        cons = ({'type': 'eq', 'fun': lambda alpha: self.constraint(alpha, y)})
        alpha0 = np.zeros((m,))
        # Solve constrained optimization using scipy.
        res = opt.minimize(lambda alpha0: self.obj_dual(X, y, alpha0), alpha0, 
                           method='SLSQP',  bounds=bound, constraints=cons,
                           options={'maxiter':100,'disp': True})
        # recover w
        alpha_opt = res.x
        q = alpha_opt.shape[0]
        alpha_opt = np.reshape(alpha_opt, (q,))
        vec1 = np.multiply(alpha_opt,y)
        # wrong, delete last column.
        X_hat = X[:,0:n-1]
        w0 = np.sum(np.multiply(vec1.reshape((-1,1)),X_hat),axis = 0)
        w0 = w0.reshape((w0.shape[0],))
        active_indx = np.where((alpha_opt > 0) & (alpha_opt < self.C))
        b =  np.mean(y[active_indx] - np.dot(X_hat[active_indx,:],w0))
        w0 = np.append(w0,b)
        return w0
    
    def gaussian_kernel(self,X1,X2,gamma):
        # X_1 and X_2 are matrix.
        # Remember to delete last column.
        X1 = X1[:,0:X1.shape[1]-1]
        X2 = X2[:,0:X2.shape[1]-1]
        K = np.zeros((X1.shape[0],X2.shape[0]))
        for i,x in enumerate(X1):
            for j,y in enumerate(X2):
                K[i,j] = np.exp(np.square(np.linalg.norm(x-y))/-gamma)
        return K
    
    def kernel_dual_obj(self,K,y,alpha):
        # Compute dual objective of kernel svm.
        alpha = np.reshape(alpha,(-1,1))
        q = alpha.shape[0]
        alpha = np.reshape(alpha,(q,))
        vector1 = np.multiply(alpha,y)
        vector1 = vector1.reshape((-1,1))
        vector1 = np.matmul(vector1, np.transpose(vector1))
        return 0.5*np.sum(np.multiply(vector1, K)) - np.sum(alpha)
    
    def kernel_svm_train(self,X,y):
        # train constrained kernel svm using scipy.
        m = X.shape[0]
        n = X.shape[1]
        bound = [(0,self.C)]*m
        cons = ({'type': 'eq', 'fun': lambda alpha: self.constraint(alpha, y)})
        alpha0 = np.zeros(m)
        K = self.gaussian_kernel(X,X, self.gamma)
        res1 = opt.minimize(lambda alpha0: self.kernel_dual_obj(K,y, alpha0), alpha0, 
                           method='SLSQP',  bounds=bound, constraints=cons,
                           options={'disp': True})
        return res1.x
    
    def kernel_prediction(self,alpha,X,y,X0):
        # X,y refers to training data
        # X0 refers to test instances.
        m = X.shape[0]
        n = X.shape[1]
        n0 = X0.shape[1]
        X_hat = X[:,0:n-1]
        X0_hat = X0[:,0:n0-1]
        #K = self.gaussian_kernel(X, X,self.gamma)
        vector1 = np.multiply(alpha,y)
        #output1 = np.sum(np.multiply(vector1.reshape((-1,1)),K),axis = 0)
        #active_indx = np.where((alpha > 0) & (alpha < self.C))
        #X_hat = X[active_indx,:]
        #X_hat = X_hat.reshape((-1,n))
        #print(X_hat.shape)
        #K2 = self.gaussian_kernel(X, X_hat, self.gamma)
        #output2 = np.sum(np.multiply(vector1.reshape((-1,1)),K2),axis = 0)
        #b =  np.mean(y[active_indx] - output2)
        #print(b)
        K3 = self.gaussian_kernel(X, X0, self.gamma)
        output3 = np.sum(np.multiply(vector1.reshape((-1,1)),K3),axis = 0)
        y_hat = np.copy(output3)
        y_hat[output3 > 0] =1
        y_hat[output3 <= 0] = -1
        return y_hat
    
    

    def kernel_perceptron(self,X,y):
        m = X.shape[0]
        n = X.shape[1]
        indx = np.arange(m)
        c = np.zeros((m,))
        t = 0
        K = self.gaussian_kernel(X, X,self.gamma)
        for t in range(self.epoch):
            np.random.shuffle(indx)
            for i in range(m):
                vector1 = np.multiply(c,y)
                K1 = K[:,indx[i]]
                score = y[indx[i]]*np.dot(np.transpose(vector1),K1)
                if score <= 0:
                    c[indx[i]] =c[indx[i]] + 1        
        return c
    
    def kernel_perceptron_predict(self,c,X,y,X0):
        c = self.kernel_perceptron(X, y)
        K = self.gaussian_kernel(X, X0, self.gamma)
        vector1 = np.multiply(c,y)
        vector2 = np.sum(np.multiply(vector1.reshape((-1,1)),K),axis = 0)
        vector2[vector2>0] = 1
        vector2[vector2<0] = -1
        return vector2