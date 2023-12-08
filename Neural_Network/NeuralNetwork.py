#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:57:53 2023

@author: ivanyang
"""

import math
import numpy as np
import scipy

class NeuralNetwork:
    def __init__(self,width_list):
        self.in_dim = width_list[0]
        self.out_dim = width_list[-1]
        self.width_list = width_list
        self.layers = len(width_list)
        self.gamma = 0.1
        self.d = 5
        self.epochs = 200
        self.batch_size = 20
        self.weights = [None for q in range(self.layers)]
        self.dweights = [None for q in range(self.layers)]
        for i in range(1,self.layers-1):
            # take care about bias term !!!!!
            # initialize weights among input layer & hidden layers(layer 0 -1)
            # as well as first hidden layers & second hidden layers (layer 1-2)
            #wi = np.random.normal(0,1,(self.width_list[i]-1,self.width_list[i-1]))
            wi = np.zeros((self.width_list[i]-1,self.width_list[i-1]))
            self.weights[i] = wi
            self.dweights[i] = np.zeros((self.width_list[i]-1,self.width_list[i-1]))
            
        # initialize layer2 to layer3's weights
        i = self.layers-1
        # row records weights w.r.t next layer, column records weights w.r.t previous layer
        wi = np.random.normal(0,1,(self.width_list[i],self.width_list[i-1]))
        self.weights[i] = wi
        self.dweights[i] = np.zeros((self.width_list[i],self.width_list[i-1]))
        #print('initial size',self.dweights[i].shape)
        # initialize nodes
        self.nodes = [np.ones([self.width_list[i],1]) for i in range(self.layers)]
        
    # set_x function to adjust hyper-parameter.
    def set_gamma(self,gamma):
        self.gamma = gamma
    
    def set_d(self,d):
        self.d = d
    
    def set_epochs(self,epochs):
        self.epochs = epochs
    
    def set_batch_size(self,batch_size):
        self.batch_size = batch_size
    # use scipy to compute sigmoid, avoid warning.
    
    def sigmoid(self,x):
        return scipy.special.expit(x)
    
    def forward(self,x):
        # forward pass
        #w_3^T\sigmoid(w_2^T\sigmoid(w1^Tx+w_0^1^Tx0)+w_0^2^Tz0)+w_0^3^Tz_0^2
        self.nodes[0] = x
        for i in range(1,self.layers-1):
            inputs = np.matmul(self.weights[i], self.nodes[i-1])
            self.nodes[i][:-1,:] = self.sigmoid(inputs.reshape([-1,1]))
        i = self.layers-1
        self.nodes[i] = np.matmul(self.weights[i], self.nodes[i-1])
    
    def backprop(self,y):
        #back-propogation of algorithm: compute gradient
        ## Last layer
        ## dl/dw_3
        dLdz = self.nodes[-1]-y #(y-y^*): n2xn
        #print('dLdy',dLdz.shape)
        nk = self.width_list[-1] # output dimension
        dzdw = np.transpose(np.tile(self.nodes[-2],[1,nk])) # n*n2
        #print('dzdw', dzdw.shape)
        # dL/dy* dy/dw = dL/dw : R^{n_output x n_2}
        self.dweights[-1] = dLdz*dzdw #n2xn
        #size_test = dLdz*dzdw
        #print(size_test.shape)
        #print('last layer weights',self.dweights[-1].shape)
        # exclude bias term (row: # of z, column: y)
        # equivalent as dy/dz = [w,...w] (and exclude bias term.)
        # in shape of R^{nx(n2-1)}
        dzdz = self.weights[-1][:,:-1]
        #print('dydz^2',dzdz.shape)
        # compute weights for hidden layers
        for i in reversed(range(1,self.layers-1)):
            nk = self.width_list[i]-1
            # nodes value of input layers
            z_inp = self.nodes[i-1]
            #print('input z',z_inp.shape)
            # exclude bias term
            z_out = self.nodes[i][:-1]
            #print('output z', z_out.shape)
            # s as intermediate value for \sigma(s)
            # ds dw in shape of R^{(n2-1)xn1}
            dsdw = np.transpose(np.tile(z_inp,[1,nk]))
            #print('dsdw',dsdw.shape)
            # dz_{out}/dw, in shape of R^{(n2-1)xn1}
            dzdw = z_out*(1-z_out) * dsdw
            #print('dzdw',dzdw.shape)
            # dL/dz_{out+1}*dz_{out+1}/dz_{out} =dL/dz_{out}
            # in shape of R^{(n2-1)x1}
            dLdz = np.matmul(np.transpose(dzdz),dLdz)
            #print('dLdz',dLdz.shape)
            # dL/dz_{out}*dz_{out}/dw
            # in shape of R^{n-1xn1}
            dLdw = dLdz * dzdw
            #print('dLdw',dLdw.shape)
            self.dweights[i] = dLdw
            # update dz_{out}/dz_{int} (view the output layer as y)
            # in shape of R^{(n2-1)x(n1-1)}
            dzdz = z_out * (1-z_out) * self.weights[i]
            #exclude bias term of z_{out}
            dzdz = dzdz[:,:-1]
            #print('dzdz',dzdz.shape)
        
    
    
    def SGD_update_w(self,lr):
        # one_step sgd update
        for i in range(1,self.layers):
            self.weights[i] = self.weights[i] - lr * self.dweights[i]
    
    def forward_backprop(self,x,y):
        self.forward(x)
        self.backprop(y)
    
    def mini_batch_SGD(self, x,y):
        m,n = x.shape
        indx = np.arange(m)
        for t in range(self.epochs):
            np.random.shuffle(indx)
            x = x[indx,:]
            y = y[indx,:]
            np.random.seed(123)
            sample_indx = np.random.choice(m,size = self.batch_size,replace = False)
            # sample index set to compute stochastic gradient.
            x_row = x[sample_indx,:]
            y_row = y[sample_indx,:]
            for i in range(len(sample_indx)):
                self.forward_backprop(x_row[i].reshape((self.in_dim, 1)), y_row[i].reshape((self.out_dim, 1)))
                lr = (self.gamma) / (1 + (self.gamma/self.d) * t)*(1/len(sample_indx))
                self.SGD_update_w(lr)
    
    def predict(self, x):
     # make predictions given input x.
      m,n = x.shape
      pred_list = []
      for i in range(m):
          self.forward(x[i,:].reshape(self.in_dim))
          y = self.nodes[-1]
          pred_list.append(np.transpose(y))
      y_pred = np.concatenate(pred_list, axis=0)
      return y_pred
        