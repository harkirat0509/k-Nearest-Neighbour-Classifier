#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:22:42 2018

@author: harkirat
"""
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

np.random.seed(2)

#reading dataset
df = pd.read_csv('teleCust1000t.csv')

mask = np.random.rand(len(df))<0.8

train = df[mask]
test = df[~mask]

#train will be m_trainx12 and test will be m_testx12
train = train.values
test = test.values
m_train = len(train)
m_test = len(test)

train_data_X = train[:,:-1]
train_data_Y = train[:,-1].reshape(m_train,1)

test_data_X = test[:,:-1]
test_data_Y = test[:,-1].reshape(m_test,1)

mean = np.mean(train_data_X,axis=0)
std = np.std(train_data_X,axis=0)
train_data_X = (train_data_X-mean)/std
test_data_X = (test_data_X-mean)/std

def kNearestNeighbour(k,train_data_X,test_data_X,train_data_Y,test_data_Y):
    '''
    kNearestNeighbour function is used to classify data based on similarity criteria of euclidean distance
    Input is value of k, training data and test data
    Returns accuracy of predictions
    '''
    m_test = len(test_data_X)
    m_train = len(train_data_X)
    predictions = np.zeros((m_test,1))
    for j in range(m_test):
        dist = np.zeros((m_train,1))
        for i in range(m_train):
            dist[i] = (np.linalg.norm(test_data_X[j]-train_data_X[i]))
        
        kindex = []
        for i in range(k):
            index = np.argmin(dist)
            kindex.append(index)
            dist[index] = 100

        elements,count = np.unique(train_data_Y[kindex],return_counts=True)
        predictions[j] = elements[np.argmax(count)]
    
    return metrics.accuracy_score(test_data_Y,predictions)

accuracy = []
for k in range(1,41,2):
    accuracy.append(kNearestNeighbour(k,train_data_X,test_data_X,train_data_Y,test_data_Y))
    
plt.plot(range(1,41,2),accuracy)

#k corresponding to maximum accuracy can be obtained from graph now
