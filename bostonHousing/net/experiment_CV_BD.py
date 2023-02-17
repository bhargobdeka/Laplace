# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:29:57 2021

@author: BD
"""

import math
from scipy.misc import logsumexp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras import backend as K
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.constraints as constraints

import time
import sys
sys.path.append('net/')
import net
# We fix the random seed

np.random.seed(1)
data = np.loadtxt('../data/data.txt')

# We load the number of hidden units

n_hidden = [int(np.loadtxt('../data/n_hidden.txt').tolist())]

# We load the number of training epocs

n_epochs = int(np.loadtxt('../data/n_epochs.txt').tolist())

# We load the indexes for the features and for the target

index_features = np.loadtxt('../data/index_features.txt').astype(int)
index_target = np.loadtxt('../data/index_target.txt').astype(int)

X = data[ : , index_features.tolist() ]
y = data[ : , index_target.tolist() ]

# We iterate over the training test splits

n_splits = int(np.loadtxt('../data/n_splits.txt'))
errors, MC_errors, lls, times = [], [], [], []

for i in range(n_splits):
    # Train index for each split
    index_train = np.loadtxt('../data/index_train_{}.txt'.format(i)).astype(int)
    index_test = np.loadtxt('../data/index_test_{}.txt'.format(i)).astype(int)
    Xtrain = X[index_train, :]
    Xtest  = X[index_test, :]
    # Splitting data into folds
    kfold = KFold(5, True, 1)
    E = []
    count = 0
    for train_index, val_index in kfold.split(Xtrain):
         count +=1
         print("Fold no. {}".format(count))
         X_train, X_val = X[train_index], X[val_index]
         y_train, y_val = y[train_index], y[val_index]
         # Normalize data
         std_X_train = np.std(X_train, 0)
         std_X_train[std_X_train == 0 ] = 1
         mean_X_train = np.mean(X_train, 0)
         # X normalized
         X_train = (X_train - np.full(X_train.shape, mean_X_train)) / \
                np.full(X_train.shape, std_X_train)
         X_val   = (X_val - np.full(X_val.shape, mean_X_train)) / \
                np.full(X_val.shape, std_X_train)     
         mean_y_train = np.mean(y_train)
         std_y_train = np.std(y_train)
         # Y normalized
         y_train_normalized = (y_train - mean_y_train) / std_y_train
         y_train_normalized = np.array(y_train_normalized, ndmin = 2).T
         # Model
         def get_model():
             N = X_train.shape[0]
             dropout = 0.05
             tau = 0.159707652696 # obtained from BO
             lengthscale = 1e-2
             reg = lengthscale**2 * (1 - dropout) / (2. * N * tau)
             model = Sequential()
             # Input layer
             model.add(Dropout(dropout, input_shape=(X_train.shape[1],)))
             model.add(Dense(n_hidden[0], activation='relu', W_regularizer=l2(reg)))
             # Hidden layer
             for i in range(len(n_hidden) - 1):
                 model.add(Dropout(dropout))
                 model.add(Dense(n_hidden[i+1], activation='relu', W_regularizer=l2(reg)))
             # Outer layer    
             model.add(Dropout(dropout))
             model.add(Dense(y_train_normalized.shape[1], W_regularizer=l2(reg)))
             # Final Step is compilation
             model.compile(loss='mean_squared_error', optimizer='adam')
             return model
         batch_size = 32
         tau = 0.159707652696
         model = get_model()
         max_epoch = 1000
         epoch = 1
         LL_list = []
         LL_list_train = []
         while epoch <max_epoch+1:
             #print("epoch:{}".format(epoch))
             history = model.fit(X_train, y_train_normalized, batch_size, epoch, verbose=0)
             #print(history.history.keys())
             # summarize history for loss
#             plt.plot(history.history['loss'])
#             plt.title('model loss')
#             plt.ylabel('loss')
#             plt.xlabel('epoch')
#             plt.legend(['train', 'test'], loc='upper left')
#             plt.show()
#             standard_pred = model.predict(X_val, batch_size=1, verbose=0)
#             standard_pred = standard_pred * std_y_train + mean_y_train
             
             T = 10000
             predict_stochastic = K.function([model.layers[0].input, K.learning_phase()], model.layers[-1].output)
             
             Yt_hat_train = np.array([predict_stochastic([X_train, 1]) for _ in range(T)])
             Yt_hat_train = Yt_hat_train * std_y_train + mean_y_train
#             # Train Log-likelihood
#             ll_t = (logsumexp(-0.5 * tau * (y_train[None] - Yt_hat_train)**2., 0) - np.log(T) 
#                - 0.5*np.log(2*np.pi) + 0.5*np.log(tau))
#             train_ll = np.mean(ll_t)
#             LL_list_train.append(train_ll)
#             print(LL_list_train)
             # We compute the test log-likelihood
             Yt_hat = np.array([predict_stochastic([X_val, 1]) for _ in range(T)])
             Yt_hat = Yt_hat * std_y_train + mean_y_train
             ll = (logsumexp(-0.5 * tau * (y_val[None] - Yt_hat)**2., 0) - np.log(T) 
                - 0.5*np.log(2*np.pi) + 0.5*np.log(tau))
             test_ll = np.mean(ll)
             LL_list.append(test_ll)
             #print(LL_list)
             epoch+=1
         ind_maxLL = LL_list.index(np.max(LL_list))+1
         E.append(ind_maxLL)
         print(E)

opt_Epoch = np.round(np.mean(E))
print(opt_Epoch)
     