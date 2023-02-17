# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 21:51:33 2021

@author: BD
"""

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
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
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
    print("Split no. {}".format(i))
    # Train index for each split
    index_train = np.loadtxt('../data/index_train_{}.txt'.format(i)).astype(int)
    index_test = np.loadtxt('../data/index_test_{}.txt'.format(i)).astype(int)
    Xtrain = X[index_train.tolist(), :]
    ytrain = y[index_train.tolist()]
    Xtest  = X[index_test.tolist(), :]
    ytest  = y[index_test.tolist()]
    # Splitting data into folds
    kfold = KFold(5, True, 1)
    E = []
    count = 0
    for train_index, val_index in kfold.split(Xtrain):
         count +=1
         print("Fold no. {}".format(count))
         X_train, X_val = Xtrain[train_index], Xtrain[val_index]
         y_train, y_val = ytrain[train_index], ytrain[val_index]
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
         # Y val normalised
         y_val_normalized = (y_val - mean_y_train) / std_y_train
         y_val_normalized = np.array(y_val_normalized, ndmin = 2).T
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
         # simple early stopping
         es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=0) #patience=50
         mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True)
         history = model.fit(X_train, y_train_normalized, validation_data=(X_val, y_val_normalized), batch_size=32, epochs=4000, verbose=0, callbacks=[es, mc])   #callbacks=[es]
         es_epoch = len(history.history['val_loss'])
         E.append(es_epoch)
         print(E)
    opt_Epoch = np.round(np.mean(E))
    print(opt_Epoch)
    # We iterate the method 
    #model = get_model()
    network = net.net(Xtrain, ytrain,
        n_hidden , normalize = True, n_epochs = int(opt_Epoch), X_test=Xtest, y_test=ytest)
    running_time = network.running_time

    # We obtain the test RMSE and the test ll

    error, MC_error, ll = network.predict(Xtest, ytest)
    errors += [error]
    MC_errors += [MC_error]
    lls += [ll]
    times += [running_time]


print('Avg. RMSE %f +- %f' % (np.mean(errors), np.std(errors)))   
print('Avg. LL %f +- %f' % (np.mean(lls), np.std(lls)))         
         
     