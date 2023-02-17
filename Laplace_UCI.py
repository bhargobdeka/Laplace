# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 00:33:51 2022

@author: BD
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:37:46 2022

@author: BD
"""

#import pandas as pd
import zipfile
import urllib.request
import os
import time
import json
import copy
import math
#import GPy
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, Dataset, TensorDataset
from laplace.baselaplace import FullLaplace
from laplace.curvature.backpack import BackPackGGN
from laplace import Laplace, marglik_training
# from torch.optim.sgd import SGD
# from sklearn.model_selection import KFold
# from scipy.io import savemat
# from torchvision import datasets, transforms
# from torchvision.utils import make_grid
# from tqdm import tqdm, trange

#####################################
# Network Class
#####################################
# Create the net
class Net(nn.Module):
    def __init__(self, input_dim, output_dim, num_units):
        super(Net, self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        
        self.l1 = nn.Linear(input_dim, num_units)
        self.l2 = nn.Linear(num_units, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

######################################
# Create model fit function
def fit_model(model, optimizer, criterion, train_loader):
    
    for x_batch, y_batch in train_loader:
        
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        # gradient descent or adam step
        loss.backward()
        optimizer.step()
        
        
    return model

def model_predict(model_instance, X_test):
    model  = model_instance
    output = model(X_test)
    sp = torch.nn.Softplus()
    mu, sig = output[:,0], sp(output[:,1])+(10)**-6
    return mu, sig

###################################
# We load the data
data_name = ["bostonHousing"]
#data_name = ["bostonHousing","kin8nm", "naval-propulsion-plant", "power-plant", "protein-tertiary-structure"]
# data_name = ["concrete","energy", "wine-quality-red", \
#               "kin8nm","naval-propulsion-plant",\
#               "power-plant","protein-tertiary-structure"]
for j in range(len(data_name)):
    data = np.loadtxt(data_name[j] + '/data/data.txt')
    # We load the indexes for the features and for the target
    
    index_features = np.loadtxt(data_name[j] +'/data/index_features.txt').astype(int)
    index_target   = np.loadtxt(data_name[j] +'/data/index_target.txt').astype(int)
    
    
    # Change only for Protein
    if data_name[j] == 'protein-tertiary-structure':
        num_units = 100
        n_splits  = 5
    else:
        num_units = 50
        n_splits  = 20
        
    # Input data and output data
    X = data[ : , index_features.tolist() ]
    Y = data[ : , index_target.tolist() ]
    input_dim = X.shape[1]
    
    if os.path.isfile("results_DNN/log_{}.txt".format(data_name[j])):
        os.remove("results_DNN/log_{}.txt".format(data_name[j]))
#    from subprocess import call
#    call(["rm", "results/log_{}.txt".format(data_name[j])], shell=True)
    _RESULTS_lltest = data_name[j] + "/results_DNN_Lap/lltest_Lap.txt"
    _RESULTS_RMSEtest = data_name[j] + "/results_DNN_Lap/RMSEtest_Lap.txt"
    
   
    train_LLs, test_LLs, train_RMSEs, test_RMSEs, runtimes = [], [], [], [], []
    
    
    runtimes,runtimes_oneE = [],[]
    lltests, rmsetests = [], []
    rmse_split, ll_split = [], []
    for i in range(n_splits):
        print("Split : " + str(i+1))
        index_train = np.loadtxt(data_name[j] +"/data/index_train_{}.txt".format(i)).astype(int)
        index_test = np.loadtxt(data_name[j] +"/data/index_test_{}.txt".format(i)).astype(int)
        
        #Check for intersection of elements
        ind = np.intersect1d(index_train,index_test)
        if len(ind)!=0:
            print('Train and test indices are not unique')
            break
        
        # Train and Test data for the current split
        X_train = X[ index_train.tolist(), ]
        Y_train = Y[ index_train.tolist() ]
        Y_train = np.reshape(Y_train,[len(Y_train),1]) #BD
        X_test  = X[ index_test.tolist(), ]
        Y_test  = Y[ index_test.tolist() ]
        Y_test = np.reshape(Y_test,[len(Y_test),1])    #BD
        
        # Normalise Data
        X_means, X_stds = X_train.mean(axis = 0), X_train.var(axis = 0)**0.5
        idx = X_stds==0
        X_stds[np.where(idx)[0]] = 1
        Y_means, Y_stds = Y_train.mean(axis = 0), Y_train.var(axis = 0)**0.5
    
        X_train = (X_train - X_means)/X_stds
        Y_train = (Y_train - Y_means)/Y_stds
        # X_val   = (X_val - X_means)/X_stds
        # Y_val   = (Y_val - Y_means)/Y_stds
        X_test  = (X_test - X_means)/X_stds
        # Y_test  = (Y_test - Y_means)/Y_stds
        # Converting to Torch objects
        X_train = torch.from_numpy(X_train).float()
        Y_train = torch.from_numpy(Y_train).float()
        # X_train = torch.Tensor(X_train, dtype = torch.float, requires_grad=True)
        # Y_train = torch.Tensor(X_train, dtype = torch.float, requires_grad=True)
        # X_val   = torch.from_numpy(X_val).float()
        # Y_val   = torch.from_numpy(Y_val).float()
        
        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test).float()
        
        
        # Creating tensor Dataset
        train_data = TensorDataset(X_train, Y_train)
        
        # Creating test tensor Dataset
        test_data = TensorDataset(X_test, Y_test)
        
        # initialize 5 networks
        model = Net(input_dim, output_dim=1, num_units=num_units)
        # Train Loader
        train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True)
        
        # Test Loader
        test_loader = DataLoader(dataset=test_data, batch_size=500)
        
        train_logliks, test_logliks, train_errors, test_errors = [], [], [], []
        val_logliks = []
        
        # start_time
        start_time = time.time()
        num_epochs = 100
        
        
        mzstack, Szstack = [],[]
        errors, lls = [],[]
        
        # optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)#1e-02
        log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
        hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)#1e-01
        runtimes_1E = []
        for epoch in range(num_epochs):
            print(epoch)
            start_1E = time.time()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                # print(loss)
                loss.backward()
                optimizer.step()
        #%% Creating the model
            # choices for weight subsets: 'all', 'subnetwork', 'last_layer'
            # choices for hessian: 'full', 'kron', 'lowrank' and 'diag'
            la = Laplace(model, 'regression', subset_of_weights='all', hessian_structure='full')
            la.fit(train_loader)
            # print(log_sigma.exp())
            hyper_optimizer.zero_grad()
            neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
            neg_marglik.backward()
            hyper_optimizer.step()
            # print(f'sigma={la.sigma_noise.item():.2f}',
            #    f'prior precision={la.prior_precision.item():.2f}')
            f_mu, f_var = la(X_test)
            f_mu = f_mu.squeeze().detach().cpu().numpy()
            f_sigma = f_var.squeeze().sqrt().cpu().numpy()
            pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item()**2)
            pred_var = pred_std**2
            # Test RMSE and LL
            ## Metrics
            Y_true     = Y_test.detach().numpy()
            test_ystd  = np.std(Y_true)
            test_ymean = np.mean(Y_true)
        
            y_mean = test_ystd*f_mu + test_ymean
            y_var  = (test_ystd**2) * pred_var # + la.sigma_noise.item()**2
            y_mean = np.reshape(y_mean,[len(y_mean),1])    #BD
            y_var = np.reshape(y_var,[len(y_var),1])
        
            rmse    = np.sqrt(np.mean(np.power((Y_true - y_mean),2)))
            test_ll = np.mean(-0.5 * np.log(2 * np.pi * (y_var)) - \
                          0.5 * (Y_true - y_mean)**2 / (y_var))
            errors += [rmse]
            lls += [test_ll]
            runtime_1E = time.time()-start_1E
            # print(runtime_1E)
            runtimes_1E.append(runtime_1E)
            print("RMSE : "+ str(rmse))
            print("Test LL : "+ str(test_ll))
        # print(str(np.mean(runtimes_1E)))
        runtime = time.time()-start_time
        lltests.append(lls)
        rmsetests.append(errors)
        runtimes.append(runtime)
    
    mean_ll   = np.mean(lltests,axis=0)
    mean_RMSE = np.mean(rmsetests,axis=0)
    print("best LL"+str(mean_ll[-1]))
    
    plt.scatter(range(100), mean_ll)
    plt.xlabel('epochs')
    plt.ylabel('LL')
    plt.show()
    plt.scatter(range(100), mean_RMSE)
    plt.xlabel('epochs')
    plt.ylabel('RMSE')
    plt.show()
    
    # with open(_RESULTS_lltest, "w") as myfile:
    #         for item in mean_ll:
    #                 myfile.write('%f\n' % item)
    # with open(_RESULTS_RMSEtest, "w") as myfile:
    #         for item in mean_RMSE:
    #             myfile.write('%f\n' % item)
    
        