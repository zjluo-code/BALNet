#!/usr/bin/env python
# coding=utf-8
import numpy as np
#import tensorflow as tf
#from tensorflow.keras.layers import Activation,Input,Dense,LSTM,Dropout,BatchNormalization,Conv1D,TimeDistributed
#from tensorflow.keras.layers import Bidirectional,Concatenate,Flatten
#from tensorflow.keras.optimizers import Adam
#from sklearn import preprocessing
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import h5py


######load dataset#########

def load_dataset():

    train_dataset = h5py.File('new_hxc_BAL_spec.h5','r')  
    n_train = 160000

    train_set_x_orig = np.array(train_dataset["spectra"][0:n_train]) 
    train_set_x = np.expand_dims(train_set_x_orig,axis=2)
    train_set_y_orig = np.array(train_dataset["labels"][0:n_train]) 
    train_set_y = np.expand_dims(train_set_y_orig,axis=2)
    test_set_x_orig = np.array(train_dataset["spectra"][n_train:]) 
    test_set_x = np.expand_dims(test_set_x_orig,axis=2)
    test_set_y_orig = np.array(train_dataset["labels"][n_train:]) 
    test_set_y = np.expand_dims(test_set_y_orig,axis=2)

    return train_set_x, train_set_y, test_set_x, test_set_y 

Tx = 1136

n_freq = 1 

X_train,Y_train,X_test,Y_test =  load_dataset()

###
x_1136 = np.arange(0,1136,1)

###
x_temp = np.round(np.arange(0,1136,1.0/377*1136))
x_377 = [int(item) for item in x_temp]

###############

model = load_model('./training_checkpoints/BAL_iden_200.h5')


for i in range(40000):
    Y_pred_temp = model.predict(X_test[i:i+1])
    Y_pred = np.squeeze(Y_pred_temp)
    Y_true = np.squeeze(Y_test[i])
    Y_true_1136 = np.zeros(Tx)
    for no in x_1136:
        for index, item in enumerate(x_377):
            if no == item:
               Y_true_1136[i] = Y_true[index]

    plt.figure(figsize=(6,6))
    plt.subplot(3,1,1)
    plt.plot(x_1136,X_test[i])
    plt.subplot(3,1,2)
    plt.plot(x_377,Y_true)
    plt.subplot(3,1,3)
    plt.plot(x_377,Y_pred)
    plt.ylim(0,1)
    plt.show()




