#!/usr/bin/env python
# coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation,Input,Dense,LSTM,Dropout,BatchNormalization,Conv1D,TimeDistributed
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import h5py


######load dataset#########

def load_dataset():

    train_dataset = h5py.File('./BAL_spec.h5','r')  
    n_train = 160000
    #n_sample = 52353
    #n_arr = np.arange(n_sample)
    #random.shuffle(n_arr)

    train_set_x_orig = np.array(train_dataset["spectra"][0:n_train]) #train set 
    train_set_x = np.expand_dims(train_set_x_orig,axis=2)
    train_set_y_orig = np.array(train_dataset["labels"][0:n_train]) #train set labels
    train_set_y = np.expand_dims(train_set_y_orig,axis=2)
    #test_dataset = h5py.File('mxm_imgs.h5',"r")
    test_set_x_orig = np.array(train_dataset["spectra"][n_train:]) #test set
    test_set_x = np.expand_dims(test_set_x_orig,axis=2)
    test_set_y_orig = np.array(train_dataset["labels"][n_train:]) #test set
    test_set_y = np.expand_dims(test_set_y_orig,axis=2)

    #classes = np.array(test_dataset["list_classes"][:])   

    #train_set_y_orig = train_set_y_orig.T
    #test_set_y_orig = test_set_y_orig.T

    return train_set_x, train_set_y, test_set_x, test_set_y

def model(input_shape):
    #input
    X_input = Input(shape = input_shape)  
    #CNN layer
    X = Conv1D(192, kernel_size=7,strides=3)(X_input) # 
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.2)(X)
    #LSTM layer
    X = Bidirectional(LSTM(units = 128, return_sequences = True))(X)
    X = Dropout(0.2)(X)
    X = BatchNormalization()(X)
    #LSTM layer
    X = Bidirectional(LSTM(units = 128, return_sequences = True))(X) #
    X = Dropout(0.2)(X)
    X = BatchNormalization()(X)
    #
    #last dense layer
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) 
    model = Model(inputs = X_input, outputs = X)
    return model
Tx = 1165
n_freq = 1 
model = model(input_shape=(Tx,n_freq))
model.summary()
opt = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./training_checkpoints/BAL_iden_{epoch}.h5',save_freq='epoch',period=100,save_best_only=True,monitor='loss')]
X_train,Y_train,X_test,Y_test =  load_dataset()
print(X_train.shape,Y_train.shape)
model.fit(X_train, Y_train, batch_size = 256, epochs=200,callbacks=callbacks)
loss, acc = model.evaluate(X_test, Y_test)
print("Test set accuracy = ", acc)


