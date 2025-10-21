#!/usr/bin/env python
# coding=utf-8

import numpy as np
#import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import pandas as pd

import math
import h5py
import tensorflow as tf
#import random

#============================================================
# This prepare the hdf5 datasets
# arr: dataset [m,nw,nh,nc]   outfile: h5 file
#============================================================

def write_hdf5(arr,outfile,arr_name = "spectra"):
    with h5py.File(outfile,"a") as f:
        f.create_dataset(arr_name,data=arr,dtype=arr.dtype)

n_Lam = 1136 #
#channels =1    #
n_l=377    #label

N_sample = 250000

spectra = np.empty((N_sample,n_Lam),dtype = np.float32)
labels  = np.empty((N_sample,n_l),dtype = np.int32)  #

file_path = '/data_pub/'
spec_temp = fits.open(file_path+"M_SPEC.fits")
spec = Table(spec_temp[0].data)
spec = np.array(spec.to_pandas())
print(spec.shape)
lab_temp = fits.open(file_path+"Mcata.fits")

lab = Table(lab_temp[0].data)
lab = np.array(lab.to_pandas())
print(lab.shape)

for galaxy in range(N_sample):
    spectra[galaxy]=spec[galaxy,:]
    spectra[galaxy]=spectra[galaxy]/np.max(spectra[galaxy]) ######
    labels[galaxy] =lab[galaxy,:] 

write_hdf5(spectra,'BAL_spec.h5','spectra')
write_hdf5(labels,'BAL_spec.h5','labels')
