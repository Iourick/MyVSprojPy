
import numpy as np


import matplotlib.pyplot as plt

from math import *

import scipy
import scipy.stats
import scipy.signal
import scipy.io

import os
import operator as op
import glob
import copy

import inspect
import zipfile
import operator
import itertools

import astropy.io.fits as pyfits
import pylab as P
import matplotlib.cm as cm
import math
import time



np.random.seed(0) 
# Dimensions
d = 3 
m = 4
p = 2
b = 378 # number of frequencies in a sparse grid, due to some clever integration trick. 
i = 5000#100 #5000
e = 10**3#10 #
f = 100



# Generate random complex-valued CuPy arrays on the GPU
w_dmpf = np.random.uniform(0.001, 1,size=(d, m, p, f)).astype(np.complex64)
h_impf = np.random.uniform(0.001, 1,size=(i, m, p, f)).astype(np.complex64)
F_dpe = np.random.uniform(0.001, 1,size=(d, p, e)).astype(np.complex64)
T_dfe = np.random.uniform(0.001, 1,size=(d, f, e)).astype(np.complex64)
D_d = np.random.uniform(0.001, 1,size=(d,)).astype(np.complex64)

# Perform the equivalent operations using CuPy functions
conj_h_impf = np.conj(h_impf)
conj_T_dfe = np.conj(T_dfe)



def FDMT_test_curve(TestFDMTFFT = False):
    f_min = 1200 #Mhz
    f_max = 1600 #MHz
    
    N_bins = 40
    N_t = 1024 #512
    N_f = 1024 #512
    N_total = N_f*N_t*N_bins
    PulseLength = N_f*N_bins
    PulseSig = 0.4
    PulsePosition = 4134567
    D = 5
    maxDT = N_t
    dataType = 'int32'  
    
    practicalD = DispersionConstant * D
    I = np.random.normal(0,1,N_total)
    I[PulsePosition:PulsePosition+PulseLength] += np.random.normal(0,PulseSig,PulseLength)
    print ("MAX Thoretical SNR:", np.sum(np.abs(I[PulsePosition:PulsePosition+PulseLength])**2 - np.mean(abs(I)**2)) / (np.sqrt(PulseLength*np.var(abs(I)**2))))
    
    X = CoherentDedispersion(I, -D, f_min,f_max,False)   
    
   
    XX = np.abs(np.fft.fft(X.reshape([N_f,N_t*N_bins]),axis = 1))**2
    
    XX = np.transpose(XX)
    XX = np.sum(XX.reshape(N_f,N_bins,N_t),axis=1)
    
    E = np.mean(XX[:,:10])
    XX -= E
    V = np.var(XX[:,:10])
    
    XX /= (0.25*np.sqrt(V))
    
    XX_1 = XX.astype(np.int32)

    iDataType = 0;
    if XX_1.dtype == np.int32:
        iDataType = 1
    else:
        if XX_1.dtype == np.int64:
            iDataType = 2

    
    

    np.save('..\\iarrShape.npy',XX_1.shape)
    np.save('..\\XX.npy',XX_1)
    arr_fmin_max =  np.zeros(2,dtype = np.float32)
    arr_fmin_max[0] = f_min
    arr_fmin_max[1] = f_max
    np.save('..\\fmin_max.npy',arr_fmin_max)

    iarrDataType_maxDT = np.zeros(2,dtype = np.int32)
    iarrDataType_maxDT[0] = iDataType
    iarrDataType_maxDT[1] = maxDT
    np.save('..\\iarrDataType_maxDT.npy',iarrDataType_maxDT)


    
   
    
    
    return XX


def CoherentDedispersion(raw_signal,d, f_min, f_max, alreadyFFTed = False):
    """
    Will perform coherent dedispersion.
    raw signal   - is assumed to be a one domensional signal
    d            - is the dispersion measure. units: pc*cm^-3
    f_min        - the minimum freq, given in Mhz
    f_max        - the maximum freq, given in Mhz
    alreadyFFTed - to reduce complexity, insert fft(raw_signal) instead of raw_signal, and indicate by this flag
    
    For future improvements:
    1) Signal partition to chunks of length N_d is not applied, and maybe it should be.
    2) No use of packing is done, though it is obvious it should be done (either in the coherent stage (and take special care of the abs()**2 operation done by other functions) or in the incoherent stage)
    
    """
    N_total = len(raw_signal)
    practicalD = DispersionConstant * d
    f = np.arange(0,f_max-f_min, float(f_max-f_min)/N_total)
    
    # The added linear term makes the arrival times of the highest frequencies be 0
    H = np.e**(-(2*np.pi*complex(0,1) * practicalD /(f_min + f) + 2*np.pi*complex(0,1) * practicalD*f /(f_max**2)))
    #H = np.e**(-(2*np.pi*complex(0,1) * practicalD /(f_min + f) + 2*np.pi*complex(0,1) * practicalD*f /(f_min**2)))
    if not alreadyFFTed:
        CD = np.fft.ifft(np.fft.fft(raw_signal) * H)
    else :
        CD = np.fft.ifft(raw_signal * H)    
    return CD


   
#This is the standard dispersion constant, with units that fit coherent dedispersion
DispersionConstant = 4.148808*10**9; ## Mhz * pc^-1 * cm^3

print(1)

ii =0               
inp = FDMT_test_curve()



