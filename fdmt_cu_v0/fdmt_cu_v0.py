#%pip install numpy 
import numpy as np
import time
import sys
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

from numba import jit, njit
import math
from numba import cuda
import cupy as cp

def get_total_threads_limit():
    device = cuda.get_current_device()
    max_threads_per_block = device.MAX_THREADS_PER_BLOCK
    num_sms = device.MULTIPROCESSOR_COUNT
    total_threads = max_threads_per_block * num_sms
    return total_threads

total_threads = get_total_threads_limit()
print("Total threads limit:", total_threads)
device = cuda.get_current_device()
CONST_NUM_SMS = device.MULTIPROCESSOR_COUNT

# Dimensions
d = 3 
m = 4
p = 2
b = 378 # number of frequencies in a sparse grid, due to some clever integration trick. 
i = 100 #5000
e = 10 #10**3
f = 100



# Generate random complex-valued CuPy arrays on the GPU
w_dmpf = cp.random.uniform(0.001, 1,size=(d, m, p, f)).astype(cp.complex64)
h_impf = cp.random.uniform(0.001, 1,size=(i, m, p, f)).astype(cp.complex64)
F_dpe = cp.random.uniform(0.001, 1,size=(d, p, e)).astype(cp.complex64)
T_dfe = cp.random.uniform(0.001, 1,size=(d, f, e)).astype(cp.complex64)
D_d = cp.random.uniform(0.001, 1,size=(d,)).astype(cp.complex64)

# Perform the equivalent operations using CuPy functions
conj_h_impf = cp.conj(h_impf)
conj_T_dfe = cp.conj(T_dfe)



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
    dataType = 'int64'
    FUNC = FDMT
    
        
    Verbose = False
    
    
    practicalD = DispersionConstant * D
    I = np.random.normal(0,1,N_total)
    I[PulsePosition:PulsePosition+PulseLength] += np.random.normal(0,PulseSig,PulseLength)
    print ("MAX Thoretical SNR:", np.sum(np.abs(I[PulsePosition:PulsePosition+PulseLength])**2 - np.mean(abs(I)**2)) / (np.sqrt(PulseLength*np.var(abs(I)**2))))
    
    X = CoherentDedispersion(I, -D, f_min,f_max,False)    
    #F = np.fft.fft(I)/np.sqrt(len(I))
    #f = np.arange(0,f_max-f_min, float(f_max-f_min)/N_total)
    #PDB("f_shape",f.shape)
    #F*= np.e**(2*np.pi*complex(0,1) * practicalD /(f_min + f) )
    #X = np.fft.ifft(F) # The input raw voltage signal...
    
    
   
    XX = np.abs(np.fft.fft(X.reshape([N_f,N_t*N_bins]),axis = 1))**2
    
    XX = np.transpose(XX)
    XX = np.sum(XX.reshape(N_f,N_bins,N_t),axis=1)
    
    E = np.mean(XX[:,:10])
    XX -= E
    V = np.var(XX[:,:10])
    
    XX /= (0.25*np.sqrt(V))
    V = np.var(XX[:,:10])
    
    #G0 = cart.cview(XX)
    DM0 = np.real(FUNC(np.ones(XX.shape,dataType),f_min,f_max,maxDT,dataType,Verbose))
    
    
    
    DM = np.real(FUNC(XX, f_min, f_max, maxDT, dataType, Verbose))
    Res = DM/np.sqrt((DM0+0.000001) * V )
    #G = cart.cview(Res)
    print("Maximum acieved SNR:", np.max(Res))
    print ("Maximum Position:", argmaxnd(Res))
    return XX,Res,DM0


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



def FDMT_initialization(Image,f_min,f_max,maxDT,dataType):
    """
    Input: Image - power matrix I(f,t)
        f_min,f_max - are the base-band begin and end frequencies.
            The frequencies can be entered in both MHz and GHz, units are factored out in all uses.
        maxDT - the maximal delay (in time bins) of the maximal dispersion.
            Appears in the paper as N_{\Delta}
            A typical input is maxDT = N_f
        dataType - To naively use FFT, one must use floating point types.
            Due to casting, use either complex64 or complex128.
    Output: 3d array, with dimensions [N_f,N_d0,Nt]
            where N_d0 is the maximal number of bins the dispersion curve travels at one frequency bin
    
    For details, see algorithm 1 in Zackay & Ofek (2014)
    """
    # Data initialization is done prior to the first FDMT iteration
    # See Equations 17 and 19 in Zackay & Ofek (2014)

    [F,T] = Image.shape

    deltaF = (f_max - f_min)/float(F)
    deltaT = int(np.ceil((maxDT-1) *(1./f_min**2 - 1./(f_min + deltaF)**2) / (1./f_min**2 - 1./f_max**2)))

    Output = np.zeros([F,deltaT+1,T],dtype=dataType)
    Output[:,0,:] = Image

    # !!!
    AA = Output[0,:,:]
    AA0 = Output[0,0,:]
    AA1 = Output[0,1,:]
    AA2 = Output[0,2,:]
    # !!!
    

    for i_dT in range(1,deltaT+1):
        Output[:,i_dT,i_dT:] = Output[:,i_dT-1,i_dT:] + Image[:,:-i_dT]
    return Output


def argmaxnd(ar):
    return np.unravel_index(np.argmax(ar),ar.shape)
def FDMT(Image, f_min, f_max,maxDT ,dataType, Verbose = True):
    """
    This function implements the  FDMT algorithm.
    Input: Input power matrix I(f,t)
           f_min,f_max are the base-band begin and end frequencies.
                   The frequencies should be entered in MHz 
           maxDT - the maximal delay (in time bins) of the maximal dispersion.
                   Appears in the paper as N_{\Delta}
                   A typical input is maxDT = N_f
           dataType - a valid numpy dtype.
                      reccomended: either int32, or int64.
    Output: The dispersion measure transform of the Input matrix.
            The output dimensions are [Input.shape[1],maxDT]
    
    For details, see algorithm 1 in Zackay & Ofek (2014)
    """
    F,T = Image.shape
    
    if (F not in [2**i for i in range(1,30)]) or (T not in [2**i for i in range(1,30)]) :
        raise NotImplementedError("Input dimensions must be a power of 2")

    x = time.time()
    State0 = FDMT_initialization(Image,f_min,f_max,maxDT,dataType)
    #PDB('initialization ended')
    State_fdmt0 = State0.copy()
    f = int(log2(F))

    
    
    
    # 1
    State_fdmt0 = solver_fdmt(State_fdmt0,maxDT,F,f_min,f_max,dataType, Verbose)    

    [Fs,dTs,Ts] = State_fdmt0.shape
    DMT0 = np.reshape(State_fdmt0,[dTs,Ts])
   
    print('original code:\n')
    
    
    
    
    State_fdmt_cu_v5 = State0
    State_fdmt_cu_v5 = solver_fdmt_cu_v5(State_fdmt_cu_v5,maxDT,F,f_min,f_max,dataType, Verbose)
    print('cu_v5:') 
    
    [Fs1,dTs1,Ts1] = State_fdmt_cu_v5.shape
    DMT_cu_v5= np.reshape(State_fdmt_cu_v5,[dTs1,Ts1])
   
    return DMT_cu_v5
    
    
  
def solver_fdmt(State,maxDT,F,f_min,f_max,dataType, Verbose):
    output = State
    f = int(log2(F))
    for i_t in range(1,f+1):
        output = FDMT_iteration(output,maxDT,F,f_min,f_max,i_t,dataType, Verbose)
    return output

def FDMT_iteration(Input,maxDT,F,f_min,f_max,iteration_num,dataType, Verbose = False):
    """
        Input: 
            Input - 3d array, with dimensions [N_f,N_d0,Nt]
            f_min,f_max - are the base-band begin and end frequencies.
                The frequencies can be entered in both MHz and GHz, units are factored out in all uses.
            maxDT - the maximal delay (in time bins) of the maximal dispersion.
                Appears in the paper as N_{\Delta}
                A typical input is maxDT = N_f
            dataType - To naively use FFT, one must use floating point types.
                Due to casting, use either complex64 or complex128.
            iteration num - Algorithm works in log2(Nf) iterations, each iteration changes all the sizes (like in FFT)
        Output: 
            3d array, with dimensions [N_f/2,N_d1,Nt]
        where N_d1 is the maximal number of bins the dispersion curve travels at one output frequency band
        
        For details, see algorithm 1 in Zackay & Ofek (2014)
    """

    input_dims = Input.shape;
    output_dims = list(input_dims);
    
    deltaF = 2**(iteration_num) * (f_max - f_min)/float(F);
    dF = (f_max - f_min)/float(F)
    # the maximum deltaT needed to calculate at the i'th iteration
    deltaT = int(np.ceil((maxDT-1) *(1./f_min**2 - 1./(f_min + deltaF)**2) / (1./f_min**2 - 1./f_max**2)));
    #PDB("deltaT = ",deltaT)
    #PDB("N_f = ",F/2.**(iteration_num))
    #PDB('input_dims', input_dims)
    
    output_dims[0] = output_dims[0]//2;
    
    
    output_dims[1] = deltaT + 1;
    #PDB('output_dims', output_dims)
    #print(output_dims, dataType)
    Output = np.zeros(output_dims,dtype=dataType)
    
    
    # No negative D's are calculated => no shift is needed
    # If you want negative dispersions, this will have to change to 1+deltaT,1+deltaTOld
    # Might want to calculate negative dispersions when using coherent dedispersion, to reduce the number of trial dispersions by a factor of 2 (reducing the complexity of the coherent part of the hybrid)
    ShiftOutput = 0
    ShiftInput = 0
    T = output_dims[2]
    F_jumps = output_dims[0]
    
    # For some situations, it is beneficial to play with this correction.
    # When applied to real data, one should carefully analyze and understand the effect of 
    # this correction on the pulse he is looking for (especially if convolving with a specific pulse profile)
    if iteration_num>0:
        correction = dF/2.
    else:
        correction = 0
    
    temp1 = (1./f_min**2 - 1./f_max**2)
    for i_F in range(F_jumps):
        
        f_start = (f_max - f_min)/float(F_jumps) * (i_F) + f_min
        f_end = (f_max - f_min)/float(F_jumps) *(i_F+1) + f_min
        f_middle = (f_end - f_start)/2. + f_start - correction
        # it turned out in the end, that putting the correction +dF to f_middle_larger (or -dF/2 to f_middle, and +dF/2 to f_middle larger)
        # is less sensitive than doing nothing when dedispersing a coherently dispersed pulse.
        # The confusing part is that the hitting efficiency is better with the corrections (!?!).
        f_middle_larger = (f_end - f_start)/2 + f_start + correction
        temp0 = (1./f_start**2 - 1./(f_end)**2)
        
        val0 = -(1./f_middle**2 - 1./f_start**2)/temp0
        val1 = -(1./f_middle_larger**2 - 1./f_start**2)/temp0
        deltaTLocal = int(np.ceil((maxDT-1) *temp0/ temp1 ))
        
        for i_dT in range(deltaTLocal+1):      
            dT_middle_index = round(i_dT * val0)
            
            
            dT_middle_larger = round(i_dT * val1)            
                     
            
            dT_rest_index = i_dT - dT_middle_larger
            
            
            i_T_min0 = 0
            
            i_T_max0 = dT_middle_larger
            #Output[i_F,i_dT + ShiftOutput,i_T_min0:i_T_max0] = Input[2*i_F, dT_middle_index,i_T_min0:i_T_max0]
            Output[i_F,i_dT ,:dT_middle_larger ] = Input[2 * i_F,dT_middle_index,:dT_middle_larger ]

            i_T_min = dT_middle_larger
            i_T_max = T         
            
            
            #Output[i_F,i_dT + ShiftOutput,i_T_min:i_T_max] = Input[2*i_F, dT_middle_index,i_T_min:i_T_max] + Input[2*i_F+1, dT_rest_index,i_T_min - dT_middle_larger:i_T_max-dT_middle_larger]
            Output[i_F,i_dT ,dT_middle_larger: ] = Input[2 * i_F,dT_middle_index,dT_middle_larger: ] + Input[2 * i_F + 1,dT_rest_index, :i_T_max - dT_middle_larger]
            c =1

            
            
    return Output   
    


def solver_fdmt_cu_v5(State,maxDT,F,f_min,f_max,dataType, Verbose):
    sss=1
    d_input = cuda.to_device(State)    
    f = int(log2(F))
    dF = (f_max - f_min)/float(F)
    for i_t in range(1,f+1):
        output_dims = d_input.shape 
        deltaF = 2**(i_t) * (f_max - f_min)/float(F)
        
        # the maximum deltaT needed to calculate at the i'th iteration
        deltaT = int(math.ceil((maxDT-1) *(1./f_min**2 - 1./(f_min + deltaF)**2) / (1./f_min**2 - 1./f_max**2)))
        
        
        out_dim0 = output_dims[0]//2     
        out_dim1 = deltaT + 1
        out_dim2 = output_dims[2]
        
        d_Output = cuda.device_array((out_dim0, out_dim1,out_dim2),dtype= d_input.dtype)       
        
        dFdiv2 = (f_max - f_min)/float(F)/2.0
        
        output = FDMT_iteration_cu5(d_input,maxDT,F,f_min,f_max,i_t, dFdiv2,d_Output)

        d_input = d_Output
        

        d_Output = None
              
    h_out = d_input.copy_to_host()
        
    return h_out

def FDMT_iteration_cu5(d_input,maxDT,F,f_min,f_max,i_t, dFdiv2,d_Output):
    """
        Input: 
            Input - 3d array, with dimensions [N_f,N_d0,Nt]
            f_min,f_max - are the base-band begin and end frequencies.
                The frequencies can be entered in both MHz and GHz, units are factored out in all uses.
            maxDT - the maximal delay (in time bins) of the maximal dispersion.
                Appears in the paper as N_{\Delta}
                A typical input is maxDT = N_f
            dataType - To naively use FFT, one must use floating point types.
                Due to casting, use either complex64 or complex128.
            iteration num - Algorithm works in log2(Nf) iterations, each iteration changes all the sizes (like in FFT)
        Output: 
            3d array, with dimensions [N_f/2,N_d1,Nt]
        where N_d1 is the maximal number of bins the dispersion curve travels at one output frequency band
        
        For details, see algorithm 1 in Zackay & Ofek (2014)
    """
    F_jumps = d_Output.shape[0]    
    correction = 0

    if i_t>0:
        correction = dFdiv2

    
    
    #deltaF = 2**(iteration_num) * (f_max - f_min)/float(F)
    #dF = (f_max - f_min)/float(F)
    
    
   
   
    # in this circle I calculate 3 arrays arr_val0, arr_val1 and arr_deltaTLocal for each i_f
    arr_val0 = np.zeros(F_jumps, np.float32)
    arr_val1 = np.zeros(F_jumps, np.float32)
    arr_deltaTLocal = np.zeros(F_jumps, np.int64) 
    

    # calc 3 massives of parametres
    temp1 = (1./f_min**2 - 1./f_max**2)
    for i_F in range(F_jumps):        
        f_start = (f_max - f_min)/float(F_jumps) * (i_F) + f_min
        f_end = (f_max - f_min)/float(F_jumps) *(i_F+1) + f_min
        f_middle = (f_end - f_start)/2. + f_start - correction
        # it turned out in the end, that putting the correction +dF to f_middle_larger (or -dF/2 to f_middle, and +dF/2 to f_middle larger)
        # is less sensitive than doing nothing when dedispersing a coherently dispersed pulse.
        # The confusing part is that the hitting efficiency is better with the corrections (!?!).
        f_middle_larger = (f_end - f_start)/2 + f_start + correction
        temp0 = (1./f_start**2 - 1./(f_end)**2)     
        arr_val0[i_F] = -(1./f_middle**2 - 1./f_start**2)/temp0
        arr_val1[i_F] = -(1./f_middle_larger**2 - 1./f_start**2)/temp0
        arr_deltaTLocal[i_F] = int(np.ceil((maxDT-1) *temp0/ temp1 ))
    # ! calc 3 massives of parametres
    #print('1) F_jumps =', F_jumps)
    d_arr_val0 = cuda.to_device(arr_val0)   
    d_arr_val1 = cuda.to_device(arr_val1) 
    d_arr_deltaTLocal = cuda.to_device(arr_deltaTLocal)   
    #print('2) F_jumps =', F_jumps)
    ###############################################################
    
    d_arr_dT_MI = cuda.device_array((F_jumps,d_Output.shape[1]),dtype= d_input.dtype)
    d_arr_dT_ML = cuda.device_array((F_jumps,d_Output.shape[1]),dtype= d_input.dtype)
    d_arr_dT_RI = cuda.device_array((F_jumps,d_Output.shape[1]),dtype= d_input.dtype)  

    #calculation 3 2 dimensional massives dT_middle_index, dT_middle_larger, dT_rest_index 
    # for each i_F and i_d
    threads = d_Output.shape[1] #device.MAX_THREADS_PER_BLOCK
    blocks  = F_jumps 
    kernel_5_0[blocks, threads](d_arr_val0,d_arr_val1,d_arr_deltaTLocal,d_arr_dT_MI,d_arr_dT_ML, d_arr_dT_RI)  
    cuda.synchronize()

    '''arrMI = d_arr_dT_MI.copy_to_host()
    arr_dT_ML = d_arr_dT_MI.copy_to_host()
    arr_dT_RI = d_arr_dT_RI.copy_to_host()'''
    d_arr_val0 = None
    d_arr_val1 = None 
    
    #################################################################
    threads = d_Output.shape[2]
    blocks  = (F_jumps , d_Output.shape[1])
    
    kernel_5_1[blocks, threads](d_input,d_arr_deltaTLocal,d_arr_dT_MI,d_arr_dT_ML, d_arr_dT_RI, d_Output)
    cuda.synchronize()  
    d_arr_dT_MI = None
    d_arr_dT_ML = None
    d_arr_dT_RI = None 
    d_arr_deltaTLocal = None      
    return d_Output
    
@cuda.jit
def kernel_5_0(arr_val0,arr_val1,arr_deltaTLocal,arr_dT_middle_index,arr_dT_middle_larger, arr_dT_rest_index):
    if cuda.threadIdx.x > (arr_deltaTLocal[cuda.blockIdx.x] ):
        return
    arr_dT_middle_index[cuda.blockIdx.x,cuda.threadIdx.x ] = round(cuda.threadIdx.x * arr_val0[cuda.blockIdx.x])       
    arr_dT_middle_larger[cuda.blockIdx.x,cuda.threadIdx.x ] = round(cuda.threadIdx.x * arr_val1[cuda.blockIdx.x])       
    arr_dT_rest_index[cuda.blockIdx.x,cuda.threadIdx.x ] = cuda.threadIdx.x - arr_dT_middle_larger[cuda.blockIdx.x,cuda.threadIdx.x ] 
    
@cuda.jit
def kernel_5_1(d_input,arr_deltaTLocal,arr_dT_MI,arr_dT_ML, arr_dT_RI, d_Output):
    
    i_F = cuda.blockIdx.x
    i_dT = cuda.blockIdx.y
    if i_dT > arr_deltaTLocal[i_F]:
        return
    
    idx = cuda.threadIdx.x     
    
    
    if idx <arr_dT_ML[i_F, i_dT]:
        d_Output[i_F][i_dT][idx ] = d_input[2 * i_F][arr_dT_MI[i_F, i_dT]][idx]
    else:
        d_Output[i_F][i_dT][idx ] = d_input[2 * i_F][arr_dT_MI[i_F, i_dT]][idx] + d_input[2 * i_F + 1][arr_dT_RI[i_F, i_dT]][ idx - arr_dT_ML[i_F, i_dT]]    
   
    
#This is the standard dispersion constant, with units that fit coherent dedispersion
DispersionConstant = 4.148808*10**9; ## Mhz * pc^-1 * cm^3
Verbose = False
print(1)

ii =0               
inp, out, out1 = FDMT_test_curve()

ii =0  
plt.figure()
plt.imshow(out)
plt.show() 

plt.imshow(inp)
plt.show()

plt.imshow(out1)
plt.show()

b =1

