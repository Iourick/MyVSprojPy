from scipy.signal import convolve
import numpy as np
import time
import sys
import struct
import array

def createRawFileImitation():
    # 0
    fileName = 'rawImit_2pow20.bin'
    nchan = 32
    npol = 4
    f_min_ = 1200.
    f_max_ = 1600.
    quantSamp =2**20
    Sig0_ = np.random.normal(0,1,quantSamp ).astype(np.float32)
    Sig0_[1000:1128] = np.random.normal(0,10.,128)
    
    Sig1_ = Sig0*np.exp(-1j)
    
    Sig0 = CoherentDedispersion(Sig0_,-1,1200,1600,False)
    Sig1 = CoherentDedispersion(Sig1_,-1,1200,1600,False)
    t = np.arange(0, 1,1/quantSamp) 
    size = nchan*npol*quantSamp*4
    binary_data = struct.pack('iiiff', nchan,npol, size, f_min_, f_max_)
    with open(fileName, 'wb') as file:
        file.write(binary_data)
    for i in range (nchan):
        f0 = f_min_ + (f_max_ - f_min_)/nchan*i
        f1 = f0 + (f_max_ - f_min_)/nchan
        sinc_func = np.sinc((t - t.mean()) * (f1 - f0))
        sinc_func /= np.sum(sinc_func)
        f_filtered0 = convolve(Sig0, sinc_func, mode='same')
        f_filtered1 = convolve(Sig1, sinc_func, mode='same')
        with open(fileName, 'ab') as file:
            for j in range (quantSamp):
                file.write(struct.pack('ffff', f_filtered0[j].real, f_filtered0[j].imag, f_filtered1[j].real, f_filtered1[j].imag))
        
 
                
createRawFileImitation() 