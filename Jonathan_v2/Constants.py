import numpy as np
dataType = np.float32

name_types = ['lnl_approx_array_create_<__fp16>','lnl_approx_array_create_<float>', 'lnl_approx_array_create_<double>']
numType = 1
if dataType == np.float32:    
    numType = 1

if dataType == np.float64:    
    numType = 2

if dataType == np.float16:    
    numType = 0



# constans for the lookup table:
REFERENCE_DISTANCE = 1.  # Luminosity distance at which h is defined (Mpc).
_Z0 = 10.  # Typical SNR of events.
_SIGMAS = 10.  # How far out the tail of the distribution to tabulate.
d_luminosity_max = 1.5e4 # maximum luminosity distance in Mpc
shape = (256, 128) # shape of the table
MIN_Z_FOR_LNL_DIST_MARG_APPROX = 4
