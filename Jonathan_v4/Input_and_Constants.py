import numpy as np


# 1.choice type of calculation between 0 and 1
# if NUM_TYPE_OF_CALCULATION == 0 then we will use stadart cupy funsc only,
# owtherwise customized kernel
NUM_TYPE_OF_CALCULATION = 0

# 2. choice type of DATA_TYPE, if only TYPE_OF_PROGRAM == STADART_CUPY
# keep in mind that if TYPE_OF_PROGRAM == WITH_CUSTOMIZED_KERNELS_CUPY so DATA_TYPE = np.float32
# to choice type of data type (global const var) insert here below 0 for np.float16, 1 for np.float32,
# 2 for np.float64:
NUM_OF_DATA_TYPE = 1 
  
# Do not touch this code:
if NUM_TYPE_OF_CALCULATION == 1:
    NUM_OF_DATA_TYPE = 1

#global const array
ARR_TYPE_OF_PROGRAM =['STADART_CUPY', 'WITH_CUSTOMIZED_KERNELS_CUPY']
TYPE_OF_PROGRAM = ARR_TYPE_OF_PROGRAM[NUM_TYPE_OF_CALCULATION]

#global const array
ARR_OF_DATA_TYPE = [np.float16,np.float32,np.float64]

# define global const var DATA_TYPE:
DATA_TYPE = ARR_OF_DATA_TYPE[NUM_OF_DATA_TYPE]

name_types = ['lnl_approx_array_create_<float>', 'lnl_approx_array_create_<double>']
numType = 1

# define global const var DATA_COMPLEX_TYPE:
DATA_COMPLEX_TYPE = np.complex128
if DATA_TYPE == np.float16 or DATA_TYPE == np.float32 :
    DATA_COMPLEX_TYPE = np.complex64
else:
    DATA_COMPLEX_TYPE = np.complex128
    




