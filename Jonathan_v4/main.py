
import numpy as np
import sys
import itertools
from evidence_funcs_cu_v2 import *
import cupyx.scipy.interpolate as interp
from cupyx.scipy.interpolate import RegularGridInterpolator
import cupy as cp
import numba
from numba import cuda
from Input_and_Constants import *
import datetime





def get_total_threads_limit():
    device = cuda.get_current_device()
    max_threads_per_block = device.MAX_THREADS_PER_BLOCK
    num_sms = device.MULTIPROCESSOR_COUNT
    total_threads = max_threads_per_block * num_sms
    return total_threads


device = cuda.get_current_device()
print('GPU NAME = ',device.name)

print('Multiprocessors count = ',device.MULTIPROCESSOR_COUNT)
print('MAX_THREADS_PER_BLOCK = ',device.MAX_THREADS_PER_BLOCK)
total_threads = get_total_threads_limit()
print("Total threads limit:", total_threads)

m_arr = np.array([2,1,3,4])
n_phi = 100
lookup_table=None

# Dimensions
d = 3 
m = 4
p = 2
b = 378 # number of frequencies in a sparse grid, due to some clever integration trick. 
i = 1000 #5000
e = 10**3
mm = 10        
# Used for terms with 2 modes indices

dh_weights_dmpb = np.random.uniform(0.001, 1, (d, m, p, b)).astype(DATA_COMPLEX_TYPE)
h_impb = np.random.uniform(0.001, 1, (i, m, p, b)).astype(DATA_COMPLEX_TYPE)
response_dpe = np.random.uniform(0.001, 1, (d, p, e)).astype(DATA_TYPE)
timeshift_dbe = np.random.uniform(0.001, 1, (d, b, e)).astype(DATA_COMPLEX_TYPE)
hh_weights_dmppb = np.random.uniform(0.001, 1, (d, mm, p, p, b)).astype(DATA_COMPLEX_TYPE) 
asd_drift_d = np.random.uniform(0.001, 0.1, (d)).astype(DATA_TYPE)
log_weights_i = np.random.uniform(0.001, 1, (i)).astype(DATA_TYPE)
log_weights_e = np.random.uniform(0.001, 1, (e)).astype(DATA_TYPE)
debug_mode = False

n_samples = i * e * n_phi

approx_dist_marg_lnl_drop_threshold = 20.
frac_evidence_threshold =1e-3
debug_mode = False

m_inds, mprime_inds = zip(
            *itertools.combinations_with_replacement(
            range(len(m_arr)), 2))



# Creation of interpolation table && interpolator
x_arr, y_arr, table = createInterpolationTable()
d_table = cp.array(table)
d_x_arr = cp.array(x_arr)
d_y_arr = cp.array(y_arr)
fncInterpolator = RegularGridInterpolator((d_x_arr, d_y_arr), d_table,
                                bounds_error=False, fill_value=None)





current_datetime = datetime.datetime.now()

current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

print('------------------------------------------------------------------------')
print('------------------------------------------------------------------------')
print('--------   Profiling report---------------------------------------------')
print('--------   Data, time:   ',current_datetime_str,'-----------------------')
print('------------------------------------------------------------------------')


(dist_marg_lnlike_k, ln_evidencedist_marg_lnlike_k, 
 inds_i_k, inds_e_k, inds_o_k) = calculate_lnlike_and_evidence_cupy(n_phi,m_arr, m_inds,mprime_inds,dh_weights_dmpb,h_impb,response_dpe, timeshift_dbe,                          
                                hh_weights_dmppb,
                                asd_drift_d, 
                                log_weights_i,
                                log_weights_e, 
                                fncInterpolator
                                ,approx_dist_marg_lnl_drop_threshold, 
                                n_samples, 
                                frac_evidence_threshold, 
                                debug_mode
                                ,bprofile = True)


#from evidence_class_11_sept import*
#ev = Evidence(n_phi=100, m_arr=np.array([2,1,3,4]))
## load npz files 
#inputs_from_file = dict(np.load('test_input.npz'))
#outputs_from_file = dict(np.load('test_output.npz'))
#output_tuple = ev.calculate_lnlike_and_evidence(**inputs_from_file)

## keep track of the outputs of the function
#evidence_func_return_kwargs = [
#    'dist_marg_lnlike_k', 'ln_evidence',
#    'inds_i_k', 'inds_e_k', 'inds_o_k',
#    'dh_k', 'hh_k']
## arrange them in a dictionary
#output_dict = {k: v for k, v in zip(evidence_func_return_kwargs, output_tuple)}
## compare output from file and output from code
#for k,v in output_dict.items():
#    print(k +' ' +  str(np.allclose(outputs_from_file[k], v)) , )

