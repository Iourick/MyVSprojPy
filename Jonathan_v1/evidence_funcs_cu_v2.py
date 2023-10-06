
import numpy as np
import cupy as cp
from scipy.special import logsumexp
from typing import Tuple, Optional
import time
import cupyx.scipy.interpolate as interp
from scipy.integrate import quad
from pathlib import Path
from cupyx.scipy.interpolate import RegularGridInterpolator

from Constants import *
# The marginalization over distance for the likelihood assumes that 
# the maximal-likelihood d_luminosity (=sqrt(<h|h>)/<d|h>) smaller than 
# the maximal allowed distance by more than the width of the Gaussian.
# This translated to z > h_norm/(15e3) + (a few)
# We approximate the RHS of this inequality with 4


    
from Constants import _SIGMAS,_Z0

# constans for the lookup table:


# Define the functions to mimic the lookup-table class
 

def _compactify(value):
    """Monotonic function from (-inf, inf) to (-1, 1)."""
    return value / (1 + np.abs(value))
#--------------------------------------------------------------------

def _uncompactify(value):
    """
    Inverse of _compactify. Monotonic function from (-1, 1) to
    (-inf, inf).
    """
    return value / (1 - np.abs(value))
#--------------------------------------------------------------------

def dh_hh_to_x_y(d_h, h_h):
    norm_h = np.sqrt(h_h)
    overlap = d_h / norm_h
    numerator = norm_h
    
    denominator = d_luminosity_max * (_SIGMAS + np.abs(overlap))
    x = np.log(numerator / denominator)
    y =_compactify(overlap / _Z0)
    return x, y
#--------------------------------------------------------------------
def x_y_to_dh_hh(x, y):
    """
    Inner products (d|h), (h|h), as a function of the interpolation
    coordinates (x, y) in which the function to interpolate is
    smooth.
    Inverse of ``_get_x_y``.
    """
    overlap = _uncompactify(y) * _Z0
    norm_h = (np.exp(x) * d_luminosity_max
                * (_SIGMAS + np.abs(overlap)))
    d_h = overlap * norm_h
    h_h = norm_h**2
    return d_h, h_h
#--------------------------------------------------------------------
def _function_integrand(d_luminosity, d_h, h_h):
    """
    Proportional to the distance posterior. The log of the integral
    of this function is stored in the lookup table.
    """
    norm_h = np.sqrt(h_h)
    prior = 3 * d_luminosity**2 / d_luminosity_max**3
    exp_arg = -(norm_h * REFERENCE_DISTANCE / d_luminosity
                        - d_h / norm_h)**2 / 2
    return (prior * np.exp(exp_arg))
#--------------------------------------------------------------------


def _get_distance_bounds(d_h, h_h, sigmas=5.):
    """
    Return ``(d_min, d_max)`` pair of luminosity distance bounds to
    the distribution at the ``sigmas`` level.
    Let ``u = REFERENCE_DISTANCE / d_luminosity``, the likelihood is
    Gaussian in ``u``. This function returns the luminosity
    distances corresponding to ``u`` +/- `sigmas` deviations away
    from the maximum.
    Note: this can return negative values for the distance. This
    behavior is intentional. These may be interpreted as infinity.
    """
    u_peak = d_h / (REFERENCE_DISTANCE * h_h)
    delta_u = sigmas / np.sqrt(h_h)
    return np.array([REFERENCE_DISTANCE / (u_peak + delta_u),
                        REFERENCE_DISTANCE / (u_peak - delta_u)])
#--------------------------------------------------------------------


def _function(d_h, h_h):
    """
    Function to interpolate with the aid of a lookup table.
    Return ``log(evidence) - overlap**2 / 2``, where ``evidence``
    is the value of the likelihood marginalized over distance.
    Add a small number (1e-100) to avoid taking the log of zero.
    """
    return np.log(quad(_function_integrand, 0., d_luminosity_max,
                        args=(d_h, h_h))[0] + 1e-100)

#----------------------------------------------------------------
def _function_old(d_h, h_h):
    """
    Function to interpolate with the aid of a lookup table.
    Return ``log(evidence) - overlap**2 / 2``, where ``evidence``
    is the value of the likelihood marginalized over distance.
    Add a small number (1e-100) to avoid taking the log of zero.
    """
    
    return np.log(quad(_function_integrand, 0, d_luminosity_max,
                        args=(d_h, h_h),
                        points=_get_distance_bounds(d_h, h_h)
                        )[0]
                    + 1e-100)



#--------------------------------------------------------------------
def calculateTable():
     
    x_arr = np.linspace(-_SIGMAS, 0, shape[0])
    y_arr = np.linspace(_compactify(- _SIGMAS / _Z0),
                        1 - 1e-8, shape[1])
    x_grid,   y_grid = np.meshgrid(  x_arr,   y_arr, indexing='ij')

    dh_grid,   hh_grid = x_y_to_dh_hh(  x_grid,   y_grid)

     #table = np.vectorize(_function)(dh_grid, hh_grid)
    table = np.vectorize(_function)(dh_grid, hh_grid)
     
     
    return x_arr, y_arr, table

#--------------------------------------------------------------------
def calculateTable_old():
     x_arr = np.linspace(-_SIGMAS, 0, shape[0])
     y_arr = np.linspace(_compactify(- _SIGMAS / _Z0),1 - 1e-8, shape[1])
     x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing='ij')
     dh_grid, hh_grid = x_y_to_dh_hh(x_grid, y_grid)
     table = np.vectorize(_function_old)(dh_grid, hh_grid)
     
     return x_arr, y_arr, table
#--------------------------------------------------------------------   
     
     
def createInterpolationTable():
        """
        Attempt to load a previously computed table with the requested
        settings. If this is not possible, compute the table and save it
        for faster access in the future.
        """
        
        if Path('lookup_table.npz').exists():
            loaded_arrays = np.load('lookup_table.npz', allow_pickle = True)
            x_arr = loaded_arrays['x_arr']
            y_arr = loaded_arrays['y_arr']
            table =  loaded_arrays['table']
            if table.shape == shape:
                return x_arr, y_arr, table
            else:
                print('Table shape mismatch. Recomputing.')
        print('Computing table.')
        x_arr, y_arr, table = calculateTable()
        np.savez('lookup_table.npz',x_arr = x_arr, y_arr = y_arr, table = table)

        return x_arr, y_arr, table
#--------------------------------------------------------------------

    
def get_hh_by_mode(h_impb,
                    response_dpe,
                    hh_weights_dmppb, 
                    asd_drift_d, 
                    m_inds, 
                    mprime_inds):
    """
    Calculate the inner priducts <h|h> for each combination of
    intrinsic sample, extrinsic sample and modes-pair.
    Modes m mean unique modes combinations: (2,2), (2,0), ..., (3,3)
    (over all 10 combinations). 
    Inputs:
    h_impb: array of waveforms.
    response_dpe: array of detector response.
    hh_weights_dmppb: array of weights for detector. Assumes the
    weights are calculated for integrand h[m]*h[mprime].conj()
    asd_drift_d: array of ASD drifts.
    m_inds: tuple. indices of modes.
    mprime_inds: tuple. indices of modes.
    Output:
    hh_iem: array of inner products.
    """
    hh_idmpP = cp.einsum('dmpPb, impb, imPb -> idmpP',
                            hh_weights_dmppb,
                            h_impb[:, m_inds, ...],
                            h_impb.conj()[:, mprime_inds, ...],
                            optimize=True)  # idmpp
    ff_dppe = cp.einsum('dpe, dPe, d -> dpPe', 
                        response_dpe, response_dpe, asd_drift_d**-2,
                        optimize=True)  # dppe
    hh_iem = cp.einsum('idmpP, dpPe -> iem', 
                        hh_idmpP, ff_dppe, optimize=True)  # iem
    return hh_iem
#--------------------------------------------------------------------

def get_dh_hh_phi_grid(n_phi,m_arr,m_inds,dh_iem,hh_iem,  mprime_inds): 
    """
    change the orbital phase of each mode by exp(1j*phi*m), and keep
    the real part assumes that factor of two from off-diagonal modes
    (e.g. (2,0) and not e.g. (2,2)) is already in the 
    hh_weights_dmppb
    """
    
    
    d_phi_grid_ = cp.linspace(0, 2 * np.pi, n_phi, endpoint=False)  # o
    
    # dh_phasor is the phase shift applied to h[m].conj, hence the 
    # minus sign
    dh_phasor = cp.exp(-1j * cp.outer(m_arr, d_phi_grid_))  # mo
    # hh_phasor is applied both to h[m] and h[mprime].conj, hence 
    # the subtraction of the two modes
    
    
    hh_phasor = cp.exp(1j * cp.outer(
        m_arr[m_inds, ] - m_arr[mprime_inds, ],
        d_phi_grid_))  # mo

    # t1 = m_arr[m_inds, ] - m_arr[mprime_inds, ]
    # t2 = 1j * t1
    # hh_phasor = cp.exp(t2)
    
    dh_ieo = cp.einsum('iem, mo -> ieo', 
                        dh_iem, dh_phasor, optimize=True).real  # ieo
    hh_ieo = cp.einsum('iem, mo -> ieo', 
                        hh_iem, hh_phasor, optimize=True).real  # ieo
    
    return dh_ieo, hh_ieo
#--------------------------------------------------------------------
  
def select_ieo_by_approx_lnlike_dist_marginalized(
        dh_ieo, 
        hh_ieo,
        log_weights_i, 
        log_weights_e,
        cut_threshold: float = 20.
        ) :
    """
    Return three arrays with intrinsic, extrinsic and phi sample 
    indices.
    """
    h_norm = cp.sqrt(hh_ieo)
    z = dh_ieo / h_norm
    i_inds, e_inds, o_inds = cp.where(z > MIN_Z_FOR_LNL_DIST_MARG_APPROX)
    
    lnl_approx = (z[i_inds, e_inds, o_inds] ** 2 / 2
                    + 3 * cp.log(h_norm[i_inds, e_inds, o_inds])
                    - 4 * cp.log(z[i_inds, e_inds, o_inds])
                    + log_weights_i[i_inds] 
                    + log_weights_e[e_inds]) # 1d
    
    flattened_inds = cp.where(
        lnl_approx >= lnl_approx.max() - cut_threshold)[0]

    return (i_inds[flattened_inds],
            e_inds[flattened_inds],
            o_inds[flattened_inds])


#--------------------------------------------------------------------
loaded_from_source = r'''
extern "C"{

__global__ void lnl_approx_array_create(const double* pdh_ieo
                        , const double* phh_ieo
                        , const double* plog_weights_i
                        , const double* plog_weights_e                        
                        , const int i0
                        , const int i1
                        , const int i2
                        ,unsigned int N
                        ,const double MIN_Z_FOR_LNL_DIST_MARG_APPROX
                        , double* plnl_approx )
{
    
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int i_inds = tid/(i1 * i2);
    int irest = tid%(i1 * i2);

    int e_inds = irest/ i2;
    int o_inds = irest% i2;
    if (tid < N)
    {
        double val_h_norm = sqrt(phh_ieo[tid] ) ;  
        double z = pdh_ieo [tid]/ val_h_norm;
        if (z <= MIN_Z_FOR_LNL_DIST_MARG_APPROX)
        {
        plnl_approx [tid] = -1.8e10;
        }
        else
        {
        
        plnl_approx [tid] = z *z / 2.
                    + 3. * log(val_h_norm)
                    - 4. * log(z)
                    + plog_weights_i[i_inds] 
                    + plog_weights_e[e_inds];
        }
    }

}

__global__ void return_3d_indices(const unsigned int* pindices                                            
                        , const int N
                        , const int i0
                        , const int i1
                        , const int i2
                        , unsigned int* pi_inds // output
                        , unsigned int* pe_inds // output
                        , unsigned int* po_inds // output   
                         )
{
    
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (tid < N)
    {
       pi_inds[tid] = pindices[tid]/(i1 * i2);
       unsigned int irest = pindices[tid] % (i1 * i2); 
       pe_inds[tid] = irest/ i2;
       po_inds[tid] = irest% i2;
    }

}
}'''
#--------------------------------------------------------------------
code = r'''
template<typename T>
__global__ void lnl_approx_array_create_(const T* pdh_ieo
                        , const T* phh_ieo
                        , const T* plog_weights_i
                        , const T* plog_weights_e                        
                        , const int i0
                        , const int i1
                        , const int i2
                        ,unsigned int N
                        ,const float MIN_Z_FOR_LNL_DIST_MARG_APPROX
                        , T* plnl_approx )


 {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int i_inds = tid/(i1 * i2);
    int irest = tid%(i1 * i2);

    int e_inds = irest/ i2;
    int o_inds = irest% i2;
    if (tid < N)
    {
        double val_h_norm = sqrt(phh_ieo[tid] ) ;  
        double z = pdh_ieo [tid]/ val_h_norm;
        if (z <= MIN_Z_FOR_LNL_DIST_MARG_APPROX)
        {
        plnl_approx [tid] = -1.8e10;
        }
        else
        {        
        plnl_approx [tid] = z *z / 2.
                    + 3. * log(val_h_norm)
                    - 4. * log(z)
                    + plog_weights_i[i_inds] 
                    + plog_weights_e[e_inds];
         }
    }

  
}
'''



def select_ieo_by_approx_lnlike_dist_marginalized_new(
        dh_ieo, 
        hh_ieo,
        log_weights_i, 
        log_weights_e,
        cut_threshold: float = 20.
        ) :
    """
    Return three arrays with intrinsic, extrinsic and phi sample 
    indices.
    """
    
    #module = cp.RawModule(code=loaded_from_source)
    #lnl_approx_array_create = module.get_function('lnl_approx_array_create')

    N = dh_ieo.shape[0]* dh_ieo.shape[1]* dh_ieo.shape[2]
    lnl_approx = cp.zeros_like(dh_ieo)
    
    #lnl_approx = cp.zeros((N,), dtype = cp.float64)
    threads_per_block = 256
    blocks_per_grid_x = (N + threads_per_block - 1) // threads_per_block
   
    dh_ieo1 = dh_ieo.reshape(-1)
    hh_ieo1 = hh_ieo.reshape(-1)
    lnl_approx = cp.zeros_like(dh_ieo1)
    #lnl_approx_array_create((blocks_per_grid_x,), (threads_per_block,), (dh_ieo1, hh_ieo1, log_weights_i, log_weights_e,dh_ieo.shape[0],dh_ieo.shape[1],dh_ieo.shape[2],N, MIN_Z_FOR_LNL_DIST_MARG_APPROX,lnl_approx))   # y = x1 + x2
    
    
    
    mod = cp.RawModule(code=code, options=('-std=c++11',),
    name_expressions = name_types)
    lnl_approx_array_create = mod.get_function(name_types[1])  # compilation happens here
    lnl_approx_array_create((blocks_per_grid_x,), (threads_per_block,), (dh_ieo1, hh_ieo1, log_weights_i, log_weights_e,dh_ieo.shape[0],dh_ieo.shape[1],dh_ieo.shape[2],N, MIN_Z_FOR_LNL_DIST_MARG_APPROX,lnl_approx))   # y = x1 + x2
    

    
    
    indices = cp.nonzero(lnl_approx >= (lnl_approx.max() - cut_threshold))[0] 
    
    #d_i_inds, d_e_inds, d_o_inds = cp.where(
    #    lnl_approx >= lnl_approx.max() - cut_threshold )[0]
    indices_0 = cp.zeros_like(indices)
    indices_1 = cp.zeros_like(indices)
    indices_2 = cp.zeros_like(indices)

    module = cp.RawModule(code=loaded_from_source)
    return_3d_indices = module.get_function('return_3d_indices')
    blocks_per_grid_x = (indices.shape[0] + threads_per_block - 1) // threads_per_block
    return_3d_indices((blocks_per_grid_x,), (threads_per_block,), (indices, indices.shape[0], dh_ieo.shape[0],dh_ieo.shape[1],dh_ieo.shape[2],indices_0,indices_1,indices_2))   # y = x1 + x2


    
    return indices_0, indices_1,indices_2
    # return (d_i_inds,
    #         d_e_inds
    #         , d_o_inds)
#----------------------------------------------------------------------------------


def evaluate_log_evidence(lnlike, log_weights, n_samples) :
    """
    Evaluate the logarithm of the evidence (ratio) integral
    """
    arrt = cp.exp(lnlike +log_weights)
    aa = arrt.sum()
    log_evidence = cp.log(aa)- cp.log(n_samples)
    #log_evidence = cp.logaddexp(cp.asarray(lnlike) + log_weights) - cp.log(n_samples)
    return float(log_evidence)

#--------------------------------------------------------------------

def calculate_lnlike_and_evidence_(n_phi,m_arr_, m_inds,mprime_inds,dh_weights_dmpb,h_impb,response_dpe, timeshift_dbe,                          
                                hh_weights_dmppb,
                                asd_drift_d, 
                                log_weights_i,
                                log_weights_e  
                                ,fncInterpolator                              
                                ,approx_dist_marg_lnl_drop_threshold: float = 20., 
                                n_samples: Optional[int] = None, 
                                frac_evidence_threshold: float =1e-3, 
                                debug_mode: bool=False                                
                                ,bprofile = True):
    def lnlike_marginalized(d_h, h_h):
        """
        Log of the likelihood marginalized over distance.
        """
        x, y = dh_hh_to_x_y(d_h, h_h)
        #return _interpolated_table(x,y, grid=False)+ d_h**2 / h_h / 2
        #return _interpolated_table((x,y), table)+ d_h**2 / h_h / 2
        return fncInterpolator((x,y)) + d_h**2 / h_h / 2.0
    
    if bprofile:
            arr_dt = np.zeros((40,), dtype=float)
            t0 = time.time()
    d_dh_weights_dmpb = cp.array(dh_weights_dmpb)
    d_h_impb = cp.array(h_impb)
    d_response_dpe = cp.array(response_dpe)
    d_timeshift_dbe = cp.array(timeshift_dbe)
    d_asd_drift_d = cp.array(asd_drift_d)
    d_dh_weights_dmpb = cp.array(dh_weights_dmpb)

    d_hh_weights_dmppb = cp.array(hh_weights_dmppb)
    d_m_inds = cp.array(m_inds)
    d_mprime_inds = cp.array(mprime_inds)
    d_hh_weights_dmppb = cp.array(hh_weights_dmppb)    

    
    if bprofile:
            t1 = time.time()
            arr_dt[0] = (t1 -t0)*1e3
            

    if bprofile:
         t2 = time.time()
    d_dh_iem = cp.einsum('dmpb, impb, dpe, dbe, d -> iem',
                        d_dh_weights_dmpb, d_h_impb.conj(), d_response_dpe, 
                        d_timeshift_dbe.conj(), d_asd_drift_d**-2,
                        optimize=True)
    
    if bprofile:
            t3 = time.time()
            arr_dt[1] = (t3 -t2)*1e3
            


    if bprofile:
         t4 = time.time()
    d_hh_iem = get_hh_by_mode(d_h_impb, 
                                    d_response_dpe, 
                                    d_hh_weights_dmppb,
                                    d_asd_drift_d, 
                                    d_m_inds, 
                                    d_mprime_inds)
    
    
            
    
    
    d_m_arr = cp.array(m_arr_)
    
            

    if bprofile:
         t8 = time.time()
    d_dh_ieo, d_hh_ieo= get_dh_hh_phi_grid(n_phi,d_m_arr,d_m_inds,d_dh_iem,d_hh_iem,  d_mprime_inds)
    
    if bprofile:
            t9 = time.time()
            arr_dt[4] = (t9 -t8)*1e3
            
    
    # dh_ieo = d_dh_ieo.get()
    # hh_ieo = d_hh_ieo.get()

    n_samples = n_samples or d_dh_ieo.size
    
    # introduce index k for serial index for (i, e, o) combinations 
    # with high enough approximated distance marginalzed likelihood
    # i, e, o = inds_i_k[k], inds_e_k[k], inds_o_k[k]
    d_log_weights_i = cp.array(log_weights_i)
    d_log_weights_e = cp.array(log_weights_e)

    if bprofile:
         t10 = time.time()    
    if bprofile:
            t11 = time.time()
            arr_dt[5] = (t11 - t10)*1e3
            


    if bprofile:
         t12 = time.time()
    d_inds_i_k, d_inds_e_k, d_inds_o_k \
        = select_ieo_by_approx_lnlike_dist_marginalized_new(            
            d_dh_ieo, d_hh_ieo, d_log_weights_i, d_log_weights_e, 
            approx_dist_marg_lnl_drop_threshold)
    
    if bprofile:
            t13 = time.time()
            arr_dt[6] = (t13 - t12)*1e3
            
    inds_i_k = d_inds_i_k.get()        
    inds_e_k = d_inds_e_k.get()
    inds_o_k = d_inds_o_k.get()  
    
    if bprofile:
         t14 = time.time()
    #d_dist_marg_lnlike_k = cp.vectorize(fncInterpolator1)(d_dh_ieo[d_inds_i_k, d_inds_e_k, d_inds_o_k],
        #d_hh_ieo[d_inds_i_k, d_inds_e_k, d_inds_o_k])
    d_dist_marg_lnlike_k = lnlike_marginalized(
        d_dh_ieo[d_inds_i_k, d_inds_e_k, d_inds_o_k],
        d_hh_ieo[d_inds_i_k, d_inds_e_k, d_inds_o_k] )
    dist_marg_lnlike_k = d_dist_marg_lnlike_k.get() #!!
    if bprofile:
            t15 = time.time()
            arr_dt[7] = (t15 - t14)*1e3
            

    

    
    if frac_evidence_threshold:
        # set a probability threshold frac_evidence_threshold = x
        # use it to throw the n lowest-probability samples, 
        # such that the combined probability of the n thrown samples 
        # is equal to x. 
        if bprofile:
             t18 = time.time()
        d_log_weights_k = (d_log_weights_e[d_inds_e_k] 
                            + d_log_weights_i[d_inds_i_k])
        
        if bprofile:
            t19 = time.time()
            arr_dt[9] = (t19 - t18)*1e3
            
        log_weights_k = d_log_weights_k.get()
        if bprofile:
             t20 = time.time()
             
        d_log_prob_unormalized_k = (d_log_weights_k + d_dist_marg_lnlike_k)        
        if bprofile:
            t21 = time.time()
            arr_dt[10] = (t21 - t20)*1e3
            
        

        if bprofile:
             t22 = time.time()
        d_arg_sort_k = cp.argsort(d_log_prob_unormalized_k)
        if bprofile:
            t23 = time.time()
            arr_dt[11] = (t23 - t22)*1e3

        log_prob_unormalized_k = d_log_prob_unormalized_k.get() #!!!    
        ########  НЕ ПОНЯТНО!!! Jonathan!!!@
        # RETURN TO HOST 
        if bprofile:
             t24 = time.time()
        
        arg_sort_k = d_arg_sort_k.get()
        #dist_marg_lnlike_k = d_dist_marg_lnlike_k.get()
        #inds_i_k = d_inds_i_k.get()
        #inds_e_k = d_inds_e_k.get()
        #inds_o_k = d_inds_o_k.get()
        #log_weights_k = d_log_weights_k.get()
        #dh_ieo = d_dh_ieo.get()
        #hh_ieo = d_hh_ieo.get()
        if bprofile:
            t25 = time.time()
            arr_dt[12] = (t25 - t24)*1e3
            
        if bprofile:
             t26 = time.time()
        log_cdf = np.logaddexp.accumulate(
            log_prob_unormalized_k[arg_sort_k])
        if bprofile:
            t27 = time.time()
            arr_dt[13] = (t27 - t26)*1e3
            
        ########

        if bprofile:
             t28 = time.time()
        log_cdf -= log_cdf[-1] # normalize within given samples
        if bprofile:
            t29 = time.time()
            arr_dt[14] = (t29 - t28)*1e3
            
        

        if bprofile:
             t30 = time.time()
        low_prob_sample_k = np.searchsorted(
            log_cdf, np.log(frac_evidence_threshold))
        if bprofile:
            t31 = time.time()
            arr_dt[15] = (t31 - t30)*1e3
            
        
        if bprofile:
             t32 = time.time()
        inds_k = arg_sort_k[low_prob_sample_k:]
        if bprofile:
            t33 = time.time()
            arr_dt[16] = (t33 - t32)*1e3
            

        if bprofile:
             t34 = time.time()
        dist_marg_lnlike_k = dist_marg_lnlike_k[inds_k]
        inds_i_k = inds_i_k[inds_k]
        inds_e_k = inds_e_k[inds_k]
        inds_o_k = inds_o_k[inds_k]
        log_weights_k = log_weights_k[inds_k]
        ln_evidence = logsumexp(dist_marg_lnlike_k
                                + log_weights_k) - np.log(n_samples)
        if bprofile:
            t35 = time.time()
            arr_dt[17] = (t35 - t34)*1e3            

        if bprofile:
             print ('TotalTime = :', (t35 - t0)*1e3, ' (ms)')
             print('Time of copying data = ', (t35 - t0) * 1e3 - arr_dt.sum(), ' (ms)')
             t_calc = arr_dt.sum() -arr_dt [12]-arr_dt [0] 
             print('Calculation time =', t_calc,' (ms)')
             
             print('\n')
             print('   Time of d_dh_iem = cp.einsum(..) =', arr_dt[1],' (ms)')
             print('   Time of d_hh_iem = get_hh_by_mode(..) =', arr_dt[2],' (ms)')
             print('   Time of get_dh_hh_phi_grid(..) =', arr_dt[4],' (ms)')
             print('   Time of copying log_weights_i and log_weights_e to device =', arr_dt[5],' (ms)')
             print('   Time of select_ieo_by_approx_lnlike_dist_marginalized(..) =', arr_dt[6],' (ms)') 
             print('   Time of lnlike_marginalized(..) =', arr_dt[7],' (ms)')
             print('   Time of concatenating d_log_weights_e and d_log_weights_i =', arr_dt[9],' (ms)')
             print('   Time of concatenating d_log_weights_k and d_dist_marg_lnlike_k =', arr_dt[10],' (ms)')
             print('   Time of cp.argsort(..) =', arr_dt[11],' (ms)')
             print('   Time of np.logaddexp.accumulate(..)  =', arr_dt[13],' (ms)')
             print('   Time of normalizing within given samples = :', arr_dt[14],' (ms)')             
             print('   Time of np.searchsorted(..) = :', arr_dt[15],' (ms)')
             print('   Time of arg_sort_k[low_prob_sample_k:] = :', arr_dt[16],' (ms)')
             print('   Time of TAIL :', arr_dt[17],' (ms)')
             print('\n')
             
             print('   Time of initial copying data to device =', arr_dt[0],' (ms)')
             print('   Time copying data to HOST =', arr_dt[12],' (ms)')
    else:
        t16 = time.time()
        d_ln_evidence = evaluate_log_evidence(
        d_dist_marg_lnlike_k,
        d_log_weights_i[d_inds_i_k] + d_log_weights_e[d_inds_e_k],
        n_samples)
        if bprofile:
            t17 = time.time()
            arr_dt[8] = (t17 - t16)*1e3
                  
        if bprofile:
             print ('TotalTime = :', (t17 - t0)*1e3, ' (ms)')
             print('Time of copying data = ', (t17 - t0) * 1e3 - arr_dt.sum(), ' (ms)')
             t_calc = arr_dt.sum() -arr_dt [12]-arr_dt [0] 
             print('Calculation time =', t_calc,' (ms)')
             
             print('\n')
             print('   Time of d_dh_iem = cp.einsum(..) =', arr_dt[1],' (ms)')
             print('   Time of d_hh_iem = get_hh_by_mode(..) =', arr_dt[2],' (ms)')
             print('   Time of get_dh_hh_phi_grid(..) =', arr_dt[4],' (ms)')
             print('   Time of copying log_weights_i and log_weights_e to device =', arr_dt[5],' (ms)')
             print('   Time of select_ieo_by_approx_lnlike_dist_marginalized(..) =', arr_dt[6],' (ms)') 
             print('   Time of lnlike_marginalized(..) =', arr_dt[7],' (ms)')
             print('Time of evaluate_log_evidence(..) =', arr_dt[8],' (ms)')              
             print('\n')
             
             print('   Time of initial copying data to device =', arr_dt[0],' (ms)')
             print('   Time copying data to HOST =', arr_dt[12],' (ms)')
    
    if debug_mode:
        return (dist_marg_lnlike_k, ln_evidence, 
                inds_i_k, inds_e_k, inds_o_k )
    else:
        return (dist_marg_lnlike_k, ln_evidence, 
                inds_i_k, inds_e_k, inds_o_k)