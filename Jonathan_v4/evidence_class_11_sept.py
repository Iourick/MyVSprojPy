import itertools
import numpy as np
from scipy.special import logsumexp
from typing import Tuple, Optional

from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import (RectBivariateSpline,
                               InterpolatedUnivariateSpline)
import time


# The marginalization over distance for the likelihood assumes that 
# the maximal-likelihood d_luminosity (=sqrt(<h|h>)/<d|h>) smaller than 
# the maximal allowed distance by more than the width of the Gaussian.
# This translated to z > h_norm/(15e3) + (a few)
# We approximate the RHS of this inequality with 4
MIN_Z_FOR_LNL_DIST_MARG_APPROX = 4

class DummyLookupTable:
    def __init__(self):
        pass
    def lnlike_marginalized(self, dh: np.ndarray[float], 
                            hh: np.ndarray[float])-> np.ndarray[float]:
        
        h_norm = np.sqrt(hh)
        z = dh / h_norm
        lnl = (z ** 2 / 2
                      + 3 * np.log(h_norm)
                      - 4 * np.log(z)) # 1d
        return lnl

class LookupTable():
    """
    Auxiliary class to marginalize the likelihood over distance.
    The instances are callable, and use interpolation to compute
    ``log(evidence) - d_h**2 / h_h / 2``, where``evidence`` is the value
    of the likelihood marginalized over distance.
    The interpolation is done in some coordinates `x`, `y` in which the
    function is smooth (see ``_get_x_y``, ``get_dh_hh``).
    """

    REFERENCE_DISTANCE = 1.  # Luminosity distance at which h is defined (Mpc).
    _Z0 = 10.  # Typical SNR of events.
    _SIGMAS = 10.  # How far out the tail of the distribution to tabulate.
    _rng = np.random.default_rng()

    def __init__(self, d_luminosity_max=1.5e4, shape=(256, 128)):
        """
        Construct the interpolation table.
        If a table with the same settings is found in the file in
        ``LOOKUP_TABLES_FNAME``, it will be loaded for faster
        instantiation. If not, the table will be computed and saved.

        Parameters
        ----------

        d_luminosity_max: float
            Maximum luminosity distance (Mpc).

        shape: (int, int)
            Number of interpolating points in x and y.
        """
        self.d_luminosity_prior = euclidean_distance_prior
        self.d_luminosity_max = d_luminosity_max
        self.shape = shape

        self._inverse_volume = 1 / quad(self.d_luminosity_prior,
                                        0, self.d_luminosity_max)[0]

        x_arr = np.linspace(-self._SIGMAS, 0, shape[0])
        y_arr = np.linspace(self._compactify(- self._SIGMAS / self._Z0),
                            1 - 1e-8, shape[1])
        
        x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing='ij')

        dh_grid, hh_grid = self._get_dh_hh(x_grid, y_grid)

        table = self._get_table(dh_grid, hh_grid)
        self._interpolated_table = RectBivariateSpline(x_arr, y_arr, table)
        self.tabulated = {'x': x_grid,
                          'y': y_grid,
                          'd_h': dh_grid,
                          'h_h': hh_grid,
                          'function': table}  # Bookkeeping, not used.

    def _get_table(self, dh_grid, hh_grid):
        """
        Attempt to load a previously computed table with the requested
        settings. If this is not possible, compute the table and save it
        for faster access in the future.
        """
        
        if Path('lookup_table.npy').exists():
            table = np.load('lookup_table.npy')
            if table.shape == self.shape:
                return table
            else:
                print('Table shape mismatch. Recomputing.')
        print('Computing table.')
        table = np.vectorize(self._function)(dh_grid, hh_grid)
        np.save(file='lookup_table.npy', arr=table)

        return table

    def __call__(self, d_h, h_h):
        """
        Return ``log(evidence) - d_h**2 / h_h / 2``, where``evidence``
        is the value of the likelihood marginalized over distance.
        This uses interpolation from a precomputed table.

        Parameters
        ----------
        d_h, h_h: float
            Inner products (d|h), (h|h) where `d` is data and `h` is the
            model strain at a fiducial distance REFERENCE_DISTANCE.
            These are scalars (detectors are summed over). A real part
            is taken in (d|h), not an absolute value (phase is not
            marginalized over so the computation is robust to higher
            modes).
        """
        return self._interpolated_table(*self._get_x_y(d_h, h_h),
                                        grid=False)[()]

    def _get_distance_bounds(self, d_h, h_h, sigmas=5.):
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
        u_peak = d_h / (self.REFERENCE_DISTANCE * h_h)
        delta_u = sigmas / np.sqrt(h_h)
        return np.array([self.REFERENCE_DISTANCE / (u_peak + delta_u),
                         self.REFERENCE_DISTANCE / (u_peak - delta_u)])

    def lnlike_marginalized(self, d_h, h_h):
        """
        Parameters
        ----------
        d_h, h_h: float
            Inner products (d|h), (h|h) where `d` is data and `h` is the
            model strain at a fiducial distance REFERENCE_DISTANCE.
            These are scalars (detectors are summed over).
        """
        return self(d_h, h_h) + d_h**2 / h_h / 2

    def sample_distance(self, d_h, h_h, num=None, resolution=256):
        """
        Return samples from the luminosity distance distribution given
        the inner products (d|h), (h|h) of a waveform at distance
        ``REFERENCE_DISTANCE``.

        Parameters
        ----------
        d_h: float
            Inner product (summed over detectors) between data and
            waveform at ``self.REFERENCE_DISTANCE``.

        h_h: float
            Inner product (summed over detectors) of waveform at
            ``self.REFERENCE_DISTANCE`` with itself.

        num: int or None
            How many samples to generate. ``None`` (default) generates a
            single (scalar) sample.

        resolution: int
            How finely to interpolate the distance distribution when
            generating samples.
        """
        u_bounds = 1 / self._get_distance_bounds(d_h, h_h, sigmas=10.)
        focused_grid = 1 / np.linspace(*u_bounds, resolution)
        focused_grid = focused_grid[(focused_grid > 0)
                                    & (focused_grid < self.d_luminosity_max)]
        broad_grid = np.linspace(0, self.d_luminosity_max, resolution)[1:]
        distances = np.sort(np.concatenate([broad_grid, focused_grid]))
        posterior = self._function_integrand(distances, d_h, h_h)
        cumulative = InterpolatedUnivariateSpline(
            distances, posterior, k=1).antiderivative()(distances)[()]
        return np.interp(self._rng.uniform(0, cumulative[-1], num),
                         cumulative, distances)

    def _function(self, d_h, h_h):
        """
        Function to interpolate with the aid of a lookup table.
        Return ``log(evidence) - overlap**2 / 2``, where ``evidence``
        is the value of the likelihood marginalized over distance.
        """
        return np.log(quad(self._function_integrand, 0, self.d_luminosity_max,
                           args=(d_h, h_h),
                           points=self._get_distance_bounds(d_h, h_h)
                           )[0]
                      + 1e-100)

    def _function_integrand(self, d_luminosity, d_h, h_h):
        """
        Proportional to the distance posterior. The log of the integral
        of this function is stored in the lookup table.
        """
        norm_h = np.sqrt(h_h)
        return (self.d_luminosity_prior(d_luminosity) * self._inverse_volume
                * np.exp(-(norm_h * self.REFERENCE_DISTANCE / d_luminosity
                           - d_h / norm_h)**2 / 2))

    def _get_x_y(self, d_h, h_h):
        """
        Interpolation coordinates (x, y) in which the function to
        interpolate is smooth, as a function of the inner products
        (d|h), (h|h).
        Inverse of ``_get_dh_hh``.
        """
        norm_h = np.sqrt(h_h)
        overlap = d_h / norm_h
        x = np.log(norm_h / (self.d_luminosity_max
                             * (self._SIGMAS + np.abs(overlap))))
        y = self._compactify(overlap / self._Z0)
        return x, y

    def _get_dh_hh(self, x, y):
        """
        Inner products (d|h), (h|h), as a function of the interpolation
        coordinates (x, y) in which the function to interpolate is
        smooth.
        Inverse of ``_get_x_y``.
        """
        overlap = self._uncompactify(y) * self._Z0
        norm_h = (np.exp(x) * self.d_luminosity_max
                  * (self._SIGMAS + np.abs(overlap)))
        d_h = overlap * norm_h
        h_h = norm_h**2
        return d_h, h_h

    @staticmethod
    def _compactify(value):
        """Monotonic function from (-inf, inf) to (-1, 1)."""
        return value / (1 + np.abs(value))

    @staticmethod
    def _uncompactify(value):
        """
        Inverse of _compactify. Monotonic function from (-1, 1) to
        (-inf, inf).
        """
        return value / (1 - np.abs(value))

class Evidence:
    """
    A class that receives as input the components of intrinsic and
    extrinsic factorizations and calculates the likelihood at each
    (int., ext., phi) combination (distance marginalized).
    Indexing and size convention:
        * i: intrinsic parameter
        * m: modes, or combinations of modes
        * p: polarization, plus (0) or cross (1)
        * b: frequency bin (as in relative binning)
        * e: extrinsic parameters
        * d: detector
    """
    def __init__(self, 
                 n_phi: int, 
                 m_arr: np.ndarray[int], 
                 lookup_table=None):
        """
        n_phi : number of points to evaluate phi on,
        m_arr : modes
        lookup_table = lookup table for distance marginalization
        """
        self.n_phi = n_phi
        self.m_arr = m_arr
        self.phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        self.m_arr = np.asarray(m_arr)
        # Used for terms with 2 modes indices
        self.m_inds, self.mprime_inds = zip(
            *itertools.combinations_with_replacement(
            range(len(self.m_arr)), 2))

        self.lookup_table = lookup_table or LookupTable()

    @staticmethod
    def get_dh_by_mode(
        dh_weights_dmpb: np.ndarray[complex],
        h_impb: np.ndarray[complex], 
        response_dpe: np.ndarray[float],
        timeshift_dbe: np.ndarray[complex], 
        asd_drift_d: np.ndarray) -> np.ndarray[complex]:
        """
        Calculate the inner product <d|h> for each combination of 
        intrinsic sample, extrinsic sample and mode. 
        Inputs:
        dh_weights_dmpb: array weights for detector, mode, polarization,
        and frequency bin. Assumes it was calculated from the integrand
        integrand = data * h.conj() 
        h_impb: array of waveforms.
        response_dpe: array of detector response.
        timeshift_dbe: array of time shifts.
        asd_drift_d: array of ASD drifts.
        Output:
        dh_iem: array of inner products (complex).
        """
        dh_iem = np.einsum('dmpb, impb, dpe, dbe, d -> iem',
                           dh_weights_dmpb, h_impb.conj(), response_dpe, 
                           timeshift_dbe.conj(), asd_drift_d**-2,
                           optimize=True)
        return dh_iem

    @staticmethod
    def get_hh_by_mode(h_impb: np.ndarray[complex], 
                       response_dpe: np.ndarray[float],
                       hh_weights_dmppb: np.ndarray[complex], 
                       asd_drift_d: np.ndarray[float], 
                       m_inds: tuple, 
                       mprime_inds: tuple)-> np.ndarray[complex]:
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
        hh_idmpP = np.einsum('dmpPb, impb, imPb -> idmpP',
                             hh_weights_dmppb,
                             h_impb[:, m_inds, ...],
                             h_impb.conj()[:, mprime_inds, ...],
                             optimize=True)  # idmpp
        ff_dppe = np.einsum('dpe, dPe, d -> dpPe', 
                            response_dpe, response_dpe, asd_drift_d**-2,
                            optimize=True)  # dppe
        hh_iem = np.einsum('idmpP, dpPe -> iem', 
                           hh_idmpP, ff_dppe, optimize=True)  # iem
        return hh_iem

    def get_dh_hh_phi_grid(self, 
                           dh_iem: np.ndarray[complex], 
                           hh_iem: np.ndarray[complex]
                           ) -> np.ndarray[float]:
        """
        change the orbital phase of each mode by exp(1j*phi*m), and keep
        the real part assumes that factor of two from off-diagonal modes
        (e.g. (2,0) and not e.g. (2,2)) is already in the 
        hh_weights_dmppb
        """
        
        phi_grid = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)  # o
        # dh_phasor is the phase shift applied to h[m].conj, hence the 
        # minus sign
        dh_phasor = np.exp(-1j * np.outer(self.m_arr, phi_grid))  # mo
        # hh_phasor is applied both to h[m] and h[mprime].conj, hence 
        # the subtraction of the two modes
        hh_phasor = np.exp(1j * np.outer(
            self.m_arr[self.m_inds, ] - self.m_arr[self.mprime_inds, ],
            phi_grid))  # mo
        dh_ieo = np.einsum('iem, mo -> ieo', 
                           dh_iem, dh_phasor, optimize=True).real  # ieo
        hh_ieo = np.einsum('iem, mo -> ieo', 
                           hh_iem, hh_phasor, optimize=True).real  # ieo

        return dh_ieo, hh_ieo

    @staticmethod
    def select_ieo_by_approx_lnlike_dist_marginalized(
            dh_ieo: np.ndarray[float], 
            hh_ieo: np.ndarray[float],
            log_weights_i: np.ndarray[float], 
            log_weights_e: np.ndarray[float],
            cut_threshold: float = 20.
            ) -> Tuple[np.ndarray[int], np.ndarray[int], np.ndarray[int]]:
        """
        Return three arrays with intrinsic, extrinsic and phi sample 
        indices.
        """
        h_norm = np.sqrt(hh_ieo).astype(np.float32)
        z = dh_ieo / h_norm
        i_inds, e_inds, o_inds = np.where(z > MIN_Z_FOR_LNL_DIST_MARG_APPROX)
        lnl_approx = (z[i_inds, e_inds, o_inds] ** 2 / 2
                      + 3 * np.log(h_norm[i_inds, e_inds, o_inds])
                      - 4 * np.log(z[i_inds, e_inds, o_inds])
                      + log_weights_i[i_inds] 
                      + log_weights_e[e_inds]) # 1d
        print('lnl_approx.max() = ', lnl_approx.max())
        flattened_inds = np.where(
            lnl_approx >= lnl_approx.max() - cut_threshold)[0]

        return (i_inds[flattened_inds],
                e_inds[flattened_inds],
                o_inds[flattened_inds])

    @staticmethod
    def evaluate_log_evidence(lnlike: np.ndarray[float],
                              log_weights: np.ndarray[float], 
                              n_samples: int) -> float:
        """
        Evaluate the logarithm of the evidence (ratio) integral
        """
        return logsumexp(lnlike + log_weights) - np.log(n_samples)

    def calculate_lnlike_and_evidence(
            self, dh_weights_dmpb: np.ndarray[complex], 
            h_impb: np.ndarray[complex], 
            response_dpe: np.ndarray[float],
            timeshift_dbe: np.ndarray[complex],
            hh_weights_dmppb: np.ndarray[complex],
            asd_drift_d: np.ndarray[float], 
            log_weights_i: np.ndarray[float],
            log_weights_e: np.ndarray[float], 
            approx_dist_marg_lnl_drop_threshold: float = 20., 
            n_samples: Optional[int] = None, 
            frac_evidence_threshold: float =1e-3, 
            debug_mode: bool=False
            ,bprofile = False):
        """
        Use stored samples to compute lnl (distance marginalized) on
        grid of intrinsic x extrinsic x phi samples.
        
        Inputs:
        dh_weights_dmpb: array relative-binning weights for the 
        integrand data * h.conj() for each detector, mode, polarization,
        mode, polarization,and frequency bin.
        h_impb: array of waveforms, with shape (n_intrinsic, n_modes,
        n_polarizations, n_fbin)
        response_dpe: array of detector response, with shape
        (n_extrinsic, n_polarizations, n_fbin)
        timeshift_dbe: array of time shifts (complex exponents), with shape
        (n_extrinsic, n_detector, n_fbin)
        hh_weights_dmppb: array of relative-binning weights for the 
        integrand h[m]*h[mprime].conj() for each detector, mode pair, 
        polarization pair, and frequency bin.
        asd_drift_d: array of ASD drifts, with shape (n_detector)
        log_weights_i: array of intrinsic log-weights due to importance 
        sampling, per intrinsic sample.
        log_weights_e: array of extrinsic log-weights due to importance
        sampling, per extrinsic sample.
        approx_dist_marg_lnl_drop_threshold: float, threshold for
        approximated distance marginalized lnlike, below which the
        sample is discarded.
        n_samples: int, overall number of samples (intrinsic x extrinsic
        x phases) used. Needed for normalization of the evidence. Could
        be different from the number of samples used in the calculation,
        due to rejection-sampling. If None, infered from `dh_ieo`.
        frac_evidence_threshold: float, set threshold on log-posterior 
        probability for throwing out low-probability samples, such 
        that (1-frac_evidence_threshold) of the total probability is
        retained.
        debug_mode: bool, if True, return additional arrays for 
        debugging purposes.
        
        Output:
        dist_marg_lnlike_k: array of lnlike for each combination of
        intrinsic, extrinsic and phi sample.
        ln_evidence: float, ln(evidence) for the given samples.
        inds_i_k: array of intrinsic sample indices.
        inds_e_k: array of extrinsic sample indices.
        inds_o_k: array of phi sample indices.
                       
        """
        if bprofile:
            arr_dt = np.zeros((40,), dtype=float)
            t0 = time.time()
            
        

        dh_iem = self.get_dh_by_mode(dh_weights_dmpb, 
                                     h_impb, 
                                     response_dpe,
                                     timeshift_dbe, 
                                     asd_drift_d)
        if bprofile:
            t1 = time.time()
            arr_dt[0] = (t1 -t0)*1e3
            print('Time of self.get_dh_by_mode(..) =', arr_dt[0],' (ms)')
        
        if bprofile:
            t2 = time.time()
        hh_iem = self.get_hh_by_mode(h_impb, 
                                     response_dpe, 
                                     hh_weights_dmppb,
                                     asd_drift_d, 
                                     self.m_inds, 
                                     self.mprime_inds)
        if bprofile:
            t3 = time.time()
            arr_dt[1] = (t3 -t1)*1e3
            print('Time of self.get_hh_by_mode(..) =', arr_dt[1],' (ms)')

        if bprofile:
            t4 = time.time()

        dh_ieo, hh_ieo = self.get_dh_hh_phi_grid(dh_iem, hh_iem)

        if bprofile:
            t5 = time.time()
            arr_dt[2] = (t5 -t4)*1e3
            print('Time of self.get_dh_hh_phi_grid(..) =', arr_dt[2],' (ms)')
        
        n_samples = n_samples or dh_ieo.size
        
        # introduce index k for serial index for (i, e, o) combinations 
        # with high enough approximated distance marginalzed likelihood
        # i, e, o = inds_i_k[k], inds_e_k[k], inds_o_k[k]
        if bprofile:
            t6 = time.time()

        # print('dh_ieo = \n',dh_ieo)
        # print('hh_ieo = \n',hh_ieo)
        # print('log_weights_i = \n',log_weights_i)
        # print('log_weights_e = \n',log_weights_e)
        # print('approx_dist_marg_lnl_drop_threshold = ',approx_dist_marg_lnl_drop_threshold)

        inds_i_k, inds_e_k, inds_o_k \
            = self.select_ieo_by_approx_lnlike_dist_marginalized(
                dh_ieo, hh_ieo, log_weights_i, log_weights_e, 
                approx_dist_marg_lnl_drop_threshold)
        
        if bprofile:
            t7 = time.time()
            arr_dt[3] = (t7 -t6)*1e3
            print('Time of self.select_ieo_by_approx_lnlike_dist_marginalized(..) =', arr_dt[3],' (ms)')

        if bprofile:
            t8 = time.time()
        dist_marg_lnlike_k = self.lookup_table.lnlike_marginalized(
            dh_ieo[inds_i_k, inds_e_k, inds_o_k],
            hh_ieo[inds_i_k, inds_e_k, inds_o_k])
        if bprofile:
            t9 = time.time()
            arr_dt[4] = (t9 -t8)*1e3
            print('Time of self.lookup_table.lnlike_marginalized(..) =', arr_dt[4],' (ms)')


        if bprofile:
            t10 = time.time()
        ln_evidence = self.evaluate_log_evidence(
            dist_marg_lnlike_k,
            log_weights_i[inds_i_k] + log_weights_e[inds_e_k],
            n_samples)
        if bprofile:
            t11 = time.time()
            arr_dt[5] = (t11 -t10)*1e3
            print('Time of self.evaluate_log_evidence(..) =', arr_dt[5],' (ms)')

        if frac_evidence_threshold:
            # set a probability threshold frac_evidence_threshold = x
            # use it to throw the n lowest-probability samples, 
            # such that the combined probability of the n thrown samples 
            # is equal to x. 
            if bprofile:
                t12 = time.time()
            log_weights_k = (log_weights_e[inds_e_k] 
                             + log_weights_i[inds_i_k])
            if bprofile:
                t13 = time.time()
                arr_dt[6] = (t13 -t12)*1e3
                print('Time of concatenating log_weights_e and log_weights_i =', arr_dt[6],' (ms)')
            
            if bprofile:
                t14 = time.time()
            log_prob_unormalized_k = (log_weights_k + dist_marg_lnlike_k)
            if bprofile:
                t15 = time.time()
                arr_dt[7] = (t13 -t12)*1e3
                print('Time of concatenating log_weights_k and dist_marg_lnlike_k  =', arr_dt[7],' (ms)')

            if bprofile:
                t16 = time.time()
            arg_sort_k = np.argsort(log_prob_unormalized_k)
            if bprofile:
                t17 = time.time()
                arr_dt[8] = (t17 -t16)*1e3
                print('Time of np.argsort(log_prob_unormalized_k)  =', arr_dt[8],' (ms)')

            if bprofile:
                t18 = time.time()
            log_cdf = np.logaddexp.accumulate(
                log_prob_unormalized_k[arg_sort_k])
            if bprofile:
                t19 = time.time()
                arr_dt[9] = (t19 -t18)*1e3
                print('Time of np.argsort(log_prob_unormalized_k)  =', arr_dt[9],' (ms)')

            if bprofile:
                t20 = time.time()
            log_cdf -= log_cdf[-1] # normalize within given samples
            if bprofile:
                t21 = time.time()
                arr_dt[10] = (t21 -t20)*1e3
                print('Time of -= log_cdf[-1]  =', arr_dt[10],' (ms)')


            if bprofile:
                t22 = time.time()
            low_prob_sample_k = np.searchsorted(
                log_cdf, np.log(frac_evidence_threshold))
            if bprofile:
                t23 = time.time()
                arr_dt[11] = (t23 -t22)*1e3
                print('Time of np.searchsorted(..)  =', arr_dt[11],' (ms)')
            
            if bprofile:
                t24 = time.time()
            inds_k = arg_sort_k[low_prob_sample_k:]
            if bprofile:
                t25 = time.time()
                arr_dt[12] = (t25 -t24)*1e3
                print('Time of arg_sort_k[low_prob_sample_k:]  =', arr_dt[12],' (ms)')

            if bprofile:
                t26 = time.time()
            dist_marg_lnlike_k = dist_marg_lnlike_k[inds_k]
            inds_i_k = inds_i_k[inds_k]
            inds_e_k = inds_e_k[inds_k]
            inds_o_k = inds_o_k[inds_k]
            log_weights_k = log_weights_k[inds_k]
            ln_evidence = logsumexp(dist_marg_lnlike_k
                                    + log_weights_k) - np.log(n_samples)
            if bprofile:
                t27 = time.time()
                arr_dt[13] = (t27 -t26)*1e3
                print('Time of TAIL =', arr_dt[13],' (ms)')

            if bprofile:
                print('Total time = ', arr_dt.sum())
        
        if debug_mode:
            return (dist_marg_lnlike_k, ln_evidence, 
                    inds_i_k, inds_e_k, inds_o_k,
                    dh_ieo[inds_i_k, inds_e_k, inds_o_k],
                    hh_ieo[inds_i_k, inds_e_k, inds_o_k])
        else:
            return (dist_marg_lnlike_k, ln_evidence, 
                    inds_i_k, inds_e_k, inds_o_k)
        
"""
Provide class ``LookupTable`` to marginalize the likelihood over
distance; and ``LookupTableMarginalizedPhase22`` to marginalize the
likelihood over both distance and phase for (l, |m|) = (2, 2) waveforms.
"""


def euclidean_distance_prior(d_luminosity):
    """
    Distance prior uniform in luminosity volume, normalized so that
    its integral is the luminosity volume in Mpc^3.
    Note: no maximum is enforced here.
    """
    return 4 * np.pi * d_luminosity**2


