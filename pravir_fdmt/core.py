import time
import logging
import numpy as np

#from fdmt import utils
import utils

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def cff(f1_start: float, f1_end: float, f2_start: float, f2_end: float) -> float:
    """Calculates the C-term (ratio) for given two frequency ranges.

    Parameters
    ----------
    f1_start : float
        Start frequency of the first frequency range in MHz.
    f1_end : float
        End frequency of the first frequency range in MHz.
    f2_start : float
        Start frequency of the second frequency range in MHz.
    f2_end : float
        End frequency of the second frequency range in MHz.

    Returns
    -------
    float
        The C-term (ratio) for the given frequency ranges.

    Notes
    -----
    For details, see Equation 20 in Zackay & Ofek (2014).
    """
    return (f1_start**-2 - f1_end**-2) / (f2_start**-2 - f2_end**-2)


def fdmt(
    waterfall: np.ndarray,
    f_min: float,
    f_max: float,
    delta_t_max: int,
    data_type: np.dtype,
) -> np.ndarray:
    """Performs the Fast Dispersion Measure Transform on the input waterfall data.

    Parameters
    ----------
    waterfall : np.ndarray
        Intensity I(f,t) or the 2D waterfall spectra.
    f_min : float
        Start frequency of the waterfall spectra in MHz.
    f_max : float
        End frequency of the waterfall spectra in MHz.
    delta_t_max : int
        The maximal delay (in time bins) corresponding to the maximum DM.
    data_type : str
        A valid numpy dtype. Recommended: either int32, or int64.

    Returns
    -------
    np.ndarray
        The dispersion measure transform of the input waterfall data with
        dimensions [delta_t_max, N_t].

    Notes
    -----
    delta_t_max appears in the paper as $N_{\Delta}$. A typical input is
    delta_t_max = N_f. For details, see algorithm 1 in Zackay & Ofek (2014).
    """
    nchans, nsamples = waterfall.shape
    niters = int(np.log2(nchans))
    assert (
        utils.bit_count(nchans) == 1 and utils.bit_count(nsamples) == 1
    ), "Input dimensions must be a power of 2"

    tstart = time.time()
    logger.info(f"Input data dimensions: {waterfall.shape}")
    state = fdmt_init(waterfall, f_min, f_max, delta_t_max, data_type)
    logger.info(f"Initial state dimensions: {state.shape}")

    logger.info(f"Iterating {niters} times to calculate to delta_t_max = {delta_t_max}")

    # For some situations, it is beneficial to play with this correction.
    # When applied to real data, one should carefully analyze and understand the
    # effect of this correction on the pulse he is looking for (especially if
    # convolving with a specific pulse profile)
    df = (f_max - f_min) / nchans
    correction = df / 2

    for i_t in range(1, niters + 1):
        logger.debug(f"Iteration = {i_t}: ")
        state = fdmt_iter(state, f_min, f_max, delta_t_max, correction, data_type)

    logger.info(f"Total time: {time.time() - tstart}")

    assert (
        state.shape[0] == 1
    ), "Channel axis should have length 1 after all FDMT iterations."
    return state.squeeze()


def fdmt_init(
    waterfall: np.ndarray,
    f_min: float,
    f_max: float,
    delta_t_max: int,
    data_type: np.dtype,
) -> np.ndarray:
    """Initializes the FDMT algorithm.

    Parameters
    ----------
    waterfall : np.ndarray
        Intensity I(f,t) or the 2D waterfall spectra.
    f_min : float
        Start frequency of the waterfall spectra in MHz.
    f_max : float
        End frequency of the waterfall spectra in MHz.
    delta_t_max : int
        The maximal delay (in time bins) corresponding to the maximum DM.
    data_type : np.dtype
        A valid numpy dtype. Recommended: either int32, or int64.

    Returns
    -------
    np.ndarray
        The initial state of the FDMT algorithm with
        dimensions [N_f, N_dt_init, N_t].

    Notes
    -----
    For details, see Equations 22--24 in Zackay & Ofek (2014).

    """

    nchans, nsamples = waterfall.shape

    df = (f_max - f_min) / nchans
    delta_t_init = int(np.ceil((delta_t_max - 1) * cff(f_min, f_min + df, f_min, f_max)))

    state = np.zeros([nchans, delta_t_init + 1, nsamples], data_type)
    state[:, 0, :] = waterfall

    for i_dt in range(1, delta_t_init + 1):
        state[:, i_dt, i_dt:] = state[:, i_dt - 1, i_dt:] + waterfall[:, :-i_dt]
    return state


def fdmt_iter(
    state: np.ndarray,
    f_min: float,
    f_max: float,
    delta_t_max: int,
    correction: float,
    data_type: np.dtype,
) -> np.ndarray:
    """Performs one iteration of the FDMT algorithm.

    Parameters
    ----------
    state : np.ndarray
        Current state of the FDMT algorithm with dimensions [N_f, N_dt, N_t].
    f_min : float
        Start frequency of the waterfall spectra in MHz.
    f_max : float
        End frequency of the waterfall spectra in MHz.
    delta_t_max : int
        The maximal delay (in time bins) corresponding to the maximum DM.
    correction : int
        Correction to the middle frequency of the current frequency band.
    data_type : np.dtype
        A valid numpy dtype. Recommended: either int32, or int64.

    Returns
    -------
    np.ndarray
        State of the FDMT algorithm after one iteration with dimensions
        [N_f/2, N_dt_new, N_t] where N_dt_new is the maximal number of
        bins the dispersion curve travels at one output frequency band.
    """
    nchans_prev, _, nsamples = state.shape

    nchans = nchans_prev // 2
    df = (f_max - f_min) / nchans
    # the maximum delta_t needed to calculate at the i'th iteration
    delta_t = int(np.ceil((delta_t_max - 1) * cff(f_min, f_min + df, f_min, f_max)))
    output = np.zeros([nchans, delta_t + 1, nsamples], data_type)

    logger.debug(f"df = {df}, delta_t = {delta_t}")
    logger.debug(f"input dims = {state.shape}")
    logger.debug(f"output dims = {output.shape}")

    # For negative dispersions
    shift_output = 0
    shift_input = 0

    for i_f in range(nchans):
        f_start = df * (i_f) + f_min
        f_end = df * (i_f + 1) + f_min
        f_middle = (f_end - f_start) / 2 + f_start - correction
        f_middle_larger = (f_end - f_start) / 2 + f_start + correction
        delta_t_local = int(
            np.ceil((delta_t_max - 1) * cff(f_start, f_end, f_min, f_max))
        )
        for i_dt in range(delta_t_local + 1):
            dt_middle = round(i_dt * cff(f_start, f_middle, f_start, f_end))
            dt_middle_index = dt_middle + shift_input
            dt_middle_larger = round(i_dt * cff(f_start, f_middle_larger, f_start, f_end))
            dt_rest_index = (i_dt - dt_middle_larger) + shift_input

            i_t_min = 0
            i_t_max = dt_middle_larger
            output[i_f, i_dt + shift_output, i_t_min:i_t_max] = state[
                2 * i_f, dt_middle_index, i_t_min:i_t_max
            ]
            i_t_min = dt_middle_larger
            i_t_max = nsamples

            output[i_f, i_dt + shift_output, i_t_min:i_t_max] = (
                state[2 * i_f, dt_middle_index, i_t_min:i_t_max]
                + state[
                    2 * i_f + 1,
                    dt_rest_index,
                    i_t_min - dt_middle_larger : i_t_max - dt_middle_larger,
                ]
            )

    return output


def hybrid_dedisp(
    raw_signal: np.ndarray,
    f_min: float,
    f_max: float,
    n_p: int,
    d_max: float,
    data_type: np.dtype = np.int64,
) -> np.ndarray:
    """Performs the coherent FDMT hybrid algorithm.

    Parameters
    ----------
    raw_signal : np.ndarray
        Raw antenna voltage time series (base-band sampled).
    f_min : float
        Start frequency of the baseband data in MHz.
    f_max : float
        End frequency of the baseband data in MHz.
    n_p : int
        Length of the pulse in time bins, i.e (t_p/\tau), or N_p in the paper.
    d_max : float
        Maximal dispersion to scan, in units of pc cm^-3.

    Returns
    -------
    np.ndarray
        Output array with dimensions [N_d, fdmt] where N_d is the number of
        coherent DMs scanned, and fdmt is the output of the FDMT algorithm

    Notes
    -----
    For details, see algorithm 3 in Zackay & Ofek (2014).
    Subject for future improvement:
    Might want to reduce trial dispersions by a factor of 2 by
    letting FDMT scan negative dispersions. Might want to use packing
    (either in the coherent stage, or in the incoherent stage)
    """
    conv_const = (
        utils.DM_CONSTANT * 10**6 * (f_min**-2 - f_max**-2) * (f_max - f_min)
    )
    n_d = d_max * conv_const
    n_coherent = int(np.ceil(n_d / (n_p**2)))
    logger.info(f"Number of coherent iterations: {n_coherent}")
    ffted_signal = np.fft.fft(raw_signal)
    res = np.empty(n_coherent, dtype=object)
    for i_coh_d in range(n_coherent):
        cur_coherent_d = i_coh_d * (d_max / n_coherent)
        logger.info(f"Coherent iteration: {i_coh_d}, DM: {cur_coherent_d}")
        dedisp = utils.coherent_dedisp(
            ffted_signal, cur_coherent_d, f_min, f_max, ffted=True
        )
        stft = np.abs(utils.stft_func(dedisp, n_p)) ** 2
        stft_norm = (stft - np.mean(stft)) / np.std(stft)
        res[i_coh_d] = fdmt(stft_norm, f_min, f_max, n_p, data_type)
    return np.array(res)
