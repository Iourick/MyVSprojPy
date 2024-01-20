import numpy as np


DM_CONSTANT = 4.148808e3  # MHz^2 * s / (pc cm^-3) - L&K Handbook of Pulsar Astronomy


def coherent_dedisp(
    raw_signal: np.ndarray, dm: float, f_min: float, f_max: float, ffted=False
) -> np.ndarray:
    """Performs coherent dedispersion on the input signal.

    Parameters
    ----------
    raw_signal : np.ndarray
        1D array of the raw signal
    dm : float
        Dispersion measure in pc*cm^-3
    f_min : float
        Minimum frequency in MHz
    f_max : float
        Maximum frequency in MHz
    ffted : bool, optional
        If True, the input signal is assumed to be already fft(raw_signal), by default False

    Returns
    -------
    np.ndarray
        The coherent dedispersed signal

    Notes
    -----
    For future improvements:
    1) Signal partition to chunks of length N_d is not applied, and maybe it should be.
    2) No use of packing is done, though it is obvious it should be done (either in the coherent stage.
    (and take special care of the abs()**2 operation done by other functions) or in the incoherent stage)
    """

    n_total = len(raw_signal)
    dm_coeff = DM_CONSTANT * 10**6 * dm
    freqs = np.arange(0, f_max - f_min, float(f_max - f_min) / n_total)

    # The added linear term makes the arrival times of the highest frequencies be 0
    # Chirp function for the coherent dedispersion
    phase_filter = np.exp(
        -1j * 2 * np.pi * dm_coeff * (1 / (f_min + freqs) + freqs / (f_max**2))
    )
    if ffted:
        return np.fft.ifft(raw_signal * phase_filter)
    return np.fft.ifft(np.fft.fft(raw_signal) * phase_filter)


def stft_func(raw_signal: np.ndarray, block_size: int) -> np.ndarray:
    """Performs the Short Time Fourier Transform on the input signal.

    Parameters
    ----------
    raw_signal : np.ndarray
        Raw antenna voltage time series
    block_size : int
        Length of each block in the STFT

    Returns
    -------
    np.ndarray
        Frequency vs. time matrix. Absolute value squared is not performed!
    """
    shape = (len(raw_signal) // block_size, block_size)
    blocks = raw_signal[: np.prod(shape)].reshape(shape)
    return np.fft.fft(np.transpose(blocks), axis=0)


def bit_count(num):
    return bin(num).count("1")
