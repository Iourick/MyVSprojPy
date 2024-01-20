import numpy as np
import logging
from matplotlib import pyplot as plt

#from fdmt import core, utils
import core, utils

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def generate_signal(n_f=256, n_t=2048, nbins=40, pulse_sig=0.4):
    size = n_f * n_t * nbins
    pulse_pos = size // 2
    pulse_len = n_f * nbins
    signal = np.random.normal(0, 1, size)
    signal[pulse_pos : pulse_pos + pulse_len] += np.random.normal(0, pulse_sig, pulse_len)
    max_snr = np.sum(
        np.abs(signal[pulse_pos : pulse_pos + pulse_len]) ** 2 - np.mean(abs(signal) ** 2)
    ) / (np.sqrt(pulse_len * np.var(abs(signal) ** 2)))
    logger.info(f"MAX Thoretical S/N: {max_snr}")
    return signal


def test_fdmt_basic(ii, n_d=1024, n_f=1024, f_min=1200, f_max=1600, data_type=np.int64):
    waterfall = np.ones([n_f, n_f * ii], data_type)
    return core.fdmt(waterfall, f_min, f_max, n_d, data_type)


def test_fdmt(
    f_min=1200,
    f_max=1600,
    n_f=256,
    n_t=2048,
    nbins=40,
    pulse_sig=0.4,
    dm=5,
    data_type=np.int64,
):
    max_dt = n_t
    logger.info("Signal preparation")
    signal = generate_signal(n_f=n_f, n_t=n_t, nbins=nbins, pulse_sig=pulse_sig)
    logger.info(f"signal shape: n_f={n_f}, n_t={n_t}, nbins={nbins}")
    dedisp = utils.coherent_dedisp(signal, -dm, f_min, f_max, ffted=False)
    stft = np.abs(utils.stft_func(dedisp, n_f * nbins)) ** 2
    stft = np.sum(stft.reshape(n_f, nbins, n_t), axis=1)
    logger.info(f"stft shape = {stft.shape}")
    stft_norm = (stft - np.mean(stft[:, :10])) / (0.25 * np.std(stft[:, :10]))
    stft_ones = np.ones_like(stft_norm)

    logger.info("Applying FDMT")
    fdmt_dm0 = core.fdmt(stft_ones, f_min, f_max, max_dt, data_type)
    fdmt_dm = core.fdmt(stft_norm, f_min, f_max, max_dt, data_type)
    res = fdmt_dm / np.sqrt(fdmt_dm0 * np.var(stft_norm[:, :10]) + 0.000001)
    logger.info(f"Maximum acieved SNR: {np.max(res)}")
    logger.info(f"Maximum Position: {np.unravel_index(np.argmax(res), res.shape)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(stft_norm, aspect="auto")
    ax2.imshow(res, aspect="auto")
    plt.show()


def test_hitting_efficiency(
    nsamples=1024, nchans=1024, n_d=1024, f_min=400, f_max=800, data_type=np.int64
):
    thickness = 2
    pulse_pos = 900
    df = (f_max - f_min) / nchans
    freqs_grid, time_grid = np.meshgrid(
        np.arange(f_min, f_max, df), np.arange(nsamples), indexing="ij"
    )
    arr_sig = np.zeros((nchans, nsamples), data_type)
    signal = time_grid - pulse_pos + 4 * 40 * 1000000 * (f_min**-2 - freqs_grid**-2)
    arr_sig[signal**2 < (thickness / 2) ** 2] = 1
    signal = time_grid - pulse_pos + 4 * 39 * 1000000 * (f_min**-2 - freqs_grid**-2)
    arr_sig[signal**2 < (thickness / 2) ** 2] = 2
    signal = time_grid - pulse_pos + 4 * 41 * 1000000 * (f_min**-2 - freqs_grid**-2)
    arr_sig[signal**2 < (thickness / 2) ** 2] = 3
    arr_ones = np.ones_like(arr_sig)

    fdmt_dm0 = core.fdmt(arr_ones, f_min, f_max, n_d, data_type)
    fdmt_dm = core.fdmt(arr_sig, f_min, f_max, n_d, data_type)
    eff = fdmt_dm / (fdmt_dm0 + 0.0000000001)
    hitting_efficiency = eff.max() / 3
    logger.info(f"hitting_efficiency: {hitting_efficiency}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    ax1.imshow(fdmt_dm0, aspect="auto", extent=[0, nsamples, f_min, f_max])
    ax2.imshow(arr_sig, aspect="auto", extent=[0, nsamples, f_min, f_max])
    ax3.imshow(fdmt_dm, aspect="auto", extent=[0, nsamples, 0, n_d])
    plt.show()


def test_fdmt_hybrid(
    f_min=1200,
    f_max=1600,
    n_f=128,
    n_t=2048,
    pulse_sig=2,
    n_p=128,
    d_max=1.5,
):
    logger.info("Signal preparation")
    signal = generate_signal(n_f=n_f, n_t=n_t, nbins=1, pulse_sig=pulse_sig)

    # Dispersion is just like dedispersion with a minus sign...
    raw_signal = utils.coherent_dedisp(signal, -1, f_min, f_max, ffted=False)

    arr_ones = np.ones([n_p, len(raw_signal) // n_p])
    fdmt_dm0 = core.fdmt(arr_ones, f_min, f_max, n_p, np.int64)
    fdmt_output = core.hybrid_dedisp(raw_signal, f_min, f_max, n_p, d_max, np.int64)

    sigma_arr = np.array(
        [np.max(fdmt_dm / np.sqrt(fdmt_dm0 + 0.000001)) for fdmt_dm in fdmt_output]
    )
    sigma_best_idx = np.argmax(sigma_arr)
    logger.info(f"achieved best score with : {sigma_arr[sigma_best_idx]} sigmas")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(fdmt_output[sigma_best_idx], aspect="auto")
    plt.show()

test_fdmt_hybrid()