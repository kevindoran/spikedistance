"""
Functions to calculate various metrics. Intended to be (mis)used with Polars' 
map functionality.
"""

import math
import warnings
import einops
import numpy as np
import scipy


def _is_constant(arr):
    return (arr == arr[0]).all()

def pcorr(actual, pred):
    """
    Pearson correlation.
    """
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if _is_constant(pred) or _is_constant(actual):
        # Undefined, but we will use 0. This is reasonable as we expect
        # correlations to be non-negative.
        pcorr = 0.0
    else:
        pcorr = scipy.stats.pearsonr(pred, actual)[0]
    return pcorr


def smooth_pcorr(signal_a, signal_b, bin_ms: float, sigma_ms: float):
    """
    Smoothed Pearson correlation.

    Gaussian-smooth two signals then calculate their Pearson correlation."""
    signal_a = np.asarray(signal_a, dtype=float)
    signal_b = np.asarray(signal_b, dtype=float)
    if sigma_ms == 0:
        smooth_a = signal_a
        smooth_b = signal_b
    else:
        sigma = sigma_ms / bin_ms
        smooth_a = scipy.ndimage.gaussian_filter1d(signal_a, sigma)
        smooth_b = scipy.ndimage.gaussian_filter1d(signal_b, sigma)
    if _is_constant(smooth_a) or _is_constant(smooth_b):
        # Undefined, but we will use 0. This is reasonable as we expect
        # correlations to be non-negative.
        pcorr = 0.0
    else:
        pcorr = scipy.stats.pearsonr(smooth_a, smooth_b)[0]
    return pcorr


def binned_pcorr(actual, pred, num_bins: int):
    """
    Binned Pearson correlation.

    Here, we don't take bin_ms and some argument like total_ms. If we did,
    the caller would have to redo the calculation anyway in order to determine
    how many bins & ms were used.
    """
    remainder = len(actual) % num_bins
    if remainder != 0:
        actual = actual[:-remainder]
        pred = pred[:-remainder]
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    actual = einops.rearrange(actual, "(t bin) -> t bin", bin=num_bins)
    pred = einops.rearrange(pred, "(t bin) -> t bin", bin=num_bins)
    actual = np.sum(actual, axis=1)
    pred = np.sum(pred, axis=1)
    if _is_constant(pred) or _is_constant(actual):
        # Undefined, but we will use 0. This is reasonable as we expect
        # correlations to be non-negative.
        pcorr = 0.0
    else:
        pcorr = scipy.stats.pearsonr(pred, actual)[0]
    return pcorr


def van_rossum(signal_a, signal_b, bin_ms, tau_ms):
    """
    Van Rossum distance between two signals.

    Quote from <em>On the efficient calculation of van Rossum distances</em>:

    <blockquote>
        "..where the normalizing factor of 2/τ is included so that there is a
        distance of one between a spike train with a single spike and one with
        no spikes."
    </blockquote>
    """
    if len(signal_a) != len(signal_b):
        raise ValueError("signal_a and signal_b must have same shape")
    signal_a = np.asarray(signal_a, dtype=float)
    signal_b = np.asarray(signal_b, dtype=float)
    # Other metrics allow zero, so adding support here too.
    # Making tau to be at least 0.25 of a bin.
    tau = max(tau_ms / bin_ms, 0.25)
    kernel_len = 2000
    half_ker = np.exp(-np.arange(kernel_len // 2 + 1) / tau)
    ker = np.concatenate([np.zeros(kernel_len // 2), half_ker])
    # Convolve the two signals with the kernel
    conv_a = scipy.signal.convolve(signal_a, ker, mode="same")
    conv_b = scipy.signal.convolve(signal_b, ker, mode="same")
    # Squared and unscalled version
    # res = (1 / tau) * np.sum((conv_a - conv_b) ** 2)
    # Sqrt and scaled version, from Houghton et al. 2012
    res = ((2 / tau) * np.sum((conv_a - conv_b) ** 2)) ** 0.5
    return res


def inv_van_rossum(signal_a, signal_b, bin_ms, tau_ms):
    """
    Inverse Van Rossum distance between two signals.
    """
    return 1 / van_rossum(signal_a, signal_b, bin_ms, tau_ms)


def schreiber(signal_a, signal_b, bin_ms, sigma_ms):
    """Schreiber correlation for two signals."""
    signal_a = np.asarray(signal_a, dtype=float)
    signal_b = np.asarray(signal_b, dtype=float)
    if sigma_ms == 0:
        smooth_a = signal_a
        smooth_b = signal_b
    else:
        sigma = sigma_ms / bin_ms
        # Pad input spike train zeros. Don't use default reflect.
        smooth_a = scipy.ndimage.gaussian_filter1d(
            signal_a, sigma, mode="constant", cval=0
        )
        smooth_b = scipy.ndimage.gaussian_filter1d(
            signal_b, sigma, mode="constant", cval=0
        )
    # correlate then dot product.
    norm_a = np.linalg.norm(smooth_a)
    norm_b = np.linalg.norm(smooth_b)
    if norm_a == 0 or norm_b == 0:
        res = 0.0
    else:
        res = np.dot(smooth_a, smooth_b) / (norm_a * norm_b)
    return res


def psnr(actual_signal, pred_signal):
    actual_signal = np.asarray(actual_signal, dtype=float)
    pred_signal = np.asarray(pred_signal, dtype=float)
    mse = np.mean((actual_signal - pred_signal) ** 2)
    max_val = np.max(actual_signal)
    if mse == 0:
        res = 0
    elif max_val == 0:
        res = 100
    else:
        res = 20 * math.log10(max_val / math.sqrt(mse))
    return res


def smooth_psnr(
    actual_signal: np.ndarray,
    pred_signal: np.ndarray,
    eval_ms: float,
    sigma_ms: float,
):
    sigma = sigma_ms / eval_ms
    # Without setting output=float, gaussian_filter1d returns an array that
    # is the same type as the input array, possibly int.
    # Alternative is to give input as float.
    actual_signal = np.asarray(actual_signal, dtype=float)
    pred_signal = np.asarray(pred_signal, dtype=float)
    smooth_actual = scipy.ndimage.gaussian_filter1d(actual_signal, sigma)
    smooth_pred = scipy.ndimage.gaussian_filter1d(pred_signal, sigma)
    return psnr(smooth_actual, smooth_pred)


def mse(signal_a, signal_b):
    signal_a = np.asarray(signal_a, dtype=float)
    signal_b = np.asarray(signal_b, dtype=float)
    return np.mean((signal_a - signal_b) ** 2)


def smooth_mse(signal_a, signal_b, bin_ms: float, sigma_ms: float):
    sigma = sigma_ms / bin_ms
    signal_a = np.asarray(signal_a, dtype=float)
    signal_b = np.asarray(signal_b, dtype=float)
    smooth_a = scipy.ndimage.gaussian_filter1d(signal_a, sigma)
    smooth_b = scipy.ndimage.gaussian_filter1d(signal_b, sigma)
    return mse(smooth_a, smooth_b)


