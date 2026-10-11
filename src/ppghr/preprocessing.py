import numpy as np
from scipy import ndimage, signal


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the first axis.

    Zero-phase filtering is non-causal. That is fine for offline evaluation;
    a streaming implementation would use the causal `sosfilt` and accept the
    group delay.
    """
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def decimate(x: np.ndarray, fs: float, fs_target: float) -> np.ndarray:
    factor = fs / fs_target
    if not float(factor).is_integer():
        raise ValueError(f"fs={fs} is not an integer multiple of fs_target={fs_target}")
    return signal.resample_poly(x, up=1, down=int(factor), axis=0)


def zscore(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / (sd + eps)


def clean_ppg(ppg: np.ndarray, fs: float, cfg: dict, invert: bool) -> np.ndarray:
    """Raw PPG -> band-passed, decimated pulse signal (systolic peaks pointing up)."""
    p = cfg["preprocessing"]
    x = -ppg if invert else ppg
    x = bandpass(x - np.median(x), fs, *p["bandpass_hz"], order=p["filter_order"])
    return decimate(x, fs, p["fs_target"])


def clean_accel(acc: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Raw 3-axis accelerometer -> despiked, decimated signal (gravity kept).

    The dataset notes occasional single-sample IMU glitches; a short median
    filter removes them before the anti-aliasing in `decimate`.
    """
    acc = ndimage.median_filter(acc, size=(5, 1))
    return decimate(acc, fs, cfg["preprocessing"]["fs_target"])
