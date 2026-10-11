"""Classic, training-free HR estimators operating on one window at a time.

peak      time domain: systolic peak detection, HR from the median inter-beat interval
spectral  frequency domain: dominant PPG frequency in the HR band
imu_mask  spectral, but PPG spectral peaks that coincide with the dominant
          accelerometer frequencies are discarded first (a stripped-down
          version of the motion-peak exclusion used in TROIKA-style methods)
"""

import numpy as np
from scipy import signal

from ppghr.preprocessing import zscore


def peak_hr(ppg: np.ndarray, fs: float, cfg: dict) -> float:
    b = cfg["baseline"]
    lo, hi = b["hr_range_bpm"]
    x = zscore(ppg)
    peaks, _ = signal.find_peaks(x, distance=max(1, int(fs * 60 / hi)), prominence=b["peak_prominence"])
    if len(peaks) < 3:
        return np.nan
    hr = 60.0 / np.median(np.diff(peaks) / fs)
    return hr if lo <= hr <= hi else np.nan


def spectrum(x: np.ndarray, fs: float, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed periodogram along the last axis; returns (freqs_bpm, power)."""
    f, p = signal.periodogram(x, fs=fs, window="hann", nfft=nfft, detrend="constant", axis=-1)
    return f * 60.0, p


def _band(f_bpm: np.ndarray, cfg: dict) -> np.ndarray:
    lo, hi = cfg["baseline"]["hr_range_bpm"]
    return (f_bpm >= lo) & (f_bpm <= hi)


def spectral_hr(ppg: np.ndarray, fs: float, cfg: dict) -> float:
    f, p = spectrum(ppg, fs, cfg["baseline"]["nfft"])
    band = _band(f, cfg)
    return float(f[band][np.argmax(p[band])])


def accel_peaks_bpm(acc: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Frequencies (bpm) of the strongest accelerometer peaks in the HR band."""
    b = cfg["baseline"]
    f, p = spectrum(acc.T, fs, b["nfft"])
    p = p.sum(axis=0)
    band = _band(f, cfg)
    f, p = f[band], p[band]
    idx, _ = signal.find_peaks(p)
    idx = idx[np.argsort(p[idx])[::-1][: b["imu_mask_peaks"]]]
    return f[idx]


def imu_mask_hr(ppg: np.ndarray, acc: np.ndarray, fs: float, cfg: dict) -> float:
    """Strongest PPG spectral peak that does not coincide with an accelerometer peak.

    Whole spectral peaks are discarded rather than zeroing a fixed band: with
    8 s windows the Hann main lobe is about +-15 bpm wide, so a narrow band
    mask leaves the flanks of a motion peak behind and they still win.
    """
    b = cfg["baseline"]
    if np.std(np.linalg.norm(acc, axis=1)) < b["imu_motion_threshold"]:
        return spectral_hr(ppg, fs, cfg)
    f, p = spectrum(ppg, fs, b["nfft"])
    band = _band(f, cfg)
    f, p = f[band], p[band]
    motion = accel_peaks_bpm(acc, fs, cfg)
    idx, _ = signal.find_peaks(p)
    keep = [i for i in idx if np.all(np.abs(f[i] - motion) > b["imu_mask_bpm"])]
    if not keep:
        return spectral_hr(ppg, fs, cfg)
    return float(f[max(keep, key=lambda i: p[i])])


METHODS = {
    "peak": lambda ppg, acc, fs, cfg: peak_hr(ppg, fs, cfg),
    "spectral": lambda ppg, acc, fs, cfg: spectral_hr(ppg, fs, cfg),
    "imu_mask": imu_mask_hr,
}
