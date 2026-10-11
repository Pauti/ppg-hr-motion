"""Per-window signal-quality and motion descriptors.

These are not used as model inputs. They describe each window so that errors
can be broken down by signal quality and motion intensity.
"""

import numpy as np
import pandas as pd

from ppghr.baseline import accel_peaks_bpm, spectrum


def spectral_purity(ppg: np.ndarray, fs: float, cfg: dict, half_width_bpm: float = 6.0) -> float:
    """Share of in-band PPG power that lies around the dominant frequency.

    Close to 1 for a clean, quasi-periodic pulse; low when power is spread out
    by motion or noise.
    """
    lo, hi = cfg["baseline"]["hr_range_bpm"]
    f, p = spectrum(ppg, fs, cfg["baseline"]["nfft"])
    band = (f >= lo) & (f <= hi)
    f, p = f[band], p[band]
    peak = f[np.argmax(p)]
    return float(p[np.abs(f - peak) <= half_width_bpm].sum() / (p.sum() + 1e-12))


def window_features(data: dict[str, np.ndarray], cfg: dict) -> pd.DataFrame:
    fs = cfg["preprocessing"]["fs_target"]
    acc_mag = np.linalg.norm(data["acc"], axis=2)
    rows = []
    for ppg, acc in zip(data["ppg"], data["acc"]):
        peaks = accel_peaks_bpm(acc, fs, cfg)
        rows.append({
            "spectral_purity": spectral_purity(ppg, fs, cfg),
            "accel_peak_bpm": float(peaks[0]) if len(peaks) > 0 else np.nan,
            "accel_peak2_bpm": float(peaks[1]) if len(peaks) > 1 else np.nan,
        })
    df = pd.DataFrame(rows)
    df.insert(0, "window_id", data["window_id"])
    df["motion"] = acc_mag.std(axis=1)
    return df
