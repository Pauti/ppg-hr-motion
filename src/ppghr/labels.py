import numpy as np


def reference_hr(r_peaks_s: np.ndarray, t0: float, t1: float, cfg: dict) -> float:
    """Mean HR (bpm) from the ECG RR intervals fully contained in [t0, t1).

    Returns NaN if the window does not yield a trustworthy reference: too few
    beats, physiologically implausible RR intervals, or an RR interval far from
    the window median (usually a missed or spurious annotation).
    """
    lab = cfg["labels"]
    beats = r_peaks_s[(r_peaks_s >= t0) & (r_peaks_s < t1)]
    if len(beats) < 3:
        return np.nan
    rr = np.diff(beats)
    if rr.min() < lab["min_rr_s"] or rr.max() > lab["max_rr_s"]:
        return np.nan
    med = np.median(rr)
    if np.max(np.abs(rr - med)) / med > lab["max_rr_deviation"]:
        return np.nan
    return 60.0 / rr.mean()
