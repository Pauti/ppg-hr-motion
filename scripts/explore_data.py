"""Exploratory figures and a per-activity summary of the windowed data."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ppghr.baseline import spectrum
from ppghr.config import load_config
from ppghr.io import load_records
from ppghr.plotting import ACTIVITIES, plt, save
from ppghr.preprocessing import clean_accel, clean_ppg, zscore
from ppghr.windows import load_windows

SUBJECT = 1
T0_S = 120.0


def plot_signals(cfg: dict, path: Path) -> None:
    """8 s of PPG and accelerometer for one subject, one column per activity."""
    recs = {r.activity: r for r in load_records(cfg["dataset"]["raw_dir"], [f"s{SUBJECT}_{a}" for a in ACTIVITIES])}
    sig = cfg["signals"]
    fs = cfg["preprocessing"]["fs_target"]
    i0, i1 = int(T0_S * fs), int((T0_S + 8) * fs)
    t = np.arange(i1 - i0) / fs

    fig, axes = plt.subplots(2, 3, figsize=(10, 4.2), sharex=True, sharey="row")
    for col, act in enumerate(ACTIVITIES):
        rec = recs[act]
        ppg = clean_ppg(rec.signals[sig["ppg"]].to_numpy(), rec.fs, cfg, sig["invert_ppg"])[i0:i1]
        acc = clean_accel(rec.signals[sig["accel"]].to_numpy(), rec.fs, cfg)[i0:i1]
        r = rec.r_peaks / rec.fs - T0_S
        axes[0, col].plot(t, zscore(ppg))
        for rt in r[(r >= 0) & (r < 8)]:
            axes[0, col].axvline(rt, color="grey", linewidth=0.6, linestyle=":")
        axes[0, col].set_title(f"s{SUBJECT}_{act}")
        mag = np.linalg.norm(acc, axis=1)
        axes[1, col].plot(t, mag - mag.mean(), color="C1")
        axes[1, col].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("PPG, band-passed (z)")
    axes[1, 0].set_ylabel("|a| - mean (m/s²)")
    fig.suptitle("Green PPG (dotted: ECG R-peaks) and accelerometer magnitude")
    save(fig, path)


def plot_spectra(data: dict, cfg: dict, path: Path) -> None:
    """PPG vs. accelerometer spectrum of one window per activity."""
    fs = cfg["preprocessing"]["fs_target"]
    lo, hi = cfg["baseline"]["hr_range_bpm"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    for ax, act in zip(axes, ACTIVITIES):
        idx = np.flatnonzero((data["subject"] == SUBJECT) & (data["activity"] == act) & (data["t_start"] >= T0_S))[0]
        f, p_ppg = spectrum(data["ppg"][idx], fs, cfg["baseline"]["nfft"])
        _, p_acc = spectrum(data["acc"][idx].T, fs, cfg["baseline"]["nfft"])
        p_acc = p_acc.sum(axis=0)
        band = (f >= lo) & (f <= hi)
        ax.plot(f[band], p_ppg[band] / p_ppg[band].max(), label="PPG")
        ax.plot(f[band], p_acc[band] / p_acc[band].max(), label="Accelerometer")
        ax.axvline(data["hr"][idx], color="k", linestyle="--", label="ECG HR")
        ax.set(title=act, xlabel="Frequency (bpm)")
    axes[0].set_ylabel("Normalised power")
    axes[2].legend(loc="upper right")
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    figures = Path(cfg["evaluation"]["figures_dir"])
    tables = Path(cfg["evaluation"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)

    data = load_windows(cfg["windows"]["processed_path"])
    df = pd.DataFrame({
        "subject": data["subject"],
        "activity": data["activity"],
        "hr": data["hr"],
        "motion": np.linalg.norm(data["acc"], axis=2).std(axis=1),
    })
    summary = df.groupby("activity").agg(
        windows=("hr", "size"),
        subjects=("subject", "nunique"),
        hr_mean=("hr", "mean"),
        hr_std=("hr", "std"),
        hr_min=("hr", "min"),
        hr_max=("hr", "max"),
        motion_median=("motion", "median"),
    ).loc[ACTIVITIES]
    summary.to_csv(tables / "eda_summary.csv")
    print(summary.round(2).to_string())

    plot_signals(cfg, figures / "eda_signals.png")
    plot_spectra(data, cfg, figures / "eda_spectra.png")


if __name__ == "__main__":
    main()
