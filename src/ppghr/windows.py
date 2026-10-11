from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from ppghr.io import Record
from ppghr.labels import reference_hr
from ppghr.preprocessing import clean_accel, clean_ppg


def record_windows(rec: Record, cfg: dict) -> dict[str, np.ndarray]:
    fs_t = cfg["preprocessing"]["fs_target"]
    win_cfg = cfg["windows"]
    n = int(round(win_cfg["length_s"] * fs_t))
    step = int(round(win_cfg["step_s"] * fs_t))

    sig = cfg["signals"]
    ppg = clean_ppg(rec.signals[sig["ppg"]].to_numpy(), rec.fs, cfg, sig["invert_ppg"])
    acc = clean_accel(rec.signals[sig["accel"]].to_numpy(), rec.fs, cfg)
    r_s = rec.r_peaks / rec.fs

    starts = np.arange(0, len(ppg) - n + 1, step)
    hr = np.array([reference_hr(r_s, s / fs_t, (s + n) / fs_t, cfg) for s in starts])
    return {
        "ppg": np.stack([ppg[s:s + n] for s in starts]).astype(np.float32),
        "acc": np.stack([acc[s:s + n] for s in starts]).astype(np.float32),
        "hr": hr.astype(np.float32),
        "t_start": (starts / fs_t).astype(np.float32),
        "subject": np.full(len(starts), rec.subject, dtype=np.int16),
        "activity": np.full(len(starts), rec.activity),
        "record": np.full(len(starts), rec.name),
    }


def build_windows(records: Iterable[Record], cfg: dict) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Window all records; windows without a valid ECG reference are dropped."""
    parts, summary = [], []
    for rec in records:
        w = record_windows(rec, cfg)
        valid = np.isfinite(w["hr"])
        summary.append({"record": rec.name, "windows": len(valid), "kept": int(valid.sum())})
        parts.append({k: v[valid] for k, v in w.items()})
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    data["window_id"] = np.arange(len(data["hr"]), dtype=np.int32)
    return data, summary


def window_table(data: dict[str, np.ndarray], folds: list[dict] | None = None) -> pd.DataFrame:
    """Per-window metadata; `fold` is the fold in which the subject is in the test set."""
    df = pd.DataFrame({k: data[k] for k in ("window_id", "record", "subject", "activity", "t_start")})
    df["hr_true"] = data["hr"]
    if folds is not None:
        test_fold = {s: k for k, f in enumerate(folds) for s in f["test"]}
        df["fold"] = df["subject"].map(test_fold)
    return df


def save_windows(data: dict[str, np.ndarray], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_windows(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
