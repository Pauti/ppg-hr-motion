"""Run the training-free DSP baselines on every window.

The baselines have no fitted parameters, so they are applied to all subjects.
Their settings were fixed from the literature and a look at the signals, not
tuned on test errors.
"""

import argparse
from pathlib import Path

import pandas as pd

from ppghr.baseline import METHODS
from ppghr.config import load_config
from ppghr.metrics import grouped_metrics
from ppghr.splits import load_folds
from ppghr.windows import load_windows, window_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    data = load_windows(cfg["windows"]["processed_path"])
    meta = window_table(data, load_folds(cfg["splits"]["path"]))
    fs = cfg["preprocessing"]["fs_target"]

    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, fn in METHODS.items():
        pred = [fn(p, a, fs, cfg) for p, a in zip(data["ppg"], data["acc"])]
        df = meta.assign(method=name, hr_pred=pred)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        frames.append(df)

    summary = grouped_metrics(pd.concat(frames), ["method", "activity"])
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
