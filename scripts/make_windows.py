"""Filter, decimate and window all records; attach ECG-derived reference HR."""

import argparse
from pathlib import Path

import pandas as pd

from ppghr.config import load_config
from ppghr.io import load_records
from ppghr.splits import make_folds, save_folds
from ppghr.windows import build_windows, save_windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    records = load_records(cfg["dataset"]["raw_dir"])
    data, summary = build_windows(records, cfg)
    save_windows(data, cfg["windows"]["processed_path"])

    folds = make_folds(data["subject"].tolist(), cfg["splits"]["n_folds"], cfg["seed"])
    save_folds(folds, cfg["splits"]["path"])

    summary = pd.DataFrame(summary)
    out = Path(cfg["evaluation"]["tables_dir"]) / "window_counts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    total, kept = summary["windows"].sum(), summary["kept"].sum()
    print(f"{kept}/{total} windows kept ({kept / total:.1%}), {len(set(data['subject']))} subjects")
    worst = summary.assign(frac=summary["kept"] / summary["windows"]).nsmallest(5, "frac")
    print("Records with the most rejected windows:\n" + worst.to_string(index=False))


if __name__ == "__main__":
    main()
