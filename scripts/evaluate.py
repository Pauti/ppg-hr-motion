"""Compare all methods on the same windows and write tables and figures.

DSP methods cover all windows; neural models cover the test subjects of the
folds they were trained on. Only windows predicted by every method are compared.
"""

import argparse
from pathlib import Path

import pandas as pd

from ppghr.config import load_config
from ppghr.metrics import grouped_metrics
from ppghr.plotting import ACTIVITIES, plt, save
from ppghr.quality import window_features
from ppghr.windows import load_windows

METHOD_LABELS = {
    "peak": "Peak detection",
    "spectral": "Spectral peak",
    "imu_mask": "Spectral + IMU mask",
    "cnn": "CNN",
    "cnn_imu": "CNN + IMU",
    "lstm": "LSTM",
}
LARGE_ERROR_BPM = 10
MATCH_BPM = 6


def load_predictions(pred_dir: Path) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for m in METHOD_LABELS if (p := pred_dir / f"{m}.csv").exists()])
    n_methods = df["method"].nunique()
    common = df.groupby("window_id")["method"].nunique()
    df = df[df["window_id"].isin(common[common == n_methods].index)].copy()
    df["abs_err"] = (df["hr_pred"] - df["hr_true"]).abs()
    return df


def error_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Where do large errors (>10 bpm) land during walking and running?

    at_accel_peak: estimate within 6 bpm of one of the two strongest accelerometer peaks
    at_double_hr:  estimate at 1.8-2.2x the reference (second PPG harmonic)
    hr_near_accel_peak: same test for the reference HR, i.e. the chance level
    """
    def near_accel(hr, g):
        return ((hr - g["accel_peak_bpm"]).abs() <= MATCH_BPM) | ((hr - g["accel_peak2_bpm"]).abs() <= MATCH_BPM)

    rows = []
    for (method, act), g in df[df["activity"] != "sit"].groupby(["method", "activity"]):
        large = g[g["abs_err"] > LARGE_ERROR_BPM]
        rows.append({
            "method": method,
            "activity": act,
            "large_error_share": len(large) / len(g),
            "at_accel_peak": near_accel(large["hr_pred"], large).mean(),
            "at_double_hr": (large["hr_pred"] / large["hr_true"]).between(1.8, 2.2).mean(),
            "hr_near_accel_peak": near_accel(g["hr_true"], g).mean(),
        })
    return pd.DataFrame(rows)


def sqi_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """MAE when only the windows with the highest spectral purity are reported."""
    rows = []
    for method, g in df.groupby("method"):
        g = g.sort_values("spectral_purity", ascending=False)
        for cov in (0.2, 0.5, 0.8, 1.0):
            rows.append({"method": method, "coverage": cov, "mae": g["abs_err"].iloc[: int(cov * len(g))].mean()})
    return pd.DataFrame(rows).pivot(index="method", columns="coverage", values="mae")


def plot_mae_by_activity(by_act: pd.DataFrame, methods: list[str], path: Path) -> None:
    mae = by_act.pivot(index="method", columns="activity", values="mae").loc[methods, ACTIVITIES]
    ax = mae.rename(index=METHOD_LABELS).T.plot.bar(figsize=(7.5, 3.4), width=0.8, rot=0)
    ax.set(xlabel="", ylabel="MAE (bpm)", title="HR error by activity (out-of-subject)", ylim=(0, mae.max().max() * 1.3))
    ax.legend(ncol=3, loc="upper left")
    save(ax.figure, path)


def plot_subject_mae(by_subj: pd.DataFrame, methods: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    data = [by_subj[by_subj["method"] == m]["mae"] for m in methods]
    ax.boxplot(data, tick_labels=[METHOD_LABELS[m].replace(" ", "\n", 1) for m in methods])
    ax.set(ylabel="MAE per subject (bpm)", title="Per-subject error, all activities")
    save(fig, path)


def plot_scatter(df: pd.DataFrame, methods: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(methods), figsize=(3.4 * len(methods), 3.4), sharex=True, sharey=True)
    lo, hi = df["hr_true"].min() - 5, df["hr_true"].max() + 5
    for ax, m in zip(axes, methods):
        g = df[df["method"] == m]
        for act in ACTIVITIES:
            a = g[g["activity"] == act]
            ax.scatter(a["hr_true"], a["hr_pred"], s=2, alpha=0.3, label=act)
        ax.plot([lo, hi], [lo, hi], "k-", linewidth=0.8)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), title=METHOD_LABELS[m], xlabel="Reference HR (bpm)")
    axes[0].set_ylabel("Estimated HR (bpm)")
    axes[0].legend(markerscale=5)
    save(fig, path)


def plot_timeline(df: pd.DataFrame, record: str, methods: list[str], path: Path) -> None:
    g = df[df["record"] == record]
    ref = g[g["method"] == methods[0]].sort_values("t_start")
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.plot(ref["t_start"], ref["hr_true"], "k-", linewidth=2, label="ECG reference")
    for m in methods:
        s = g[g["method"] == m].sort_values("t_start")
        ax.plot(s["t_start"], s["hr_pred"], linewidth=1.0, label=METHOD_LABELS[m])
    ax.set(xlabel="Time (s)", ylabel="HR (bpm)", title=record)
    ax.legend(ncol=4, loc="lower left")
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ev = cfg["evaluation"]
    tables, figures = Path(ev["tables_dir"]), Path(ev["figures_dir"])
    tables.mkdir(parents=True, exist_ok=True)

    data = load_windows(cfg["windows"]["processed_path"])
    df = load_predictions(Path(ev["predictions_dir"])).merge(window_features(data, cfg), on="window_id")
    methods = [m for m in METHOD_LABELS if m in set(df["method"])]

    overall = grouped_metrics(df, ["method"]).set_index("method").loc[methods]
    by_act = grouped_metrics(df, ["method", "activity"])
    by_subj = grouped_metrics(df, ["method", "subject"])
    errors = error_analysis(df)
    sqi = sqi_coverage(df).loc[methods]

    overall.to_csv(tables / "overall.csv")
    by_act.to_csv(tables / "by_activity.csv", index=False)
    by_subj.to_csv(tables / "by_subject.csv", index=False)
    grouped_metrics(df, ["method", "subject", "activity"]).to_csv(tables / "by_subject_activity.csv", index=False)
    errors.to_csv(tables / "error_analysis.csv", index=False)
    sqi.to_csv(tables / "sqi_coverage.csv")

    plot_mae_by_activity(by_act, methods, figures / "mae_by_activity.png")
    plot_subject_mae(by_subj, methods, figures / "mae_by_subject.png")
    plot_scatter(df, [m for m in ("spectral", "cnn", "lstm") if m in methods], figures / "scatter.png")

    # Example: the walking record with the median spectral-peak error (typical, not picked).
    rec_mae = df[(df["method"] == "spectral") & (df["activity"] == "walk")].groupby("record")["abs_err"].mean()
    example = rec_mae.sort_values().index[len(rec_mae) // 2]
    plot_timeline(df, example, [m for m in ("peak", "spectral", "lstm") if m in methods],
                  figures / f"timeline_{example}.png")

    pd.set_option("display.width", 160)
    print(overall.round(2), "\n")
    print(by_act.pivot(index="method", columns="activity", values="mae").loc[methods, ACTIVITIES].round(2), "\n")
    print(by_subj.groupby("method")["mae"].describe().loc[methods].round(2), "\n")
    print(errors.round(2).to_string(index=False), "\n")
    print("MAE by share of windows kept (ranked by spectral purity):\n", sqi.round(2))


if __name__ == "__main__":
    main()
