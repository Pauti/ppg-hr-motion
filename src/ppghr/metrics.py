import numpy as np
import pandas as pd


def hr_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE/RMSE/bias in bpm over windows with an estimate; `coverage` is that share."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_pred)
    err = y_pred[ok] - y_true[ok]
    return {
        "n": int(ok.sum()),
        "coverage": float(ok.mean()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "within_5bpm": float(np.mean(np.abs(err) <= 5)),
    }


def grouped_metrics(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Metrics per group; `df` needs columns hr_true, hr_pred and the `by` columns."""
    rows = [
        {**dict(zip(by, key if isinstance(key, tuple) else (key,))), **hr_metrics(g["hr_true"], g["hr_pred"])}
        for key, g in df.groupby(by, sort=True)
    ]
    return pd.DataFrame(rows)
