"""Train the neural HR regressors with subject-wise cross-validation.

For every fold the model is trained on the train subjects, early-stopped on the
val subjects and used to predict the test subjects. Concatenating the test
predictions of all folds gives one out-of-subject prediction per window.
"""

import argparse
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from ppghr.config import load_config
from ppghr.models import build_model, prepare_inputs, uses_imu
from ppghr.splits import load_folds
from ppghr.windows import load_windows, window_table


def train_fold(name: str, x: np.ndarray, y: np.ndarray, subjects: np.ndarray, fold: dict, cfg: dict, seed: int):
    tr = np.isin(subjects, fold["train"])
    va = np.isin(subjects, fold["val"])
    t = cfg["training"]

    keras.utils.set_random_seed(seed)
    model = build_model(name, x.shape[1:], float(y[tr].mean()), float(y[tr].std()))
    model.compile(optimizer=keras.optimizers.Adam(t["learning_rate"]), loss="mae")
    callbacks = [
        keras.callbacks.EarlyStopping("val_loss", patience=t["early_stopping_patience"], restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]
    history = model.fit(
        x[tr], y[tr],
        validation_data=(x[va], y[va]),
        epochs=t["epochs"],
        batch_size=t["batch_size"],
        callbacks=callbacks,
        verbose=0,
    )
    return model, history.history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="+", help="default: training.models from the config")
    parser.add_argument("--folds", nargs="+", type=int, help="default: all folds")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tf.config.experimental.enable_op_determinism()
    data = load_windows(cfg["windows"]["processed_path"])
    folds = load_folds(cfg["splits"]["path"])
    meta = window_table(data, folds)
    y = data["hr"].astype(np.float32)

    model_dir = Path(cfg["training"]["model_dir"])
    pred_dir = Path(cfg["evaluation"]["predictions_dir"])
    table_dir = Path(cfg["evaluation"]["tables_dir"])
    for d in (model_dir, pred_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)

    fold_ids = args.folds if args.folds is not None else range(len(folds))
    for name in args.models or cfg["training"]["models"]:
        x = prepare_inputs(data, uses_imu(name))
        preds, log = [], []
        for k in fold_ids:
            start = time.perf_counter()
            model, hist = train_fold(name, x, y, data["subject"], folds[k], cfg, cfg["seed"] + k)
            elapsed = time.perf_counter() - start
            model.save(model_dir / f"{name}_fold{k}.keras")

            te = np.isin(data["subject"], folds[k]["test"])
            hr_pred = model.predict(x[te], batch_size=512, verbose=0).ravel()
            preds.append(meta[te].assign(method=name, hr_pred=hr_pred))

            best = int(np.argmin(hist["val_loss"]))
            log.append({
                "model": name, "fold": k, "params": model.count_params(),
                "epochs": len(hist["val_loss"]), "best_epoch": best + 1,
                "val_mae": hist["val_loss"][best],
                "test_mae": float(np.mean(np.abs(hr_pred - y[te]))),
                "train_time_s": round(elapsed, 1),
            })
            print(f"{name} fold {k}: val MAE {log[-1]['val_mae']:.2f}, test MAE {log[-1]['test_mae']:.2f} "
                  f"({log[-1]['epochs']} epochs, {elapsed:.0f} s)", flush=True)

        pd.concat(preds).to_csv(pred_dir / f"{name}.csv", index=False)
        pd.DataFrame(log).to_csv(table_dir / f"training_{name}.csv", index=False)


if __name__ == "__main__":
    main()
