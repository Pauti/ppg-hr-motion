"""Export a trained model to TFLite and check what quantization costs.

Three variants are produced: float32, dynamic-range quantized (int8 weights,
float activations) and full-integer int8 (int8 weights, activations and I/O).
The int8 calibration set is drawn from the fold's *training* subjects only.
Each variant is run on the fold's test subjects and compared with the Keras
model. Latency is measured on the desktop CPU with one thread; it says nothing
precise about a microcontroller, but shows relative cost between variants.
"""

import argparse
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from ppghr.config import load_config
from ppghr.models import prepare_inputs, uses_imu
from ppghr.splits import load_folds
from ppghr.windows import load_windows

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # older TensorFlow installs ship the interpreter themselves
    Interpreter = tf.lite.Interpreter


VARIANTS = ("float32", "dynamic_range", "int8")


def count_macs(model: keras.Model) -> int:
    """Multiply-accumulates per inference for Conv1D/Dense/LSTM layers."""
    macs = 0
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv1D):
            _, length, filters = layer.output.shape
            macs += length * layer.kernel_size[0] * layer.input.shape[-1] * filters
        elif isinstance(layer, keras.layers.Dense):
            macs += layer.input.shape[-1] * layer.units
        elif isinstance(layer, keras.layers.LSTM):
            _, steps, n_in = layer.input.shape
            macs += steps * 4 * (n_in + layer.units) * layer.units
    return int(macs)


def convert(model: keras.Model, variant: str, rep_x: np.ndarray) -> bytes:
    # Fixed batch size 1, as on a device. A dynamic batch dimension prevents
    # the converter from fusing the LSTM into a builtin TFLite op.
    inp = keras.Input(rep_x.shape[1:], batch_size=1)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras.Model(inp, model(inp)))
    if variant in ("dynamic_range", "int8"):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if variant == "int8":
        converter.representative_dataset = lambda: ([x[None]] for x in rep_x)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    return converter.convert()


def run_tflite(model_bytes: bytes, x: np.ndarray, latency_runs: int) -> tuple[np.ndarray, float]:
    interp = Interpreter(model_content=model_bytes, num_threads=1)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    def quantize(v: np.ndarray) -> np.ndarray:
        if inp["dtype"] == np.int8:
            scale, zero = inp["quantization"]
            return np.clip(np.round(v / scale + zero), -128, 127).astype(np.int8)
        return v.astype(np.float32)

    def dequantize(v: np.ndarray) -> float:
        if out["dtype"] == np.int8:
            scale, zero = out["quantization"]
            return float((v.astype(np.float32) - zero) * scale)
        return float(v)

    preds = np.empty(len(x), dtype=np.float32)
    for i, window in enumerate(x):
        interp.set_tensor(inp["index"], quantize(window[None]))
        interp.invoke()
        preds[i] = dequantize(interp.get_tensor(out["index"]).ravel()[0])

    sample = quantize(x[:1])
    start = time.perf_counter()
    for _ in range(latency_runs):
        interp.set_tensor(inp["index"], sample)
        interp.invoke()
    latency_ms = (time.perf_counter() - start) / latency_runs * 1000
    return preds, latency_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", help="default: export.model from the config")
    parser.add_argument("--fold", type=int, help="default: export.fold from the config")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ex = cfg["export"]
    name = args.model or ex["model"]
    k = ex["fold"] if args.fold is None else args.fold

    data = load_windows(cfg["windows"]["processed_path"])
    fold = load_folds(cfg["splits"]["path"])[k]
    x = prepare_inputs(data, uses_imu(name))
    y = data["hr"]
    tr = np.flatnonzero(np.isin(data["subject"], fold["train"]))
    te = np.isin(data["subject"], fold["test"])

    rng = np.random.default_rng(cfg["seed"])
    rep_x = x[rng.choice(tr, size=min(ex["representative_windows"], len(tr)), replace=False)]

    model = keras.models.load_model(Path(cfg["training"]["model_dir"]) / f"{name}_fold{k}.keras")
    keras_pred = model.predict(x[te], batch_size=512, verbose=0).ravel()

    out_dir = Path(ex["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{
        "variant": "keras (reference)", "size_kb": np.nan,
        "test_mae": float(np.mean(np.abs(keras_pred - y[te]))),
        "max_abs_diff_vs_keras": 0.0, "mean_abs_diff_vs_keras": 0.0, "latency_ms": np.nan,
    }]
    for variant in args.variants:
        model_bytes = convert(model, variant, rep_x)
        path = out_dir / f"{name}_fold{k}_{variant}.tflite"
        path.write_bytes(model_bytes)
        pred, latency = run_tflite(model_bytes, x[te], ex["latency_runs"])
        diff = np.abs(pred - keras_pred)
        rows.append({
            "variant": variant,
            "size_kb": len(model_bytes) / 1024,
            "test_mae": float(np.mean(np.abs(pred - y[te]))),
            "max_abs_diff_vs_keras": float(diff.max()),
            "mean_abs_diff_vs_keras": float(diff.mean()),
            "latency_ms": latency,
        })

    table = pd.DataFrame(rows)
    table_dir = Path(cfg["evaluation"]["tables_dir"])
    table_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_dir / f"tflite_{name}.csv", index=False)

    print(f"Model {name}, fold {k}: {model.count_params():,} parameters, "
          f"{count_macs(model) / 1e6:.2f} M MACs per {cfg['windows']['length_s']:.0f} s window, "
          f"input {tuple(x.shape[1:])}, {int(te.sum())} test windows")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
