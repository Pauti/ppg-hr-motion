"""Small HR regressors, kept TFLite-friendly (no custom ops, fixed input length).

The target is standardised with train-fold statistics; the inverse transform is
part of the model (final Rescaling layer), so the exported model outputs bpm.
"""

import keras
import numpy as np
from keras import layers

from ppghr.preprocessing import zscore

GRAVITY = 9.81


def prepare_inputs(data: dict[str, np.ndarray], use_imu: bool) -> np.ndarray:
    """Per-window normalisation only, so nothing is fitted on the training set.

    PPG is z-scored per window. Acceleration keeps its amplitude (motion
    intensity is informative) and is only mean-removed and expressed in g.
    """
    ppg = zscore(data["ppg"], axis=1)[..., None]
    if not use_imu:
        return ppg.astype(np.float32)
    acc = data["acc"] - data["acc"].mean(axis=1, keepdims=True)
    return np.concatenate([ppg, acc / GRAVITY], axis=-1).astype(np.float32)


def _conv_block(x, filters: int, kernel: int, pool: bool = True):
    x = layers.Conv1D(filters, kernel, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return layers.MaxPooling1D(2)(x) if pool else x


def build_cnn(input_shape: tuple[int, int], hr_mean: float, hr_std: float) -> keras.Model:
    inp = keras.Input(input_shape, name="window")
    x = _conv_block(inp, 16, 7)
    x = _conv_block(x, 32, 7)
    x = _conv_block(x, 32, 5)
    x = _conv_block(x, 64, 5, pool=False)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1)(x)
    out = layers.Rescaling(hr_std, offset=hr_mean, name="hr_bpm")(x)
    return keras.Model(inp, out, name="cnn")


def build_lstm(input_shape: tuple[int, int], hr_mean: float, hr_std: float) -> keras.Model:
    inp = keras.Input(input_shape, name="window")
    x = layers.AveragePooling1D(4)(inp)          # 50 Hz -> 12.5 Hz, 100 steps
    x = layers.LSTM(32)(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(1)(x)
    out = layers.Rescaling(hr_std, offset=hr_mean, name="hr_bpm")(x)
    return keras.Model(inp, out, name="lstm")


MODELS = {
    "cnn": (build_cnn, False),
    "cnn_imu": (build_cnn, True),
    "lstm": (build_lstm, False),
}


def build_model(name: str, input_shape: tuple[int, int], hr_mean: float, hr_std: float) -> keras.Model:
    builder, _ = MODELS[name]
    return builder(input_shape, hr_mean, hr_std)


def uses_imu(name: str) -> bool:
    return MODELS[name][1]
