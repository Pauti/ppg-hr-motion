import numpy as np
import pytest

from ppghr.baseline import imu_mask_hr, peak_hr, spectral_hr
from ppghr.config import load_config

FS = 50
T = np.arange(0, 8, 1 / FS)


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/default.yaml")


def pulse(hr_bpm: float) -> np.ndarray:
    f = hr_bpm / 60
    return np.sin(2 * np.pi * f * T) + 0.3 * np.sin(2 * np.pi * 2 * f * T + 0.8)


@pytest.mark.parametrize("hr", [55, 72, 110, 150])
def test_clean_pulse(cfg, hr):
    x = pulse(hr) + 0.05 * np.random.default_rng(0).standard_normal(len(T))
    assert peak_hr(x, FS, cfg) == pytest.approx(hr, abs=3)
    assert spectral_hr(x, FS, cfg) == pytest.approx(hr, abs=2)


def test_imu_mask_removes_motion_peak(cfg):
    hr, motion_bpm = 80, 120
    motion = np.sin(2 * np.pi * motion_bpm / 60 * T)
    ppg = pulse(hr) + 2.0 * motion
    acc = np.stack([motion, 0.5 * motion, np.zeros_like(T)], axis=1)

    assert spectral_hr(ppg, FS, cfg) == pytest.approx(motion_bpm, abs=2)
    assert imu_mask_hr(ppg, acc, FS, cfg) == pytest.approx(hr, abs=2)


def test_imu_mask_inactive_at_rest(cfg):
    ppg = pulse(65)
    acc = np.zeros((len(T), 3))
    assert imu_mask_hr(ppg, acc, FS, cfg) == spectral_hr(ppg, FS, cfg)
