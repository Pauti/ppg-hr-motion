import numpy as np
import pytest

from ppghr.config import load_config
from ppghr.labels import reference_hr
from ppghr.splits import make_folds


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/default.yaml")


def test_reference_hr_regular_beats(cfg):
    beats = np.arange(0.3, 20, 0.75)  # 80 bpm
    assert reference_hr(beats, 2.0, 10.0, cfg) == pytest.approx(80.0)


def test_reference_hr_rejects_missed_beat(cfg):
    beats = np.arange(0.3, 20, 0.75)
    beats = np.delete(beats, 6)  # one missing annotation -> one doubled RR interval
    assert np.isnan(reference_hr(beats, 2.0, 10.0, cfg))


def test_reference_hr_needs_beats(cfg):
    assert np.isnan(reference_hr(np.array([1.0, 1.8]), 0.0, 8.0, cfg))


def test_folds_are_subject_disjoint_and_cover_everyone():
    subjects = list(range(1, 23))
    folds = make_folds(subjects, n_folds=5, seed=42)
    tested = []
    for f in folds:
        tr, va, te = map(set, (f["train"], f["val"], f["test"]))
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert tr | va | te == set(subjects)
        tested += f["test"]
    assert sorted(tested) == subjects


def test_folds_are_deterministic():
    assert make_folds(list(range(22)), 5, seed=1) == make_folds(list(range(22)), 5, seed=1)
