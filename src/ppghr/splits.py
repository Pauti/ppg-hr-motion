"""Subject-wise cross-validation folds.

Windows overlap in time and HR changes slowly, so any split at window or
record level would put near-duplicate samples in train and test. All splits
here are by subject: subjects are shuffled once and cut into `n_folds` groups.
In fold k, group k is the test set, group k+1 the validation set (early
stopping only) and the rest is training data. Every subject is tested exactly
once across all folds.
"""

import json
from pathlib import Path

import numpy as np


def make_folds(subjects: list[int], n_folds: int, seed: int) -> list[dict[str, list[int]]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(sorted(set(subjects))).tolist()
    groups = [sorted(order[i::n_folds]) for i in range(n_folds)]
    folds = []
    for k in range(n_folds):
        test, val = groups[k], groups[(k + 1) % n_folds]
        train = sorted(s for s in order if s not in test and s not in val)
        folds.append({"train": train, "val": val, "test": test})
    return folds


def save_folds(folds: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(folds, indent=2))


def load_folds(path: str | Path) -> list[dict[str, list[int]]]:
    return json.loads(Path(path).read_text())
