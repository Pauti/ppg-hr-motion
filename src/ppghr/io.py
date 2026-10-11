import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

RECORD_NAME = re.compile(r"s(?P<subject>\d+)_(?P<activity>sit|walk|run)$")


@dataclass
class Record:
    name: str
    subject: int
    activity: str
    fs: float
    signals: pd.DataFrame   # one column per channel
    r_peaks: np.ndarray     # ECG R-peak sample indices


def parse_name(name: str) -> tuple[int, str]:
    m = RECORD_NAME.match(name)
    if m is None:
        raise ValueError(f"Unexpected record name: {name}")
    return int(m["subject"]), m["activity"]


def load_records(raw_dir: str | Path, names: list[str] | None = None) -> Iterator[Record]:
    """Read WFDB records (.hea/.dat) and their R-peak annotations (.atr)."""
    raw_dir = Path(raw_dir)
    all_names = (raw_dir / "RECORDS").read_text().split()
    for name in sorted(names or all_names, key=parse_name):
        rec = wfdb.rdrecord(str(raw_dir / name))
        ann = wfdb.rdann(str(raw_dir / name), "atr")
        subject, activity = parse_name(name)
        yield Record(
            name=name,
            subject=subject,
            activity=activity,
            fs=float(rec.fs),
            signals=pd.DataFrame(rec.p_signal, columns=rec.sig_name),
            r_peaks=np.asarray(ann.sample, dtype=np.int64),
        )
