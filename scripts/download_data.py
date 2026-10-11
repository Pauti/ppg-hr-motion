"""Download the PhysioNet Pulse Transit Time PPG dataset (WFDB format).

Only the WFDB files (.hea/.dat/.atr, ~430 MB) are fetched; the CSV export
contains the same signals but is ~2.6 GB. Files are taken from PhysioNet's
AWS Open Data mirror, which is much faster than physionet.org, with
physionet.org as fallback. Every file is checked against the SHA256SUMS.txt
published with the dataset.
"""

import argparse
import hashlib
import urllib.request
from pathlib import Path

from ppghr.config import load_config

SOURCES = (
    "https://physionet-open.s3.amazonaws.com/{slug}/{version}/",
    "https://physionet.org/files/{slug}/{version}/",
)
WFDB_SUFFIXES = (".hea", ".dat", ".atr")
EXTRA_FILES = ("RECORDS", "README.txt", "LICENSE.txt", "SHA256SUMS.txt")


def fetch(name: str, dest: Path, bases: list[str]) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for i, base in enumerate(bases):
        try:
            with urllib.request.urlopen(base + name, timeout=60) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return
        except OSError:
            if i == len(bases) - 1:
                raise


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    bases = [s.format(slug=ds["physionet_slug"], version=ds["version"]) for s in SOURCES]
    out_dir = Path(ds["raw_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in EXTRA_FILES:
        if not (out_dir / name).exists():
            fetch(name, out_dir / name, bases)

    checksums = {}
    for line in (out_dir / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        checksums[name.strip()] = digest

    records = (out_dir / "RECORDS").read_text().split()
    files = [r + s for r in records for s in WFDB_SUFFIXES]
    for i, name in enumerate(files, 1):
        dest = out_dir / name
        if dest.exists() and sha256(dest) == checksums.get(name):
            continue
        print(f"[{i}/{len(files)}] {name}", flush=True)
        fetch(name, dest, bases)
        if name in checksums and sha256(dest) != checksums[name]:
            raise RuntimeError(f"Checksum mismatch for {name}")

    print(f"{len(records)} records available in {out_dir}")


if __name__ == "__main__":
    main()
