"""Спільні утиліти скриптів."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")     # паралельні процеси інакше вичерпують пам'ять
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "external" / "asvspoof2019"

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def print_table(rows: list[list], header: list[str]) -> None:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))
