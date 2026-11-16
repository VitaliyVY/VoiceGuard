"""Завантаження зовнішніх українських наборів для незалежної перевірки (scripts/uk_external.py).

  ML-ITW (nccm2p2/ML-ITW, Hugging Face) — лише українська частина: з parquet читаються
      тільки групи рядків, де є language == "ukrainian" (≈ 90 записів, ~15 МБ замість 3,3 ГБ);
  DRSSU (github.com/bohutahennadii/audio_database, CC BY 4.0) — випадкові --drssu записів
      з generated/ (1…10000) і real/ (1…30000), поштучно (весь репозиторій ≈ 12 ГБ);
  FLEURS uk test (google/fleurs, CC BY 4.0) — перші --fleurs записів після перемішування.

Приклад:
    python scripts/uk_external_download.py --drssu 300 --fleurs 300
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import io
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import ROOT  # noqa: E402

import soundfile as sf  # noqa: E402

EXT = ROOT / "data" / "external"
DRSSU = "https://raw.githubusercontent.com/bohutahennadii/audio_database/main/"


def ml_itw() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    out = EXT / "ml_itw_uk"
    out.mkdir(parents=True, exist_ok=True)
    fs, rows = HfFileSystem(), []
    for i in range(4):
        pf = pq.ParquetFile(fs.open(f"datasets/nccm2p2/ML-ITW/data/eval-0000{i}-of-00004.parquet"))
        for g in range(pf.num_row_groups):
            if "ukrainian" not in pf.read_row_group(g, columns=["language"]).column(0).to_pylist():
                continue
            for r in pf.read_row_group(g).to_pylist():
                if r["language"] != "ukrainian":
                    continue
                name = f"{len(rows):03d}_{r['label']}_{r['speaker']}.wav"
                x, sr = sf.read(io.BytesIO(r["file"]["bytes"]))
                sf.write(out / name, x, sr)
                rows.append([name, r["label"], r["speaker"], r["platform"], sr, round(len(x) / sr, 2)])
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "label", "speaker", "platform", "sr", "seconds"])
        w.writerows(rows)
    print(f"ML-ITW uk: {len(rows)} записів")


def drssu(n: int, seed: int) -> None:
    rng = random.Random(seed)
    jobs = [("generated", i) for i in rng.sample(range(1, 10001), n)] + \
           [("real", i) for i in rng.sample(range(1, 30001), n)]

    def get(job):
        d, i = job
        p = EXT / "drssu" / d / f"{i}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            urllib.request.urlretrieve(DRSSU + f"{d}/{i}.wav", p)

    with cf.ThreadPoolExecutor(12) as ex:
        list(ex.map(get, jobs))
    print(f"DRSSU: {n} підроблених + {n} справжніх")


def fleurs(n: int, seed: int) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out = EXT / "fleurs_uk"
    p = hf_hub_download("google/fleurs", "parquet-data/uk_ua/test-00000-of-00001.parquet",
                        repo_type="dataset", local_dir=out / "raw")
    rows = pq.read_table(p).to_pylist()
    random.Random(seed).shuffle(rows)
    for k, r in enumerate(rows[:n]):
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
        sf.write(out / f"{k:03d}.wav", x, sr)
    print(f"FLEURS uk: {min(n, len(rows))} записів")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drssu", type=int, default=300)
    ap.add_argument("--fleurs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ml_itw()
    drssu(a.drssu, a.seed)
    fleurs(a.fleurs, a.seed)


if __name__ == "__main__":
    main()
