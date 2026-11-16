"""Справжнє українське мовлення з Common Voice 19 (без завантаження всього архіву 5,7 ГБ).

Архів wavs.zip читається на сервері частинами (HTTP Range): спершу центральний
каталог ZIP, потім лише вибрані файли. Для кожного мовця беруться щонайбільше
--per-speaker записів — так у наборі багато різних голосів і мікрофонів.
Частини test / train джерела не перетинаються за мовцями: test — для перевірки,
train — для можливої адаптації детектора (мовці інші).

Приклад:
    python scripts/uk_real.py --test 400 --train 300
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import urllib.request
import zipfile

from _common import ROOT

BASE = "https://huggingface.co/datasets/mouseyy/common_voice_19_uk_cropped/resolve/main/"
UA = {"User-Agent": "VoiceGuard-research/0.1 (student diploma project)"}


class HTTPFile(io.RawIOBase):
    """Файл на сервері з довільним доступом через HTTP Range."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[-1])
            self.url = r.url                                   # після переадресації на CDN

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def choose(meta_name: str, n: int, per_speaker: int, seed: int) -> list[tuple[str, str]]:
    text = urllib.request.urlopen(urllib.request.Request(BASE + meta_name, headers=UA), timeout=60).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text), delimiter="|"))
    rng = random.Random(seed)
    rng.shuffle(rows)
    per, out = {}, []
    for r in rows:
        if per.get(r["speaker_name"], 0) < per_speaker:
            per[r["speaker_name"]] = per.get(r["speaker_name"], 0) + 1
            out.append((r["speaker_name"], r["audio_file"]))
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--per-speaker", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = ROOT / "data" / "uk" / "real"
    out.mkdir(parents=True, exist_ok=True)
    plan = [("test", "metadata_test.csv", args.test), ("train", "metadata_train.csv", args.train)]
    zf = zipfile.ZipFile(io.BufferedReader(HTTPFile(BASE + "wavs.zip"), buffer_size=1 << 20))
    names = set(zf.namelist())
    print(f"у архіві {len(names)} файлів", flush=True)
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["split", "speaker", "file"])
        for split, meta, n in plan:
            items = choose(meta, n, args.per_speaker, args.seed)
            for i, (spk, path) in enumerate(items):
                dst = out / split / path.split("/")[-1]
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    member = path if path in names else next(x for x in names if x.endswith(path.split("/")[-1]))
                    dst.write_bytes(zf.read(member))
                w.writerow([split, spk[:16], f"{split}/{dst.name}"])
                if (i + 1) % 50 == 0:
                    print(f"  {split}: {i + 1}/{len(items)}", flush=True)
            print(f"{split}: {len(items)} записів, мовців {len({s for s, _ in items})}", flush=True)


if __name__ == "__main__":
    main()
