"""Синтезоване українське мовлення двома відкритими синтезаторами.

  Piper (VITS, голоси rhasspy/piper-voices uk_UA: lada, tetiana, mykyta,
         ukrainian_tts — 3 диктори);
  MMS-TTS (Meta, facebook/mms-tts-ukr, ліцензія CC BY-NC — некомерційне використання).
Речення — набір speech-uk/opentts-sentences (наголоси прибрано).

Частини:
  test  — 200 фраз Piper + 200 MMS (перевірка);
  adapt — фрази лише Piper з іншими реченнями (адаптація детектора). MMS у
          адаптації немає, тож на ньому перевіряється узагальнення на невідомий синтезатор.

Приклад:
    python scripts/uk_fake.py --test-per-engine 200 --adapt 118
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
import unicodedata

import numpy as np

from _common import ROOT
from voiceguard import audio

VOICES = [("uk_UA-lada-x_low", [None]), ("uk_UA-tetiana-high", [None]), ("uk_UA-mykyta-high", [None]),
          ("uk_UA-ukrainian_tts-medium", [0, 1, 2])]


def sentences(seed: int) -> list[str]:
    with open(ROOT / "data" / "uk" / "tts" / "sentences.csv", encoding="utf-8") as fh:
        rows = [r["sentence"] for r in csv.DictReader(fh)]
    out = []
    for s in rows:
        s = unicodedata.normalize("NFD", s).replace("́", "")          # прибрати наголоси
        s = unicodedata.normalize("NFC", s)
        s = re.sub(r"^[—–-]\s*", "", s.strip().strip('"')).strip()
        if 25 <= len(s) <= 140:
            out.append(s)
    random.Random(seed).shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-per-engine", type=int, default=200)
    ap.add_argument("--adapt", type=int, default=118)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from piper import PiperVoice
    from piper.config import SynthesisConfig

    out = ROOT / "data" / "uk" / "fake"
    sents = sentences(args.seed)
    n_t = args.test_per_engine
    plan = {"test_piper": sents[:n_t], "test_mms": sents[n_t:2 * n_t], "adapt_piper": sents[2 * n_t:2 * n_t + args.adapt]}
    rows = []
    t0 = time.perf_counter()

    voices = []
    for name, speakers in VOICES:
        v = PiperVoice.load(str(ROOT / "data" / "uk" / "tts" / "piper" / f"{name}.onnx"))
        voices += [(name, v, s) for s in speakers]
    for part in ("test_piper", "adapt_piper"):
        (out / part).mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(plan[part]):
            name, v, spk = voices[i % len(voices)]
            dst = out / part / f"{part}_{i:04d}.wav"
            if not dst.exists():
                chunks = list(v.synthesize(text, SynthesisConfig(speaker_id=spk) if spk is not None else None))
                x = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
                audio.save(dst, audio.resample(x, chunks[0].sample_rate, 16000))
            rows.append([part, f"piper:{name}:{spk}", f"{part}/{dst.name}", text])
        print(f"{part}: {len(plan[part])} ({time.perf_counter() - t0:.0f} с)", flush=True)

    import torch
    from transformers import AutoTokenizer, VitsModel

    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-ukr")
    mms = VitsModel.from_pretrained("facebook/mms-tts-ukr").eval()
    (out / "test_mms").mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(plan["test_mms"]):
        dst = out / "test_mms" / f"test_mms_{i:04d}.wav"
        if not dst.exists():
            torch.manual_seed(args.seed + i)                       # VITS стохастичний — відтворюваність
            with torch.no_grad():
                w = mms(**tok(text.lower(), return_tensors="pt")).waveform[0].numpy().astype(np.float32)
            audio.save(dst, audio.resample(w, mms.config.sampling_rate, 16000))
        rows.append(["test_mms", "mms:ukr", f"test_mms/{dst.name}", text])
    print(f"test_mms: {len(plan['test_mms'])} ({time.perf_counter() - t0:.0f} с)", flush=True)

    with open(out / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["part", "engine", "file", "text"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
