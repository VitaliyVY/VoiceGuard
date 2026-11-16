"""Частина adapt_mms: синтез MMS-TTS для адаптації (речення, яких немає в test і adapt_piper).

Потрібна для перехресної перевірки синтезаторів: адаптувати на MMS — перевірити на Piper.
Речення беруться з того самого перемішаного списку, що й у uk_fake.py, одразу після adapt_piper.

Приклад:
    python scripts/uk_fake_mms_adapt.py --n 118
"""

from __future__ import annotations

import argparse

import numpy as np

from _common import ROOT
from uk_fake import sentences
from voiceguard import audio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=118)
    ap.add_argument("--offset", type=int, default=200 + 200 + 118, help="після test_piper, test_mms, adapt_piper")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, VitsModel

    sents = sentences(args.seed)[args.offset: args.offset + args.n]
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-ukr")
    mms = VitsModel.from_pretrained("facebook/mms-tts-ukr").eval()
    out = ROOT / "data" / "uk" / "fake" / "adapt_mms"
    out.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(sents):
        dst = out / f"adapt_mms_{i:04d}.wav"
        if not dst.exists():
            torch.manual_seed(args.seed + 10_000 + i)
            with torch.no_grad():
                w = mms(**tok(text.lower(), return_tensors="pt")).waveform[0].numpy().astype(np.float32)
            audio.save(dst, audio.resample(w, mms.config.sampling_rate, 16000))
        if (i + 1) % 20 == 0:
            print(f"adapt_mms: {i + 1}/{len(sents)}", flush=True)
    with open(out / "sentences.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(sents))


if __name__ == "__main__":
    main()
