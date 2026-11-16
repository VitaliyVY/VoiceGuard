"""Демонстраційний набір файлів для показу програми (папка demo/).

  мовець/        — 5 записів одного диктора Common Voice для реєстрації + 2 для перевірки
  інший_мовець/  — 2 записи іншого диктора (перевірка «чужий голос»)
  1_справжнє_голосове.wav        — справжній запис (Common Voice), канал Telegram (Opus 16)
  2_синтезований_голос.wav       — синтез Piper зі звичайним змістом, канал Telegram
  3_шахрайське_голосове.wav      — синтез Piper зі схемою «родич у біді», канал Telegram
  4_дзвінок_банку_синтез.wav     — синтез Piper зі схемою «служба безпеки банку», мобільний канал (AMR-NB)
  5_справжнє_межовий_випадок.wav — справжній запис, у якому детектор сумнівається (чесний приклад помилки)

Справжні записи — Common Voice 19 uk (CC0), голоси дикторів відкритого набору.

Приклад:
    python scripts/make_demo.py
"""

from __future__ import annotations

import csv
import io
import random
import urllib.request
import zipfile

import numpy as np

from _common import ROOT
from uk_real import BASE, UA, HTTPFile
from voiceguard import audio, channel

OUT = ROOT / "demo"
SCAM_RELATIVE = ("Мамо, привіт, це я. Я потрапив у ДТП, мене затримала поліція. Терміново скинь двадцять тисяч "
                 "на картку адвоката, нікому не кажи, я потім усе поясню.")
SCAM_BANK = ("Добрий день, вас турбує служба безпеки банку. По вашій картці зафіксовано підозрілу операцію. "
             "Щоб її скасувати, продиктуйте, будь ласка, код з SMS.")
ORDINARY = "Привіт, я сьогодні трохи затримаюся на роботі. Купи, будь ласка, хліба й молока, добре?"


def median_f0(x: np.ndarray, sr: int = 16000) -> float:
    """Медіанна частота основного тону (автокореляція по гучних кадрах) — щоб підписати мовця."""
    n, f0s = 640, []
    e = [float(np.mean(x[i:i + n] ** 2)) for i in range(0, len(x) - n, n)]
    thr = np.percentile(e, 70)
    for k, i in enumerate(range(0, len(x) - n, n)):
        if e[k] < thr:
            continue
        fr = x[i:i + n] - x[i:i + n].mean()
        ac = np.correlate(fr, fr, "full")[n - 1:]
        lo, hi = sr // 400, sr // 60
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.3 * ac[0]:
            f0s.append(sr / lag)
    return float(np.median(f0s)) if f0s else 0.0


def piper(text: str, voice: str) -> np.ndarray:
    from piper import PiperVoice

    v = PiperVoice.load(str(ROOT / "data" / "uk" / "tts" / "piper" / f"{voice}.onnx"))
    chunks = list(v.synthesize(text))
    x = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    return audio.normalize(audio.resample(x, chunks[0].sample_rate, 16000))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    meta = urllib.request.urlopen(urllib.request.Request(BASE + "metadata_test.csv", headers=UA), timeout=60).read()
    rows = list(csv.DictReader(io.StringIO(meta.decode("utf-8")), delimiter="|"))
    by_spk: dict[str, list[dict]] = {}
    for r in rows:
        by_spk.setdefault(r["speaker_name"], []).append(r)
    rich = sorted((s for s, v in by_spk.items() if len(v) >= 20), key=lambda s: -len(by_spk[s]))
    rng = random.Random(0)
    zf = zipfile.ZipFile(io.BufferedReader(HTTPFile(BASE + "wavs.zip"), buffer_size=1 << 20))
    names = set(zf.namelist())

    def fetch(r: dict) -> np.ndarray:
        member = r["audio_file"] if r["audio_file"] in names else next(
            n for n in names if n.endswith(r["audio_file"].split("/")[-1]))
        import soundfile as sf

        x, sr = sf.read(io.BytesIO(zf.read(member)), dtype="float32", always_2d=True)
        return audio.normalize(audio.resample(x.mean(axis=1), sr, audio.SAMPLE_RATE))

    # мовець для реєстрації: перший «багатий» диктор, чиї записи досить довгі
    spk_a, spk_b = rich[0], rich[1]
    clips_a = [r for r in by_spk[spk_a] if len(r["text"]) >= 40]
    rng.shuffle(clips_a)
    xa = [fetch(r) for r in clips_a[:7]]
    f0 = median_f0(np.concatenate(xa))
    name_a = "Олена" if f0 > 165 else "Андрій"
    d = OUT / f"мовець_{name_a}"
    d.mkdir(exist_ok=True)
    for i, x in enumerate(xa[:5], 1):
        audio.save(d / f"реєстрація_{i}.wav", x)
    for i, x in enumerate(xa[5:7], 1):
        audio.save(d / f"перевірка_{i}.wav", channel.CHANNELS["opus16"](x))
    clips_b = by_spk[spk_b][:2]
    xb = [fetch(r) for r in clips_b]
    d = OUT / "інший_мовець"
    d.mkdir(exist_ok=True)
    for i, x in enumerate(xb, 1):
        audio.save(d / f"чужий_голос_{i}.wav", channel.CHANNELS["opus16"](x))
    print(f"мовець {name_a} (F0 ≈ {f0:.0f} Гц), інший мовець — {len(xb)} записи", flush=True)

    # справжнє голосове: типовий запис (детектор упевнений) і межовий випадок (детектор сумнівається,
    # так буває приблизно з кожним 20-м справжнім записом), канал Telegram
    typical = ROOT / "data" / "uk" / "real" / "test" / "common_voice_uk_23548819.wav"
    audio.save(OUT / "1_справжнє_голосове.wav", channel.CHANNELS["opus16"](audio.normalize(audio.load(typical))))
    other = next(r for s in rich[2:] for r in by_spk[s] if len(r["text"]) >= 60)
    audio.save(OUT / "5_справжнє_межовий_випадок.wav", channel.CHANNELS["opus16"](fetch(other)))
    audio.save(OUT / "2_синтезований_голос.wav", channel.CHANNELS["opus16"](piper(ORDINARY, "uk_UA-tetiana-high")))
    audio.save(OUT / "3_шахрайське_голосове.wav",
               channel.CHANNELS["opus16"](piper(SCAM_RELATIVE, "uk_UA-mykyta-high")))
    audio.save(OUT / "4_дзвінок_банку_синтез.wav",
               channel.CHANNELS["amrnb"](piper(SCAM_BANK, "uk_UA-tetiana-high")))
    print("готово:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
