"""Прикладний рівень: реєстрація мовців, перевірка запису, журнал.

Рішення приймається каскадом (як у повній системі з дослідження):
  1. детектор підробки (CM): чи голос справжній;
  2. верифікація мовця (ASV, GMM-UBM): чи це заявлений мовець.
Спроба приймається, лише якщо обидва кроки пройдено.

Детектор — найкращий наявний: WavLM + логістична регресія (models/cm_ssl.pkl),
інакше LFCC 0–4 кГц + аугментація, GMM (variant D). Пороги — точки рівних
помилок, визначені на dev (CM) та eval «свій/чужий» (ASV) ASVspoof 2019 LA.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
import pickle
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import audio, features, metrics

ROOT = Path(__file__).resolve().parents[1]
APP_DATA = ROOT / "app_data"
GMM_TAG = "lfcc4k_trim_g128_aug-clean-opus16-telephone-noise20"


@dataclass
class Verdict:
    speaker: str | None
    cm_score: float
    cm_threshold: float
    cm_genuine: bool
    cm_confidence: float                 # 0..1 — наскільки впевнено «справжній»
    asv_score: float | None = None
    asv_threshold: float | None = None
    asv_match: bool | None = None
    accepted: bool | None = None
    detector: str = ""
    seconds: float = 0.0
    duration: float = 0.0
    notes: list[str] = field(default_factory=list)
    analysis: dict = field(default_factory=dict)     # властивості сигналу (смуга, SNR, перевантаження)
    timeline: dict = field(default_factory=dict)     # підозрілість у часі: t, confidence, speech
    transcript: list = field(default_factory=list)   # [{start, end, text}] — розпізнане мовлення
    language: str = ""
    content: dict = field(default_factory=dict)      # аналіз змісту (scam.ContentReport)
    risk: dict = field(default_factory=dict)         # загальна оцінка ризику (risk.Risk)
    timings: dict = field(default_factory=dict)      # час кожного кроку, с


def analyze_signal(x: np.ndarray, sr: int = audio.SAMPLE_RATE) -> dict:
    """Властивості запису для пояснення рішення.

    bandwidth_hz — частота, нижче якої 99 % енергії мовлення (телефон і мобільні кодеки
    дають ≈ 3,4–4 кГц, широкосмугові канали — 7–8 кГц); snr_db — різниця гучних і тихих
    кадрів (оцінка відношення сигнал/шум); clipped — частка відліків на межі діапазону.
    """
    from scipy import signal as sps

    speech = audio.trim_silence(x)
    f, pxx = sps.welch(speech if len(speech) > 1024 else x, fs=sr, nperseg=1024)
    cum = np.cumsum(pxx) / max(pxx.sum(), 1e-20)
    bandwidth = float(f[min(np.searchsorted(cum, 0.99), len(f) - 1)])
    e = audio.frame_energy_db(x)
    snr = float(np.percentile(e, 95) - np.percentile(e, 10)) if len(e) > 10 else 0.0
    return {"duration": len(x) / sr, "speech": len(speech) / sr, "bandwidth_hz": bandwidth,
            "narrowband": bandwidth < 4300, "snr_db": snr,
            "clipped": float(np.mean(np.abs(x) >= 0.999)), "peak": float(np.abs(x).max()) if len(x) else 0.0}


def _dev_threshold(csv_path: Path, variant: str | None = None) -> tuple[dict[str, float], float]:
    """Пороги, визначені на dev (усі канали разом), і розкид оцінок для шкали впевненості.

    «м'яка» — точка рівних помилок; «звичайна» / «сувора» — поріг, що на dev відхиляє
    10 % / 20 % справжніх записів. На нових методах атак (eval) м'який поріг пропускає
    20–30 % підробок, суворий — 6–11 % ціною 2–5 % відхилених справжніх (results/tandem_wavlm*.json).
    """
    s, b = [], []
    with open(csv_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if r["split"] == "dev" and (variant is None or r.get("variant") == variant):
                s.append(float(r["score"]))
                b.append(r["bonafide"] == "1")
    s, b = np.array(s), np.array(b)
    thr = {"м'яка": metrics.eer(s[b], s[~b])[1], "звичайна": float(np.quantile(s[b], 0.10)),
           "сувора": float(np.quantile(s[b], 0.20))}
    return thr, float(s.std())


class Detector:
    """Детектор підробки голосу: score(x) — більше означає «справжніший»."""

    def __init__(self):
        ssl_path = ROOT / "models" / "cm_ssl.pkl"
        uk_path = ROOT / "models" / "cm_ssl_uk.pkl"         # адаптований до української (scripts/uk_eval.py)
        uk = None
        if uk_path.exists():
            with open(uk_path, "rb") as fh:
                uk = pickle.load(fh)
        if uk is not None and "thresholds" in uk:          # пороги — scripts/uk_calibrate.py
            self.kind, self._clf, self._layers = "ssl", uk["model"], tuple(l - 1 for l in uk["layers"])
            self.thresholds, self.scale = uk["thresholds"], uk["scale"]
            self.name = f"WavLM Base+ (шари {uk['layers']}), адаптовано до української"
        elif ssl_path.exists() and (ROOT / "results" / "cm_ssl_scores.csv").exists():
            with open(ssl_path, "rb") as fh:
                d = pickle.load(fh)
            self.kind, self._clf, self._layers = "ssl", d["model"], tuple(l - 1 for l in d["layers"])
            self.thresholds, self.scale = _dev_threshold(ROOT / "results" / "cm_ssl_scores.csv", d["regime"])
            self.name = f"WavLM Base+ (шари {d['layers']}) + логістична регресія"
        else:
            with open(ROOT / "models" / f"cm_gmm_{GMM_TAG}.pkl", "rb") as fh:
                self._gmm = pickle.load(fh)
            self.kind = "gmm"
            self.thresholds, self.scale = _dev_threshold(ROOT / "results" / f"cm_scores_{GMM_TAG}.csv")
            self.name = "LFCC 0–4 кГц + аугментація каналами, GMM"
        self.strictness = "звичайна"

    @property
    def threshold(self) -> float:
        return self.thresholds[self.strictness]

    def score(self, x: np.ndarray) -> float:
        x = audio.trim_silence(x)
        if self.kind == "ssl":
            from . import ssl

            stats = ssl.layer_stats(x).astype(np.float32)
            return float(self._clf.decision_function(stats[list(self._layers)].mean(axis=0)[None])[0])
        return self._gmm.score(features.lfcc(x, fmax=4000.0))

    def confidence(self, s: float) -> float:
        return float(1.0 / (1.0 + np.exp(-(s - self.threshold) / (0.25 * self.scale + 1e-9))))

    def timeline(self, x: np.ndarray, window: float = 3.0, hop: float = 0.5, max_seconds: float = 30.0) -> dict:
        """Впевненість «справжній» для ковзних вікон запису (наочне пояснення рішення).

        WavLM проганяється по запису один раз; для кожного вікна з кадрових станів шарів
        рахуються ті самі ознаки, що й для всього запису (середнє, ст. відхилення), і
        оцінюються тим самим класифікатором. Для GMM беруться логарифмічні відношення
        правдоподібностей кадрів, усереднені у вікні. Вікна, де майже немає мовлення,
        позначаються speech = False (оцінка там малоінформативна).
        """
        x = np.asarray(x[: int(audio.SAMPLE_RATE * max_seconds)], dtype=np.float32)
        dur = len(x) / audio.SAMPLE_RATE
        win = min(window, dur)
        starts = np.arange(0.0, max(dur - win, 0.0) + 1e-9, hop)
        e = audio.frame_energy_db(x)                                   # кадри 25 мс з кроком 10 мс
        voiced = e >= e.max() - 35.0
        if self.kind == "ssl":
            from . import ssl

            st = ssl.frame_states(x, list(self._layers))
            rate = st.shape[0] / max(dur, 1e-9)
            scores = []
            for t0 in starts:
                a, b = int(t0 * rate), max(int((t0 + win) * rate), int(t0 * rate) + 2)
                feat = ssl.window_stats(st[a:b])
                scores.append(float(self._clf.decision_function(feat[None])[0]))
        else:
            f = features.lfcc(x, fmax=4000.0).astype(np.float64)
            llr = self._gmm.bonafide.score_samples(f) - self._gmm.spoof.score_samples(f)
            rate = len(llr) / max(dur, 1e-9)
            scores = [float(np.mean(llr[int(t0 * rate): max(int((t0 + win) * rate), int(t0 * rate) + 1)]))
                      for t0 in starts]
        speech = [bool(voiced[int(t0 * 100): int((t0 + win) * 100)].mean() > 0.3) if len(voiced) else False
                  for t0 in starts]
        centers = (starts + win / 2).round(2)
        return {"t": centers.tolist(), "confidence": [round(self.confidence(s), 4) for s in scores],
                "speech": speech, "window": win, "duration": dur}


class Verifier:
    """Верифікація мовця GMM-UBM; моделі мовців — у app_data/speakers/*.npz."""

    def __init__(self):
        with open(ROOT / "models" / "asv_ubm_mfcc_trim_g256.pkl", "rb") as fh:
            self.ubm = pickle.load(fh)
        rep = json.loads((ROOT / "results" / "asv_gmm_ubm_mfcc_trim_g256_eval.json").read_text(encoding="utf-8"))
        self.default_threshold = float(rep["channels"]["clean"]["threshold"])
        self.dir = APP_DATA / "speakers"
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _feat(x: np.ndarray) -> np.ndarray:
        return features.cmvn(features.mfcc(audio.trim_silence(x)))

    def speakers(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.npz"))

    def info(self, name: str) -> dict:
        d = np.load(self.dir / f"{name}.npz")
        return {"speech_seconds": float(d["speech_seconds"]), "created": float(d["created"]),
                "n_recordings": int(d["n_recordings"])}

    def enroll(self, name: str, recordings: list[np.ndarray]) -> float:
        """Реєстрація: модель мовця з кількох записів; повертає загальну тривалість мовлення, с."""
        feats = [self._feat(x) for x in recordings]
        means = self.ubm.enroll(feats)
        speech = sum(len(f) for f in feats) / 100.0
        np.savez(self.dir / f"{name}.npz", means=means, speech_seconds=speech, created=time.time(),
                 n_recordings=len(recordings))
        return speech

    def delete(self, name: str) -> None:
        (self.dir / f"{name}.npz").unlink(missing_ok=True)

    def score(self, name: str, x: np.ndarray) -> float:
        means = np.load(self.dir / f"{name}.npz")["means"]
        return self.ubm.score(means, self._feat(x))


class VoiceGuard:
    def __init__(self):
        self.detector = Detector()
        self.verifier = Verifier()
        self.asv_threshold = self.verifier.default_threshold
        APP_DATA.mkdir(exist_ok=True)
        self._db = sqlite3.connect(APP_DATA / "journal.sqlite", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS checks (time REAL, source TEXT, verdict TEXT)")

    def check(self, x: np.ndarray, speaker: str | None = None, source: str = "",
              progress: Callable[[str], None] | None = None, transcribe: bool = True,
              language: str | None = "uk") -> Verdict:
        """progress(крок) викликається перед кроками: analysis, detector, timeline, verifier, asr, content, done."""
        step = progress or (lambda _: None)
        t0 = time.perf_counter()
        timings = {}
        step("analysis")
        info = analyze_signal(x)
        x = audio.normalize(x)
        timings["analysis"] = time.perf_counter() - t0
        step("detector")
        t1 = time.perf_counter()
        s = self.detector.score(x)
        timings["detector"] = time.perf_counter() - t1
        v = Verdict(speaker, s, self.detector.threshold, s >= self.detector.threshold, self.detector.confidence(s),
                    detector=self.detector.name, duration=len(x) / audio.SAMPLE_RATE, analysis=info)
        if info["speech"] < 2.0:
            v.notes.append(f"Мовлення лише {info['speech']:.1f} с — для надійного рішення бажано 3–5 с.")
        if info["clipped"] > 0.001:
            v.notes.append("Запис перевантажений (кліпінг) — говоріть тихіше або далі від мікрофона.")
        if info["snr_db"] < 15:
            v.notes.append(f"Шумно (≈ {info['snr_db']:.0f} дБ) — точність детектора в шумі нижча.")
        step("timeline")
        t1 = time.perf_counter()
        v.timeline = self.detector.timeline(x)
        timings["timeline"] = time.perf_counter() - t1
        tl = [c for c, sp in zip(v.timeline["confidence"], v.timeline["speech"]) if sp]
        if v.cm_genuine and tl and min(tl) < 0.5:
            worst = v.timeline["t"][int(np.argmin([c if sp else 1 for c, sp in
                                                   zip(v.timeline["confidence"], v.timeline["speech"])]))]
            v.notes.append(f"Запис загалом справжній, але біля {worst:.1f} с є підозрілий фрагмент — "
                           "можлива вставка синтезованого мовлення.")
        if speaker:
            step("verifier")
            t1 = time.perf_counter()
            a = self.verifier.score(speaker, x)
            timings["verifier"] = time.perf_counter() - t1
            v.asv_score, v.asv_threshold, v.asv_match = a, self.asv_threshold, a >= self.asv_threshold
            v.accepted = v.cm_genuine and v.asv_match
        report = None
        if transcribe:
            from . import asr, scam

            step("asr")
            t1 = time.perf_counter()
            segs, lang = asr.transcribe(x, language)
            timings["asr"] = time.perf_counter() - t1
            v.transcript = [{"start": s.start, "end": s.end, "text": s.text} for s in segs]
            v.language = lang
            step("content")
            t1 = time.perf_counter()
            report = scam.analyze(" ".join(s.text for s in segs), segs)
            v.content = report.to_dict()
            timings["content"] = time.perf_counter() - t1
        from . import risk

        r = risk.combine(voice_fake=1.0 - v.cm_confidence,
                         speaker_mismatch=(not v.asv_match) if speaker else None, content=report)
        v.risk = asdict(r)
        v.timings = timings
        step("done")
        v.seconds = time.perf_counter() - t0
        self._db.execute("INSERT INTO checks VALUES (?, ?, ?)", (time.time(), source, json.dumps(asdict(v), ensure_ascii=False)))
        self._db.commit()
        return v

    def check_text(self, text: str, source: str = "текст") -> dict:
        """Перевірка повідомлення без голосу (SMS, Telegram, Viber): зміст, посилання, номери."""
        from . import risk, scam

        report = scam.analyze(text)
        r = risk.combine(content=report)
        result = {"text": text, "content": report.to_dict(), "risk": asdict(r)}
        self._db.execute("INSERT INTO checks VALUES (?, ?, ?)",
                         (time.time(), f"{source} (текст)", json.dumps({"text_only": True, **result}, ensure_ascii=False)))
        self._db.commit()
        return result

    def journal(self, limit: int = 200) -> list[tuple[float, str, dict]]:
        rows = self._db.execute("SELECT time, source, verdict FROM checks ORDER BY time DESC LIMIT ?", (limit,)).fetchall()
        return [(t, s, json.loads(v)) for t, s, v in rows]
