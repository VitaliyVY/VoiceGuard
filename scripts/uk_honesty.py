"""Чи адаптований детектор розпізнає синтез, а не «чистоту запису»?

Справжні записи Common Voice зроблені побутовими мікрофонами (шум кімнати, MP3),
синтезовані — «студійно» чисті. Детектор міг навчитися розрізняти саме це.
Перевірка: синтезованим записам test (Piper, MMS) надаються умови Common Voice —
стиснення MP3 і справжній фоновий шум кімнати, вирізаний із пауз записів
Common Voice частини train (інші мовці, ніж у test). Якщо детектор далі ловить
підробки — він реагує на ознаки синтезу; якщо пропускає — на умови запису.

Додатково: зв'язок оцінки справжніх записів з їхнім відношенням сигнал/шум.

Приклад:
    python scripts/uk_honesty.py
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _common import ROOT, print_table
from voiceguard import audio, channel, metrics, ssl

UK = ROOT / "data" / "uk"
CACHE = ROOT / "data" / "cache"
CONDS = ["mp3", "mp3_room20", "mp3_room10"]


def room_noise(path: str) -> np.ndarray:
    """Фон кімнати: найтихіші кадри (паузи) запису Common Voice."""
    x = audio.normalize(audio.load(path))
    n = 320
    fr = x[: len(x) // n * n].reshape(-1, n)
    e = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    quiet = fr[e <= np.percentile(e, 20)].ravel()
    return quiet if len(quiet) > 1600 else np.zeros(0, np.float32)


def degrade(x: np.ndarray, cond: str, noise: np.ndarray) -> np.ndarray:
    if cond != "mp3":
        snr = 20.0 if cond == "mp3_room20" else 10.0
        x = channel.add_noise(x, snr, noise=np.resize(noise, len(x)) if len(noise) else None)
    return channel.codec(x, "libmp3lame", "mp3", 16000, "32k")


def task(args):
    path, cond, noise_path = args
    x = audio.normalize(audio.load(path))
    x = audio.trim_silence(degrade(x, cond, room_noise(noise_path)))
    return ssl.layer_stats(x)


def snr_task(path: str) -> float:
    from voiceguard.system import analyze_signal

    return analyze_signal(audio.normalize(audio.load(path)))["snr_db"]


def files(part: str) -> list[str]:
    return sorted(str(p) for p in (UK / part).glob("*.wav"))


def main() -> None:
    from scipy.stats import spearmanr

    real_train = files("real/train")
    models = {}
    for name, fn in (("WavLM (англ.)", "cm_ssl.pkl"), ("WavLM адаптований", "cm_ssl_uk.pkl")):
        with open(ROOT / "models" / fn, "rb") as fh:
            m = pickle.load(fh)
        layers = [l - 1 for l in m["layers"]]
        models[name] = lambda X, m=m, layers=layers: m["model"].decision_function(
            X[:, layers].astype(np.float32).mean(axis=1))
    F = {}
    with ProcessPoolExecutor(2) as pool:
        for part, key in (("fake/test_piper", "fake_test_piper"), ("fake/test_mms", "fake_test_mms")):
            fs = files(part)
            for c in CONDS:
                path = CACHE / f"uk_{key}_{c}.npy"
                if not path.exists():
                    jobs = [(f, c, real_train[i % len(real_train)]) for i, f in enumerate(fs)]
                    np.save(path, np.stack(list(pool.map(task, jobs, chunksize=4))))
                    print(f"  ознаки {key} {c}", flush=True)
                F[(key, c)] = np.load(path)
        snr_real = np.array(list(pool.map(snr_task, files("real/test"), chunksize=8)))
        snr_fake = np.array(list(pool.map(snr_task, files("fake/test_piper") + files("fake/test_mms"), chunksize=8)))
    for key in ("real_test", "fake_test_piper", "fake_test_mms"):
        F[(key, "clean")] = np.load(CACHE / f"uk_{key}_clean.npy")

    report = {"snr_db_median": {"real": float(np.median(snr_real)), "fake": float(np.median(snr_fake))}}
    print(f"\nВідношення сигнал/шум (медіана): справжні {np.median(snr_real):.0f} дБ, синтез {np.median(snr_fake):.0f} дБ")
    for name, score in models.items():
        r = score(F[("real_test", "clean")])
        fakes_clean = np.concatenate([score(F[(k, "clean")]) for k in ("fake_test_piper", "fake_test_mms")])
        _, thr = metrics.eer(r, fakes_clean)
        rows, res = [], {}
        for c in ["clean"] + CONDS:
            fp, fm = score(F[("fake_test_piper", c)]), score(F[("fake_test_mms", c)])
            f = np.concatenate([fp, fm])
            res[c] = {"eer_piper": metrics.eer(r, fp)[0], "eer_mms": metrics.eer(r, fm)[0], "eer": metrics.eer(r, f)[0],
                      "fake_accepted_at_clean_thr": float(np.mean(f >= thr))}
            rows.append([c, f"{100 * res[c]['eer_piper']:.1f}", f"{100 * res[c]['eer_mms']:.1f}",
                         f"{100 * res[c]['eer']:.1f}", f"{100 * res[c]['fake_accepted_at_clean_thr']:.1f}"])
        rho, p = spearmanr(r, snr_real)
        res["spearman_score_vs_snr_real"] = {"rho": float(rho), "p": float(p)}
        report[name] = res
        print(f"\n{name}  (справжні — Common Voice test, як є)")
        print_table(rows, ["умови підробок", "EER Piper", "EER MMS", "EER усі", "підробок пропущено (поріг EER чистих)"])
        print(f"  кореляція оцінки «справжній» зі сигнал/шум у справжніх записів: ρ = {rho:.2f} (p = {p:.3f})")
    (ROOT / "results" / "uk_honesty.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
