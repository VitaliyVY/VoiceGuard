"""Верифікація мовця GMM-UBM на ASVspoof 2019 LA і її вразливість до підробок.

UBM навчається на справжніх записах train (мовці train не трапляються в
dev/eval). Мовців dev/eval реєструють за протоколом ASV (кілька записів
кожного), далі оцінюються перевірки трьох типів:
  target    — той самий мовець (має пройти);
  nontarget — інша людина (має не пройти);
  spoof     — синтез / перетворення голосу того самого мовця (має не пройти).
Звітується EER «свій / чужий» і частка підробок, які пройшли поріг EER —
тобто скільки атак пропустила б система без детектора підробки.

Приклад:
    python scripts/asv_gmm_ubm.py --trim --per-key 1500 --channels clean opus16
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _common import DATA, ROOT, print_table
from voiceguard import audio, channel, data, features, metrics
from voiceguard.models import GMMUBMVerifier

_STATE: dict = {}


def extract(path: str, trim: bool, chan: str = "clean") -> np.ndarray:
    x = channel.CHANNELS[chan](audio.load(path))
    if trim:
        x = audio.trim_silence(x)
    return features.cmvn(features.mfcc(x))


def train_frames(task) -> np.ndarray:
    path, trim, per_utt, seed = task
    f = extract(path, trim)
    rng = np.random.default_rng(seed)
    return f if len(f) <= per_utt else f[rng.choice(len(f), per_utt, replace=False)]


def enroll_task(task) -> tuple[str, np.ndarray]:
    speaker, paths, trim = task
    return speaker, _STATE["model"].enroll([extract(p, trim) for p in paths])


def score_task(task) -> float:
    speaker, path, trim, chan = task
    return _STATE["model"].score(_STATE["speakers"][speaker], extract(path, trim, chan))


def _init(model_path: str, speakers_path: str | None = None) -> None:
    with open(model_path, "rb") as fh:
        _STATE["model"] = pickle.load(fh)
    if speakers_path:
        with open(speakers_path, "rb") as fh:
            _STATE["speakers"] = pickle.load(fh)


def subset(trials: list[data.ASVTrial], per_key: int | None, seed: int) -> list[data.ASVTrial]:
    if not per_key:
        return trials
    rng = np.random.default_rng(seed)
    groups: dict[str, list] = defaultdict(list)
    for t in trials:
        groups[t.key if t.key != "spoof" else f"spoof:{t.attack}"].append(t)
    out = []
    for k in sorted(groups):
        g = groups[k]
        n = per_key if not k.startswith("spoof") else max(per_key // 10, 50)
        out += [g[i] for i in sorted(rng.choice(len(g), min(n, len(g)), replace=False))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--trim", action="store_true")
    ap.add_argument("--components", type=int, default=256)
    ap.add_argument("--frames-per-utt", type=int, default=60)
    ap.add_argument("--ubm-frames", type=int, default=300_000)
    ap.add_argument("--split", choices=["dev", "eval"], default="eval")
    ap.add_argument("--per-key", type=int, default=1500)
    ap.add_argument("--channels", nargs="+", default=["clean"], choices=list(channel.CHANNELS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    tag = f"mfcc{'_trim' if args.trim else ''}_g{args.components}"
    model_path = ROOT / "models" / f"asv_ubm_{tag}.pkl"
    spk_path = ROOT / "models" / f"asv_speakers_{tag}_{args.split}.pkl"
    model_path.parent.mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    t0 = time.perf_counter()

    if args.retrain or not model_path.exists():
        items = [it for it in data.cm_protocol(args.data, "train") if it.bonafide]
        tasks = [(str(data.audio_path(args.data, "train", it.utt)), args.trim, args.frames_per_utt, args.seed + i)
                 for i, it in enumerate(items)]
        with ProcessPoolExecutor(args.workers) as pool:
            frames = np.concatenate(list(pool.map(train_frames, tasks, chunksize=16)))
        rng = np.random.default_rng(args.seed)
        if len(frames) > args.ubm_frames:
            frames = frames[rng.choice(len(frames), args.ubm_frames, replace=False)]
        print(f"UBM: {len(frames)} кадрів з {len(items)} справжніх записів train", flush=True)
        model = GMMUBMVerifier.fit_ubm(frames, args.components, args.seed)
        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)
        print(f"UBM навчено ({time.perf_counter() - t0:.0f} с)", flush=True)

    if args.retrain or not spk_path.exists():
        enroll = data.asv_enrollment(args.data, args.split)
        tasks = [(spk, [str(data.audio_path(args.data, args.split, u)) for u in utts], args.trim)
                 for spk, utts in enroll.items()]
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(model_path),)) as pool:
            speakers = dict(pool.map(enroll_task, tasks))
        with open(spk_path, "wb") as fh:
            pickle.dump(speakers, fh)
        print(f"Зареєстровано мовців: {len(speakers)} ({time.perf_counter() - t0:.0f} с)", flush=True)

    trials = subset(data.asv_trials(args.data, args.split), args.per_key, args.seed)
    report: dict = {"config": vars(args), "channels": {}}
    rows = []
    scores_csv = ROOT / "results" / f"asv_scores_{tag}_{args.split}.csv"
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(model_path), str(spk_path))) as pool, \
            open(scores_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["channel", "speaker", "utt", "attack", "key", "score"])
        for chan in args.channels:
            tasks = [(t.speaker, str(data.audio_path(args.data, args.split, t.utt)), args.trim, chan) for t in trials]
            scores = np.array(list(pool.map(score_task, tasks, chunksize=16)))
            for t, s in zip(trials, scores):
                w.writerow([chan, t.speaker, t.utt, t.attack, t.key, f"{s:.6f}"])
            key = np.array([t.key for t in trials])
            tar, non, spf = scores[key == "target"], scores[key == "nontarget"], scores[key == "spoof"]
            value, thr = metrics.eer(tar, non)
            res = {"eer_target_nontarget": value, "threshold": thr,
                   "spoof_pass_rate": float((spf >= thr).mean()),
                   "eer_target_spoof": metrics.eer(tar, spf)[0],
                   "n": {"target": len(tar), "nontarget": len(non), "spoof": len(spf)}}
            res["spoof_pass_per_attack"] = {a: float((scores[(key == "spoof") & (np.array([t.attack for t in trials]) == a)] >= thr).mean())
                                            for a in sorted({t.attack for t in trials if t.key == "spoof"})}
            report["channels"][chan] = res
            rows.append([chan, f"{value * 100:.2f}", f"{res['spoof_pass_rate'] * 100:.1f}", f"{res['eer_target_spoof'] * 100:.2f}"])
            print(f"{chan}: готово ({time.perf_counter() - t0:.0f} с)", flush=True)
    print()
    print_table(rows, ["канал", "EER свій/чужий, %", "підробок пройшло, %", "EER свій/підробка, %"])
    out = ROOT / "results" / f"asv_gmm_ubm_{tag}_{args.split}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Звіт:", out.relative_to(ROOT), "| оцінки:", scores_csv.relative_to(ROOT))


if __name__ == "__main__":
    main()
