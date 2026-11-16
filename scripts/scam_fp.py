"""Хибні тривоги аналізу змісту на звичайних українських текстах.

Корпус UA-GEC (Syvokon et al., 2023, CC BY 4.0): есе, листи, дописи, переклади,
написані звичайними людьми — шахрайських повідомлень там немає, тож будь-який
рівень ризику «середній» чи «високий» — хибна тривога. Одиниця — абзац.
Правила налаштовувались лише на частині train; підсумок — на test.

Приклад:
    python scripts/scam_fp.py --split test --show 10
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from _common import ROOT
from voiceguard import scam

GEC = ROOT / "data" / "external" / "ua-gec" / "data" / "gec-fluency"


def paragraphs(split: str) -> list[str]:
    out = []
    for p in sorted((GEC / split / "target").glob("*.a1.txt")):
        out += [t.strip() for t in p.read_text(encoding="utf-8").split("\n") if len(t.strip()) >= 20]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--show", type=int, default=0, help="показати N абзаців із найбільшим ризиком")
    args = ap.parse_args()
    texts = paragraphs(args.split)
    reps = [scam.analyze(t) for t in texts]
    levels = Counter("високий" if r.score >= 0.6 else "середній" if r.score >= 0.3 else "низький" for r in reps)
    tactics = Counter(k for r in reps if r.score >= 0.3 for k in r.tactics)
    n = len(texts)
    res = {"split": args.split, "paragraphs": n, **{k: levels[k] / n for k in ("низький", "середній", "високий")},
           "tactics_in_alarms": dict(tactics.most_common())}
    print(f"UA-GEC {args.split}: {n} абзаців")
    for k in ("низький", "середній", "високий"):
        print(f"  {k:9s} {100 * levels[k] / n:5.1f} %  ({levels[k]})")
    print("  тактики в тривогах:", dict(tactics.most_common()))
    for r, t in sorted(zip(reps, texts), key=lambda z: -z[0].score)[: args.show]:
        print(f"\n[{r.score:.2f}] {list(r.tactics)} {r.combos}\n  {t[:220]}")
    (ROOT / "results").mkdir(exist_ok=True)
    Path(ROOT / "results" / f"scam_fp_{args.split}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                                     encoding="utf-8")


if __name__ == "__main__":
    main()
