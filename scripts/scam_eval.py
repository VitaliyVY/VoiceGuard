"""Виявлення шахрайських розмов аналізом змісту (scam.py) на перекладених діалогах.

Дані — scripts/scam_translate.py. Правила на цих діалогах не налаштовувались.
Тривога — рівень ризику «середній» або вищий (оцінка ≥ 0,3), як у програмі.

Приклад:
    python scripts/scam_eval.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

from _common import ROOT, print_table
from voiceguard import scam

SRC = ROOT / "data" / "uk" / "scam_dialogue_uk.csv"


def main() -> None:
    with open(SRC, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_type: dict[str, list[float]] = defaultdict(list)
    tactics: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        rep = scam.analyze(r["dialogue_uk"])
        by_type[(r["label"], r["type"])].append(rep.score)
        tactics[r["label"]].update(rep.tactics.keys())
    out, table = {}, []
    for (label, t), s in sorted(by_type.items(), key=lambda z: (z[0][0] != "1", z[0][1])):
        alarm = sum(x >= 0.3 for x in s) / len(s)
        high = sum(x >= 0.6 for x in s) / len(s)
        out[t] = {"label": int(label), "n": len(s), "alarm": alarm, "high": high}
        table.append(["шахрайство" if label == "1" else "звичайна", t, len(s), f"{100 * alarm:.0f}", f"{100 * high:.0f}"])
    print_table(table, ["клас", "тип", "N", "тривога ≥ середній, %", "високий, %"])
    for label, name in (("1", "шахрайські"), ("0", "звичайні")):
        s = [x for (l, _), v in by_type.items() if l == label for x in v]
        if s:
            out[f"all_{label}"] = {"n": len(s), "alarm": sum(x >= 0.3 for x in s) / len(s)}
            print(f"{name}: тривога в {100 * out[f'all_{label}']['alarm']:.0f} % з {len(s)}; "
                  f"тактики: {dict(tactics[label].most_common())}")
    (ROOT / "results" / "scam_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
