"""Перевірка без графічного інтерфейсу: голосовий запис, текст повідомлення, реєстрація мовця.

Приклади (з кореня репозиторію):
    python examples/check.py audio demo/3_шахрайське_голосове.wav
    python examples/check.py audio demo/1_справжнє_голосове.wav --no-asr
    python examples/check.py audio demo/мовець_Олена/перевірка_1.wav --speaker "Олена (демо)"
    python examples/check.py text "Вам нараховано виплату від Дії! Перейдіть: diia-vyplata.xyz"
    python examples/check.py enroll "Ім'я" запис1.wav запис2.wav запис3.wav
    python examples/check.py speakers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from voiceguard import audio  # noqa: E402
from voiceguard.system import VoiceGuard  # noqa: E402


def _print_risk(risk: dict) -> None:
    print(f"Ризик:           {risk['level']} ({risk['score'] * 100:.0f} %)")
    for r in risk.get("reasons", []):
        print(f"  • {r}")
    if risk.get("advice"):
        print("Поради:")
        for a in risk["advice"]:
            print(f"  – {a}")


def cmd_audio(vg: VoiceGuard, a: argparse.Namespace) -> None:
    x = audio.load(a.file)
    v = vg.check(x, speaker=a.speaker, source=str(a.file), transcribe=not a.no_asr,
                 progress=lambda s: print(f"  … {s}", file=sys.stderr))
    if a.json:
        from dataclasses import asdict
        print(json.dumps(asdict(v), ensure_ascii=False, indent=2, default=float))
        return
    print(f"\nФайл:            {a.file}  ({v.duration:.1f} с)")
    print(f"Детектор:        {v.detector}")
    print(f"Справжність:     {v.cm_confidence * 100:.0f} %  → "
          f"{'справжній голос' if v.cm_genuine else 'ознаки синтезу / підробки'}")
    if a.speaker:
        print(f"Мовець «{a.speaker}»: {'збіг' if v.asv_match else 'НЕ збігається'} "
              f"(оцінка {v.asv_score:.2f}, поріг {v.asv_threshold:.2f})")
    if v.transcript:
        print("Розпізнано:      " + " ".join(s["text"].strip() for s in v.transcript))
    for n in v.notes:
        print(f"Примітка:        {n}")
    _print_risk(v.risk)
    print(f"Час перевірки:   {v.seconds:.1f} с")


def cmd_text(vg: VoiceGuard, a: argparse.Namespace) -> None:
    r = vg.check_text(a.text)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=float))
        return
    _print_risk(r["risk"])


def cmd_enroll(vg: VoiceGuard, a: argparse.Namespace) -> None:
    secs = vg.verifier.enroll(a.name, [audio.load(f) for f in a.files])
    print(f"Мовця «{a.name}» зареєстровано: {len(a.files)} записів, {secs:.1f} с мовлення.")


def cmd_speakers(vg: VoiceGuard, a: argparse.Namespace) -> None:
    names = vg.verifier.speakers()
    print("\n".join(names) if names else "Зареєстрованих мовців немає.")


def main() -> None:
    p = argparse.ArgumentParser(description="VoiceGuard — перевірка з командного рядка")
    p.add_argument("--json", action="store_true", help="повний результат у форматі JSON")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("audio", help="перевірити аудіофайл (wav, mp3, ogg, m4a…)")
    s.add_argument("file", type=Path)
    s.add_argument("--speaker", help="ім'я зареєстрованого мовця для верифікації")
    s.add_argument("--no-asr", action="store_true", help="без розпізнавання мовлення (швидше)")
    s = sub.add_parser("text", help="перевірити текст повідомлення (SMS, Telegram, Viber)")
    s.add_argument("text")
    s = sub.add_parser("enroll", help="зареєструвати мовця за 3–5 записами")
    s.add_argument("name")
    s.add_argument("files", nargs="+", type=Path)
    sub.add_parser("speakers", help="список зареєстрованих мовців")
    a = p.parse_args()
    vg = VoiceGuard()
    {"audio": cmd_audio, "text": cmd_text, "enroll": cmd_enroll, "speakers": cmd_speakers}[a.cmd](vg, a)


if __name__ == "__main__":
    main()
