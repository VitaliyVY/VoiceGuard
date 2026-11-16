"""Загальна оцінка ризику шахрайства з кількох незалежних сигналів.

Сигнали та їхні ваги (наскільки сильно кожен сам по собі свідчить про шахрайство):
  voice    — голос синтезований або перетворений (1 − впевненість детектора);
  speaker  — голос не збігається із заявленим мовцем;
  content  — зміст розмови містить тактики шахрайських схем;
  links    — у тексті підозрілі посилання, номери карток.
Поєднання — імовірнісне «або»: R = 1 − Π (1 − w_i · s_i). Так один сильний
сигнал уже дає високий ризик, а кілька слабких підсилюють один одного.
Ваги задано експертно й прозоро; їх можна відкалібрувати на зібраному наборі.
"""

from __future__ import annotations

from dataclasses import dataclass, field

WEIGHTS = {"voice": 0.9, "speaker": 0.7, "content": 0.9, "links": 0.7}
LEVELS = [(0.3, "низький", "good"), (0.6, "середній", "warn"), (1.01, "високий", "bad")]


@dataclass
class Risk:
    score: float
    level: str
    kind: str                               # good | warn | bad — для кольору
    signals: dict[str, float]
    reasons: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)


def combine(voice_fake: float | None = None, speaker_mismatch: bool | None = None, content=None) -> Risk:
    signals: dict[str, float] = {}
    reasons, advice = [], []
    if voice_fake is not None:
        signals["voice"] = float(voice_fake)
        if voice_fake >= 0.5:
            reasons.append(f"Голос має ознаки синтезу ({100 * voice_fake:.0f} %)")
    if speaker_mismatch is not None:
        signals["speaker"] = 1.0 if speaker_mismatch else 0.0
        if speaker_mismatch:
            reasons.append("Голос не збігається із заявленою людиною")
    if content is not None:
        signals["content"] = content.score
        signals["links"] = content.links_score
        reasons += content.combos
        reasons += [f"{name}" for name in content.tactics.values()]
        reasons += [f"Посилання/номер {l.value}: {'; '.join(l.reasons)}" for l in content.links if l.risk >= 0.3]
    p_safe = 1.0
    for k, v in signals.items():
        p_safe *= 1.0 - WEIGHTS[k] * min(max(v, 0.0), 1.0)
    score = 1.0 - p_safe
    level, kind = next((name, kind) for thr, name, kind in LEVELS if score < thr)
    if score >= 0.3:
        advice.append("Не називайте коди з SMS, PIN, CVV і паролі — банк і поліція їх ніколи не питають.")
        advice.append("Завершіть розмову й передзвоніть людині чи в банк самі, на відомий вам номер.")
    if content is not None and "relative" in content.tactics:
        advice.append("Поставте питання, відповідь на яке знає лише ваш родич, або домовтеся про «кодове слово».")
    if content is not None and any(l.kind == "url" and l.risk >= 0.3 for l in content.links):
        advice.append("Не переходьте за посиланням і не вводьте там дані картки.")
    return Risk(round(score, 3), level, kind, signals, reasons, advice)
