"""ШІ-аналітик: пояснення висновку людською мовою локальною мовною моделлю (Ollama).

Рішення приймають виміряні модулі (детектор голосу, верифікація мовця, аналіз
змісту, перевірка посилань). Мовна модель отримує їхні результати й текст
розмови та лише пояснює висновок і дає пораду — вона не може змінити оцінку
ризику. Модель працює локально (сервер Ollama на 127.0.0.1:11434), тож записи
й тексти не залишають комп'ютер. Якщо Ollama не запущено — програма працює
без цього модуля.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable

URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma3:4b"

SYSTEM = (
    "Ти — помічник програми VoiceGuard, яка захищає людей від телефонного шахрайства. Тобі дають результати "
    "вимірювальних модулів і текст розмови чи повідомлення. Поясни користувачу висновок українською мовою, просто "
    "й коротко (до 120 слів).\n"
    "Правила: 1) не змінюй і не заперечуй числові результати модулів — рівень ризику вже визначено; "
    "2) не вигадуй фактів, яких немає в тексті; цитуй лише реальні фрази з тексту; 3) якщо доказів мало — так і скажи; "
    "4) не давай юридичних порад; 5) поради добирай до схеми: якщо видають себе за родича — порадь покласти "
    "слухавку й передзвонити йому на звичний номер або поставити запитання, відповідь на яке знає лише він; "
    "якщо за банк чи державу — передзвонити на номер з картки чи офіційного сайту.\n"
    "Формат відповіді:\n**Висновок:** одне речення.\n**Чому:** 2–3 речення з короткими цитатами.\n"
    "**Що робити:** 2–3 конкретні поради списком."
)


def available(model: str = DEFAULT_MODEL, timeout: float = 2.0) -> tuple[bool, str]:
    """(чи можна користуватися, пояснення)."""
    try:
        with urllib.request.urlopen(f"{URL}/api/tags", timeout=timeout) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
    except Exception:                                            # noqa: BLE001
        return False, "Сервер Ollama не запущено. Запустіть програму Ollama й повторіть."
    if not any(n == model or n.split(":")[0] == model.split(":")[0] and model in n for n in names):
        return False, f"Модель {model} не завантажено (у терміналі: ollama pull {model})."
    return True, "готово"


def facts_from_verdict(v) -> dict:
    """Стислий опис результатів перевірки голосу для моделі."""
    d = {"рівень_ризику": v.risk["level"], "ризик_відсотків": round(100 * v.risk["score"]),
         "причини": v.risk["reasons"],
         "голос": f"справжній з упевненістю {100 * v.cm_confidence:.0f} %" if v.cm_genuine
         else f"має ознаки синтезу (впевненість «справжній» лише {100 * v.cm_confidence:.0f} %)",
         "смуга_частот_кГц": round(v.analysis.get("bandwidth_hz", 0) / 1000, 1)}
    if v.speaker:
        d["мовець"] = f"заявлено «{v.speaker}», голос {'збігається' if v.asv_match else 'НЕ збігається'}"
    if v.transcript:
        d["текст_розмови"] = " ".join(s["text"] for s in v.transcript)[:1500]
    return d


def facts_from_text(result: dict) -> dict:
    return {"рівень_ризику": result["risk"]["level"], "ризик_відсотків": round(100 * result["risk"]["score"]),
            "причини": result["risk"]["reasons"], "текст_повідомлення": result["text"][:1500]}


def explain(facts: dict, on_token: Callable[[str], None] | None = None, model: str = DEFAULT_MODEL,
            timeout: float = 300.0) -> str:
    """Пояснення висновку. on_token — для поступового виведення відповіді."""
    body = {"model": model, "stream": True,
            "options": {"temperature": 0.2, "num_ctx": 2048, "num_predict": 350},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "Результати модулів і текст:\n" +
                          json.dumps(facts, ensure_ascii=False, indent=1)}]}
    req = urllib.request.Request(f"{URL}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    out = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.strip():
                continue
            msg = json.loads(line)
            tok = msg.get("message", {}).get("content", "")
            if tok:
                out.append(tok)
                if on_token:
                    on_token(tok)
            if msg.get("done"):
                break
    return "".join(out)
