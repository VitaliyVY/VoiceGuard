"""Аналіз змісту розмови чи повідомлення на ознаки шахрайських схем.

Підхід прозорий і пояснюваний: словник ознак, згрупованих за тактиками
соціальної інженерії (запит кодів, переказ коштів, терміновість, секретність,
видавання себе за банк чи державний орган, «родич у біді», виграші й виплати,
віддалений доступ, інвестиції). Кожна ознака — регулярний вираз за основами
слів (українська, російська, суржик), бо розпізнаний текст містить різні
словоформи й помилки. Окремо — посилання, номери телефонів і карток із
евристиками фішингових доменів.

Оцінка зважує не лише кількість ознак, а й їх поєднання: «служба безпеки
банку» + «код з SMS» набагато небезпечніше, ніж кожна ознака окремо.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse

# ----------------------------------------------------------------------------- тактики
TACTICS: dict[str, dict] = {
    # strong — прохання або пряма ознака схеми (велика вага);
    # weak — лише згадка теми, яка часто трапляється і в звичайних текстах (мала вага;
    # сила такої ознаки проявляється тільки в поєднанні з проханням — див. COMBOS)
    "codes": {
        "name": "Просять код або пароль", "strong_w": 0.9, "weak_w": 0.2,
        "strong": [r"код\w*\s+(з|із|из|в|у)\s+(смс|sms|повідомлен|сообщен)", r"(смс|sms)[- ]?код",
                   r"код\w* підтвердж", r"код\w* подтвержд", r"\bcvv\b|\bcvc\b|три цифр\w* на (звороті|обороте|зворот)",
                   r"одноразов\w* (код|парол)",
                   r"(назвіть|назви|продиктуйте|продиктуй|скажіть|скажи|надішліть|надішли|повідомте|введіть|введи|скиньте|скинь|"
                   r"назовите|скажите|сообщите|пришлите)[,.\s]+(\w+[,.\s]+){0,3}(код|парол|пін|пин|code)"],
        "weak": [r"парол\w*", r"\bпін[- ]?код\w*|\bпин[- ]?код\w*"],
    },
    "money": {
        "name": "Просять переказати гроші або дані картки", "strong_w": 0.6, "weak_w": 0.15,
        "strong": [r"переки(нь|ньте|нути)\s+(\w+\s+)?(гроші|грошей|кошти|деньги|на карт|\d)",
                   r"ски(нь|ньте|нути)\s+(\w+\s+)?(гроші|грошей|деньги|на карт|\d)",
                   # сума «на картку» — стійко до помилок розпізнавання дієслова («воскинь» замість «скинь»)
                   r"\d+\s*(тисяч\w*|тис\.?|грн|гривень|гривні)\s+(\w+\s+)?на\s+(картк|карт|рахун)",
                   r"(перека(жіть|жи|зати)|переве(діть|ди|сти)|переведите)\s+(\w+\s+){0,3}(на|кошти|гроші|деньги)",
                   r"(безпечн|безопасн|резервн|захищен|защищен)\w* (рахун|сч[её]т)\w*",
                   r"(назвіть|назви|продиктуйте|надішліть|надішли|введіть|скиньте|скинь|назовите|пришлите)[,\s]+(\w+[,\s]+){0,3}(номер\w* карт|дан\w* карт)",
                   r"(оплат|сплат)\w*\s+(доставк|комісі|комисси|мит|штраф)", r"термін дії карт|срок действия карт"],
        "weak": [r"перека(з|ж)\w*", r"поповн\w*|пополн\w*", r"номер\w* карт\w*", r"\biban\b"],
    },
    "personal": {
        "name": "Просять «підтвердити» особисті дані", "strong_w": 0.25, "weak_w": 0.05,
        "strong": [r"(підтвердіть|підтвердити|надайте|надати|назвіть|продиктуйте|уточніть|перевірити|"
                   r"подтвердите|подтвердить|назовите|уточните|проверить)[,\s]+(\w+[,\s]+){0,4}"
                   r"(дані|даних|інформаці\w*|ім'я|адрес\w*|дат\w* народж\w*|паспорт\w*|ідентифікаці\w*|рнокпп|"
                   r"данные|информаци\w*|имя|дат\w* рожд\w*)",
                   r"останні\w*\s+(чотири|4)\s+цифр|последни\w*\s+(четыре|4)\s+цифр",
                   r"номер\w* (паспорт|соціальн|ідентифікаційн|рнокпп)"],
        "weak": [r"особист\w* дан\w*|персональн\w* дан\w*"],
    },
    "urgency": {
        "name": "Тиснуть терміновістю", "strong_w": 0.25, "weak_w": 0.1,
        "strong": [r"не кладіть трубку|не кладите трубку", r"останн\w* шанс|последн\w* шанс",
                   r"(протягом|в течение)\s+\w*\s*(годин|хвилин|часа|минут)",
                   r"(карт|рахун)\w*\s+(\w+\s+)?(буде|будуть|будет)\s+заблок"],
        "weak": [r"терміново|срочно", r"негайно|немедленно", r"прямо зараз|прямо сейчас", r"заблоку\w*|заблокир\w*"],
    },
    "secrecy": {
        "name": "Просять нікому не казати", "strong_w": 0.55, "weak_w": 0.55,
        "strong": [r"ні\s?кому не (кажи|кажіть|говори|говоріть|розповідай)", r"ни\s?кому не (говори|говорите|рассказывай)",
                   r"не (кажи|кажіть|говори)\s+(мамі|татові|батькам|рідним|в банку)", r"(це|это)\s+(таємниц|секрет|конфіденц)",
                   r"не (дзвоніть|звоните)\s+(в банк|у банк|на гарячу)"],
        "weak": [],
    },
    "authority": {
        "name": "Видають себе за банк, поліцію чи державу", "strong_w": 0.5, "weak_w": 0.1,
        "strong": [r"служб\w* (безпеки|безопасности|без\s?пеки)\s*(\w+\s+)?(банк|приват|моно|ощад)", r"(працівник|співробітник|сотрудник)\w* (банк|поліці|полици)",
                   r"(ваш|ваша|вашу|вашій)\s+(карт|рахун)\w*\s+(\w+\s+)?(заблок|скомпр)\w*",
                   r"підозріл\w* (операц|транзакц)|подозрительн\w* (операц|транзакц)",
                   r"(від|из|з)\s+(імені|имени)\s+(банк|держав)",
                   r"несанкціонован\w* (транзакц|операц|покуп|доступ|списан)|несанкционированн\w* (транзакц|операц|доступ)",
                   r"(обліков\w* запис|акаунт|аккаунт|рахун)\w*\s+(\w+\s+)?(скомпрометован|зламан|взломан)",
                   r"(вас|вам)\s+(турбу|дзвон|беспоко|звон)\w*\s+(\w+\s+)?(банк|поліці|полици|служб)"],
        "weak": [r"служб\w* безпеки|служб\w* безопасности", r"кіберполіц\w*|киберполиц\w*", r"\bсбу\b|\bнбу\b|національн\w* банк", r"слідч\w*|следовател\w*",
                 r"(приват|моно|ощад|райффайзен|пумб)\w*\s*банк|приватбанк|монобанк|ощадбанк"],
    },
    "relative": {
        "name": "«Родич у біді»", "strong_w": 0.5, "weak_w": 0.05,
        "strong": [r"(потрапи|попа)\w*\s+(в|у)\s+(дтп|аварі|аварию|біду|беду)",
                   r"мене\s+(затрима|забра|арешт)\w*|меня\s+(задержа|забра|арест)\w*",
                   r"(мамо|мама|тату|папа|бабусю|бабушка)[,!]?\s+(це|это)\s+я\b", r"\b(я|ми)\s+(в|у)\s+(біді|беде)"],
        "weak": [r"\bмамо\b|\bмама\b|\bтату\b|\bпапа\b|\bбабусю\b|\bбабушка\b",
                 r"\b(син|сина|сину|синок|синочку|донька|доню|доня|дочь|дочка|доця|онук|онуче|внук|внучка)\b",
                 r"\bдтп\b", r"лікарн\w*|больниц\w*", r"адвокат\w*"],
    },
    "prize": {
        "name": "Виграш, виплата або допомога", "strong_w": 0.45, "weak_w": 0.1,
        "strong": [r"(вам|ви|вы)\s+(\w+\s+)?(нарахован|виграл|начислен|выиграл)\w*", r"(вам|вами)\s+нараховано",
                   r"(щоб|аби|чтобы)\s+(отримати|получить|забрати|забрать)\s+(\w+\s+)?(виплату|выплату|компенсаці|компенсаци|приз|виграш|выигрыш)",
                   r"(ви|вы)\s+(стали|став)\s+(переможц|победител)"],
        "weak": [r"вигра(ли|в|ш)\w*|выигра\w*", r"(приз|подарунок|подарок)\w*", r"компенсаці\w*|компенсаци\w*",
                 r"виплат\w*|выплат\w*", r"поверне\w* (кошт|грош)|возврат\w* (средств|денег)", r"гуманітарн\w* допомог|гуманитарн\w* помощ", r"\bдія\b|\bдии\b|\bдия\b",
                 r"\bоон\b|червон\w* хрест|красн\w* крест"],
    },
    "remote": {
        "name": "Просять встановити програму чи перейти за посиланням", "strong_w": 0.6, "weak_w": 0.15,
        "strong": [r"anydesk|енідеск|эни ?деск|teamviewer|тім ?в'?ювер|rustdesk|quick ?support",
                   r"(встановіть|встанови|установіть|установите|завантажте|завантаж|скачайте|скачай)\s+(\w+\s+)?(програм|додат|приложен|застосун)",
                   r"(перейд|перейти|натисн|нажм)\w*\s+(за|по|на)\s+(посилан|ссылк)",
                   r"демонстраці\w* екран|демонстраци\w* экран"],
        "weak": [r"телеграм[- ]?бот|бот\w* в телеграм"],
    },
    "investment": {
        "name": "«Гарантований прибуток»", "strong_w": 0.45, "weak_w": 0.05,
        "strong": [r"гарантован\w* (дохід|прибут)|гарантированн\w* (доход|прибыл)",
                   r"(подвої|удвої|удвои|умнож)\w*\s+(\w+\s+)?(гроші|кошти|капітал|деньги|вклад|депозит|заощадж)",
                   r"(пасивн|пассивн)\w* дох\w*\s+(від|от)\s+\d"],
        "weak": [r"криптовалют\w*|біткоїн|биткоин|\busdt\b", r"(біржа|биржа|брокер)\w*", r"інвест\w*|инвест\w*",
                 r"(пасивн|пассивн)\w* дох"],
    },
}

# поєднання тактик, характерні для відомих схем (множник ризику)
COMBOS = [({"authority", "codes"}, "«Служба безпеки банку» просить код — класична схема"),
          ({"relative", "money"}, "«Родич у біді» просить гроші — класична схема"),
          ({"authority", "money"}, "Нібито банк чи держава просить переказ на «безпечний рахунок»"),
          ({"prize", "codes"}, "«Виплата чи виграш» в обмін на код"),
          ({"prize", "money"}, "«Виплата чи виграш» в обмін на дані картки чи «оплату комісії»"),
          ({"remote", "authority"}, "Нібито банк просить встановити програму віддаленого доступу"),
          ({"authority", "personal"}, "Нібито банк чи служба просить «підтвердити» особисті дані"),
          ({"prize", "personal"}, "«Повернення коштів чи виплата» в обмін на особисті дані"),
          ({"secrecy", "money"}, "Просять гроші й наполягають, щоб нікому не казали")]

# ----------------------------------------------------------------------------- посилання й номери
OFFICIAL = {"privatbank.ua", "privat24.ua", "monobank.ua", "oschadbank.ua", "diia.gov.ua", "gov.ua", "novaposhta.ua",
            "ukrposhta.ua", "olx.ua", "prom.ua", "rozetka.com.ua", "raiffeisen.ua", "pumb.ua", "abank.com.ua",
            "telegram.org", "t.me", "google.com", "youtube.com", "viber.com", "kyivstar.ua", "vodafone.ua", "lifecell.ua"}
BRANDS = ["privat", "privat24", "mono", "monobank", "oschad", "diia", "novaposhta", "nova-poshta", "ukrposhta", "olx",
          "rozetka", "raiffeisen", "pumb", "kyivstar", "vodafone", "lifecell", "gov", "viplata", "vyplata", "dopomoga"]
SUSPICIOUS_TLD = {"xyz", "top", "click", "icu", "online", "site", "shop", "live", "buzz", "monster", "rest", "link",
                  "store", "info", "su", "ru", "tk", "ml", "ga", "cf", "gq", "cfd", "sbs", "bond"}
SHORTENERS = {"bit.ly", "tinyurl.com", "cutt.ly", "goo.gl", "t.co", "is.gd", "clck.ru", "shorturl.at", "rb.gy", "tiny.cc"}

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+|\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>\"']*)?", re.I)
PHONE_RE = re.compile(r"\+?38[\s-]?\(0\d{2}\)[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b|\+?380[\s-]?\(?\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b|\b0\d{2}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}\b|\+\d{1,3}[\s-]?\d{2,4}[\s-]?\d{3}[\s-]?\d{2,4}")
CARD_RE = re.compile(r"\b(?:\d[ -]?){15,18}\d\b")


@dataclass
class Finding:
    tactic: str
    name: str
    phrase: str
    start: float | None = None          # секунда в записі (якщо текст із розпізнавання)


@dataclass
class LinkFinding:
    kind: str                           # url | phone | card
    value: str
    risk: float                         # 0..1
    reasons: list[str] = field(default_factory=list)


@dataclass
class ContentReport:
    score: float                        # 0..1 — ризик за змістом
    findings: list[Finding]
    tactics: dict[str, str]             # тактика → назва
    combos: list[str]
    links: list[LinkFinding]
    links_score: float

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(t: str) -> str:
    return t.lower().replace("’", "'").replace("ʼ", "'").replace("ё", "е")


def check_url(raw: str) -> LinkFinding:
    url = raw if "://" in raw else "http://" + raw
    host = (urlparse(url).hostname or "").lower().strip(".")
    reasons, risk = [], 0.0
    if not host:
        return LinkFinding("url", raw, 0.0)
    official = any(host == d or host.endswith("." + d) for d in OFFICIAL)
    tld = host.rsplit(".", 1)[-1]
    if not official:
        brand = next((b for b in BRANDS if b in host.replace(".", "-")), None)
        if brand:
            reasons.append(f"схоже на «{brand}», але це не офіційний домен")
            risk += 0.6
        if tld in SUSPICIOUS_TLD:
            reasons.append(f"підозріла доменна зона .{tld}")
            risk += 0.3
        if host in SHORTENERS:
            reasons.append("скорочене посилання приховує справжню адресу")
            risk += 0.4
        if host.startswith("xn--") or ".xn--" in host:
            reasons.append("домен з підмінними символами (punycode)")
            risk += 0.5
        if re.fullmatch(r"[\d.]+", host):
            reasons.append("замість назви сайту — IP-адреса")
            risk += 0.5
        if host.count("-") >= 2 or len(host) > 30:
            reasons.append("довга адреса з багатьма дефісами")
            risk += 0.15
        if "@" in raw:
            reasons.append("символ @ в адресі (маскування справжнього сайту)")
            risk += 0.4
        if not reasons:
            reasons.append("невідомий сайт — перевірте, перш ніж вводити дані")
            risk += 0.1
    return LinkFinding("url", raw, min(risk, 1.0), reasons)


def find_links(text: str) -> list[LinkFinding]:
    out: list[LinkFinding] = []
    for m in URL_RE.finditer(text):
        v = m.group(0).rstrip(".,;:!?)")
        if re.fullmatch(r"\d+(\.\d+)+", v):            # числа на кшталт 2.5 — не адреси
            continue
        out.append(check_url(v))
    for m in CARD_RE.finditer(text):
        out.append(LinkFinding("card", m.group(0), 0.7, ["у повідомленні номер платіжної картки — "
                                                         "ніколи не надсилайте й не диктуйте його незнайомим"]))
    for m in PHONE_RE.finditer(text):
        if any(m.group(0) in c.value for c in out if c.kind == "card"):
            continue
        out.append(LinkFinding("phone", m.group(0).strip(), 0.2, ["номер телефону — передзвонюйте лише на номер "
                                                                  "з офіційного сайту чи картки банку"]))
    return out


def spans(text: str, keep: set[str] | None = None) -> list[tuple[int, int, str]]:
    """Позиції підозрілих фрагментів для підсвічування: (початок, кінець, тактика або 'link').
    keep — лише ці тактики (ті, що аналіз показав); None — усі."""
    nt = _norm(text)                     # та сама довжина, що й text (заміни 1:1)
    out = []
    for key, tac in TACTICS.items():
        if keep is not None and key not in keep:
            continue
        for pat in tac["strong"] + tac["weak"]:
            out += [(m.start(), m.end(), key) for m in re.finditer(pat, nt)]
    out += [(m.start(), m.end(), "link") for m in URL_RE.finditer(text)]
    out += [(m.start(), m.end(), "link") for m in CARD_RE.finditer(text)]
    out += [(m.start(), m.end(), "link") for m in PHONE_RE.finditer(text)]
    # межі — до кінця слова, щоб не різати «ПриватБанк|у»
    grown = []
    for s_, e_, k in out:
        while k != "link" and e_ < len(text) and (text[e_].isalpha() or text[e_] == "'"):
            e_ += 1
        grown.append((s_, e_, k))
    out = sorted(grown)
    merged: list[tuple[int, int, str]] = []
    for s, e, k in out:                  # перекриття — залишаємо перший
        if merged and s < merged[-1][1]:
            continue
        merged.append((s, e, k))
    return merged


def analyze(text: str, segments: list | None = None) -> ContentReport:
    """Ризик за змістом. segments — сегменти розпізнавання (start, text) для прив'язки до часу."""
    parts = [(getattr(s, "start", None), getattr(s, "text", "")) for s in segments] if segments else [(None, text)]
    findings: list[Finding] = []
    weight: dict[str, float] = {}       # тактика → вага найсильнішої знайденої ознаки
    # розпізнавання ділить мовлення на сегменти, і прохання може розірватися паузою
    # («продиктуйте, будь ласка… | код з SMS») — тому наостанок перевіряється й увесь текст
    whole = [(None, " ".join(t for _, t in parts))] if len(parts) > 1 else []
    for k, (start, t) in enumerate(parts + whole):
        nt = _norm(t)
        for key, tac in TACTICS.items():
            for group in ("strong", "weak"):
                m = next((m for pat in tac[group] if (m := re.search(pat, nt))), None)
                if m:
                    w = tac[f"{group}_w"]
                    if k < len(parts) or w > weight.get(key, 0.0):
                        findings.append(Finding(key, tac["name"], t[max(m.start() - 25, 0): m.end() + 25].strip(), start))
                    weight[key] = max(weight.get(key, 0.0), w)
                    break
    found = set(weight)
    strong = {k for k in found if weight[k] >= TACTICS[k]["strong_w"]}
    # поєднання рахується, лише якщо хоча б одна тактика пари — прохання, а не просто згадка
    hit = [(need, desc) for need, desc in COMBOS if need <= found and need & strong]
    combos = [desc for _, desc in hit]
    # імовірнісне «або»: кожна тактика — незалежний доказ; поєднання з COMBOS підсилює
    p_safe = 1.0
    for w in weight.values():
        p_safe *= 1.0 - w
    score = 1.0 - p_safe
    if combos:
        score = 1.0 - (1.0 - score) * 0.4 ** len(combos)
    # показуємо прохання й ті згадки, що входять у знайдене поєднання; решта згадок — лише в оцінці
    shown = strong | {k for need, _ in hit for k in need}
    links = find_links(text)
    links_score = max((l.risk for l in links), default=0.0)
    return ContentReport(round(score, 3), [f for f in findings if f.tactic in shown],
                         {k: TACTICS[k]["name"] for k in sorted(shown)}, combos, links, round(links_score, 3))
