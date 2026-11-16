# Експерименти: дані, протокол і відтворення

Для **запуску програми** нічого з цього не потрібно — навчені моделі вже є в `models/`, а результати — у `results/`.
Цей документ — для перевірки та відтворення результатів кваліфікаційної роботи.

## Протокол оцінювання

* **train** — навчання; **dev** — вибір налаштувань і порогів; **eval** — лише підсумковий звіт (нові мовці й частково нові методи атак A07–A19).
  Рішення, що виграє лише на eval, не обирається.
* Метрики: **EER** (рівна частка хибних прийнять і відхилень), minDCF; для каскаду — FRR, FAR і частка прийнятих підробок.
* Стійкість до каналу: кожен запис dev/eval пропускається через канал і лише потім оцінюється. 10 каналів:
  `clean opus32 opus16 opus8 telephone amrnb amrwb gsm noise20 noise10`.
* Аугментація під час навчання: `clean opus16 telephone noise20`; кодеки `opus32 opus8 amrnb amrwb gsm` у навчанні не трапляються.
* Випадковість фіксована (`--seed 0`); для LightCNN — три зерна (0, 1, 2).

## Дані

| Набір | Ліцензія | Звідки | Куди покласти |
|---|---|---|---|
| ASVspoof 2019 LA (Wang et al., 2020) | ODC-By | https://datashare.ed.ac.uk/handle/10283/3336 — `LA.zip`, 7,6 ГБ, MD5 `30c98f11d8b2bc21f2c257bfd78bb5c5` | розпакувати в `data/external/asvspoof2019/` |
| Common Voice 19 uk | CC0 | завантажує `scripts/uk_real.py` (частинами через HTTP Range) | `data/uk/real/` |
| Piper uk_UA, MMS-TTS ukr | rhasspy/piper-voices; CC BY-NC 4.0 | синтезує `scripts/uk_fake.py` | `data/uk/fake/` |
| UA-GEC (Syvokon et al., 2023) | CC BY 4.0 | https://github.com/grammarly/ua-gec | `data/external/ua-gec/` |
| BothBosu/scam-dialogue | Apache-2.0 | https://huggingface.co/datasets/BothBosu/scam-dialogue — `scam-dialogue_test.csv` | `data/external/scam_dialogue/` |

Зовнішня перевірка (розділ 7) — набори, які не використовувались ні в навчанні, ні в адаптації:

| Набір | Ліцензія | Звідки | Що взято |
|---|---|---|---|
| ML-ITW (Multilingual In-The-Wild, 2026), українська частина | не вказана | https://huggingface.co/datasets/nccm2p2/ML-ITW | 73 справжні записи і 17 дипфейків українських публічних осіб (Facebook, X) |
| DRSSU (Lipianina-Honcharenko et al., *Scientific Data*, 2025) | CC BY 4.0 | https://github.com/bohutahennadii/audio_database | 300 підробок (Edge-TTS `uk-UA-OstapNeural` + RVC) і 300 справжніх (Common Voice 16.1) |
| FLEURS uk, test | CC BY 4.0 | https://huggingface.co/datasets/google/fleurs | 300 справжніх записів |

Завантаження: `python scripts/uk_external_download.py --drssu 300 --fleurs 300`.

Українські записи — лише з відкритих наборів, голоси — зі згоди авторів записів (Common Voice) або синтетичні.

## Відтворення

```bash
pip install -r requirements-experiments.txt
export PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=1     # Windows PowerShell: $env:PYTHONIOENCODING="utf-8"
CH="clean opus32 opus16 opus8 telephone amrnb amrwb gsm noise20 noise10"
```

Готові черги запусків (bash; на Windows — Git Bash) лежать у `scripts/queue_*.sh`; журнали оригінальних запусків — `results/run*_log.txt`.

### 1. Базові системи на ASVspoof 2019 LA, ефект тиші, стійкість до каналу

```bash
# детектор LFCC-GMM: без відкидання тиші / з ним (Müller et al., 2021)
python scripts/cm_gmm.py --feature lfcc --components 128 --channels $CH
python scripts/cm_gmm.py --feature lfcc --trim --components 128 --channels $CH
# LFCC 0–4 кГц, навчання на чистих і з аугментацією каналами
python scripts/cm_gmm.py --feature lfcc4k --trim --components 128 --dev-per-attack 200 --eval-per-attack 200 --channels $CH
python scripts/cm_gmm.py --feature lfcc4k --trim --components 128 --dev-per-attack 200 --eval-per-attack 200 \
       --channels $CH --train-channels clean opus16 telephone noise20
# верифікація мовця GMM-UBM і частка підробок, що її проходять
python scripts/asv_gmm_ubm.py --trim --components 256 --split eval --channels $CH
```

Результати: `results/cm_gmm_*.json`, `results/cm_scores_*.csv`, `results/asv_gmm_ubm_mfcc_trim_g256_eval.json`, моделі — `models/`.

### 2. Нейромережеві детектори

```bash
# LightCNN на LFCC: чисті / аугментація; три зерна (queue_seeds.sh)
python scripts/cm_cnn.py --feature lfcc4k --channels $CH
python scripts/cm_cnn.py --feature lfcc4k --channels $CH --train-channels clean opus16 telephone noise20 --seed 0
# об'єднання GMM + CNN на рівні оцінок
python scripts/fuse.py results/cm_scores_lfcc4k_trim_g128_aug-clean-opus16-telephone-noise20.csv \
                       results/cm_cnn_scores_lfcc4k_trim_aug-clean-opus16-telephone-noise20.csv
# WavLM Base+ + логістична регресія (основний детектор): ознаки → класифікатор
python scripts/ssl_cm.py --stage all
python scripts/ssl_cpnap.py                    # проєкція завад каналу (CP-NAP), порівняння
```

Результат: `models/cm_ssl.pkl`, `results/cm_ssl.json`, `results/cm_ssl_scores.csv`.
Ознаки WavLM для всіх каналів кешуються в `data/cache/` (кілька ГБ); на CPU етап `extract` триває десятки годин.

### 3. Повна система «верифікація + детектор» (каскад)

```bash
python scripts/tandem.py --cm models/cm_gmm_lfcc_trim_g128.pkl \
       models/cm_gmm_lfcc4k_trim_g128_aug-clean-opus16-telephone-noise20.pkl models/cm_ssl.pkl   # queue_tandem.sh
python scripts/tandem.py --cm models/cm_ssl.pkl --feature ssl --names "WavLM" --workers 3 --out results/tandem_wavlm_scores.json
python scripts/tandem_points.py
```

Результат: `results/tandem*.json` — частка підробок, що проходять верифікацію, 71,9 % → 11,4 %.

### 4. Українська мова

```bash
python scripts/uk_real.py --test 400 --train 300     # справжні записи Common Voice 19 uk
python scripts/uk_fake.py                            # синтез Piper і MMS-TTS (test 200+200, adapt 118)
python scripts/uk_eval.py                            # англомовні детектори на українських записах + адаптація → models/cm_ssl_uk.pkl
python scripts/uk_calibrate.py                       # пороги адаптованої моделі для програми
python scripts/uk_honesty.py                         # перевірка «синтез чи умови запису»: підробки з MP3 і шумом кімнати
python scripts/uk_fake_mms_adapt.py --n 118          # MMS для перехресної перевірки синтезаторів
python scripts/uk_extra.py                           # крива навчання, бутстреп-інтервали, Piper↔MMS
```

Результати: `results/uk_eval.json` (EER 64 % → 3,2 %), `results/uk_honesty.json`, `results/uk_extra.json`.

### 5. Аналіз змісту

```bash
python scripts/scam_fp.py --split train              # налаштування правил (лише train)
python scripts/scam_fp.py --split test --show 10     # хибні тривоги на звичайних текстах UA-GEC → 0,1 %
python scripts/scam_translate.py --per-type 10       # переклад діалогів українською (потрібна Ollama)
python scripts/scam_eval.py                          # виявлення шахрайських розмов → 58 % без хибних тривог
```

### 6. Демонстраційні файли і швидкодія

```bash
python scripts/make_demo.py                          # набір demo/ з даних Common Voice і Piper
```

### 7. Зовнішня перевірка на незалежних українських наборах

```bash
python scripts/uk_external_download.py --drssu 300 --fleurs 300
python scripts/uk_external.py          # → results/uk_external.json, results/uk_external_scores.csv
```

EER (журнал — `results/run14_log.txt`):

| Пара «справжні vs підробки» | WavLM, лише ASVspoof (англ.) | WavLM, адаптований (програма) |
|---|---|---|
| B. FLEURS vs DRSSU (Edge-TTS + RVC) | 34,3 % | **1,0 %** |
| C. DRSSU-справжні (Common Voice 16.1) vs DRSSU-підробки | 44,3 % | 12,3 % |
| A. ML-ITW: справжні vs дипфейки з соцмереж | 82,3 % | **69,9 %** |

Помилки адаптованого детектора на порогах програми (м'яка / звичайна / сувора):
FLEURS — відхилено справжніх 0,3 / 1,0 / 2,7 %; DRSSU-підробки — пропущено 13,7 / 1,0 / 0,0 %;
ML-ITW — відхилено справжніх 11,0 / 19,2 / 27,4 %, пропущено дипфейків **17 з 17** на всіх порогах.

Висновки:
* Адаптація узагальнюється на **інші синтезатори й перетворення голосу** (Edge-TTS + RVC не було в адаптації — лише Piper) і на **незалежні справжні записи** (FLEURS): EER 34,3 % → 1,0 %.
* Справжні записи з Common Voice детектор частіше вважає підозрілими (12,3 %), як і в основних експериментах: побутові мікрофони й шум.
* **Дипфейки з соцмереж детектор не виявляє.** Записи ML-ITW пройшли обробку платформами (стиснення відео, музика, монтаж), а дипфейки, ймовірно, створено комерційними сервісами клонування голосу. Підробки Залужного отримали вищі оцінки «справжності», ніж його справжні записи. Вибірка мала (17 дипфейків, 3 особи), але результат однозначний: це головне обмеження системи і напрям подальшої роботи (адаптація на записах із соцмереж і сучасних сервісах клонування).

## Головні результати

| Експеримент | Файл | Значення |
|---|---|---|
| Середній EER на 10 каналах, базовий LFCC-GMM → WavLM | `cm_gmm_lfcc_trim_g128.json`, `cm_ssl.json` | 34,3 % → 7,4 % |
| EER WavLM на кодеках, яких не було в навчанні | `cm_ssl.json` | 8,4 % |
| Підробки, що проходять верифікацію: без детектора → з WavLM (clean) | `tandem.json` → `tandem_wavlm.json` | 71,9 % → 11,4 % |
| WavLM (англ.) на українських записах, clean / opus16 / amrnb | `uk_eval.json` | 64,3 % / 62,8 % / 65,5 % |
| Після адаптації (118 пар, лише Piper) | `uk_eval.json`, `uk_extra.json` | 3,2–7,0 % |
| Адаптація на 30 парах | `uk_extra.json` | ≈ 13 % |
| Перехресно: адаптація Piper → тест MMS / MMS → Piper | `uk_extra.json` | ≈ 1 % / 39–50 % |
| Хибні тривоги аналізу змісту (UA-GEC test, 1033 абзаци) | `scam_fp_test.json` | 0,1 % |
| Виявлені шахрайські розмови | `scam_eval.json` | 58 % |

## Обладнання й час

Усі експерименти виконано на CPU (AMD Ryzen 5, 6 ГБ RAM, без відеокарти). Найдовші етапи — витяг ознак WavLM (десятки годин) і навчання GMM на ASVspoof (≈ 10–20 хв на конфігурацію на канал).
