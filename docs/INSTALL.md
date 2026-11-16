# Протокол встановлення та запуску

Перевірено на: Windows 10 (x64), Python 3.14, ноутбук AMD Ryzen 5, 6 ГБ RAM, без відеокарти.
Також має працювати на Linux і macOS з Python 3.10+.

## 1. Вимоги

| Компонент | Обов'язково | Примітка |
|---|---|---|
| Python 3.10 або новіший | так | https://www.python.org/downloads/ — під час встановлення на Windows позначити «Add python.exe to PATH» |
| Git | так (або завантажити ZIP) | https://git-scm.com |
| ≈ 2 ГБ на диску | так | бібліотеки + моделі WavLM (≈ 360 МБ) і Whisper small (≈ 470 МБ) |
| Інтернет | лише перший запуск | моделі WavLM і Whisper завантажуються автоматично й кешуються |
| Мікрофон | ні | для запису з мікрофона в програмі |
| Ollama + `gemma3:4b` | ні | лише для кнопки «✨ Пояснити» (пояснення від ШІ) |

ffmpeg окремо ставити **не потрібно** — він входить у пакет `imageio-ffmpeg`.

## 2. Отримання коду

```bash
git clone https://github.com/VitaliyVY/VoiceGuard.git
cd VoiceGuard
```

Або на сторінці репозиторію: **Code → Download ZIP**, розпакувати й відкрити термінал у цій теці.

## 3. Віртуальне середовище та залежності

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Якщо PowerShell забороняє запуск скриптів: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
> На Linux для `sounddevice` може знадобитися `sudo apt install libportaudio2`.

Встановлення PyTorch лише для CPU (менший обсяг, необов'язково):

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 4. Перевірка встановлення

```bash
python -m unittest discover -s tests
```

Очікувано:

```
.......................
----------------------------------------------------------------------
Ran 23 tests in 3.6s

OK
```

## 5. Запуск десктопної програми

```bash
python VoiceGuard.pyw
```

На Windows можна просто двічі клацнути `VoiceGuard.pyw` (вікно консолі не з'являється).

1. Дочекатися зеленого індикатора **«● Готово»** внизу ліворуч (перший раз — до кількох хвилин через завантаження моделей, далі ≈ 20–45 с).
2. **Перевірка голосу** → «Відкрити файл» (або «Записати» з мікрофона) → за потреби обрати мовця → **«Перевірити»**.
3. Результат: вердикт і рівень ризику, шкали «Справжність голосу» / «Збіг із мовцем», графік «Підозрілість у часі», розпізнаний текст із підсвіченими тактиками, причини й поради.

Сторінки програми:

| Сторінка | Призначення |
|---|---|
| Перевірка голосу | аналіз аудіофайлу або запису з мікрофона |
| Перевірка тексту | SMS / Telegram / Viber: тактики, фішингові посилання, номери карток |
| Мовці | реєстрація голосу людини за 3–5 фразами (≈ 15 с мовлення), видалення |
| Журнал | історія перевірок (SQLite, `app_data/journal.sqlite`) |
| Статистика | зведення перевірок і якість моделей |
| Пакетна перевірка | перевірка всієї теки з записами і зведений звіт |
| Про систему | використані моделі, рівень суворості (м'яка / звичайна / сувора) |

Без PySide6 програма автоматично запускає простіший інтерфейс на Tkinter.

## 6. Запуск з командного рядка

```bash
python examples/check.py audio <файл>                    # перевірити запис
python examples/check.py audio <файл> --speaker "Ім'я"   # + верифікація мовця
python examples/check.py audio <файл> --no-asr           # лише голос, без розпізнавання (швидше)
python examples/check.py text "<текст повідомлення>"     # перевірити текст
python examples/check.py enroll "Ім'я" a.wav b.wav c.wav # зареєструвати мовця
python examples/check.py speakers                        # список мовців
python examples/check.py --json audio <файл>             # повний результат у JSON
```

Приклади з очікуваним виводом — [USAGE.md](USAGE.md).

## 7. Пояснення від ШІ (необов'язково)

1. Встановити Ollama: https://ollama.com/download
2. Завантажити модель (≈ 3,4 ГБ):
   ```bash
   ollama pull gemma3:4b
   ```
   Швидший запасний варіант — `ollama pull qwen2.5:3b`.
3. Переконатися, що Ollama запущено (сервер `http://127.0.0.1:11434`), і перезапустити VoiceGuard.
   Кнопка «✨ Пояснити» стане активною. Пояснення на CPU займає 1–2 хв.

Без Ollama програма повністю працює, лише без текстового пояснення.

## 8. Типові проблеми

| Симптом | Рішення |
|---|---|
| `ModuleNotFoundError` | активувати віртуальне середовище й повторити `pip install -r requirements.txt` |
| Довгий перший запуск / помилка завантаження | потрібен інтернет: WavLM завантажується з download.pytorch.org, Whisper — з huggingface.co; моделі кешуються (`~/.cache/torch`, `~/.cache/huggingface`) |
| `UnicodeEncodeError` у консолі Windows | виконати `set PYTHONIOENCODING=utf-8` (cmd) або `$env:PYTHONIOENCODING="utf-8"` (PowerShell) |
| Не працює мікрофон | перевірити дозвіл на мікрофон у налаштуваннях ОС; на Linux — `libportaudio2` |
| Не вистачає пам'яті | закрити інші програми; перевірка потребує ≈ 1,5–2 ГБ RAM |
| «✨ Пояснити» неактивна | не запущено Ollama або не завантажено модель `gemma3:4b` |
