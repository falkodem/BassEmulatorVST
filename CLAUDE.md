# BassEmulatorVST

VST3 плагин (JUCE, Windows, Reaper): трансформация гитарного звука в звук баса в реальном времени.
Цель — решить проблему неточного интонирования на атаке ноты (слабость Guitar Rig и Ampero Stomp).
Сценарий: монофонная игра на гитаре, запись бас-партий.
Текущий фокус: интеграция PESTO в плагин через ANIRA + ONNX Runtime (Phase 2A, Шаг 2). PESTO выбран pitch-детектором по итогам Шага 1.

## Диапазоны нот (научная нотация, SPN)

| Инструмент | Нижняя нота | Верхняя нота | Примечание |
|---|---|---|---|
| Гитара (стандарт) | **E2** (82 Гц) — ми большой октавы | **E4** (330 Гц) — ми первой октавы | Открытые струны |
| Бас-гитара (стандарт) | **E1** (41 Гц) — ми контроктавы | **G3** (196 Гц) | Открытые струны |

Гитара играет **на октаву выше баса**. При конвертации: F₀(гитары) / 2 = F₀(баса).
Детекция ведётся на гитарном сигнале (E2–E4), синтез — на F₀/2 (E1–E3).

## Сборка

```bash
# Первый раз (или после переименования/переноса проекта)
# ANIRA требует явного CMAKE_BUILD_TYPE — без него configure падает с ошибкой
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Пересборка
cmake --build build --config Release

# Деплой в Reaper (закрыть плагин в Reaper перед копированием)
cp -r build/BassEmulatorVST_artefacts/Release/VST3/BassEmulatorVST.vst3 "D:/Music/Plugins/BassEmulatorVST/"
```

Артефакт: `build/BassEmulatorVST_artefacts/Release/VST3/BassEmulatorVST.vst3`

## Структура проекта

```
BassEmulatorVST/
├── CMakeLists.txt              — сборка VST3, JUCE 7.0.12 через FetchContent
├── ROADMAP.md                  — план проекта по фазам
├── RESEARCH.md                 — обзор подходов guitar→bass (рендерится в RESEARCH.html)
├── REVIEW.md                   — ревью RESEARCH.md: сверка с кодом и согласованность
├── src/                        — DSP плагин (C++/JUCE)
│   ├── PluginProcessor.h/cpp   — вся DSP логика, APVTS с параметрами
│   ├── PluginEditor.h/cpp      — GUI
│   ├── YinPitchDetector.h      — YIN pitch detection (header-only)
│   ├── OnsetDetector.h         — детектор атаки по энергии (header-only)
│   └── EnvelopeFollower.h      — RC-цепь огибающей (header-only)
├── ml/                         — весь ML-код (Python)
│   ├── configs/train_v0.json   — гиперпараметры и версии данных/модели
│   ├── nn_architectures/       — модели; REGISTRY dict для выбора по имени
│   │   ├── __init__.py         — REGISTRY: WaveConvNet, DilatedConvNet
│   │   ├── bassnet.py          — WaveConvNet (1D CNN, waveform domain)
│   │   └── dilated.py          — DilatedConvNet (dilated 1D CNN, RF ≈ 23 мс)
│   ├── pitch_eval/             — оффлайн-сравнение pitch-детекторов (Phase 2A, Шаг 1)
│   │   ├── run_eval.py         — CLI: прогон детекторов по WAV → метрики + графики
│   │   ├── compare.py          — выравнивание F0 на сетку, метрики расхождения
│   │   ├── synthesize.py       — озвучка F0-кривых пилой для слуховой оценки
│   │   └── detectors/          — обёртки детекторов; REGISTRY: yin, pesto
│   ├── train_config.py         — dataclass TrainConfig
│   ├── train.py                — трейн-луп (запускать отсюда)
│   ├── losses.py               — лоссы (MultiScaleSTFTLoss и др.)
│   ├── transforms.py           — waveform/stft-трансформы для WindowDataset
│   ├── process_audio.py        — оффлайн-инференс (overlap-add)
│   ├── import_dataset.py       — импорт WAV из Reaper → data/v0/
│   └── slice_dataset.py        — нарезка окон → data/v0/windows/
├── scripts/                    — вспомогательные скрипты
│   ├── render_docs.py          — RESEARCH.md → RESEARCH.html
│   ├── yin_pseudo.py           — учебный псевдокод YIN
│   └── pesto_pseudo.py         — учебный псевдокод PESTO
├── data/v0/                    — аудиоданные (в .gitignore)
│   ├── guitar/ bass/           — сырые WAV-пары
│   ├── index.csv
│   └── windows/                — guitar.npy  bass.npy  meta.csv
├── runs/                       — артефакты обучения и оценки (в .gitignore)
│   ├── v0/YYYYMMDD_HHMMSS/     — раны обучения
│   │   ├── best.pt             — чекпойнт лучшей эпохи
│   │   ├── config.json         — снимок гиперпараметров этого рана
│   │   └── events.out.*        — TensorBoard events
│   └── pitch_eval/YYYYMMDD_HHMMSS/  — раны pitch_eval (F0-кривые CSV, графики, summary)
├── processed/                  — выход process_audio.py (в .gitignore)
│   └── {data}_{arch}_{model}/  — напр. v0_WaveConvNet_v0/ — WAV-результаты инференса
└── models/                     — экспортированные модели для плагина (в .gitignore)
```

### ML-команды

Python-окружение: в корне проекта лежит `venv/`, внутри установлен `poetry` (`venv/bin/poetry`) и все зависимости. Перед любыми poetry-командами активируй venv:

```bash
source venv/bin/activate
```

После этого `poetry`, `poetry run`, `poetry add` работают как обычно. Без активации команды `poetry` нет в PATH — это не системный poetry, он живёт внутри venv.

> **Сеть нестабильна.** В окружении интернет может быть недоступен или обрываться — автоматическая установка пакетов (`poetry add`, `pip install`) часто падает. Не пытайся ставить пакеты сам: выдай пользователю точную команду установки (`poetry add <пакеты>`), он поставит вручную и подтвердит, после чего можно продолжать.

```bash
# Подготовка данных
poetry run python ml/import_dataset.py
poetry run python ml/slice_dataset.py

# Обучение
poetry run python ml/train.py

# TensorBoard (все раны)
tensorboard --logdir runs/

# Оффлайн-инференс (--run = папка рана, содержит best.pt + config.json)
poetry run python ml/process_audio.py \
  --run runs/v0/YYYYMMDD_HHMMSS \
  --input data/v0/guitar/
```

## Phase 1: Параметры и пайплайн
_(DSP-baseline, shipped; будет заменён PESTO + ML-тембр в Phase 2A)_

### Параметры (APVTS)

| ID               | Название         | Диапазон        | Дефолт |
|------------------|------------------|-----------------|--------|
| filterCutoff     | Filter Cutoff    | 100–2000 Гц     | 800    |
| filterResonance  | Filter Resonance | 0–1             | 0.3    |
| envAttack        | Env Attack       | 1–50 мс         | 10     |
| envRelease       | Env Release      | 10–500 мс       | 100    |
| dryWet           | Dry/Wet          | 0–1             | 1.0    |

### Пайплайн processBlock

```
Вход (гитара, channel 0)
  │
  ├─→ OnsetDetector       — скачок RMS > 6 дБ → triggerAttack()
  ├─→ YinPitchDetector    — F0 каждые ~21 мс → currentPitch = F0/2
  │
  │   (если pitchIsValid == false → dry pass-through)
  │
  ├─→ Oscillator (sawtooth, currentPitch)
  │       × EnvelopeFollower (driven by input amplitude)
  │       → LadderFilter LPF12 (cutoff, resonance)
  │
  └─→ mix: out = dry*(1-wet) + bass*wet
```

## Ключевые решения

- **Детекция на гитарном сигнале**, не на синтезируемом: более короткие периоды, меньше требуемая латентность.
- **`pitchIsValid` флаг**: бас не выходит до первого стабильного F0 от YIN — исключает артефакты на старте.
- **Все DSP классы header-only**: не нужно добавлять в CMakeLists.txt.
- **Параметры читаются в каждом processBlock** через `getRawParameterValue()->load()` (atomic read, thread-safe).

## Зависимости

- JUCE 7.0.12 (GPL, только личное использование)
- MSVC / Visual Studio Build Tools
- CMake 3.22+
- ANIRA + ONNX Runtime — запланированы для Phase 2A (inference engine для PESTO в плагине)