# BassEmulatorVST

VST3 плагин (JUCE, Windows, Reaper): трансформация гитарного звука в звук баса в реальном времени.
Цель — решить проблему неточного интонирования на атаке ноты (слабость Guitar Rig и Ampero Stomp).
Сценарий: монофонная игра на гитаре, запись бас-партий.

## Сборка

```bash
# Первый раз (или после переименования/переноса проекта)
cmake -B build -S .

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
├── src/                        — DSP плагин (C++/JUCE)
│   ├── PluginProcessor.h/cpp   — вся DSP логика, APVTS с параметрами
│   ├── PluginEditor.h/cpp      — GUI
│   ├── YinPitchDetector.h      — YIN pitch detection (header-only)
│   ├── OnsetDetector.h         — детектор атаки по энергии (header-only)
│   └── EnvelopeFollower.h      — RC-цепь огибающей (header-only)
├── ml/                         — весь ML-код (Python)
│   ├── configs/train_v0.json   — гиперпараметры и версии данных/модели
│   ├── nn_architectures/       — модели; REGISTRY dict для выбора по имени
│   │   ├── __init__.py
│   │   └── bassnet.py          — WaveConvNet (1D CNN, waveform domain)
│   ├── train_config.py         — dataclass TrainConfig
│   ├── train.py                — трейн-луп (запускать отсюда)
│   ├── process_audio.py        — оффлайн-инференс (overlap-add)
│   ├── import_dataset.py       — импорт WAV из Reaper → data/v0/
│   └── slice_dataset.py        — нарезка окон → data/v0/windows/
├── data/v0/                    — аудиоданные (в .gitignore)
│   ├── guitar/ bass/           — сырые WAV-пары
│   ├── index.csv
│   └── windows/                — guitar.npy  bass.npy  meta.csv
└── runs/                       — артефакты обучения (в .gitignore)
    └── v0/
        └── YYYYMMDD_HHMMSS/
            ├── best.pt         — чекпойнт лучшей эпохи
            ├── config.json     — снимок гиперпараметров этого рана
            └── events.out.*    — TensorBoard events
```

### ML-команды

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
- RTNeural — запланирован для Phase 2