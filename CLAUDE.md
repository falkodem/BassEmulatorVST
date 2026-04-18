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
├── CMakeLists.txt          — сборка, JUCE 7.0.12 через FetchContent
└── src/
    ├── PluginProcessor.h/cpp   — вся DSP логика, APVTS с параметрами
    ├── PluginEditor.h/cpp      — GUI (5 ручек: Cutoff, Resonance, Attack, Release, Dry/Wet)
    ├── YinPitchDetector.h      — YIN pitch detection (header-only)
    ├── OnsetDetector.h         — детектор атаки по энергии (header-only)
    └── EnvelopeFollower.h      — RC-цепь огибающей (header-only)
```

## Текущее состояние: Phase 1 (классический bass emulator)

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

### DSP компоненты

**YinPitchDetector** — кольцевой буфер 1536 сэмплов (1024 + 512), обновление питча каждые 1024 новых сэмпла. 4 шага: difference function → CMNDF → absolute threshold (0.15) → parabolic interpolation. Диапазон: 70–1300 Гц (гитарный регистр).

**OnsetDetector** — RMS текущего блока vs предыдущего. Порог: ratio > 2.0 (6 дБ), пол тишины: 1e-4. Cooldown 50 мс после срабатывания.

**EnvelopeFollower** — one-pole filter с раздельными attack/release коэффициентами. `coeff = exp(-1 / (sampleRate × ms / 1000))`. `triggerAttack()` сбрасывает envelope в 0 для чёткой атаки.

**Oscillator** — `juce::dsp::Oscillator`, sawtooth: `f(x) = x / π`. Частота = F0/2 (октава вниз).

**Filter** — `juce::dsp::LadderFilter`, режим LPF12.

## Roadmap

- [x] MVP: JUCE plugin wrapper + реверб (VST3, работает в Reaper)
- [x] Phase 1: Классический bass emulator (YIN + onset + envelope + sawtooth + LadderFilter)
- [ ] Phase 2: RTNeural — TCN для real-time (требует парные данные гитара/бас)
- [ ] Phase 3: RAVE — VAE для offline постобработки

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