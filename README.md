# BassEmulatorVST

VST plugin для Windows/Reaper: трансформация гитарного звука в звук баса.
Цель — решить проблему неточного интонирования при атаке (характерно для Guitar Rig, Ampero Stomp).

## Архитектурные заметки

### RTNeural
https://github.com/jatinchowdhury18/RTNeural

Специализированная C++ библиотека для real-time инференса нейросетей в аудио-плагинах.
Используется в amp-sim плагинах (BYOD, GuitarML и др.).

**Почему актуально для этого проекта:**
- Поддерживает LSTM, GRU, Conv1D — подходящие архитектуры для end-to-end guitar→bass
- Спроектирована под минимальную латентность (real-time audio thread safe)
- Хорошо интегрируется с JUCE

**Планируемые режимы:**
- Real-time: лёгкая модель для мониторинга во время записи
- Offline: тяжёлая модель для пост-обработки аудиофайла

## Архитектура кода

### Структура файлов

```
BassEmulatorVST/
├── CMakeLists.txt          ← инструкция для сборки
└── src/
    ├── PluginProcessor     ← МОЗГ: вся логика обработки звука
    └── PluginEditor        ← ЛИЦО: GUI с ручками
```

### Классы

```mermaid
classDiagram
    class BassEmulatorVSTProcessor {
        +apvts AudioProcessorValueTreeState
        -reverb Reverb
        +prepareToPlay()
        +processBlock()
        +createEditor()
    }
    class BassEmulatorVSTEditor {
        -roomSizeSlider
        -dampingSlider
        -wetSlider
        -drySlider
        +resized()
        +paint()
    }
    class APVTS {
        roomSize float
        damping float
        wetLevel float
        dryLevel float
    }
    BassEmulatorVSTProcessor "1" --> "1" BassEmulatorVSTEditor : createEditor()
    BassEmulatorVSTProcessor "1" --> "1" APVTS : владеет
    BassEmulatorVSTEditor --> APVTS : SliderAttachment
```

### Поток аудиосигнала

```mermaid
flowchart LR
    DAW([Reaper]) -->|float* buffer| PB

    subgraph processBlock
        PB[получить буфер] --> URP[прочитать параметры из APVTS]
        URP --> CH{каналов?}
        CH -->|1| MONO[processMono]
        CH -->|2| STEREO[processStereo]
    end

    MONO --> OUT([выход в DAW])
    STEREO --> OUT
```

### Поток параметров GUI → DSP

```mermaid
flowchart LR
    USER([пользователь крутит ручку]) --> SL[Slider]

    subgraph "GUI thread"
        SL -->|SliderAttachment| APVTS[(APVTS\natomic float)]
    end

    subgraph "Audio thread"
        APVTS -->|atomic read| URP[updateReverbParameters]
        URP --> REV[Reverb.setParameters]
    end
```

> **Почему atomic?** GUI и аудио работают в разных потоках. `atomic<float>` — thread-safe передача значения без блокировок, что критично для real-time аудио.

### Сборка проекта

```mermaid
flowchart TD
    CML[CMakeLists.txt] -->|FetchContent| JUCE[JUCE 7.0.12]
    CML -->|juce_add_plugin| TGT[BassEmulatorVST target]
    CML -->|juce_generate_juce_header| HDR[JuceHeader.h]
    JUCE --> HDR
    HDR --> SRC[src/*.cpp]
    SRC -->|MSVC| VST3[BassEmulatorVST.vst3]
    VST3 --> REAPER([Reaper])
```

## Roadmap

- [ ] MVP: JUCE plugin wrapper + простой реверб
- [ ] Исследование архитектуры end-to-end модели (guitar→bass)
- [ ] Интеграция RTNeural
- [ ] Real-time режим
- [ ] Offline режим