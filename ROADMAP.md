# BassEmulatorVST — Roadmap

## Цель проекта

Трансформировать гитарный звук в звук баса в реальном времени.
Основная проблема Guitar Rig / Ampero Stomp: неточное интонирование при атаке ноты.
Сценарий использования: монофонная игра, запись бас-партий с помощью гитары.

Подробный обзор области, анализ подходов, статьи и gap-анализ — в [`RESEARCH.md`](RESEARCH.md).

---

## Стратегия: два подхода

### Подход A: Conditioned ML (приоритет)

```
guitar ──→ [Pitch Detector] ──→ F0 ──→ [Synth: f₀/2] ──→ synth_bass ──┐
                                                                         ├──→ [ML тембр] ──→ realistic_bass
guitar ──────────────────────────────────────────────────────────────────┘
```

Явно разделяем задачи: питч решает детектор, нейросеть занимается только тембром.
Проще обучать, быстрее сходится, атака контролируется детерминированно.
Pitch detector — **PESTO** (pretrained, ONNX, < 5 мс; см. [RESEARCH.md §8](RESEARCH.md)).

### Подход B: End-to-end (side-experiment)

```
guitar ──→ [Neural Network] ──→ bass
```

Сеть сама учится определять питч и синтезировать тембр.
Сложнее, требует больше данных и ёмких архитектур, но потенциально захватывает
артикуляцию и стилистику вне явного F0.
Не блокирует Подход A, идёт параллельно по мере появления ресурсов.

---

## Текущее состояние

- [x] MVP: JUCE plugin wrapper (VST3, работает в Reaper)
- [x] Phase 1: YIN + onset + envelope + sawtooth + LadderFilter — реализован, деплой в Reaper работает
- [x] Phase 2 exploration: WaveConvNet v0 (stateless 1D CNN) — пайплайн и датасет проверены, не финальная архитектура
- [x] Phase 2A: PESTO evaluation — завершено; PESTO выбран как pitch detector
- [~] Phase 2A: интеграция PESTO в плагин (streaming-режим, mirror=0.8 zeros — собран на Linux, ждёт сборки/проверки на Windows)
- [ ] Phase 2A: ML-синтез тембра
- [ ] Phase 2B: end-to-end TCN/GRU с состоянием (side-experiment)
- [ ] Phase 3: RAVE / DDSP offline постобработка

---

## Next steps

### Шаг 1 — Оценить PESTO на наших данных (выполнено)

- [x] Запустить pretrained PESTO через ONNX inference на записях из `data/v0/guitar/` (без обучения)
- [x] Сравнить с YIN по тем же сигналам: измерить расхождение F0 на атаке, бендах, низкой E2, single-coil шуме
- [x] Зафиксировать случаи сбоя: тип сигнала, нота, SNR — понять, где PESTO ошибается и насколько это критично

Инструментарий: `ml/pitch_eval/` — `run_eval.py` (метрики, summary CSV/MD, графики F0), `synthesize.py` (sawtooth-синтез для аудиальной оценки). Артефакты: `runs/pitch_eval/<timestamp>/`.

Результат на 18 файлах `data/v0/guitar/`:

| детектор | recall | precision | F1 |
|---|---|---|---|
| YIN | 1.000 | 0.845 | 0.916 |
| PESTO | 0.917 | 0.997 | 0.954 |

YIN даёт ~15% false positives (питч в тишине/затухании). PESTO почти не галлюцинирует; пропуски (~8% кадров) — тихие хвосты нот, не игровые. Расхождение YIN↔PESTO: медиана 11 центов (согласие хорошее), mean 299 центов из-за редких octave errors PESTO; 5.8% кадров расходятся >50 центов. Аудиальная оценка подтверждает: у YIN много слышимых ложных нот, у PESTO ~1 заметная октавная ошибка в минуту.

Решение: **PESTO выбран как pitch detector проекта.** YIN остаётся как baseline для регрессионных сравнений. `voiced_threshold` PESTO не требует тюнинга на текущем датасете. Идея ground-truth-метрики «% попадания в ноту из имени файла» отклонена: записи содержат бенды, номинальная нота из имени файла ненадёжна как ground truth.

Референс: [RESEARCH.md §8](RESEARCH.md) — сравнение алгоритмов, PESTO ONNX streaming export.

### Шаг 2 — Интегрировать PESTO в плагин (выполнено в offline-режиме, заменено Шагом 2.5)

- [x] Подключить ANIRA + ONNX Runtime в CMakeLists.txt (см. [RESEARCH.md §12.5](RESEARCH.md))
- [x] Заменить YIN на PESTO для pitch detection в `PluginProcessor`
- [x] Latency budget: ~5–6 мс на inference (укладывается)

Первая итерация интеграции (offline-PESTO с overlapping windows) собрана и работала в Reaper, но качество F0 оказалось спорным — обнаружили что мы по сути запускаем offline-модель с reflect-pad на каждом 10-мс хопе, что генерирует артефакты. Переход на streaming-режим — Шаг 2.5.

### Шаг 2.5 — Правильный инференс PESTO по статье (ожидает сборки на Windows)

- [x] Изучить streaming-режим PESTO: `CachedConv1d` хранит последние `kernel_width - hop_length` сэмплов реального прошлого как левый паддинг свёртки (заменяет offline reflect-pad), `mirror_fn` (zeros/refill/reflection) заполняет правый край фейком
- [x] Перейти на streaming-экспорт PESTO в `ml/utils/export_pesto_onnx.py`: `load_model(streaming=True, max_batch_size=1, mirror=0.8)`, обёртка `StatelessPESTO` выносит cache (`CachedPadding1d.pad`) как явный input/output ONNX-тензор. Финальные параметры: `mirror=0.8`, `mirror_fn=zeros`, `cache_size=4651`, `chunk_size=441`
- [x] Замерить trade-off mirror vs точность на гитарном WAV (`runs/pitch_eval/streaming_compare_*/`): mirror=1.0 даёт **16% octave errors**, **mirror=0.8 — 2%** за +18 мс алгоритмической задержки, mirror=0.5 — 1.2% за +44 мс. Выбран mirror=0.8 как elbow кривой. Mirror_fn=refill оказался ХУЖЕ дефолтного zeros на всех значениях mirror (видимо потому что pretrained checkpoint обучен на offline-условиях с reflect-pad; на дообученной realtime-модели refill должен быть лучше — см. backlog "finetune PESTO")
- [x] Переписать `src/PestoPitchDetector.h` под streaming: убрал `kWindowSamples=11025` (overlapping windows), теперь `kHopSamples=441` (chunk на вызов) и `kCacheSize=4651`. Multi-input ANIRA config (audio streamable 441 + cache non-streamable 4651), 5 outputs (f0/conf/vol/acts/cache_out, все non-streamable). Кастомный `PestoProcessor::pre_process` подкладывает cache из member-vector, `post_process` читает `cache_out` обратно. F0 читается из единственного фрейма (kFramesPerCall=1) вместо kFramesPerWindow-1
- [ ] Собрать VST3 на Windows, проверить в Reaper аудиально (нет ли застываний питча, корректность работы реверса нот, общее качество vs предыдущая offline-итерация)
- [ ] `setLatencySamples` теперь включает `kMirrorLagSamples=775` (18 мс mirror algorithmic + handler latency) — проверить PDC в Reaper

Бюджет задержки «струна → ухо» с mirror=0.8 и Reaper buffer 256 сэмплов: ~45 мс (DAW round-trip ~12 + PESTO ~28 + analog ~5). Подробнее в комментариях `src/PestoPitchDetector.h`.

### Шаг 3 — ML-синтез тембра (v2, conditioned)

- [ ] Подготовить тройки `(guitar, synth_bass, real_bass)` — `synth_bass` генерируется оффлайн через YIN + осциллятор из VST-кода
- [ ] Обучить модель на паре `(guitar, synth_bass)` → `real_bass`: задача — только тембральная коррекция, субгармоника уже в `synth_bass`
- [ ] Добавить `envelope_loss` к текущему MRSTFT (см. [RESEARCH.md §10](RESEARCH.md))
- [ ] Оценить результат субъективно, сравнить с Phase 1 baseline

Архитектурные кандидаты: NEWT/FastNEWT (260k параметров, RT), DDSP-style (см. [RESEARCH.md §6](RESEARCH.md)).
Датасет: свои парные записи по клику в Reaper; Slakh2100 как дополнительный pre-train при нехватке данных (см. [RESEARCH.md §11](RESEARCH.md)).

---

## Backlog / отложенное

- ~~**Streaming PESTO (оптимизация инференса)**~~ — выполнено в Шаге 2.5. Streaming CQT через `CachedConv1d` решает задачу полностью: у `mir-1k_g7` единственная `CachedConv1d` живёт в CQT (1 гармоника), а энкодер `Resnet1d` обрабатывает каждый CQT-кадр **независимо** (нет временной свёртки → состояние не нужно), так что отдельно стримить энкодер не пришлось.
- **finetune PESTO под realtime-режим** — запланировано после набора большего датасета. Текущий плагин использует pretrained `mir-1k_g7`, который обучен в offline-режиме (whole-file CQT с reflect-pad с обеих сторон), а мы инференсим в streaming с `mirror=0.8, mirror_fn=zeros` (правая половина CQT-окна = фейк zeros). Замеры (`runs/pitch_eval/streaming_compare_*/`) показывают деградацию **~2% octave errors** vs offline на гитарных записях. При fine-tuning нужно: (1) собирать training данные в **streaming-режиме** с тем же `mirror` и `mirror_fn`, что используются в плагине (сейчас 0.8 / zeros); (2) попробовать `mirror_fn=refill` — наши замеры на zero-shot модели показали что refill ХУЖЕ zeros на всех mirror (30% vs 16% при mirror=1.0; 2.8% vs 2.0% при mirror=0.8), но на дообученной модели refill теоретически должен быть лучше для квазипериодических сигналов как утверждают авторы статьи; (3) использовать тот же sample_rate (44100) и chunk_size (441). В `ml/utils/export_pesto_onnx.py` уже есть флаг `--mirror-fn refill` для будущего использования. Без realtime-aware fine-tune'а refill включать не имеет смысла. Не блокирует Шаги 2–3. (см. [RESEARCH.md §8](RESEARCH.md))
- **TCN/GRU end-to-end (v3)** — рекуррентная модель с hidden state, решает фазовую когерентность субгармоники без явного pitch. Требует больше парных данных. (см. [RESEARCH.md §7, §9](RESEARCH.md))
- **RAVE / DDSP offline (v4, Phase 3)** — VAE для высококачественной постобработки без ограничений по латентности. Работает с непарными данными через WaveTransfer / Sony Diffusion Bridges. (см. [RESEARCH.md §7](RESEARCH.md))
- ~~**setLatencySamples()**~~ — выполнено: `PluginProcessor::prepareToPlay` дёргает `setLatencySamples(pesto.getLatencySamples())`. После Шага 2.5 latency включает `kMirrorLagSamples=775` (mirror algorithmic) + ANIRA handler latency. Проверка PDC в Reaper — в чек-листе Шага 2.5.
- **PolyBLEP осциллятор** — текущий sawtooth даёт aliasing при F₀/2 > 500 Гц. Некритично до перехода на ML-синтез. (см. [RESEARCH.md §5](RESEARCH.md))

---

## Open questions

1. ~~Насколько PESTO устойчив к single-coil шуму и атаке на E2 на наших конкретных записях?~~ — решено в Шаге 1: устойчив, precision 0.997.
2. ~~YIN или PESTO?~~ — решено в Шаге 1: PESTO выбран, YIN остаётся baseline.
3. Достаточно ли текущего объёма датасета (`data/v0/`) для v2 conditioned ML, или нужен pre-train на Slakh2100?
4. Какой inference engine использовать для ML-тембра в плагине: ANIRA + LibTorch или ANIRA + ONNX? (зависит от финальной архитектуры)
5. Как оценивать качество тембра объективно — метрика "звучит как бас"? MRSTFT достаточно или нужен perceptual/adversarial loss?
6. Когда onset срабатывает на новой ноте, а PESTO ещё не обновил F0 — нужен ли быстрый fade-out при onset для маскировки? (актуальнее после Шага 2.5: mirror=0.8 даёт +18 мс алгоритмического лага F0 относительно реального аудио → onset реально опережает обновление питча)
7. Стоит ли вынести `mirror` в настройку плагина (multi-model bundle с Low/Balanced/Quality пресетами, +32 МБ к VST3) или оставить фиксированным? Пока зафиксирован mirror=0.8.
