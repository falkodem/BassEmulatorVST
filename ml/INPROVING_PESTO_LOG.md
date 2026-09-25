# PESTO improvement experiment log

Этот файл фиксирует последовательность экспериментов по улучшению streaming PESTO.
Общий список гипотез и мотивация находятся в `ml/IMPROVING_PESTO.md`. Здесь
хранятся только принятые решения, конкретные шаги, параметры запусков и результаты.

## Цель текущей серии

Перенести качество offline PESTO в causal streaming PESTO без дополнительного
lookahead. Сначала проверяется чистая покадровая distillation. Temporal loss,
новая архитектура, hysteresis и изменение CQT frontend рассматриваются только
как отдельные последующие гипотезы.

## Зафиксированные решения

- Teacher: offline PESTO `mir-1k_g7`.
- Initial student: fine-tuned checkpoint from run `20260706_110636`.
- Student frontend: `sample_rate=44100`, `chunk_size=441`, `mirror=1.0`,
  `mirror_fn=refill`.
- Один prediction соответствует одному chunk длительностью 10 ms.
- Teacher и student labels сопоставляются по одному номинальному frame index:
  временной offset равен нулю.
- Недоступный student future context не заменяется сдвинутым teacher frame.
- Confidence и pitch обучаются последовательно, поскольку у веток нет общих
  обучаемых параметров.
- Ручная voiced/unvoiced-разметка на первом этапе не используется.
- Temporal loss и sequence batches на первом этапе не используются.
- Scalar F0 loss на первом этапе не используется.
- `random_offset=False`: teacher labels и streaming frames используют
  фиксированную сетку с шагом 441 sample.
- Train/validation split выполняется по непрерывным участкам аудио, без
  перемешивания соседних frames между выборками.
- `19-PESTO`, `21-PESTO` и `22-PESTO` используются для train. `20-PESTO`
  делится пополам по границе полного 441-sample chunk: первая половина идёт в
  train, вторая половина целиком используется для validation.
- Все новые веса и режимы обучения задаются через train config и сохраняются в
  `config.json` рана.

## Контракт teacher labels

Для каждого WAV и каждого 10-ms frame сохраняются:

- `frame_index` и `time_s`;
- `activations`: абсолютное post-shift pitch-распределение teacher;
- `confidence`: soft target в диапазоне `[0, 1]`;
- `f0_hz`: диагностическое значение, не участвующее в первом loss;
- метаданные teacher checkpoint, sample rate, hop size и параметры HCQT;
- число исходных samples и правило обработки неполного последнего chunk.

Предлагаемый формат: один сжатый `.npz` на WAV плюс общий `manifest.json`.
Teacher labels генерируются один раз и не пересчитываются каждую эпоху.

## Losses первого этапа

### Confidence distillation

Student confidence получает энергию полного streaming HCQT до `PitchShiftCQT`.
Target — offline `teacher_confidence` для того же frame index.

```text
L_confidence = soft BCE(student_confidence, teacher_confidence)
```

Во время этого запуска pitch encoder заморожен. Confidence head инициализируется
весами `confidence.*` из исходного PESTO checkpoint.

### Pitch distillation

Teacher activations уже находятся в абсолютной pitch-системе. Сырые student
activations сначала переводятся в абсолютную систему с помощью текущего
`student.shift`:

```text
student_absolute = roll(student_raw, -round(student.shift * bins_per_semitone))
L_distill = KL(teacher_activations || student_absolute)
```

`student.shift` загружается из checkpoint и заново оценивается на известных
синтетических нотах каждую эпоху. Teacher confidence используется как вес или
маска pitch loss. Confidence head во время pitch-distillation заморожен.

В первом чистом pitch-distillation run существующие invariance, shift-entropy и
equivariance losses не используются. Их возврат выполняется отдельной следующей
ablation, чтобы не смешивать эффект teacher KL с self-supervised regularization.

## План работ

| Шаг | Статус | Результат |
|---|---|---|
| 0. Зафиксировать student checkpoint и train/validation split | выполнен | `20260706_110636`; validation = вторая половина `20-PESTO` |
| 1. Реализовать генератор offline teacher labels | не начат | `.npz` на WAV и `manifest.json` |
| 2. Проверить frame alignment и целостность labels | не начат | Совпадающие frame count/time, валидные distributions |
| 3. Добавить confidence-only dataloader и train mode | не начат | Checkpoint с обученной `confidence.*` |
| 4. Запустить короткий confidence smoke test | не начат | Loss уменьшается, encoder не меняется |
| 5. Обучить confidence и выполнить ONNX evaluation | не начат | Сравнение recall/false-voiced/confidence sweep |
| 6. Добавить pitch activation distillation | не начат | KL по абсолютным activations с teacher-confidence weighting |
| 7. Запустить короткий pitch smoke test | не начат | KL уменьшается, confidence не меняется |
| 8. Обучить pitch encoder и выполнить ONNX evaluation | не начат | Pitch/attack/stability metrics против baseline |
| 9. Собрать итоговую модель из distilled pitch и confidence | не начат | Экспортированный ONNX и полный quality report |

## Проверки корректности

- Teacher activations не содержат NaN, неотрицательны и суммируются в 1.
- Teacher confidence находится в `[0, 1]`.
- Число labels совпадает с числом полных streaming chunks.
- Первый и последний десяток frames каждого WAV проверяются отдельно.
- Во время confidence training параметры encoder остаются побитово неизменными.
- Во время pitch training параметры confidence остаются побитово неизменными.
- Экспортированный ONNX совпадает с PyTorch streaming inference в пределах
  существующих допусков экспортера.
- Все сравниваемые модели используют одинаковые `mirror` и `mirror_fn`.

## Метрики принятия результата

Confidence experiment:

- validation soft BCE относительно offline teacher;
- voiced recall и false-voiced rate на существующем note-labelled eval;
- confidence threshold sweep;
- attack recall и voiced-toggle rate.

Pitch experiment:

- validation KL относительно offline teacher;
- MAE/median/RMSE в cents;
- доля ошибок в пределах 20 и 50 cents;
- octave error rate;
- attack settling time, dropout и jump rate.

Основное правило: следующий механизм не добавляется, пока результат предыдущего
не экспортирован и не сравнен с тем же baseline на одном eval protocol.

## Согласованные параметры первого эксперимента

1. Исходный student checkpoint: run `20260706_110636`.
2. Validation: вторая половина `20-PESTO_1-260607_1530.wav`. Первая половина
   этого WAV и полные `19-PESTO`, `21-PESTO`, `22-PESTO` образуют train split.
3. Первый pitch-distillation run использует только teacher KL. Три старых
   self-supervised loss возвращаются только следующей отдельной ablation.

## Журнал запусков

| Дата | Run | Изменение относительно baseline | Статус | Результат |
|---|---|---|---|---|
| 2026-08-08 | protocol-v1 | Зафиксирован план framewise offline-to-streaming distillation | завершён | Протокол согласован 2026-08-09 |

Для каждого выполненного запуска ниже добавляются:

- checkpoint и полный config;
- teacher-label manifest;
- train/validation split;
- seed и длительность обучения;
- путь к ONNX и eval results;
- краткий вывод: подтверждена или отвергнута гипотеза.
