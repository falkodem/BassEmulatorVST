# PESTO: fine-tune и distillation

Offline PESTO с настоящим будущим контекстом адаптируется к гитаре и выдаёт
покадровые метки для streaming student. Student видит только пришедшее аудио
(`mirror=1.0/refill`), как плагин. Команды ниже выполняются **внутри** контейнера
`trainloop-pesto-train`, из `/workspace`, обычным `python`. Запуск контейнера и
просмотр логов описаны в [Docker README](../../../docker/pesto-train/README.md).

## Данные и разбиение

Четыре WAV лежат на сервере в `data/pesto_train/`, внутри контейнера — в `/data`.
Во всех командах используются 44 100 Гц и шаг 441 сэмпл (10 мс).

```bash
WAVS=(
  /data/19-PESTO_0-260606_1408.wav
  /data/20-PESTO_1-260607_1530.wav
  /data/21-PESTO_2-260726_2105.wav
  /data/22-PESTO_3-260726_2117.wav
)
```

Offline fine-tune обучается на `19`, `20`, `22`; скрипт **не имеет настоящей
валидации**. Файл `21` туда не передаётся. Для distillation метки создаются на
всех четырёх WAV, но `--validation-wav /data/21-...` целиком исключает `21` из
train: в текущем `DistillConfig.validation_start_fraction=0.0`. Внутри train
покадровые примеры перемешиваются; validation не перемешивается.

## 1. Offline fine-tune teacher

```bash
python -u -m ml.pesto.finetune.train \
  --frontend offline --pretrained mir-1k_g7 \
  --wav "${WAVS[0]}" "${WAVS[1]}" "${WAVS[3]}" \
  --epochs 150 --accelerator gpu \
  --run-name offline_mir_upstream_ce_next
```

`train.py` строит offline HCQT блоками с настоящим контекстом слева и справа,
считает три self-supervised loss PESTO (`invariance`, `shift_entropy`,
`equivariance`) и оптимизирует pitch encoder. Confidence-head из `mir-1k_g7`
сохраняется, но не обучается. Текущий `TrainConfig` задаёт batch 512, LR `1e-5`,
80 эпох по умолчанию (здесь явно 150), `loss_weighting=gradients` и начальные
веса loss `0/1/0` в порядке invariance/shift-entropy/equivariance. Весами затем
управляет нормализация по градиентам; это не фиксированные коэффициенты.
`ssl_cross_entropy=upstream` применяет второй softmax в двух CE-loss и не
выбирается отдельным CLI-флагом для fine-tune. Каждый epoch выбирается случайный
sample offset; validation hook пересчитывает абсолютный `shift` на синтетических
нотах, а не на наших WAV.

Альтернативный запуск меняет только `shift_entropy` на одномерное W₂:

```bash
python -u -m ml.pesto.finetune.train \
  --frontend offline --shift-loss wasserstein2 --pretrained mir-1k_g7 \
  --wav "${WAVS[0]}" "${WAVS[1]}" "${WAVS[3]}" \
  --epochs 150 --accelerator gpu \
  --run-name offline_mir_wasserstein2_next
```

Тот же `train.py` может дообучать модель сразу в streaming-условиях, без
teacher и KL; это отдельный SSL-only эксперимент, не шаг получения offline
teacher:

```bash
python -u -m ml.pesto.finetune.train \
  --frontend streaming --pretrained mir-1k_g7 \
  --wav "${WAVS[0]}" "${WAVS[1]}" "${WAVS[3]}" \
  --epochs 150 --accelerator gpu \
  --run-name streaming_mir_ssl_next
```

В `runs/finetune_pesto/<run-name>/` пишутся `config.json`, TensorBoard events,
`last.ckpt`, `best-epoch=...-train_loss=....ckpt` и финальный
`finetuned-<run-name>.ckpt`. Для teacher выбирай **best**, но помни: это
минимум train loss, не независимая validation-метрика; качество следует отдельно
проверить на отложенном `21` и нотных WAV. Offline checkpoint сам по себе не
предназначен для плагина.

## 2. Offline teacher-метки

Пример для уже обученного лучшего `upstream_ce` checkpoint:

```bash
python -u -m ml.pesto.finetune.generate_teacher_labels \
  --teacher-model 'runs/finetune_pesto/offline_mir_upstream_ce_20260929/best-epoch=084-train_loss=5.449664.ckpt' \
  --output-dir runs/pesto_teacher/upstream_ce_best_e084 \
  --wav "${WAVS[@]}" --device cuda
```

Если дообучил нового teacher, замени `--teacher-model` на его best checkpoint и
**выбери новый `--output-dir`**. Генератор переиспользует существующие `.npz`,
когда совпали только длины WAV/меток: он не сверяет, каким checkpoint они были
созданы. `--overwrite` принудительно пересчитает метки в старом каталоге.
Для MIR-teacher baseline можно указать `--teacher-model mir-1k_g7` и отдельный
каталог `runs/pesto_teacher/mir-1k_g7`.

Teacher запускается offline, с реальным будущим контекстом. Для каждого WAV
получается `.npz`: `activations` — post-shift pitch-распределения `float16`,
`confidence` — soft confidence `float32`, `f0_hz` — диагностическое значение,
`frame_index`, `time_s`, `num_samples`. `manifest.json` содержит пути к исходным
WAV, параметры сетки кадров, checkpoint teacher и его SHA-256. Неполный последний
chunk не размечается. Для короткой проверки `--max-minutes` ограничивает **каждый**
WAV; такие метки не подходят для полного distillation-запуска.

## 3. Distillation в streaming student

Чистый teacher-weighted KL, начиная с исходного `mir-1k_g7`:

```bash
python -u -m ml.pesto.finetune.distill \
  --mode pitch --student-checkpoint mir-1k_g7 \
  --teacher-labels-dir runs/pesto_teacher/upstream_ce_best_e084 \
  --wav "${WAVS[@]}" --validation-wav "${WAVS[2]}" \
  --epochs 150 --accelerator gpu \
  --run-name upstream_ce_best_e084_pitch_next
```

Вариант KL + три self-supervised loss с формулой CE как в upstream PESTO:

```bash
python -u -m ml.pesto.finetune.distill \
  --mode pitch_kl_ssl --ssl-cross-entropy upstream \
  --student-checkpoint mir-1k_g7 \
  --teacher-labels-dir runs/pesto_teacher/upstream_ce_best_e084 \
  --wav "${WAVS[@]}" --validation-wav "${WAVS[2]}" \
  --epochs 150 --accelerator gpu \
  --weight-invariance 0.2 --weight-shift-entropy 0.2 --weight-equivariance 0.1 \
  --run-name upstream_ce_best_e084_pitch_kl_ssl_next
```

В обоих режимах student HCQT получает реальное прошлое и `refill` вместо
будущего; pitch encoder обучается, confidence-head заморожен. KL сравнивает
абсолютные pitch-распределения, взвешивая кадры teacher confidence. В
`pitch_kl_ssl` к нему на train добавляются `invariance`, `shift_entropy` и
`equivariance` с указанными **фиксированными** весами. Валидация считает только
KL; сравнивать её число между **разными teacher** напрямую нельзя. Без
`--ssl-cross-entropy upstream` дистилляция использует `probability` — CE без
второго softmax. Текущий `DistillConfig`: batch 512, pitch LR `1e-5`, 150 эпох,
`random_offset=False`; DataLoader перемешивает train-кадры.

Есть также `--mode confidence`: soft BCE учит только confidence-head по
`confidence` из тех же teacher-меток, заморозив pitch encoder. Если указать
лучший pitch-student checkpoint как `--student-checkpoint`, итоговый checkpoint
содержит обе ветки. Это псевдометки offline teacher, **не** ручная разметка
voiced/unvoiced. Например:

```bash
python -u -m ml.pesto.finetune.distill \
  --mode confidence \
  --student-checkpoint 'runs/distill_pesto/upstream_ce_best_e084_pitch/best-pitch-epoch=114-val_loss=0.259167.ckpt' \
  --teacher-labels-dir runs/pesto_teacher/upstream_ce_best_e084 \
  --wav "${WAVS[@]}" --validation-wav "${WAVS[2]}" \
  --epochs 150 --accelerator gpu \
  --run-name upstream_ce_best_e084_pitch_confidence_next
```

Для экспорта после такого запуска укажи его `best-confidence-*.ckpt` в
`--model-name`: checkpoint уже содержит и pitch, и обученную confidence-ветку.

Артефакты находятся в `runs/distill_pesto/<run-name>/`: `config.json`, events,
`last.ckpt`, `best-<mode>-epoch=...-val_loss=....ckpt` и финальный
`distilled-<mode>-<run-name>.ckpt`. Для сравнения используй best. TensorBoard
пишет `train_loss`, `val_loss`, `loss/pitch_kl_weighted/*`, а для KL+SSL ещё
сырые и взвешенные `loss/{invariance,shift_entropy,equivariance}*/train`.
`--max-minutes 1 --epochs 2` подходит для smoke test под **новым** `run-name`.

Важная особенность provenance: `distill.py` не принимает `--teacher-model`;
`teacher_model` в его `config.json` остаётся `mir-1k_g7` и используется для
создания streaming HCQT. Реальный teacher определяется `teacher_labels_dir` и
`manifest.json` внутри этого каталога.

## 4. Экспорт и оценка

В плагин экспортируется **streaming student**, не offline teacher. Экспортёр
читает `config.json` рядом с checkpoint (44,1 кГц, 441, `mirror=1.0/refill`):

```bash
python -m ml.pesto.export_onnx \
  --model-name 'runs/distill_pesto/upstream_ce_best_e084_pitch/best-pitch-epoch=114-val_loss=0.259167.ckpt' \
  --output models/eval_upstream_ce_best_e084_pitch/pesto.onnx

python -m ml.pitch_eval.eval_pesto_onnx \
  --models models/eval_upstream_ce_best_e084_pitch/pesto.onnx \
  --input data/v0/guitar \
  --output runs/pitch_eval/upstream_ce_best_e084_pitch
```

Экспортёр пишет `pesto.onnx` и `pesto_onnx_meta.json`, по умолчанию сверяет
ONNX с PyTorch на одном WAV. `eval_pesto_onnx` пишет `summary.csv`,
`per_file.csv`, `confidence_sweep.csv` и покадровый `frames.csv.gz`. Для
сравнения моделей нужны одинаковые входные WAV, порог confidence и настройки
оценки. Выбранные для слуховой проверки ONNX перечислены в
[корневом README](../../../README.md#сборка-с-одной-из-трёх-pesto-моделей).
