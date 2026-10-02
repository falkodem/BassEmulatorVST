# PESTO — журнал экспериментов

Актуальные команды обучения находятся в [README пайплайна](finetune/README.md),
открытые задачи — в [ROADMAP](../../ROADMAP.md). Этот файл фиксирует уже
выполненные эксперименты и их границы; `runs/` исключён из Git, поэтому пути к
логам и checkpoints относятся к серверу обучения.

## Общий протокол текущей серии

- Аудио: `/data/19-PESTO_0-260606_1408.wav`, `20-PESTO_1-260607_1530.wav`,
  `21-PESTO_2-260726_2105.wav`, `22-PESTO_3-260726_2117.wav`.
- Offline fine-tune teacher: train на `19/20/22`, файл `21` не передаётся в
  скрипт. В самом fine-tune нет validation loss; `best` выбирается по train loss.
- Teacher labels: offline PESTO на всех четырёх WAV; один `.npz` на WAV и
  `manifest.json` с checkpoint и SHA-256. `activations` уже содержат абсолютный
  post-shift pitch, `confidence` — soft target, `f0_hz` — диагностика.
- Distillation: `21` целиком отложен для validation, остальные три — train;
  фиксированная сетка по 441 сэмплу (`random_offset=False`), training frames
  перемешиваются. Student использует `mirror=1.0/refill`, teacher — реальный
  будущий контекст. Temporal offset между метками и student равен нулю.
- Pitch loss — KL по pitch-распределениям с весом teacher confidence; в
  `pitch_kl_ssl` добавляются три PESTO loss. Confidence учится отдельно по
  soft BCE. Значение `student.shift` загружается из начального checkpoint и в
  `distill.py` не переоценивается; в offline fine-tune `shift` пересчитывается
  каждый epoch на синтетических нотах.
- ONNX-eval: одни и те же 18 нотных WAV E2–A3, потоковый frontend
  `mirror=1.0/refill`, порог confidence 0.5. Метрики pitch считаются по
  активным кадрам, прошедшим voiced-гейт; это не слуховой тест Reaper.

## Выполненные запуски

| Вариант | Лучший checkpoint / результат |
|---|---|
| MIR → streaming pitch, только KL | `runs/distill_pesto/20260926_115241_pitch/`, best epoch 129 |
| MIR → streaming confidence | `runs/distill_pesto/20260926_133222_confidence/`, best epoch 146 |
| MIR → KL+SSL, equivariance 0.1, двойной softmax | `runs/distill_pesto/20260927_140350_pitch_kl_ssl/`, best epoch 226 |
| MIR → KL+SSL, equivariance 1.0, двойной softmax | `runs/distill_pesto/20260928_073654_pitch_kl_ssl/`, best epoch 226 |
| MIR → KL+SSL, equivariance 1.0, CE по вероятностям | `runs/distill_pesto/20260928_140151_pitch_kl_ssl/`, best epoch 226 |
| Offline MIR fine-tune, upstream CE | `runs/finetune_pesto/offline_mir_upstream_ce_20260929/`, best epoch 84 |
| Offline MIR fine-tune, shift W₂ | `runs/finetune_pesto/offline_mir_wasserstein2_20260929/`, best epoch 25; финальный checkpoint после 150 эпох деградировал |
| Offline upstream_ce teacher → streaming pitch, KL | `runs/distill_pesto/upstream_ce_best_e084_pitch/`, best epoch 114 |
| Offline upstream_ce teacher → streaming pitch, KL+SSL/upstream CE | `runs/distill_pesto/upstream_ce_best_e084_pitch_kl_ssl_upstreamce/`, best epoch 131 |

Лучший offline `upstream_ce` teacher:
`runs/finetune_pesto/offline_mir_upstream_ce_20260929/best-epoch=084-train_loss=5.449664.ckpt`.
Его метки: `runs/pesto_teacher/upstream_ce_best_e084/manifest.json` и четыре
одноимённых `.npz`. Оба новых student стартовали с `mir-1k_g7`, а не с
предыдущего streaming checkpoint; во всех сравниваемых pitch-моделях
confidence-head остался от `mir-1k_g7`.

## ONNX-сравнение лучших student

На одинаковом нотном eval при confidence ≥0.5:

| Student | В пределах 20 центов | В пределах 50 центов | Ошибка >600 центов |
|---|---:|---:|---:|
| MIR → KL | 55.42% | 95.44% | 0.226% |
| MIR → KL+SSL, equivariance 0.1 | 55.35% | 95.40% | 0.244% |
| MIR → KL+SSL, equivariance 1.0, двойной softmax | 55.43% | 95.40% | 0.242% |
| MIR → KL+SSL, equivariance 1.0, CE по вероятностям | 56.09% | 95.35% | 0.242% |
| Upstream CE teacher → KL | **89.03%** | **96.70%** | **0.088%** |
| Upstream CE teacher → KL+SSL/upstream CE | 88.78% | 96.69% | 0.088% |

Полный отчёт последних двух вариантов:
`runs/pitch_eval/20261001_upstream_ce_two_distills/summary.csv`; прежние отчёты
находятся в `runs/pitch_eval/20260926_pitch_kl_best_vs_base/` и каталогах
`20260928_pitch_kl_ssl_best/`, `20260928_pitch_kl_ssl_equiv1_best/`,
`20260929_pitch_kl_ssl_probce_equiv1_best/`. На этом тесте основной выигрыш
дал адаптированный offline teacher; преимущество KL+SSL над чистым KL для него
не подтверждено.

Три ONNX для ближайшей слуховой проверки перечислены в
[корневом README](../../README.md#сборка-с-одной-из-трёх-pesto-моделей).
Совместный вариант с дополнительно обученной confidence-веткой поверх
выбранного нового pitch-student ещё не обучен и не оценён. Проверка атак,
переходов, затуханий и PDC в Reaper остаётся открытой.
