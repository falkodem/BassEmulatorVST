# Fine-tune PESTO для streaming-инференса

Дообучение `mir-1k_g7` на гитарном датасете в условиях, идентичных тому что видит плагин:
streaming HCQT через `CachedConv1d`, фиксированные `mirror` / `mirror_fn`, тот же sample_rate / chunk_size / gamma.

## Структура

```
ml/finetune_pesto/
├── vendor/                  — копия минимума из pesto-full (LightningModule, encoder, losses, callback)
├── streaming_datamodule.py  — НАШ DataModule: on-the-fly streaming HCQT с random offset каждую эпоху
├── config.py                — dataclass со всеми гиперами
└── train.py                 — entry point
```

## Запуск

```bash
# дефолт: 10 эпох, lr=1e-5, mirror=1.0, mirror_fn=refill
poetry run python -m ml.finetune_pesto.train

# быстрая sanity-check на 2 эпохи
poetry run python -m ml.finetune_pesto.train --epochs 2
```

После обучения в `runs/finetune_pesto/<timestamp>/` сохраняются:
- `config.json` — полный снимок конфигурации запуска.
- `last.ckpt` — последний checkpoint для resume.
- `best-epoch=...-train_loss=....ckpt` — лучший checkpoint по среднему итоговому `train_loss` за эпоху.
- `finetuned-<timestamp>.ckpt` — финальное состояние после завершения обучения.

## Улучшение обучения

`invariance` и `shift_entropy` около `5.4-5.5` находятся близко к entropy baseline
для почти равномерных pitch activations, поэтому важнее смотреть, двигаются ли они
ниже этого уровня и не происходит ли collapse. Дефолтный режим в `config.py`
теперь ближе к `pesto-full`: `GradientsLossWeighting` со стартовыми весами
`shift_entropy=1`, `invariance=0`, `equivariance=0`.

Эти настройки живут в `TrainConfig`: `loss_weighting`, `loss_weighting_ema`,
`weight_invariance`, `weight_equivariance`, `weight_shift_entropy`. В режиме
`gradients` веса являются начальными и дальше обновляются каждый batch по нормам
градиентов; в режиме `fixed` они остаются постоянными.

Рекомендуемые первые прогоны:

```bash
# upstream-like weighting из config.py, более смелый LR для fine-tune
poetry run python -m ml.finetune_pesto.train --lr 3e-5

# если стабильно, попробовать reference LR из pesto-full
poetry run python -m ml.finetune_pesto.train --lr 1e-4
```

Для старого поведения нужно выставить в `config.py`:

```python
loss_weighting = "fixed"
weight_invariance = 1.0
weight_shift_entropy = 1.0
weight_equivariance = 1.0
```

## Деплой

Реэкспортировать ONNX с дообученным чекпойнтом:

```bash
poetry run python ml/utils/export_pesto_onnx.py \
    --model-name runs/finetune_pesto/<timestamp>/finetuned-<timestamp>.ckpt \
    --confidence-model mir-1k_g7
```

`--confidence-model` опционален. Если он не указан, confidence остаётся тем,
который загрузился из `--model-name`. Для checkpoint текущего fine-tune он нужен,
поскольку training-модель сохраняет encoder и shift, но не confidence-head.

Экспортёр автоматически читает `config.json` рядом с checkpoint и использует
из него `sample_rate`, `chunk_size`, `mirror` и `mirror_fn`. Можно передать другой
файл через `--train-config`. Явные параметры CLI имеют приоритет над конфигом,
например `--mirror 0.8 --mirror-fn zeros`.

`models/pesto.onnx` обновится, дальше — пересборка плагина на Windows.

## Зависимости

Помимо текущих:
- `pytorch-lightning` (для Trainer / LightningModule)

```bash
poetry add pytorch-lightning
```

## Что не делается (но было бы хорошо)

- **Валидация**: нет ground-truth F0. После обучения — слушать в Reaper, или сравнивать ONNX-выход с offline-PESTO на dev WAV из `data/v0/guitar/` (offline = reference).
- **Mixed precision**: `--precision 16-mixed` может ускорить, но HCQT может быть нестабилен в fp16. Дефолт fp32.
