# ML Ideas & Future Work

Идеи, не реализованные сразу — чтобы не потерять и вернуться при необходимости.

---

## D. Mixed Precision Training (`torch.cuda.amp`)

**Что:** Обернуть forward-pass в `torch.autocast("cuda")` и использовать `GradScaler`
для backward. Добавить флаг `"mixed_precision": false` в `TrainConfig`.

**Почему сейчас не нужно:** ~85k параметров, маленький датасет, вероятно CPU или небольшая GPU.
Реальный выигрыш будет при переходе к архитектурам Phase 2 (RTNeural-style RNN, спектральные
энкодеры на длинных окнах, большие свёрточные сети).

**Как реализовать:**
1. `TrainConfig`: добавить `mixed_precision: bool = False`
2. В `main()` создать `scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)`
3. В `run_epoch` обернуть forward+loss:
   ```python
   with torch.autocast("cuda", enabled=cfg.mixed_precision):
       loss = criterion(model(guitar), bass)
   scaler.scale(loss).backward()
   scaler.unscale_(optimizer)      # нужно для корректного grad_norm
   gnorm = ...                     # считаем после unscale
   if clip_grad_norm:
       nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
   scaler.step(optimizer)
   scaler.update()
   ```
4. Логировать `scaler.get_scale()` в TensorBoard как `train/amp_scale`

**Когда возвращаться:** Phase 2 — более тяжёлые архитектуры, датасет > 10k окон или длинные окна.

---

## E. Temporal Context — стек предыдущего окна

**Что:** Давать модели два окна как input `(2, 1024)` — текущее + предыдущее.
Выход по-прежнему `(1, 1024)`. Даёт 46 мс контекста против 23 мс, без RNN.

**Инференс — не проблема:** VST-плагин обрабатывает кадры последовательно.
Предыдущий кадр просто лежит в буфере плагина (circular buffer). Первый кадр → нули.
Это стандартная практика для любого stateful audio-плагина.

**Сложность — в тренинге:**
Текущий датасет — плоский перемешанный массив окон. Для sequential context нужно:
1. Кастомный `Sampler`, который группирует окна по файлу и перемешивает только **файлы**,
   но внутри файла сохраняет порядок окон (используя `window_in_file` из meta.csv).
2. Специальная обработка первого окна каждого файла — предыдущее = нули.
3. `WindowDataset` переделать: хранить пары `(prev_window, cur_window)` или строить на лету.

**Когда делать:** Когда базовая тимбральная трансформация стабильно работает и нужно
улучшить воспроизведение атаки ноты.
