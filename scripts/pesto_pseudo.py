"""
PESTO pitch detection — псевдокод с комментариями.

Источник: Riou, Lattner, Hadjeres, Peeters,
"PESTO: Pitch Estimation with Self-Supervised Transposition-Equivariant
Objective", ICASSP 2023.

Алгоритм time-frequency: CQT-спектрограмма → CNN → распределение вероятностей
по pitch-бинам → argmax + interpolation.

В отличие от YIN, PESTO ОБУЧАЕТСЯ. Но обучается без F0-лейблов
(self-supervised) — единственное требование: сдвиг входа по частотной оси
должен приводить к такому же сдвигу выхода (equivariance).

Этот файл — учебный псевдокод. Для production:
- pip install pesto-pitch (PyTorch-инференс)
- ONNX-export для real-time JUCE-плагина
- ~28K params, ~0.5 мс/кадр на CPU
"""

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────
# Шаг 1: CQT (Constant-Q Transform)
# ──────────────────────────────────────────────────────────────────────────
def compute_cqt(x: np.ndarray, sample_rate: int,
                f_min: float = 32.7,  # C1
                bins_per_octave: int = 36,
                n_octaves: int = 7) -> np.ndarray:
    """
    CQT — лог-частотная спектрограмма с фиксированным числом бинов на октаву.

    Отличие от STFT:
    - STFT: линейная сетка частот (0, Δf, 2Δf, …) — на низких частотах разрешение
            хуже (бины ноты A1=55 и B1=62 могут попасть в один бин), на высоких
            — избыточно (десятки бинов между соседними нотами).
    - CQT:  логарифмическая сетка (f_k = f_min · 2^(k / bins_per_octave)) — каждая
            нота занимает одинаковое число бинов, какая бы ни была её абсолютная
            частота. Это естественное представление для музыки.

    Параметры PESTO:
    - bins_per_octave = 36 (3 бина на полутон → ~33 cents разрешение)
    - n_octaves = 7 (от ~33 Гц до ~4200 Гц — покрывает все музыкальные ноты)
    - всего бинов: 36 × 7 = 252

    Свойство, которое использует PESTO:
        транспонирование сигнала на k полутонов
            ≡ сдвиг CQT-спектрограммы по частотной оси на (k · 3) бинов
    Это и есть equivariance, на которой строится self-supervised loss.

    На практике: librosa.cqt(x, sr=sr, fmin=f_min, n_bins=252, bins_per_octave=36).
    Реализация требует filter bank и FFT, нетривиальна — используем библиотеку.

    Returns:
        cqt: shape (n_bins, n_frames), magnitude (или log-magnitude)
    """
    # Псевдо-вызов библиотеки. В реальности:
    # cqt = np.abs(librosa.cqt(x, sr=sample_rate, fmin=f_min,
    #                          n_bins=bins_per_octave * n_octaves,
    #                          bins_per_octave=bins_per_octave))
    # cqt = np.log1p(cqt)  # log-magnitude — стандарт для нейросетей
    n_bins = bins_per_octave * n_octaves
    n_frames = len(x) // 256  # hop_length=256
    cqt = np.zeros((n_bins, n_frames))  # placeholder
    return cqt


# ──────────────────────────────────────────────────────────────────────────
# Шаг 2: CNN encoder
# ──────────────────────────────────────────────────────────────────────────
class PESTONet(nn.Module):
    """
    Маленькая 1D-CNN, принимающая один CQT-кадр и выдающая распределение
    по pitch-бинам.

    Архитектура из статьи:
    - Input: CQT-кадр (252 бина) + контекстное окно ~3 кадра
    - 4-5 conv1d-слоёв с small kernels (3-5)
    - GAP / flatten
    - FC → 384 выходных бинов (pitch-классы)
    - Softmax → pmf

    ~28K параметров — крошечная сеть, можно держать на CPU без затрат.

    Pitch-сетка на выходе:
    - 384 бина
    - покрывают диапазон 8 октав (можно настроить)
    - 384 / 8 = 48 бин на октаву → 4 бина на полутон → 25 cents разрешение
    - argmax даёт грубый ответ, interpolation уточняет
    """

    def __init__(self, n_pitch_bins: int = 384):
        super().__init__()
        # цепочка conv1d по частотной оси
        # input: (batch, 1, 252_freq_bins)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        # после трёх pool: 252 → 126 → 63 → 31
        self.fc = nn.Linear(64 * 31, n_pitch_bins)

    def forward(self, cqt_frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cqt_frame: (batch, 1, 252) — один CQT-кадр

        Returns:
            logits: (batch, 384) — нет softmax, его делаем снаружи
        """
        x = self.relu(self.conv1(cqt_frame))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        logits = self.fc(x)
        return logits


# ──────────────────────────────────────────────────────────────────────────
# Шаг 3: Argmax + parabolic interpolation
# ──────────────────────────────────────────────────────────────────────────
def decode_pitch(probabilities: np.ndarray,
                 bin_to_hz_func) -> tuple[float, float]:
    """
    Превращаем pmf по pitch-бинам в одно число F0.

    1. argmax → грубый бин
    2. parabolic interpolation вокруг argmax → уточнение (тот же приём, что в YIN)
    3. бин → Гц через bin_to_hz_func

    Confidence = max(probabilities) — используется как «voicing-detector».
    Если max < 0.3 (например) → unvoiced.

    Args:
        probabilities: (384,) — softmax-выход CNN
        bin_to_hz_func: функция bin_index (float) → Гц

    Returns:
        f0: частота в Гц
        confidence: max-probability ∈ [0, 1]
    """
    bin_argmax = int(np.argmax(probabilities))
    confidence = float(probabilities[bin_argmax])

    # parabolic interpolation для sub-bin точности
    if 0 < bin_argmax < len(probabilities) - 1:
        s_minus = probabilities[bin_argmax - 1]
        s_zero = probabilities[bin_argmax]
        s_plus = probabilities[bin_argmax + 1]
        denom = 2.0 * (s_minus - 2.0 * s_zero + s_plus)
        offset = (s_minus - s_plus) / denom if denom != 0 else 0.0
    else:
        offset = 0.0

    bin_refined = bin_argmax + offset
    f0 = bin_to_hz_func(bin_refined)
    return f0, confidence


# ──────────────────────────────────────────────────────────────────────────
# Шаг 3b: Soft-argmax с локальной маской (как в реальном PESTO inference)
# ──────────────────────────────────────────────────────────────────────────
def decode_with_local_softargmax(probabilities: np.ndarray,
                                 bin_to_hz_func,
                                 window_radius: int = 5) -> tuple[float, float]:
    """
    Робастная альтернатива decode_pitch — то, что используется в
    реальном PESTO inference (и в обучении, через дифференцируемый E[bin]).

    Зачем: «наивный» soft-argmax по всей pmf
            ŷ = Σ_k  k · p[k]
    ломается на бимодальных распределениях. Если у сети есть octave-ambiguity
    и она выдаёт два пика — на бине F и на бине 2F — expected value встанет
    между ними и выдаст совершенно неправильный pitch.

    Решение — двухшаговый decode:
        1. argmax → грубый бин (выбираем «главный» пик)
        2. soft-argmax только в окне ±window_radius вокруг argmax → уточняем

    Это уже идейно ближе к YIN parabolic interpolation: «локальная интерполяция
    вокруг найденного минимума/максимума», только окно шире (±5 vs ±1 у YIN)
    и веса берутся из pmf, а не из формы параболы.

    window_radius:
    - 1 → по сути та же parabolic interpolation (3 точки)
    - 5 → стандарт PESTO; ширина «нормального» пика ~3-7 бинов после Gaussian
          label-smoothing, окно должно покрывать весь пик
    - >10 → начинают подмешиваться хвосты, expected value «дрейфует»

    Args:
        probabilities: (N_bins,) — softmax-выход CNN
        bin_to_hz_func: bin_index (float) → Hz
        window_radius: радиус окна вокруг argmax

    Returns:
        f0: уточнённая частота в Гц
        confidence: суммарная вероятность в окне (после маски)
    """
    n_bins = len(probabilities)
    bin_argmax = int(np.argmax(probabilities))

    # 1. определяем границы окна
    lo = max(0, bin_argmax - window_radius)
    hi = min(n_bins, bin_argmax + window_radius + 1)

    # 2. вырезаем pmf в окне и перенормируем (чтобы Σp = 1 локально)
    p_local = probabilities[lo:hi]
    p_sum = p_local.sum()
    if p_sum < 1e-9:
        # вырожденный случай — окно пустое
        return float(bin_to_hz_func(bin_argmax)), 0.0
    p_local_norm = p_local / p_sum

    # 3. soft-argmax в локальных координатах
    bin_indices = np.arange(lo, hi, dtype=np.float32)
    bin_refined = float((p_local_norm * bin_indices).sum())

    # confidence: сколько вероятностной массы попало в окно
    # (если pmf «сосредоточена» — будет ≈1, если размазана — меньше)
    confidence = float(p_sum)

    f0 = bin_to_hz_func(bin_refined)
    return f0, confidence


# ──────────────────────────────────────────────────────────────────────────
# Шаг 4 — главное в PESTO: self-supervised обучение
# ──────────────────────────────────────────────────────────────────────────
def equivariance_loss(model: PESTONet,
                       cqt_original: torch.Tensor,
                       k_shift_semitones: int,
                       bins_per_octave: int = 36) -> torch.Tensor:
    """
    Главный трюк PESTO: training БЕЗ pitch-лейблов.

    Идея:
    Если мы возьмём CQT и сдвинем его по оси частот на (k · bins_per_octave/12)
    бинов, это равносильно транспонированию сигнала на k полутонов.

    Значит, если модель честно предсказывает F0:
        F0_pred(shifted_input) - F0_pred(original_input) = k полутонов

    Loss:
        L = | (F0_pred_shifted - F0_pred_original) - k_semitones |²

    Этот loss НЕ требует знать истинную F0! Он требует только знать,
    как мы сдвинули вход — что мы контролируем (random augmentation).

    Дополнительный loss (entropy regularization):
    - Чтобы pmf не размазалась по всему диапазону, добавляют энтропию pmf
      как регуляризатор: -H(p) → минимизируем, заставляя сеть быть уверенной.

    Всё обучение — на parsed audio из YouTube/Slakh без лейблов F0.
    Сеть учится распознавать «выглядит как нота на этой частоте» по форме спектра.

    Args:
        model: PESTONet
        cqt_original: (batch, 1, 252)
        k_shift_semitones: насколько сдвигаем (random int, например ±12)

    Returns:
        loss: скаляр, на который делаем backward()
    """
    # bins_per_semitone = bins_per_octave / 12
    bins_per_semitone = bins_per_octave // 12  # 3 для дефолтных 36 бин/окт
    shift_bins = k_shift_semitones * bins_per_semitone

    # сдвигаем CQT по частотной оси (torch.roll или slice + pad)
    # ВАЖНО: shift в CQT-домене — это просто permutation бинов, НЕ pitch-shift аудио
    # (последний дорогой, первый — почти бесплатный)
    cqt_shifted = torch.roll(cqt_original, shifts=shift_bins, dims=-1)

    # forward через сеть
    logits_orig = model(cqt_original)
    logits_shift = model(cqt_shifted)

    pmf_orig = torch.softmax(logits_orig, dim=-1)
    pmf_shift = torch.softmax(logits_shift, dim=-1)

    # «soft argmax» — вычисляем ожидаемый pitch-бин
    bin_indices = torch.arange(pmf_orig.shape[-1], dtype=torch.float32)
    expected_bin_orig = (pmf_orig * bin_indices).sum(dim=-1)
    expected_bin_shift = (pmf_shift * bin_indices).sum(dim=-1)

    # ожидаем: expected_bin_shift - expected_bin_orig ≈ shift_bins
    predicted_shift = expected_bin_shift - expected_bin_orig
    target_shift = float(shift_bins)

    loss = ((predicted_shift - target_shift) ** 2).mean()
    return loss


# ──────────────────────────────────────────────────────────────────────────
# Главная функция PESTO (inference)
# ──────────────────────────────────────────────────────────────────────────
def pesto_inference(x: np.ndarray, sample_rate: int,
                    model: PESTONet) -> tuple[np.ndarray, np.ndarray]:
    """
    Полный PESTO pipeline для готовой обученной модели.

    Args:
        x: аудио, shape (n_samples,)
        sample_rate: 44100
        model: обученная PESTONet

    Returns:
        f0_per_frame: (n_frames,) — F0 в Гц для каждого CQT-кадра
        confidence_per_frame: (n_frames,) — confidence ∈ [0, 1]

    Streaming замечание:
    PESTO в реальном времени работает кадр-за-кадром. Латентность определяется
    окном CQT (~80 мс для нижних октав), а не forward-pass-ом сети (он < 1 мс).

    Есть варианты «причинного» CQT с укороченными окнами для ВЧ-контента —
    тогда латентность падает до ~10 мс ценой деградации на низких нотах.
    """
    # 1. CQT
    cqt = compute_cqt(x, sample_rate)  # (n_bins, n_frames)

    # 2. инференс кадр-за-кадром
    n_frames = cqt.shape[1]
    f0_per_frame = np.zeros(n_frames)
    confidence_per_frame = np.zeros(n_frames)

    # bin → Hz: pitch-сетка модели (зависит от того, как обучали)
    # допустим, модель покрывает [27.5 Гц, 4186 Гц] на 384 бинах
    f_min, f_max = 27.5, 4186.0
    log_f_min, log_f_max = np.log2(f_min), np.log2(f_max)

    def bin_to_hz(bin_idx: float) -> float:
        # линейно по log-частоте
        log_f = log_f_min + (bin_idx / 384) * (log_f_max - log_f_min)
        return float(2 ** log_f)

    model.eval()
    with torch.no_grad():
        for t in range(n_frames):
            cqt_frame = torch.from_numpy(cqt[:, t]).float().unsqueeze(0).unsqueeze(0)
            # shape: (1, 1, n_bins)
            logits = model(cqt_frame)
            pmf = torch.softmax(logits, dim=-1).squeeze(0).numpy()

            # decode_pitch — простая parabolic interpolation (3 точки)
            # decode_with_local_softargmax — то, что реально использует PESTO
            f0, conf = decode_with_local_softargmax(pmf, bin_to_hz, window_radius=5)
            f0_per_frame[t] = f0
            confidence_per_frame[t] = conf

    return f0_per_frame, confidence_per_frame


# ──────────────────────────────────────────────────────────────────────────
# Сравнение с YIN
# ──────────────────────────────────────────────────────────────────────────
"""
                    YIN                       PESTO
───────────────────────────────────────────────────────────────────
Domain              Time                      Time-Frequency (CQT)
Принцип             min CMNDF                 CNN-classification
Параметры           0                         28K
Обучение            нет                       self-supervised (equivariance)
Знание о гармониках нет — только периодичность да — выучено из данных
Шум, форманты       чувствителен              устойчив
Octave errors       часто на низких/слабых    редко
Чистая синусоида    идеально                  может промахнуться (out-of-dist)
Латентность         W/2 = ~12 мс              CQT window + ~0.5 мс forward
Inference           C++ header-only           ONNX (~1.5 MB)
Voicing             d'(τ) < threshold         max(pmf) > threshold

Главная мысль:
    YIN — детектор повторяемости. Math-based, агностичный к смыслу сигнала.
    PESTO — распознаватель шаблонов. Учится «как выглядит нота с такой F0».

YIN молотит автокорреляцию и берёт минимум — он не знает, что такое
«спектр гитары» или «формант». Поэтому он одинаково хорошо/плохо работает
на любом сигнале, где есть периодичность.

PESTO выучил, что у настоящих нот есть характерный гармонический рисунок
в CQT — ровные ряды бинов на частотах k·F0. Поэтому он устойчив к шуму
и формантам (если они попали в training set), но может сломаться на
сигнале вне распределения (например, чистая синусоида у пианиссимо-флейты,
которой не было в обучении).
"""
