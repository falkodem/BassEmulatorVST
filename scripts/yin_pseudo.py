"""
YIN pitch detection — псевдокод с комментариями.

Источник: de Cheveigné & Kawahara, "YIN, a fundamental frequency estimator
for speech and music", JASA 2002.

Алгоритм time-domain: ищет периодичность сигнала через autocorrelation-like
метрику (squared difference function), нормализует её и берёт первый минимум.

Никакого обучения нет — чистая математика. Все параметры (порог, размер окна)
эвристические.

Этот файл — учебный псевдокод. Для production используются:
- C++ реализация в src/YinPitchDetector.h (наша)
- librosa.yin (Python, NumPy-векторизованный)
- aubio (C, fastest)
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Шаг 1: Difference function
# ──────────────────────────────────────────────────────────────────────────
def difference_function(x: np.ndarray, tau_max: int) -> np.ndarray:
    """
    Считаем "насколько сигнал отличается от своей сдвинутой копии" для каждого lag τ.

        d(τ) = Σ_{n=0}^{W-τ-1} (x[n] - x[n+τ])²

    Интуиция: если в сигнале есть период T, то x[n] ≈ x[n+T] для всех n,
    значит d(T) будет близка к нулю. Минимум d(τ) = период.

    Это родственник autocorrelation: при разворачивании квадрата получаем
        d(τ) = Σ x[n]² + Σ x[n+τ]² - 2·Σ x[n]·x[n+τ]
                                     ↑ это и есть autocorrelation r(τ)
    Но d(τ) лучше работает на коротких окнах: меньше bias к τ=0.

    Args:
        x: входной кадр, shape (W,)
        tau_max: максимальный lag для проверки (W/2 обычно)

    Returns:
        d: массив shape (tau_max,), где d[τ] = squared difference на лаге τ
    """
    W = len(x)
    d = np.zeros(tau_max)

    for tau in range(tau_max):
        # суммируем (x[n] - x[n+tau])² по доступным n
        # n+tau не должно выйти за границу окна
        diff = x[:W - tau] - x[tau:W]
        d[tau] = np.sum(diff ** 2)

    return d


# ──────────────────────────────────────────────────────────────────────────
# Шаг 2: Cumulative Mean Normalized Difference Function (CMNDF)
# ──────────────────────────────────────────────────────────────────────────
def cmndf(d: np.ndarray) -> np.ndarray:
    """
    Нормализуем d(τ) так, чтобы:
    - для случайного шума d'(τ) ≈ 1 (на всех τ)
    - для периодического сигнала d'(τ) ≈ 0 в точке периода
    - не было ложного минимума на τ=0

        d'(0) = 1
        d'(τ) = d(τ) / [ (1/τ) · Σ_{j=1}^{τ} d(j) ]

    Знаменатель — средняя difference на лагах [1..τ].
    Если d(τ) меньше этого среднего → точка значимо ниже фона → возможен период.

    Зачем это: без нормализации d(τ) монотонно растёт с τ (просто потому,
    что sum по большему диапазону), и абсолютный порог не работает.
    После CMNDF можно использовать фиксированный threshold (~0.1) на любой
    частоте — это и есть главный вклад YIN над autocorr.

    Args:
        d: difference function из шага 1

    Returns:
        d_prime: CMNDF, shape (len(d),). d_prime[0] = 1 по определению.
    """
    d_prime = np.ones_like(d)
    cumulative_sum = 0.0

    # начинаем с τ=1, потому что для τ=0 определение d'(0)=1
    for tau in range(1, len(d)):
        cumulative_sum += d[tau]
        # средняя difference на [1..τ] = cumulative_sum / τ
        # CMNDF = d(τ) / mean
        d_prime[tau] = d[tau] / (cumulative_sum / tau)

    return d_prime


# ──────────────────────────────────────────────────────────────────────────
# Шаг 3: Absolute threshold
# ──────────────────────────────────────────────────────────────────────────
def absolute_threshold(d_prime: np.ndarray, threshold: float = 0.1) -> int:
    """
    Ищем ПЕРВЫЙ локальный минимум CMNDF, который ниже порога.

    Идея: первый «достаточно глубокий» провал — это и есть период.
    Без поиска локального минимума можно случайно зацепиться за нисходящий
    склон следующего, более глубокого, минимума на 2τ → octave-down ошибка.

    threshold = 0.1 — стандартное значение из статьи.
    Меньше → строже (больше unvoiced-кадров), больше → мягче (больше ошибок).

    Args:
        d_prime: CMNDF
        threshold: порог (0.1–0.15)

    Returns:
        tau*: лаг с первым минимумом ниже порога, или -1 если не нашли (unvoiced)
    """
    tau = 2  # начинаем с τ=2, чтобы было место для проверки локального минимума

    while tau < len(d_prime) - 1:
        if d_prime[tau] < threshold:
            # нашли точку ниже порога — теперь спускаемся в локальный минимум
            # (если CMNDF продолжает падать — идём вниз вместе с ней)
            while tau + 1 < len(d_prime) and d_prime[tau + 1] < d_prime[tau]:
                tau += 1
            return tau
        tau += 1

    return -1  # unvoiced или сигнал без явной периодичности


# ──────────────────────────────────────────────────────────────────────────
# Шаг 4: Parabolic interpolation для sub-sample точности
# ──────────────────────────────────────────────────────────────────────────
def parabolic_interpolation(d_prime: np.ndarray, tau: int) -> float:
    """
    Период обычно НЕ кратен sample period. Если sample_rate = 44100 Гц
    и нота A2 (110 Гц) → период = 401 семпл с долей. YIN же выдаёт целое τ,
    значит F0 квантуется грубо (особенно на высоких частотах).

    Решение: вокруг найденного минимума τ* мы знаем 3 точки:
        (τ-1, d'(τ-1)), (τ, d'(τ)), (τ+1, d'(τ+1))
    Через них проводим параболу и берём её аналитический минимум.

    Формула вершины параболы по 3 точкам с равными x-шагами:
        offset = (d'(τ-1) - d'(τ+1)) / [2 · (d'(τ-1) - 2·d'(τ) + d'(τ+1))]
        τ_refined = τ + offset

    Точность улучшается на порядок (от 1 семпла до ~0.1 семпла).

    Args:
        d_prime: CMNDF
        tau: целый lag из absolute_threshold

    Returns:
        tau_refined: уточнённый лаг (float)
    """
    if tau <= 0 or tau >= len(d_prime) - 1:
        return float(tau)

    s_minus = d_prime[tau - 1]
    s_zero = d_prime[tau]
    s_plus = d_prime[tau + 1]

    # знаменатель может быть 0 в патологических случаях
    denom = 2.0 * (s_minus - 2.0 * s_zero + s_plus)
    if denom == 0:
        return float(tau)

    offset = (s_minus - s_plus) / denom
    return tau + offset


# ──────────────────────────────────────────────────────────────────────────
# Главная функция YIN
# ──────────────────────────────────────────────────────────────────────────
def yin(x: np.ndarray, sample_rate: int,
        f_min: float = 80.0, f_max: float = 800.0,
        threshold: float = 0.1) -> tuple[float, bool]:
    """
    Полный YIN pipeline на одном кадре.

    Args:
        x: кадр аудио, shape (W,). Обычно W = 1024 или 2048.
        sample_rate: частота дискретизации (44100)
        f_min: нижняя граница ожидаемых частот (для гитары E2 ≈ 82 Гц)
        f_max: верхняя граница (для нашей задачи E4 ≈ 330 Гц)
        threshold: CMNDF threshold (0.1)

    Returns:
        f0: оценённая F0 в Гц (или 0.0 если unvoiced)
        is_voiced: True если найден период

    Замечание про латентность:
        Минимальный период = sample_rate / f_max  (короткий)
        Максимальный период = sample_rate / f_min  (длинный)
        Окно W должно быть >= 2 · максимального периода,
        иначе на низких нотах не успеем увидеть полный период.

        Для E2 (82 Гц): период ≈ 538 семплов → W >= 1076.
        Возьмём W = 1024 → еле-еле проходим E2 ≈ 23 мс латентность.
    """
    # лаги, соответствующие f_min..f_max
    tau_min = int(sample_rate / f_max)  # короткий лаг = высокая частота
    tau_max = int(sample_rate / f_min)  # длинный лаг = низкая частота

    # ограничиваем сверху размером кадра
    tau_max = min(tau_max, len(x) // 2)

    # 1. Difference function
    d = difference_function(x, tau_max)

    # 2. CMNDF
    d_prime = cmndf(d)

    # 3. Absolute threshold (искажаем только в диапазоне [tau_min, tau_max])
    # обнуляем d_prime[:tau_min] чтобы не нашли слишком высокую частоту по ошибке
    d_prime_search = d_prime.copy()
    d_prime_search[:tau_min] = 1.0  # делаем эти лаги «не-минимумами»

    tau = absolute_threshold(d_prime_search, threshold)

    if tau < 0:
        return 0.0, False  # unvoiced

    # 4. Parabolic interpolation
    tau_refined = parabolic_interpolation(d_prime, tau)

    # 5. F0 = sr / период
    f0 = sample_rate / tau_refined

    return f0, True


# ──────────────────────────────────────────────────────────────────────────
# Пример использования
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sr = 44100
    duration = 0.05  # 50 мс
    f0_true = 220.0  # A3

    t = np.arange(int(sr * duration)) / sr
    # генерим сигнал из 3 гармоник (имитация гитарной ноты)
    signal = (np.sin(2 * np.pi * f0_true * t) +
              0.5 * np.sin(2 * np.pi * 2 * f0_true * t) +
              0.3 * np.sin(2 * np.pi * 3 * f0_true * t))

    # окно ~23 мс
    window = signal[:1024]

    f0_estimated, voiced = yin(window, sr)

    print(f"Истинная F0:    {f0_true:.2f} Гц")
    print(f"Оценённая F0:   {f0_estimated:.2f} Гц")
    print(f"Voiced:         {voiced}")
    print(f"Ошибка:         {abs(f0_estimated - f0_true):.3f} Гц "
          f"({100 * abs(f0_estimated - f0_true) / f0_true:.2f}%)")
