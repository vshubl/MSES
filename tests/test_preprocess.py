import numpy as np
import pytest
from mses.preprocess import apply_asls_matrix, apply_savgol, baseline_als


def test_apply_savgol():
    """Проверка фильтра Савицкого–Голея на сигнале с шумом."""
    t = np.linspace(0, 10, 100)
    pure_signal = np.sin(t)
    noisy_signal = pure_signal + np.random.normal(0, 0.1, size=t.shape)

    # Применяем сглаживание к матрице (столбец = сигнал)
    signal_matrix = noisy_signal.reshape(-1, 1)
    smoothed = apply_savgol(signal_matrix, window_length=11, polyorder=2)

    assert smoothed.shape == signal_matrix.shape
    # Среднеквадратичное отклонение от чистого сигнала должно уменьшиться после сглаживания
    rmse_before = np.sqrt(np.mean((noisy_signal - pure_signal) ** 2))
    rmse_after = np.sqrt(np.mean((smoothed.flatten() - pure_signal) ** 2))
    assert rmse_after < rmse_before


def test_baseline_als_correction():
    """Проверка AsLS коррекции на сигнале с трендом (наклонной базовой линией)."""
    t = np.linspace(0, 10, 200)
    baseline_true = 0.5 * t  # Линейный дрейф
    peak = 2.0 * np.exp(-((t - 5) ** 2) / 0.5)
    y = peak + baseline_true

    # Коррекция одиночной векторной базовой линии
    estimated_baseline = baseline_als(y, lam=1e5, p=0.01)

    assert estimated_baseline.shape == y.shape
    # Проверяем, что найденная базовая линия близка к истинной вне области пика
    off_peak_mask = (t < 3) | (t > 7)
    assert np.allclose(estimated_baseline[off_peak_mask], baseline_true[off_peak_mask], atol=0.2)


def test_apply_asls_matrix():
    """Проверка применения AsLS ко всей матрице сигналов."""
    t = np.linspace(0, 5, 100)
    signal1 = np.sin(t) + 0.1 * t
    signal2 = np.cos(t) + 0.2 * t

    matrix = np.column_stack([signal1, signal2])
    corrected = apply_asls_matrix(matrix, lam=1e4, p=0.01)

    assert corrected.shape == matrix.shape
    assert not np.isnan(corrected).any()