import logging
import numpy as np
import pywt

logger = logging.getLogger(__name__)


def wavelet_compression(signal_matrix: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """Сжатие данных с помощью дискретного вейвлет-преобразования (DWT)."""
    logger.info(f"Сжатие DWT с вейвлетом {wavelet}, уровень {level}")
    compressed_list = []
    for i in range(signal_matrix.shape[1]):
        coeffs = pywt.wavedec(signal_matrix[:, i], wavelet, level=level)
        # Объединяем коэффициенты в единый вектор признаков для образца
        flat_coeffs = np.concatenate([c for c in coeffs])
        compressed_list.append(flat_coeffs)
    return np.array(compressed_list)


def generate_synthetic_data(n_samples: int = 200, n_sensors: int = 8, n_points: int = 2000) -> tuple[
    np.ndarray, np.ndarray]:
    """Генератор синтетических данных для отладки кода."""
    logger.info(f"Генерация синтетических данных: образцов={n_samples}, сенсоров={n_sensors}")
    np.random.seed(42)
    t = np.linspace(0, 10, n_points)

    # Истинные концентрации (например, 2 аналита)
    concentrations = np.random.rand(n_samples, 2)

    # Синтетические сигналы с перекрывающимися пиками и шумом
    X = np.zeros((n_samples, n_points * n_sensors))
    for i in range(n_samples):
        sample_signals = []
        for s in range(n_sensors):
            peak1 = concentrations[i, 0] * np.exp(-((t - 3.0 - s * 0.1) ** 2) / 0.5)
            peak2 = concentrations[i, 1] * np.exp(-((t - 6.0 - s * 0.1) ** 2) / 0.8)
            noise = np.random.normal(0, 0.02, n_points)
            signal = peak1 + peak2 + noise + 0.05 * t  # наклонная базовая линия
            sample_signals.append(signal)
        X[i, :] = np.concatenate(sample_signals)

    return X, concentrations