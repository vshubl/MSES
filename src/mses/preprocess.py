import logging
import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


def apply_savgol(signal_matrix: np.ndarray, window_length: int = 15, polyorder: int = 2) -> np.ndarray:
    """Сглаживание фильтром Савицкого–Голея."""
    logger.info("Применение фильтра Савицкого–Голея")
    return savgol_filter(signal_matrix, window_length=window_length, polyorder=polyorder, axis=0)


def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Вычисление базовой линии методом AsLS (асимметричные наименьшие квадраты)."""
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def apply_asls_matrix(signal_matrix: np.ndarray, lam: float = 1e5, p: float = 0.01) -> np.ndarray:
    """Вычитание базовой линии методом AsLS для матрицы сигналов."""
    logger.info(f"Коррекция базовой линии AsLS (lam={lam}, p={p})")
    corrected = np.empty_like(signal_matrix)
    for i in range(signal_matrix.shape[1]):
        base = baseline_als(signal_matrix[:, i], lam=lam, p=p)
        corrected[:, i] = signal_matrix[:, i] - base
    return corrected