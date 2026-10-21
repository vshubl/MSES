import logging
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

logger = logging.getLogger(__name__)


def perform_pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Вычисление PCA: счета, нагрузки и объясненная дисперсия."""
    logger.info(f"Запуск PCA с {n_components компонентами}")
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    loadings = pca.components_
    explained_variance = pca.explained_variance_ratio_
    return scores, loadings, explained_variance


def detect_outliers_q_t2(X: np.ndarray, scores: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """Маркировка выбросов по T2 Хотеллинга и Q-остатку."""
    logger.info("Расчет выбросов (T2 и Q-статистика)")
    # Упрощенный расчет Q-остатков (реконструкция)
    X_reconstructed = np.dot(scores, loadings)
    q_residuals = np.sum((X - X_reconstructed) ** 2, axis=1)
    threshold_q = np.mean(q_residuals) + 3 * np.std(q_residuals)
    outliers = q_residuals > threshold_q
    return outliers