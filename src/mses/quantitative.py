import logging
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def create_pls_pipeline(n_components: int = 5) -> Pipeline:
    """Создает безопасный Pipeline для PLS без утечки данных."""
    logger.info(f"Создание PLS Pipeline с латентными переменными: {n_components}")
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pls', PLSRegression(n_components=n_components))
    ])


def evaluate_pls_model(model: Pipeline, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Оценка качества модели и расчет метрик RMSEC, RMSEP, R^2."""
    logger.info("Оценка PLS модели на train/test")
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    rmsec = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmsep = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)

    return {"RMSEC": rmsec, "RMSEP": rmsep, "R2": r2}


def calculate_lod(y_residuals_std: float, slope: float) -> float:
    """Расчет предела обнаружения LOD = 3 * Sy/x / b."""
    return 3.0 * y_residuals_std / slope