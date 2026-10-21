import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_voltammetry_data(file_path: str | Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Загружает CSV/XLSX файл с поддержкой десятичной запятой.

    Ожидает формат, где столбцы содержат потенциалы/токи для сенсоров.
    """
    path = Path(file_path)
    logger.info(f"Загрузка данных из файла: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, sep=None, engine="python", decimal=",")
    elif path.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")

    # Пример простейшего парсинга: выделяем метаданные и матрицу сигналов
    # (предполагается, что первые столбцы — метаданные, остальные — точки вольтамперограммы)
    logger.debug(f"Успешно прочитано строк: {len(df)}, колонок: {len(df.columns)}")

    # Возвращаем заглушки для сетки потенциалов, массива сигналов и датафрейма метаданных
    potentials = np.linspace(-1.0, 1.0, 2000)
    signals = df.select_dtypes(include=[np.number]).to_numpy()
    return potentials, signals, df