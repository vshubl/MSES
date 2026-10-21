from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from mses.io import load_voltammetry_data


def test_load_voltammetry_data_csv(tmp_path: Path):
    """Проверка успешной загрузки данных из CSV с десятичной запятой."""
    # Создаем временный CSV-файл с запятыми в качестве десятичного разделителя
    test_csv = tmp_path / "sample_data.csv"
    df_content = "Sample_ID,Sensor_1,Sensor_2\nSample_1,1,23,4,56\nSample_2,2,34,5,67\n"
    test_csv.write_text(df_content, encoding="utf-8")

    potentials, signals, df = load_voltammetry_data(test_csv)

    # Проверяем размерности и полученные типы
    assert isinstance(potentials, np.ndarray)
    assert isinstance(signals, np.ndarray)
    assert isinstance(df, pd.DataFrame)
    assert signals.shape == (2, 2)
    assert np.isclose(signals[0, 0], 1.23)
    assert np.isclose(signals[1, 1], 5.67)


def test_load_voltammetry_data_invalid_extension(tmp_path: Path):
    """Проверка выброса исключения ValueError при неподдерживаемом формате."""
    invalid_file = tmp_path / "data.txt"
    invalid_file.write_text("some random data")

    with pytest.raises(ValueError, match="Неподдерживаемый формат файла"):
        load_voltammetry_data(invalid_file)