import logging
import typer

app = typer.Typer(help="CLI для обработки сигналов мультисенсорной электрохимической системы")
logging.basicConfig(level=logging.INFO)


@app.command()
def fit(data_path: str, model_output: str = "model.pkl"):
    """Калибровка модели по файлу данных."""
    typer.echo(f"Обучение модели на основе данных из {data_path}...")
    # Здесь вызываются функции из io.py, preprocess.py и quantitative.py


@app.command()
def predict(data_path: str, model_path: str = "model.pkl"):
    """Предсказание концентраций для новых образцов."""
    typer.echo(f"Предсказание для файла {data_path}...")


@app.command()
def report(output_file: str = "report.html"):
    """Генерация сводного HTML-отчета."""
    typer.echo(f"Генерация отчета в файл {output_file}...")


if __name__ == "__main__":
    app()