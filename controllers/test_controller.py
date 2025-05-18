import os
import sys

from src.data_analyzer import DataAnalyzer
from src.model_trainer import TaxiModel
from src.preprocessing_pipeline import FeatureEngineer
from utils.io_helpers import load_data

LATEST_MODEL_PATH = "model_storage/latest_model.pkl"

def test_model(
    model_name: str,
    file_path: str,
    metric: str = "RMSE",
    source_type: str = "csv"
) -> None:
    """
    Evaluate a saved model against labeled data and report performance.

    Parameters
    ----------
    model_name : str
        Identifier of the model to test. Use "latest" for the most recently trained model.
    file_path : str
        Path to file with evaluation data (CSV or other supported format).
    metric : str
        Evaluation metric: "RMSE" or "MAE". Default is "RMSE".
    source_type : str
        Format of the evaluation file: "csv", "json", etc.
    """
    if not os.path.exists(file_path):
        sys.exit(f"[test] Test data not found: {file_path}")

    model_path = LATEST_MODEL_PATH if model_name == "latest" else f"models/{model_name}_model.pkl"
    if not os.path.exists(model_path):
        sys.exit(f"[test] Model file not found: {model_path}")

    model = TaxiModel.load(model_path)
    df = load_data(file_path, source_type)

    analyzer = DataAnalyzer(df)
    clean_df = analyzer.fit_transform()
    pre_df = FeatureEngineer(clean_df).fit_transform()

    y_true = pre_df["log_trip_duration"]
    X = pre_df.drop(columns=["log_trip_duration"])

    score = model.predict(X, y_true, metric)
    print(f"[test] {metric}: {score:.2f} seconds on {len(y_true)} samples.")
