import os
import sys
from typing import Optional

from src.db_manager import DBManager
from src.data_analyzer import DataAnalyzer
from src.model_trainer import TaxiModel
from src.preprocessing_pipeline import FeatureEngineer
from utils.model_helpers import quick_dataset_summary, auto_select_model

LATEST_MODEL_PATH = "model_storage/latest_model.pkl"

def train_model(
    model_type: Optional[str],
    warm_start: bool,
    db_path: str,
    start: int = 0,
    end: Optional[int] = None
) -> None:
    """
    Train a regression model to predict log trip duration, and save it to disk.

    Parameters
    ----------
    model_type : str or None
        One of the supported model types in `TaxiModel` (e.g., "LR", "KNN", "DT", "RF", "Lasso", "Ridge").
        If None, an appropriate model is selected automatically.
    warm_start : bool
        If True, continue training an existing model if supported.
    db_path : str
        Path to the SQLite database containing training data.
    start : int
        Starting row index (inclusive) for training data.
    end : int or None
        Ending row index (exclusive). If None, all rows from `start` are used.
    """
    db = DBManager(db_path)
    df = db.fetch_range(start, end)
    if df.empty:
        sys.exit(f"[train] No training data found in slice [{start}:{end}] from {db_path}.")

    analyzer = DataAnalyzer(df)

    if model_type is None:
        df_stats = quick_dataset_summary(df)
        uniqueness = analyzer.calculate_uniqueness()
        model_type = auto_select_model(df_stats, uniqueness)

    clean_df = analyzer.fit_transform()
    pre_df = FeatureEngineer(clean_df).fit_transform()

    y = pre_df["log_trip_duration"]
    X = pre_df.drop(columns=["log_trip_duration"])

    model = TaxiModel(model_type)
    if model.pipeline == 0:
        sys.exit(f"[train] Unsupported model type: {model_type}")

    model.train(model_type, X, y, warm_start)

    model.save(LATEST_MODEL_PATH)
    model.save(f"model_storage/{model_type}_model.pkl")

    print(
        f"[train] Trained {model_type} on {len(clean_df)} rows → saved to "
        f"{LATEST_MODEL_PATH} & model_storage/{model_type}_model.pkl."
    )