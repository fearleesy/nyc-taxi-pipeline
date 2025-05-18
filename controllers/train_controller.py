import os
import sys
from typing import Optional

from src.db_manager import DBManager
from src.data_analyzer import DataAnalyzer
from src.model_trainer import TaxiModel
from src.preprocessing_pipeline import FeatureEngineer
from utils.model_helpers import quick_dataset_summary, auto_select_model
from utils.logger import get_logger

logger = get_logger(__name__)

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
    logger.debug(f"Starting training: model_type={model_type}, warm_start={warm_start}, db_path={db_path}, start={start}, end={end}")


    db = DBManager(db_path)
    df = db.fetch_range(start, end)
    logger.debug(f"Fetched {len(df)} rows from DB slice [{start}:{end}]")

    if df.empty:
        logger.debug(f"No training data found in slice [{start}:{end}] from {db_path}.")
        sys.exit(f"[train] No training data found in slice [{start}:{end}] from {db_path}.")

    analyzer = DataAnalyzer(df)
    logger.debug("Initialized DataAnalyzer")

    if model_type is None:
        df_stats = quick_dataset_summary(df)
        logger.debug(f"Dataset summary: {df_stats}")

        uniqueness = analyzer.calculate_uniqueness()
        logger.debug(f"Uniqueness stats: {uniqueness}")

        model_type = auto_select_model(df_stats, uniqueness)
        logger.debug(f"Auto-selected model type: {model_type}")
    else:
        logger.debug(f"Using provided model type: {model_type}")

    clean_df = analyzer.fit_transform()
    logger.debug(f"DataAnalyzer fit_transform output rows: {len(clean_df)}")

    pre_df = FeatureEngineer(clean_df).fit_transform()
    logger.debug(f"FeatureEngineer fit_transform output rows: {len(pre_df)}")

    y = pre_df["log_trip_duration"]
    X = pre_df.drop(columns=["log_trip_duration"])
    logger.debug(f"Prepared features X with shape {X.shape} and target y with length {len(y)}")

    model = TaxiModel(model_type)
    logger.debug(f"Created TaxiModel instance with model_type={model_type}")

    if model.pipeline == 0:
        logger.debug(f"Unsupported model type: {model_type}")
        sys.exit(f"[train] Unsupported model type: {model_type}")

    model.train(model_type, X, y, warm_start)
    logger.debug("Model training completed")

    model.save(LATEST_MODEL_PATH)
    model.save(f"model_storage/{model_type}_model.pkl")

    logger.info(
        f"Trained {model_type} on {len(clean_df)} rows → saved to "
        f"{LATEST_MODEL_PATH} & model_storage/{model_type}_model.pkl."
    )