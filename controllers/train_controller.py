import os
import sys
import time
import tracemalloc
import json
from typing import Optional

from models.src.db_manager import DBManager
from models.src.data_analyzer import DataAnalyzer
from models.src.model_trainer import TaxiModel
from models.src.preprocessing_pipeline import FeatureEngineer
from utils.model_helpers import quick_dataset_summary, auto_select_model
from utils.logger import get_logger

logger = get_logger(__name__)
with open('views/params.json', 'r') as f:
    config = json.load(f)

models_storage_path = config["paths"]["models_storage_path"]

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

    if model_type is None:
        df_stats = quick_dataset_summary(df)
        logger.debug(f"Dataset summary: {df_stats}")

        uniqueness = analyzer.calculate_uniqueness()
        logger.debug(f"Uniqueness stats: {uniqueness}")

        model_type = auto_select_model(df_stats, uniqueness)
        logger.debug(f"Auto-selected model type: {model_type}")
    else:
        logger.debug(f"Using provided model type: {model_type}")
    
    tracemalloc.start()

    start = time.perf_counter()
    clean_df = analyzer.fit_transform()
    elapsed_analyzer = time.perf_counter() - start
    logger.debug(f"DataAnalyzer.fit_transform completed in {elapsed_analyzer:.4f} seconds with {len(clean_df)} output rows")

    start = time.perf_counter()
    pre_df = FeatureEngineer(clean_df).fit_transform()
    elapsed_fe = time.perf_counter() - start
    logger.debug(f"FeatureEngineer.fit_transform completed in {elapsed_fe:.4f} seconds with {len(pre_df)} output rows")

    y = pre_df["log_trip_duration"]
    X = pre_df.drop(columns=["log_trip_duration"])
    logger.debug(f"Prepared features X with shape {X.shape} and target y with length {len(y)}")

    model = TaxiModel(model_type)
    logger.debug(f"Created TaxiModel instance with model_type={model_type}")

    start = time.perf_counter()
    model.train(model_type, X, y, warm_start)
    elapsed_train = time.perf_counter() - start
    logger.debug(f"Model training completed in {elapsed_train:.4f} seconds")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.debug(f"Memory usage: Current = {current / 1024 / 1024:.2f} MB, Peak = {peak / 1024 / 1024:.2f} MB")

    latest_model_path = os.path.join(models_storage_path, "latest_model.pkl")
    model.save(latest_model_path)
    model.save(os.path.join(models_storage_path, f"{model_type}_model.pkl"))

    logger.info(
        f"[train] Trained {model_type} on {len(clean_df)} rows → saved to "
        f"{latest_model_path} & {models_storage_path}/{model_type}_model.pkl."
    )