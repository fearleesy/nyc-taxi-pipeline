import os
import sys
import time
import json
import tracemalloc

from models.src.data_analyzer import DataAnalyzer
from models.src.model_trainer import TaxiModel
from models.src.preprocessing_pipeline import FeatureEngineer
from utils.io_helpers import load_data
from utils.logger import get_logger

logger = get_logger(__name__)
with open('views/params.json', 'r') as f:
    config = json.load(f)

models_storage_path = config["paths"]["models_storage_path"]

LATEST_MODEL_PATH = "models/latest_model.pkl"

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
    logger.debug(f"Starting test_model with model_name={model_name}, file_path={file_path}, metric={metric}, source_type={source_type}")


    if not os.path.exists(file_path):
        logger.debug(f"Test data not found: {file_path}")
        sys.exit(f"[test] Test data not found: {file_path}")

    model_path = os.path.join(models_storage_path, "latest_model.pkl") if model_name == "latest" else f"{models_storage_path}/{model_name}_model.pkl"
    logger.debug(f"Resolved model path: {model_path}")

    if not os.path.exists(model_path):
        logger.debug(f"Model file not found: {model_path}")
        sys.exit(f"[test] Model file not found: {model_path}")

    logger.debug("Loading model...")
    model = TaxiModel.load(model_path)
    logger.debug("Loading test data...")
    df = load_data(file_path, source_type)
    logger.debug(f"Loaded test data with {len(df)} rows")

    tracemalloc.start()

    start = time.perf_counter()
    analyzer = DataAnalyzer(df)
    clean_df = analyzer.fit_transform()
    elapsed_analyzer = time.perf_counter() - start
    logger.debug(f"DataAnalyzer.fit_transform completed in {elapsed_analyzer:.4f} seconds with {len(clean_df)} output rows")

    start = time.perf_counter()
    pre_df = FeatureEngineer(clean_df).fit_transform()
    elapsed_fe = time.perf_counter() - start
    logger.debug(f"FeatureEngineer.fit_transform completed in {elapsed_fe:.4f} seconds with {len(pre_df)} output rows")

    y_true = pre_df["log_trip_duration"]
    X = pre_df.drop(columns=["log_trip_duration"])
    logger.debug(f"Prepared features X with shape {X.shape} and target y with length {len(y_true)}")

    start = time.perf_counter()
    score1 = model.predict(X, y_true, "RMSE")
    score2 = model.predict(X, y_true, "MAE")
    elapsed_predict = time.perf_counter() - start
    logger.debug(f"Model prediction completed in {elapsed_predict:.4f} seconds")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.debug(f"Memory usage: Current = {current / 1024 / 1024:.2f} MB, Peak = {peak / 1024 / 1024:.2f} MB")
    if metric == "MAE":
        logger.info(f"[test] {metric}: {score2:.2f} seconds on {len(y_true)} samples.")
    else:
        logger.info(f"[test] {metric}: {score1:.2f} seconds on {len(y_true)} samples.")
    
    return score1, score2
