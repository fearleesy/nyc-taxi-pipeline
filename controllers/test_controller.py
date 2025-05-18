import os
import sys

from src.data_analyzer import DataAnalyzer
from src.model_trainer import TaxiModel
from src.preprocessing_pipeline import FeatureEngineer
from utils.io_helpers import load_data
from utils.logger import get_logger

logger = get_logger(__name__)

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
    logger.debug(f"Starting test_model with model_name={model_name}, file_path={file_path}, metric={metric}, source_type={source_type}")


    if not os.path.exists(file_path):
        logger.debug(f"Test data not found: {file_path}")
        sys.exit(f"[test] Test data not found: {file_path}")

    model_path = LATEST_MODEL_PATH if model_name == "latest" else f"models/{model_name}_model.pkl"
    logger.debug(f"Resolved model path: {model_path}")

    if not os.path.exists(model_path):
        logger.debug(f"Model file not found: {model_path}")
        sys.exit(f"[test] Model file not found: {model_path}")

    logger.debug("Loading model...")
    model = TaxiModel.load(model_path)
    logger.debug("Loading test data...")
    df = load_data(file_path, source_type)
    logger.debug(f"Loaded test data with {len(df)} rows")

    analyzer = DataAnalyzer(df)
    logger.debug("Initialized DataAnalyzer")

    clean_df = analyzer.fit_transform()
    logger.debug(f"DataAnalyzer fit_transform output rows: {len(clean_df)}")

    pre_df = FeatureEngineer(clean_df).fit_transform()
    logger.debug(f"FeatureEngineer fit_transform output rows: {len(pre_df)}")

    y_true = pre_df["log_trip_duration"]
    X = pre_df.drop(columns=["log_trip_duration"])
    logger.debug(f"Prepared features X with shape {X.shape} and target y with length {len(y_true)}")

    logger.debug("Running model prediction")
    score = model.predict(X, y_true, metric)
    
    logger.info(f"[test] {metric}: {score:.2f} seconds on {len(y_true)} samples.")
