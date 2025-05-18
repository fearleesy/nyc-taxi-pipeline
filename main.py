import argparse
import os
import sys
from typing import Optional

import pandas as pd

from src.data_analyzer import DataAnalyzer
from src.model_trainer import TaxiModel
from src.db_manager import DBManager
from src.preprocessing_pipeline import FeatureEngineer


LATEST_MODEL_PATH = "models/latest_model.pkl"


def add_data(file_path: str, db_path: str = "work.db", start: int = 0, end: Optional[int] = None) -> None:
    """Load rows [start:end] from CSV and append them to the SQLite database.

    Parameters
    ----------
    file_path : str
        Path to the CSV file with raw trip data to add.
    db_path : str
        Target DB file that accumulates all training data.
    start : int
        Starting index of the rows to include from file_path.
    end : int, optional
        Ending index (exclusive) of the rows to include from file_path.
    """
    if not os.path.exists(file_path):
        sys.exit(f"[add_data] File not found: {file_path}")

    new_df = pd.read_csv(file_path)
    if new_df.empty:
        sys.exit(f"[add_data] No rows found in {file_path}.")

    sliced_df = new_df.iloc[start:end]
    if sliced_df.empty:
        sys.exit(f"[add_data] No data in slice [{start}:{end}].")

    db = DBManager(db_path)
    db.insert_df(sliced_df)
    print(f"[add_data] Inserted {len(sliced_df)} rows into {db_path}. Total length: {db.get_length()}")

def train_model(model_type: str, warm_start: bool, db_path: str, start: int = 0, end: Optional[int] = None) -> None:
    """Train *model_type* on data at *csv_path* and save to disk.

    Parameters
    ----------
    model_type : str
        One of the models supported by ``TaxiModel`` (e.g. LR, KNN, DT, RF, Lasso, Ridge).
    warm_start : bool
        Whether to continue training an existing model if supported.
    db_path : str
        Path to the DB file containing the training data.
    start : int
        Starting index of the rows to train on from database.
    end : int, optional
        Ending index (exclusive) of the rows to train on from database.
    """
    db = DBManager(db_path)
    df = db.fetch_range(start, end)
    if df.empty:
        sys.exit(
            f"[train] No training data found in slice [{start}:{end}] of {db_path}."
        )

    analyzer = DataAnalyzer(df)
    clean_df = analyzer.fit_transform()

    preprocesser = FeatureEngineer(clean_df)
    preprocessed_df = preprocesser.fit_transform()

    print(preprocessed_df)

    y = preprocessed_df["log_trip_duration"]
    X = preprocessed_df.drop(columns=["log_trip_duration"])

    model = TaxiModel(model_type)
    if model.pipeline == 0:
        sys.exit(f"[train] Unsupported model type: {model_type}")

    model.train(model_type, X, y, warm_start)

    model.save(LATEST_MODEL_PATH)
    model.save(f"models/{model_type}_model.pkl")

    print(
        f"[train] Trained {model_type} on {len(clean_df)} rows → saved to {LATEST_MODEL_PATH} & models/{model_type}_model.pkl."
    )

def test_model(model_name: str, csv_path: str, metric: str = "MAE") -> None:
    """Evaluate a previously-saved model against *csv_path*.

    Parameters
    ----------
    model_name : str
        Model identifier ("latest" or the base name used when training).
    csv_path : str
        Path to CSV with evaluation data.
    metric : str, optional
        Evaluation metric (RMSE or MAE). Defaults to MAE.
    """
    if not os.path.exists(csv_path):
        sys.exit(f"[test] Test data not found: {csv_path}")

    model_path = LATEST_MODEL_PATH if model_name == "latest" else f"models/{model_name}_model.pkl"
    if not os.path.exists(model_path):
        sys.exit(f"[test] Model file not found: {model_path}")

    model = TaxiModel.load(model_path)
    df = pd.read_csv(csv_path)

    analyzer = DataAnalyzer(df)
    clean_df = analyzer.fit_transform()

    preprocesser = FeatureEngineer(clean_df)
    preprocessed_df = preprocesser.fit_transform()

    y_true = preprocessed_df["log_trip_duration"]
    X = preprocessed_df.drop(columns=["log_trip_duration"])

    score = model.predict(X, y_true, metric)
    print(f"[test] {metric}: {score:.2f} seconds on {len(y_true)} samples.")

def summarize(db_path: str = "work.db", start: int = 0, end: Optional[int] = None) -> None:
    """Print basic statistics for a slice of *csv_path*."""
    db = DBManager(db_path)
    df_slice = db.fetch_range(start, end)
    if df_slice.empty:
        sys.exit(f"[summ] No data in slice [{start}:{end}].")

    analyzer = DataAnalyzer(df_slice)
    try:
        stats = analyzer._calculate_basic_stats()
    except AttributeError:
        sys.exit("[summ] DataAnalyzer does not implement _calculate_basic_stats().")

    print("Statistics for the current sample:")
    for k, v in stats.items():
        print(f"{k} :")
        for w, ans in v.items():
            print(f"\t{w} : {ans}")
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI for taxi-trip duration ML pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # add_data
    p_add = subparsers.add_parser("add_data", help="Append raw data to the main training SQLite database.")
    p_add.add_argument("--file_path", type=str, default="train.csv", help="Path to CSV file with new data to add.")
    p_add.add_argument("--train_db", type=str, default="work.db", help="Path to the current database.")
    p_add.add_argument("-s", "--start", type=int, default=0, help="Start index of rows to add.")
    p_add.add_argument("-e", "--end", type=int, default=None, help="End index (exclusive) of rows to add.")

    # train
    p_train = subparsers.add_parser("train", help="Train a model on given CSV data.")
    p_train.add_argument("model", type=str, help="Model type (LR, KNN, DT, RF, Lasso, Ridge, ...).")
    p_train.add_argument(
        "-w", "--warm_start",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether to warm-start training if the model supports it (0 = no, 1 = yes).",
    )
    p_train.add_argument("--db", type=str, default="work.db", help="DB file containing training data.")
    p_train.add_argument("-s", "--start", type=int, default=0, help="Start index of rows to train.")
    p_train.add_argument("-e", "--end", type=int, default=None, help="End index (exclusive) of rows to train.")

    # test
    p_test = subparsers.add_parser("test", help="Evaluate a trained model.")
    p_test.add_argument("--model", type=str, default="latest", help='Model identifier ("latest" or base model name).')
    p_test.add_argument("--db", type=str, default="test.csv", help="DB file with test/evaluation data.")
    p_test.add_argument(
        "--metric",
        type=str,
        default="RMSE",
        choices=["RMSE", "MAE"],
        help="Evaluation metric to report.",
    )

    # summ
    p_sum = subparsers.add_parser("summ", help="Show basic statistics for a SQLite database.")
    p_sum.add_argument("--db", nargs="?", default="work.db", help="DB file to summarize.")
    p_sum.add_argument("-s", "--start", nargs="?", type=int, default=0, help="Start index of slice.")
    p_sum.add_argument("-e", "--end", nargs="?", type=int, default=None, help="End index (exclusive) of slice.")

    return parser

def main(argv: Optional[list[str]] = None) -> None:
    parser = _create_parser()
    args = parser.parse_args(argv)
  

    if args.command == "add_data":
        add_data(args.file_path, args.train_db, args.start, args.end)
    elif args.command == "train":
        train_model(args.model, bool(args.warm_start), args.db, args.start, args.end)
    elif args.command == "test":
        test_model(args.model, args.db, args.metric)
    elif args.command == "summ":
        summarize(args.db, args.start, args.end)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()

