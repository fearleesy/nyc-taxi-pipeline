import argparse
import os
import sys
from typing import Optional

import pandas as pd

from src.data_collector import DataCollector
from src.data_analyzer import DataAnalyzer
from src.model_trainer import TaxiModel
from src.preprocessing_pipeline import FeatureEngineer


LATEST_MODEL_PATH = "models/latest_model.pkl"


def add_data(file_path: str, train_csv: str = "work.csv", start: int = 0, end: Optional[int] = None) -> None:
    """Append a slice of new data from *file_path* to *train_csv* (creates the file if missing).

    Parameters
    ----------
    file_path : str
        Path to the CSV file with raw trip data to add.
    train_csv : str
        Target CSV that accumulates all training data.
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

    if os.path.exists(train_csv):
        train_df = pd.read_csv(train_csv)
        combined_df = pd.concat([train_df, sliced_df], ignore_index=True)
    else:
        combined_df = sliced_df

    combined_df.to_csv(train_csv, index=False)
    print(f"[add_data] Added {len(sliced_df)} rows → {train_csv} now has {len(combined_df)} rows.")

def train_model(model_type: str, warm_start: bool, csv_path: str, start: int = 0, end: Optional[int] = None) -> None:
    """Train *model_type* on data at *csv_path* and save to disk.

    Parameters
    ----------
    model_type : str
        One of the models supported by ``TaxiModel`` (e.g. LR, KNN, DT, RF, Lasso, Ridge).
    warm_start : bool
        Whether to continue training an existing model if supported.
    csv_path : str
        Path to the CSV file containing the training data.
    start : int
        Starting index of the rows to train on from csv_path.
    end : int, optional
        Ending index (exclusive) of the rows to train on from csv_path.
    """
    if not os.path.exists(csv_path):
        sys.exit(f"[train] Training data not found: {csv_path}")

    df = pd.read_csv(csv_path)
    sliced_df = df.iloc[start:end]
    analyzer = DataAnalyzer(sliced_df)
    clean_df = analyzer.fit_transform()

    y = clean_df["log_trip_duration"]
    X = clean_df.drop(columns=["log_trip_duration"])

    model = TaxiModel(model_type)
    if model.pipeline == 0:
        sys.exit(f"[train] Unsupported model type: {model_type}")

    model.train(model_type, X, y, warm_start)

    model.save(LATEST_MODEL_PATH)
    model.save(f"models/{model_type}_model.pkl")

    print(
        f"[train] Trained {model_type} on {len(sliced_df)} rows. Saved to {LATEST_MODEL_PATH} and models/{model_type}_model.pkl."
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

    y_true = clean_df["log_trip_duration"]
    X = clean_df.drop(columns=["log_trip_duration"])

    score = model.predict(X, y_true, metric)
    print(f"[test] {metric}: {score:.2f} seconds on {len(y_true)} samples.")

def summarize(csv_path: str = "work.csv", start: int = 0, end: Optional[int] = None) -> None:
    """Print basic statistics for a slice of *csv_path*."""
    if not os.path.exists(csv_path):
        sys.exit(f"[summ] CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df_slice = df.iloc[start:end]
    if df_slice.empty:
        sys.exit(f"[summ] No data in slice [{start}:{end}].")

    analyzer = DataAnalyzer(df_slice)
    try:
        stats = analyzer._calculate_basic_stats()
    except AttributeError:
        sys.exit("[summ] DataAnalyzer does not implement _calculate_basic_stats().")

    print("Статистика по текущей выборке:")
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
    p_add = subparsers.add_parser("add_data", help="Append raw data to the main training CSV.")
    p_add.add_argument("--file_path", type=str, default="train.csv", help="Path to CSV file with new data to add.")
    p_add.add_argument("--train_csv", type=str, default="work.csv", help="Path to the cumulative training CSV.")
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
    p_train.add_argument("--csv", type=str, default="work.csv", help="CSV file containing training data.")
    p_train.add_argument("-s", "--start", type=int, default=0, help="Start index of rows to train.")
    p_train.add_argument("-e", "--end", type=int, default=None, help="End index (exclusive) of rows to train.")

    # test
    p_test = subparsers.add_parser("test", help="Evaluate a trained model.")
    p_test.add_argument("model", type=str, help='Model identifier ("latest" or base model name).')
    p_test.add_argument("--csv", type=str, default="test.csv", help="CSV file with test/evaluation data.")
    p_test.add_argument(
        "--metric",
        type=str,
        default="MAE",
        choices=["RMSE", "MAE"],
        help="Evaluation metric to report.",
    )

    # summ
    p_sum = subparsers.add_parser("summ", help="Show basic statistics for a CSV slice.")
    p_sum.add_argument("csv", nargs="?", default="work.csv", help="CSV file to summarize.")
    p_sum.add_argument("start", nargs="?", type=int, default=0, help="Start index of slice.")
    p_sum.add_argument("end", nargs="?", type=int, default=None, help="End index (exclusive) of slice.")

    return parser

def main(argv: Optional[list[str]] = None) -> None:
    parser = _create_parser()
    args = parser.parse_args(argv)
  

    if args.command == "add_data":
        add_data(args.file_path, args.train_csv, args.start, args.end)
    elif args.command == "train":
        train_model(args.model, bool(args.warm_start), args.csv)
    elif args.command == "test":
        test_model(args.model, args.csv, args.metric)
    elif args.command == "summ":
        summarize(args.csv, args.start, args.end)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()

