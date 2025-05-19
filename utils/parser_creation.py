import argparse


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI for taxi-trip duration ML pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # add_data
    p_add = subparsers.add_parser("add_data", help="Append raw data to the training SQLite database.")
    p_add.add_argument("--file_path", type=str, default="train.csv", help="Path to file with new data.")
    p_add.add_argument("--db", type=str, default="work.db", help="Path to the SQLite database.")
    p_add.add_argument("-s", "--start", type=int, default=0, help="Start index of rows to add.")
    p_add.add_argument("-e", "--end", type=int, default=None, help="End index (exclusive) of rows to add.")
    p_add.add_argument("--source_type", type=str, default="csv", help="Format of the input data.")

    # train
    p_train = subparsers.add_parser("train", help="Train a model on database records.")
    p_train.add_argument("--model", type=str, help="Model type (LR, KNN, DT, RF, Lasso, Ridge, ...).")
    p_train.add_argument("-w", "--warm_start",  type=int, choices=[0, 1], default=0, help="Continue training if model supports it.")
    p_train.add_argument("--db", type=str, default="work.db", help="SQLite DB file with training data.")
    p_train.add_argument("-s", "--start", type=int, default=0, help="Start index for training data.")
    p_train.add_argument("-e", "--end", type=int, default=None, help="End index (exclusive) for training data.")

    # test
    p_test = subparsers.add_parser("test", help="Evaluate a trained model.")
    p_test.add_argument("--model", type=str, default="latest", help='Model identifier ("latest" or specific name).')
    p_test.add_argument("--db", type=str, default="test.csv", help="CSV or DB file with evaluation data.")
    p_test.add_argument("--metric", type=str, default="RMSE", choices=["RMSE", "MAE"], help="Evaluation metric.")
    p_test.add_argument("--source_type", type=str, default="csv", help="Input data format.")

    # summ
    p_sum = subparsers.add_parser("summ", help="Show basic statistics from the database.")
    p_sum.add_argument("--db", type=str, default="work.db", help="SQLite DB file to summarize.")
    p_sum.add_argument("-s", "--start", type=int, default=0, help="Start index for summary.")
    p_sum.add_argument("-e", "--end", type=int, default=None, help="End index (exclusive) for summary.")

    return parser
