import os
import sys
import time
from typing import Optional

from src.db_manager import DBManager
from src.data_analyzer import DataAnalyzer
from utils.io_helpers import load_data
from utils.batch_logging import compute_batch_meta, append_log_entry


def add_data(
    file_path: str,
    db_path: str = "work.db",
    start: int = 0,
    end: Optional[int] = None,
    source_type: str = "csv"
) -> None:
    """
    Load data from file and append a slice of it into the SQLite database.

    Parameters
    ----------
    file_path : str
        Path to the input data file (CSV, JSON, etc.).
    db_path : str
        SQLite database file to insert data into. Defaults to "work.db".
    start : int
        Starting index of the slice to be inserted.
    end : int or None
        Ending index (exclusive) of the slice to be inserted. If None, all data from `start` onward is used.
    source_type : str
        Format of the data source (e.g., "csv", "json").
    """
    if not os.path.exists(file_path):
        sys.exit(f"[add_data] File not found: {file_path}")

    new_df = load_data(file_path, source_type)
    if new_df.empty:
        sys.exit(f"[add_data] No rows found in {file_path}.")

    sliced_df = new_df.iloc[start:end]
    if sliced_df.empty:
        sys.exit(f"[add_data] No data in slice [{start}:{end}].")

    start_time = time.time()

    db = DBManager(db_path)
    db.insert_df(sliced_df)
    total_length = db.get_length()

    elapsed_time_sec = round(time.time() - start_time, 2)

    meta = compute_batch_meta(
        file_path=file_path,
        batch_size=len(sliced_df),
        db_total_size=total_length,
        csv_total_size=len(new_df),
        start=start,
        end=end,
        elapsed_time_sec=elapsed_time_sec,
        num_duplicates=sliced_df.duplicated().sum()
    )
    append_log_entry(meta)

    print(
        f"[add_data] Inserted {len(sliced_df)} rows into {db_path}. "
        f"Total length: {total_length}"
    )


def summarize(
    db_path: str = "work.db",
    start: int = 0,
    end: Optional[int] = None
) -> None:
    """
    Print basic statistics for a data slice from the database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    start : int
        Starting index of the slice to summarize.
    end : int or None
        Ending index (exclusive) of the slice. If None, summarize from `start` to the end of the table.
    """
    db = DBManager(db_path)
    df_slice = db.fetch_range(start, end)
    if df_slice.empty:
        sys.exit(f"[summarize] No data in slice [{start}:{end}].")

    analyzer = DataAnalyzer(df_slice)

    try:
        stats = analyzer._calculate_basic_stats()
    except AttributeError:
        sys.exit("[summarize] DataAnalyzer does not implement _calculate_basic_stats().")

    print("Statistics for the current sample:")
    for category, metrics in stats.items():
        print(f"{category}:")
        for name, value in metrics.items():
            print(f"\t{name}: {value}")
        print()
