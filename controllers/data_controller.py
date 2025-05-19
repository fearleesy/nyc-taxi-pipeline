import os
import sys
import time
import json
from typing import Optional

from models.src.db_manager import DBManager
from models.src.data_analyzer import DataAnalyzer
from utils.io_helpers import load_data
from utils.batch_logging import compute_batch_meta
from utils.logger import get_logger

logger = get_logger(__name__)

with open('views/params.json', 'r') as f:
    config = json.load(f)

base_data_type = config["data_parameters"]["base_data_type"]

def add_data(
    file_path: str,
    db_path: str = "work.db",
    start: int = 0,
    end: Optional[int] = None,
    source_type: str = base_data_type
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
    logger.debug(f"add_data started with file_path={file_path}, db_path={db_path}, slice=[{start}:{end}], source_type={source_type}")


    if not os.path.exists(file_path):
        logger.debug(f"File not found: {file_path}")
        sys.exit(f"[add_data] File not found: {file_path}")

    new_df = load_data(file_path, source_type)
    logger.debug(f"Loaded data with {len(new_df)} rows from {file_path}")

    if new_df.empty:
        logger.debug(f"No rows found in file: {file_path}")
        sys.exit(f"[add_data] No rows found in {file_path}.")

    sliced_df = new_df.iloc[start:end]
    logger.debug(f"Sliced data frame size: {len(sliced_df)} rows")

    if sliced_df.empty:
        logger.debug(f"No data in slice: {start}:{end}")
        sys.exit(f"[add_data] No data in slice [{start}:{end}].")

    start_time = time.time()

    logger.debug("Initializing DBManager...")
    db = DBManager(db_path)
    db.insert_df(sliced_df)
    total_length = db.get_length()
    logger.debug("Insert complete.")
    logger.debug(f"Total DB size after insertion: {total_length} rows.")

    elapsed_time_sec = round(time.time() - start_time, 2)
    logger.debug(f"Time taken for DB insertion: {elapsed_time_sec} seconds.")

    logger.debug("Computing metadata for batch log...")
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
    logger.debug(f"Metadata: {meta}")

    logger.info(
        f"[add_data] Inserted {len(sliced_df)} rows into {db_path}. "
        f"Total length: {total_length}, time: {elapsed_time_sec}s"
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
    logger.debug(f"summarize started with db_path={db_path}, slice=[{start}:{end}]")

    logger.debug("Initializing DBManager...")
    db = DBManager(db_path)
    df_slice = db.fetch_range(start, end)
    logger.debug(f"Fetched {len(df_slice)} rows from DB slice")

    if df_slice.empty:
        logger.debug(f"No data in range: {start}:{end}")
        sys.exit(f"[summarize] No data in slice [{start}:{end}].")
    
    logger.debug("Creating DataAnalyzer instance...")
    analyzer = DataAnalyzer(df_slice)

    logger.debug("Calculating basic statistics...")
    try:
        stats = analyzer._calculate_basic_stats()
        logger.debug(f"[summarize] Computed stats: {stats}")
    except AttributeError as e:
        logger.exception("DataAnalyzer missing '_calculate_basic_stats'")
        sys.exit("[summarize] DataAnalyzer does not implement _calculate_basic_stats().")

    logger.info("[summarize] Statistics for the current sample writed to stats/data_quality_report.json")
    # for category, metrics in stats.items():
    #     logger.debug(f"{category}:")
    #     for name, value in metrics.items():
    #         logger.debug(f"\t{name}: {value}")
    #     logger.debug("") 
