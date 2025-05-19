from datetime import datetime
from typing import Optional

def compute_batch_meta(
    file_path: str,
    batch_size: int,
    db_total_size: int,
    csv_total_size: int,
    start: int,
    end: Optional[int],
    elapsed_time_sec: float,
    num_duplicates: int
) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "file_path": file_path,
        "batch_size": batch_size,
        "db_total_size": db_total_size,
        "batch_ratio": round(batch_size / db_total_size * 100, 2) if db_total_size else 0,
        "csv_total_size": csv_total_size,
        "slice_start": start,
        "slice_end": end,
        "elapsed_time_sec": elapsed_time_sec,
        "num_duplicates": int(num_duplicates),
    }
