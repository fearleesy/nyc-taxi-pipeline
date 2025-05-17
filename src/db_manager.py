import sqlite3
import pandas as pd
from typing import Optional


class DBManager:
    def __init__(self, db_path: str, table_name: str = "work"):
        self.db_path = db_path
        self.table_name = table_name

    def _connect(self):
        return sqlite3.connect(self.db_path)
    
    def get_length(self) -> int:
        """Return the number of rows in the main data table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM work")
            result = cursor.fetchone()
        return result[0] if result else 0

    def _ensure_table_exists(self, df_sample: pd.DataFrame) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.table_name}'")
            exists = cursor.fetchone()

            if not exists:
                column_defs = ", ".join(
                    f'"{col}" {self._map_dtype(dtype)}'
                    for col, dtype in df_sample.dtypes.items()
                )
                sql = f'CREATE TABLE {self.table_name} ({column_defs});'
                cursor.execute(sql)
                conn.commit()

    def _map_dtype(self, dtype) -> str:
        if pd.api.types.is_integer_dtype(dtype):
            return "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            return "REAL"
        elif pd.api.types.is_bool_dtype(dtype):
            return "BOOLEAN"
        else:
            return "TEXT"

    def insert_df(self, df: pd.DataFrame) -> None:
        self._ensure_table_exists(df)
        with self._connect() as conn:
            df.to_sql(self.table_name, conn, if_exists="append", index=False)

    def fetch_range(self, start: int = 0, end: Optional[int] = None) -> pd.DataFrame:
        with self._connect() as conn:
            query = f"SELECT * FROM {self.table_name}"
            if end is not None:
                query += f" LIMIT {end - start} OFFSET {start}"
            elif start > 0:
                query += f" LIMIT -1 OFFSET {start}"
            return pd.read_sql(query, conn)
