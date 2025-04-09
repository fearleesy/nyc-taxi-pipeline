import sqlite3
import logging

import pandas as pd

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_db()
        self.dtype_mapping = {
            'int64': 'INTEGER',
            'float64': 'REAL',
            'object': 'TEXT',
            'bool': 'INTEGER',
            'datetime64[ns]': 'TEXT'
        }

    def _map_dtypes(self, df):
        return {col: self.dtype_mapping.get(str(dtype), 'TEXT') 
                for col, dtype in df.dtypes.items()}

    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS raw_trips")
            conn.commit()

    def save_batch(self, batch):
        with sqlite3.connect(self.db_path) as conn:
            columns = self._map_dtypes(batch)
            create_table = f"""
            CREATE TABLE IF NOT EXISTS raw_trips (
                {', '.join(f'{k} {v}' for k, v in columns.items())}
            )"""
            conn.execute(create_table)
            
            placeholders = ', '.join(['?'] * len(batch.columns))
            sql = f"INSERT INTO raw_trips VALUES ({placeholders})"
            data = [tuple(x) for x in batch.to_numpy()]
            conn.executemany(sql, data)
            conn.commit()
        # logger.info(f"Успешно сохранено {len(batch)} записей")
    def load_batch(self, iter, batch):
        conn = sqlite3.connect('data/taxi.db')
        query = f"SELECT * FROM raw_trips limit {batch},{iter*500}"
        df = pd.read_sql(query, conn)
        conn.close()
        return df