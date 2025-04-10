import pandas as pd

class DataCollector:
    def __init__(self, csv_path: str, initial_size: int):
        self.csv_path = csv_path
        self.current_index = 0
        self.batch_data = pd.read_csv(self.csv_path, nrows=initial_size)
        self.current_index = initial_size
        self.total_rows = sum(1 for _ in open(self.csv_path)) - 1

    def get_batch(self, batch_size: int) -> pd.DataFrame:
        if self.current_index >= self.total_rows:
            print("Весь датасет уже загружен.")
            return self.batch_data

        read_rows = min(batch_size, self.total_rows - self.current_index)

        new_data = pd.read_csv(
            self.csv_path,
            skiprows=range(1, self.current_index + 1),
            nrows=read_rows,
            header=0
        )
        self.batch_data = pd.concat([self.batch_data, new_data], ignore_index=True)
        self.current_index += read_rows
        return self.batch_data
