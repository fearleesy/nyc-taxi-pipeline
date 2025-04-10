import pandas as pd

class DataAnalyzer:
    def __init__(self):
        self.quality_thresholds = {
            'missing_rate': 0.5,
            'min_trip_duration': 60
        }

    def clean_data(self, batch):
        clean_batch = batch[
            (batch['trip_duration'] > self.quality_thresholds['min_trip_duration']) &
            (batch['pickup_latitude'].between(40, 42)) &
            (batch['dropoff_latitude'].between(40, 42))
        ]
        return clean_batch.dropna()

    def calculate_statistics(self, df: pd.DataFrame) -> dict:
        """Расчет статистик для очищенных данных"""
        stats = {
            'mean_duration': df['trip_duration'].mean(),
            'median_duration': df['trip_duration'].median(),
            'std_duration': df['trip_duration'].std(),
            'min_duration': df['trip_duration'].min(),
            'max_duration': df['trip_duration'].max(),
            'total_records': len(df),
            'vendor_distribution': df['vendor_id'].value_counts().to_dict(),
            'passenger_stats': {
                'mean': df['passenger_count'].mean(),
                'max': df['passenger_count'].max()
            }
        }
        return stats