import pandas as pd

class MetadataCalculator:
    @staticmethod
    def calculate(batch):
        return {
            "batch_size": len(batch),
            "avg_trip_duration": batch["trip_duration"].mean(),
            "min_trip_duration": batch["trip_duration"].min(),
            "max_trip_duration": batch["trip_duration"].max(),
            "unique_vendors": batch["vendor_id"].nunique(),
            "total_passengers": batch["passenger_count"].sum()
        }
    
# Зачем staticmethod