import numpy as np
import pandas as pd

class FeatureEngineer:
    @staticmethod
    def add_time_features(df):
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['hour'] = df['pickup_datetime'].dt.hour
        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
        return df