import pandas as pd
import numpy as np
import json
from datetime import datetime
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from haversine import haversine, Unit
import folium


class DataAnalyzer:
    def __init__(self, df: pd.DataFrame,
                 cleaning_thresholds= None):
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_thresholds = cleaning_thresholds or {
            'completeness': 0.75,
            'uniqueness': 0.25
        }
        self.metrics = {}
        self.cleaning_report = {}
        self.historical_stats = None
        self.scaler = StandardScaler()
        self.added_columns = []
        self.removed_columns = []
        self.categorical_columns = []
    
    def calculate_completeness(self):
        completeness = {}
        total = len(self.df)
        for col in self.df.columns:
            non_null = self.df[col].count()
            value = non_null / total if total > 0 else 0
            completeness[col] = {
                'non_null': non_null,
                'null': total - non_null,
                'value': round(value, 4)
            }
        self.metrics['completeness'] = completeness
        return completeness
    
    def calculate_uniqueness(self):
        uniqueness = {}
        total = len(self.df)
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            value = unique_count / total if total > 0 else 0
            uniqueness[col] = {
                'unique': unique_count,
                'duplicates': total - unique_count,
                'value': round(value, 4)
            }
        self.metrics['uniqueness'] = uniqueness
        return uniqueness
    
    def calculate_all_metrics(self):
        self.calculate_completeness()
        self.calculate_uniqueness()
        return self.metrics
    
    def clean_data(self):
        df = self.df[self.df.notna().all(axis=1)].copy()

        max_log_trip_duration = df['trip_duration'].quantile(0.995)
        min_log_trip_duration = df['trip_duration'].quantile(0.01)
        df = df[(df['trip_duration'] <= max_log_trip_duration) & (df['trip_duration'] >= min_log_trip_duration)]

        maxim = df['haversine'].quantile(0.995)
        minim = df['haversine'].quantile(0.015)
        df = df[(df['haversine'] <= maxim) & (df['haversine'] >= minim)]

        if not self.metrics:
            self.calculate_all_metrics()
        
        for col, metrics in self.metrics['uniqueness'].items():
            if metrics['value'] < self.cleaning_thresholds['uniqueness']:
                self.categorical_columns.append(col)
        
        self.cleaned_df = df
        self.cleaning_report = {
            'original_shape': self.original_shape,
            'cleaned_shape': df.shape,
        }
        
        return df
    
    def save_quality_report(self, file_path: str):
        
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            raise TypeError(f"Object is not writable in JSON")

        report = {
            'dataset_shape': list(self.original_shape),
            'quality_metrics': self.metrics,
            'cleaning_report': getattr(self, 'cleaning_report', None)
        }
    
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=convert)
    
    def get_quality_summary(self):
        summary = []
        for metric_type, cols in self.metrics.items():
            for col, values in cols.items():
                if col not in self.df.columns:
                    continue

                nan_count = self.df[col].isna().sum()
                summary.append({
                    'column': col,
                    'metric': metric_type,
                    'value': values['value'],
                    'status': 'PASS' if values['value'] >= self.cleaning_thresholds.get(metric_type, 0) else 'FAIL',
                    'NaNs': self.df[col].isna().sum() 
                })
        metric_df = pd.DataFrame(summary)
        metric_df['completeness'] = metric_df['value'][metric_df['metric']== 'completeness']
        metric_df['uniqueness'] = metric_df['value'][metric_df['metric']== 'uniqueness']
        metric_df = metric_df.drop(columns = ['metric','value'])

        metric_df = metric_df.groupby('column', as_index=False).first()
        metric_df['completeness'] = metric_df['completeness'].fillna('Not checked')
        metric_df['uniqueness'] = metric_df['uniqueness'].fillna('Not checked')
        
        return metric_df
    
    def _calculate_basic_stats(self):
        stats = {}
        
        for col in self.numeric_cols:
            stats[col] = {
                'mean': self.data[col].mean(),
                'std': self.data[col].std(),
                'skewness': self.data[col].skew(),
                'kurtosis': self.data[col].kurtosis(),
                'percentiles': self.data[col].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
            }

        for col in self.categorical_cols:
            stats[col] = {
                'n_unique': self.data[col].nunique(),
                'top_value': self.data[col].mode()[0],
                'top_freq': (self.data[col] == self.data[col].mode()[0]).mean(),
                'value_counts': self.data[col].value_counts(normalize=True).head(10).to_dict()
            }
        
        self.historical_stats = stats

    def feature_engineering(self):
        df = self.df

        df['log_trip_duration'] = np.log1p(df['trip_duration'])
        self.added_columns.append('log_trip_duration')
        df = df.drop(columns=['trip_duration'])
        self.removed_columns.append('trip_duration')

        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['day_of_week'] = df['pickup_datetime'].dt.day_of_week
        df['hour'] = df['pickup_datetime'].dt.hour
        df['month'] = df['pickup_datetime'].dt.month
        self.added_columns.append('day_of_week')
        self.added_columns.append('hour')
        self.added_columns.append('month')

        def calculate_haversine(row):
            start_point = (row['pickup_latitude'], row['pickup_longitude'])
            end_point = (row['dropoff_latitude'], row['dropoff_longitude'])
            return haversine(start_point, end_point)

        df['haversine'] = df.apply(calculate_haversine, axis=1)
        df['log_haversine'] = np.log1p(df['haversine'])
        df = df.drop(columns=['haversine'])
        self.added_columns.append('log_haversine')

        def check_jamm(hour):
            if 7 <= hour <= 9 or 16 <= hour <= 19:
                return True
            else:
                return False

        df['is_jamm'] = df['hour'].apply(check_jamm)
        df['is_free'] = ~df['is_jamm']
        self.added_columns.append('is_jamm')
        self.added_columns.append('if_free')

        airport1_coords = (40.6413, -73.7781)
        airport2_coords = (40.6895, -74.1745)
        radius = 2

        df['start_at_airport1'] = df.apply(lambda row: haversine((row['pickup_latitude'], row['pickup_longitude']), airport1_coords, unit=Unit.KILOMETERS) <= radius, axis=1)
        df['end_at_airport1'] = df.apply(lambda row: haversine((row['dropoff_latitude'], row['dropoff_longitude']), airport1_coords, unit=Unit.KILOMETERS) <= radius, axis=1)

        df['start_at_airport2'] = df.apply(lambda row: haversine((row['pickup_latitude'], row['pickup_longitude']), airport2_coords, unit=Unit.KILOMETERS) <= radius, axis=1)
        df['end_at_airport2'] = df.apply(lambda row: haversine((row['dropoff_latitude'], row['dropoff_longitude']), airport2_coords, unit=Unit.KILOMETERS) <= radius, axis=1)

        self.added_columns.append('start_at_airport1')
        self.added_columns.append('start_at_airport2')
        self.added_columns.append('end_at_airport1')
        self.added_columns.append('end_at_airport2')

        df['is_airport'] = (
            df['start_at_airport1'] |
            df['end_at_airport1'] |
            df['start_at_airport2'] |
            df['end_at_airport2']
        )
        self.added_columns.append('is_airport')

        df['vendor_id'] = df['vendor_id'] - 1

        def check_flag(flag):
            if flag == 'N':
                return 0
            else:
                return 1
        df['store_and_fwd_flag'] = df['store_and_fwd_flag'].apply(check_flag)

        transformer = MapGridTransformer(n_rows=7, n_cols=7)
        transformer.fit(df)
        df = transformer.transform(df)

        self.df = df

    def fit_transform(self):
        self.calculate_all_metrics()
        

        with open("DataAnalisis.txt", "w", encoding="utf-8") as file:
            file.write("Качество данных до очистки:")
            file.write(f"\n{self.get_quality_summary()}")
            self.feature_engineering()
        
            cleaned_df = self.clean_data()
            self.save_quality_report("data_quality_report.json")
        
            file.write("\n\nОтчет об очистке:")
            file.write(f"\nИсходный размер: {self.cleaning_report['original_shape']}")
            file.write(f"\nОчищенный размер: {self.cleaning_report['cleaned_shape']}")
            file.write(f"\nУдаленные столбцы: {self.removed_columns}")
            file.write(f"\nДобавленные столбцы: {self.added_columns}")

        return cleaned_df
        

class MapGridTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_rows, n_cols, lat_center=40.7831, lon_center=-73.9712, lat_range=0.15, lon_range=0.15):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.lat_center = lat_center
        self.lon_center = lon_center
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.max_lat = None
        self.min_lat = None
        self.max_lon = None
        self.min_lon = None
        self.step_lat = None
        self.step_lon = None

    def fit(self, X):
        self.min_lat = self.lat_center - self.lat_range / 2
        self.max_lat = self.lat_center + self.lat_range / 2
        self.min_lon = self.lon_center - self.lon_range / 2
        self.max_lon = self.lon_center + self.lon_range / 2

        self.step_lat = (self.max_lat - self.min_lat) / self.n_rows
        self.step_lon = (self.max_lon - self.min_lon) / self.n_cols
        return self

    def _get_cell(self, lat, lon):
        lat_idx = ((lat - self.min_lat) // self.step_lat).astype(int)
        lon_idx = ((lon - self.min_lon) // self.step_lon).astype(int)

        inside_grid = (lat_idx >= 0) & (lat_idx < self.n_rows) & (lon_idx >= 0) & (lon_idx < self.n_cols)
        cell_number = np.where(inside_grid, lat_idx * self.n_cols + lon_idx, -1)
        return cell_number

    def plot_rectangle(self):
        m = folium.Map(location=[(self.min_lat + self.max_lat) / 2, (self.min_lon + self.max_lon) / 2], zoom_start=12)

        rectangle = folium.Rectangle(
            bounds=[(self.min_lat, self.min_lon), (self.max_lat, self.max_lon)],
            color="blue",
            fill=True,
            fill_opacity=0.2
        )
        rectangle.add_to(m)

        return m

    def transform(self, X):
        pickup_cells = self._get_cell(X['pickup_latitude'].values, X['pickup_longitude'].values)
        dropoff_cells = self._get_cell(X['dropoff_latitude'].values, X['dropoff_longitude'].values)
        return X.assign(pickup_cell=pickup_cells, dropoff_cell=dropoff_cells)


if __name__ == "__main__":
    conn = sqlite3.connect('taxi.db')
    query = f"SELECT * FROM raw_trips"
    df = pd.read_sql(query, conn)
    conn.close()
    
    preproccess = DataAnalyzer(df)
    preproccess.fit_transform()
