import json
import numpy as np
import pandas as pd
from haversine import haversine, Unit
import folium
from sklearn.base import BaseEstimator, TransformerMixin
from utils.logger import get_logger

logger = get_logger(__name__)

with open('utils/params.json', 'r') as f:
    config = json.load(f)

jams_hours = config['trip_constraints']['jams_hours']

airport1_coords = config['geo_parameters']['airport1_coords']
airport2_coords = config['geo_parameters']['airport2_coords']


top_quantile1 = config["data_parameters"]["top_quantile1"]
low_quantile1 = config["data_parameters"]["low_quantile1"]

lat = config["geo_parameters"]["city_center"]["lat"]
lon = config["geo_parameters"]["city_center"]["lon"]
lat_range = config["geo_parameters"]["city_center"]["lat_range"]
lon_range = config["geo_parameters"]["city_center"]["lon_range"]
radius = config['geo_parameters']['radius']


class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.added_columns = []
        self.removed_columns = []
        self.original_shape = self.df.shape
        self.df_shape = self.df.shape

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
            if jams_hours[0] <= hour <= jams_hours[1] or jams_hours[2] <= hour <= jams_hours[3]:                return True
            else:
                return False

        df['is_jamm'] = df['hour'].apply(check_jamm)
        df['is_free'] = ~df['is_jamm']
        self.added_columns.append('is_jamm')
        self.added_columns.append('if_free')

        # airport1_coords = (40.6413, -73.7781)
        # airport2_coords = (40.6895, -74.1745)
        # radius = 2

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
        
        self.categorical_cols = ['day_of_week', 'month', 'hour', 'is_jamm', 'is_airport', 'vendor_id', 'store_and_fwd_flag', 'passenger_count']
        self.numeric_cols = ['log_haversine', 'pickup_cell', 'dropoff_cell']

        self.df = df

    def clean_data(self):
        df = self.df

        maxim = df['log_haversine'].quantile(top_quantile1)
        minim = df['log_haversine'].quantile(low_quantile1)
        df = df[(df['log_haversine'] <= maxim) & (df['log_haversine'] >= minim)]

        self.df_shape = df.shape

        
        self.df = df

    def fit_transform(self):
        self.feature_engineering()
        logger.debug("Feature engineering completed")
        self.clean_data()
        logger.debug("Data cleaning completed")
        logger.debug(f"Size before cleaning: {self.original_shape}")
        logger.debug(f"Size after cleaning: {self.df_shape}")
        logger.debug(f"Added columns: {self.added_columns}")
        logger.debug(f"Removed columns: {self.removed_columns}")
        return self.df

        
class MapGridTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_rows, n_cols, lat_center=lat, lon_center=lon, lat_range=lat_range, lon_range=lon_range):
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

