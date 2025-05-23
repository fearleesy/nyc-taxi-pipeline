import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from haversine import haversine, Unit
import folium
import os
import datetime
from utils.logger import get_logger

logger = get_logger(__name__)
with open('views/params.json', 'r') as f:
    config = json.load(f)

data_quality_report_path = config["paths"]["data_quality_report_path"]
imgs_before_cleaning_path = config["paths"]["imgs_before_cleaning_path"]
imgs_after_cleaning_path = config["paths"]["imgs_after_cleaning_path"]


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
        self.scaler = StandardScaler()
        self.numeric_cols = []
        self.categorical_cols = []
        self._detect_column_types()
        self.historical_stats = None

    def _detect_column_types(self):
        self.numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        self.categorical_cols.append('vendor_id')
        self.categorical_cols.append('passenger_count')

        self.numeric_cols.remove('vendor_id')
        self.numeric_cols.remove('passenger_count')

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

        max_trip_duration = df['trip_duration'].quantile(0.995)
        min_trip_duration = df['trip_duration'].quantile(0.01)
        df = df[(df['trip_duration'] <= max_trip_duration) & (df['trip_duration'] >= min_trip_duration)]
        df = df[df["passenger_count"] <= 6]

        df = df[df["trip_duration"] <= 12 * 3600]
        df = df[df["trip_duration"] >= 60]

        if not self.metrics:
            self.calculate_all_metrics()
        
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
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report["timestamp"] = timestamp

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        existing_data.append(report)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False, default=convert)
    
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
                    'NaNs': self.df[col].isna().sum() 
                })
        metric_df = pd.DataFrame(summary)
        metric_df['completeness'] = metric_df['value'][metric_df['metric']== 'completeness']
        metric_df['uniqueness'] = metric_df['value'][metric_df['metric']== 'uniqueness']
        metric_df = metric_df.drop(columns = ['metric','value'])

        metric_df = metric_df.groupby('column', as_index=False).first()
        metric_df['completeness'] = metric_df['completeness'].fillna('Not checked')
        metric_df['uniqueness'] = metric_df['uniqueness'].fillna('Not checked')
        
        return metric_df, nan_count
    
    def _calculate_basic_stats(self):
        stats = {}

        for col in self.numeric_cols:
            stats[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'percentiles': self.df[col].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
            }

        for col in self.categorical_cols:
            stats[col] = {
                'n_unique': self.df[col].nunique(),
                'top_value': self.df[col].mode()[0],
                'top_freq': (self.df[col] == self.df[col].mode()[0]).mean(),
                'value_counts': self.df[col].value_counts(normalize=True).head(10).to_dict()
            }
        
        self.historical_stats = stats
        return stats

    
    def make_plot(self, df, imgs_folder_path):
        def calculate_haversine(row):
            start_point = (row['pickup_latitude'], row['pickup_longitude'])
            end_point = (row['dropoff_latitude'], row['dropoff_longitude'])
            return haversine(start_point, end_point)

        df['haversine'] = df.apply(calculate_haversine, axis=1)
        df['log_haversine'] = np.log1p(df['haversine'])
        df = df.drop(columns=['haversine'])
        
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['day_of_week'] = df['pickup_datetime'].dt.day_of_week
        df['month'] = df['pickup_datetime'].dt.month
        df['hour'] = df['pickup_datetime'].dt.hour
        data = df.groupby(['hour', 'month', 'day_of_week']).size().reset_index(name='trip_count')

        fig, axes = plt.subplots(3, 2, figsize=(18, 18))
        fig.suptitle('Анализ данных поездок', fontsize=16)

        '''sns.lineplot(data=data, x='hour', y='trip_count', hue="month", 
                    palette='viridis', errorbar=None, ax=axes[0, 0])'''
        if 'month' in df.columns:
            sns.lineplot(data=data, x='hour', y='trip_count', hue="month",
                        palette='viridis', errorbar=None, ax=axes[0, 0])
        else:
            sns.lineplot(data=data, x='hour', y='trip_count',
                        errorbar=None, ax=axes[0, 0])
        axes[0, 0].set_title('Количество поездок по часам в зависимости от месяца')
        axes[0, 0].set_xlabel('Время')
        axes[0, 0].set_ylabel('Количество поездок')

        sns.lineplot(x=df['day_of_week'], y=df['trip_duration'], 
                    errorbar=None, ax=axes[0, 1])
        axes[0, 1].set_title('Время поездки в зависимости от дня недели')
        axes[0, 1].set_xlabel('День недели')
        axes[0, 1].set_ylabel('Время поездки (лог)')

        sns.histplot(df['trip_duration'], bins=50, kde=True, 
                    color='blue', ax=axes[1, 0])
        axes[1, 0].set_title('Распределение времени поездки')
        axes[1, 0].set_xlabel('Время поездки')
        axes[1, 0].set_ylabel('Frequency')

        sns.histplot(df['log_haversine'], bins=50, kde=True, 
                    color='blue', ax=axes[1, 1])
        axes[1, 1].set_title('Распределение расстояния поездки')
        axes[1, 1].set_xlabel('Расстояние поездки')
        axes[1, 1].set_ylabel('Frequency')

        sns.boxplot(data=df, x="vendor_id", y="passenger_count", ax=axes[2, 0])
        axes[2, 0].set_title('passenger_count vs. vendor_id')

        sns.boxplot(data=df, x="passenger_count", y="trip_duration", ax=axes[2, 1])
        axes[2, 1].set_title('passenger_count vs. trip_duration')

        plt.tight_layout()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.png"
        image_path = os.path.join(imgs_folder_path, filename)
        plt.savefig(image_path, dpi=300, bbox_inches='tight')

        df = df.drop(columns=['log_haversine', 'day_of_week', 'month', 'hour'], )

        return image_path


    def fit_transform(self):
        self.calculate_all_metrics()
        logger.debug("All metrics calculated")
        image_path = self.make_plot(self.df, imgs_before_cleaning_path)
        logger.debug(f"Saved plot before cleaning to: {image_path}")
        
        metric_df, nan_count = self.get_quality_summary()
        logger.debug("Data quality before cleaning:")
        logger.debug(f"\n{metric_df}")
        logger.debug(f"Number of NaN values: {nan_count}")
    
        cleaned_df = self.clean_data()
        logger.debug("Data cleaned")

        self.save_quality_report(data_quality_report_path)
        logger.debug(f"Saved data quality report to: {data_quality_report_path}")

        image_path = self.make_plot(cleaned_df, imgs_after_cleaning_path)
        logger.debug(f"Saved plot after cleaning to: {image_path}")
    
        logger.debug("Cleaning summary:")
        logger.debug(f"Original shape: {self.cleaning_report['original_shape']}")
        logger.debug(f"Cleaned shape: {self.cleaning_report['cleaned_shape']}")

        self.cleaned_df = cleaned_df

        return cleaned_df
