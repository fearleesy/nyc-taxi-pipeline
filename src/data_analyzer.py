import pandas as pd
import numpy as np
import json
from datetime import datetime
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from haversine import haversine, Unit
import folium
import os

if not os.path.exists("stats"):
    os.mkdir("stats")
data_quiality_path = "stats/DataAnalisis.txt"
data_quality_report_path = "stats/data_quality_report.json"

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
        #self.added_columns = []
        #self.removed_columns = []
        self.numeric_cols = []
        self.categorical_cols = []

        self._detect_column_types()


    def _detect_column_types(self):
            self.numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
    
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

        if not self.metrics:
            self.calculate_all_metrics()
        
        '''for col, metrics in self.metrics['uniqueness'].items():
            if metrics['value'] < self.cleaning_thresholds['uniqueness']:
                self.categorical_cols.append(col)
            else:
                self.numeric_cols.append(col)'''
        
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
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'skewness': self.df[col].skew(),
                'kurtosis': self.df[col].kurtosis(),
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

    def fit_transform(self):
        self.calculate_all_metrics()
        

        with open(data_quiality_path, "w", encoding="utf-8") as file:
            file.write("Качество данных до очистки:")
            file.write(f"\n{self.get_quality_summary()}")
            #self.feature_engineering()
        
            cleaned_df = self.clean_data()
            self.save_quality_report(data_quality_report_path)
        
            file.write("\n\nОтчет об очистке:")
            file.write(f"\nИсходный размер: {self.cleaning_report['original_shape']}")
            file.write(f"\nОчищенный размер: {self.cleaning_report['cleaned_shape']}")

        return cleaned_df
