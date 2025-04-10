from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.linear_model import Lasso, LinearRegression, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np

class TaxiModel:
    def __init__(self):
        self.pipeline = self._build_pipeline()
    
    def _build_pipeline(self):
        numeric_features = ['passenger_count', 'pickup_longitude', 
                          'pickup_latitude', 'dropoff_longitude',
                          'dropoff_latitude']
        categorical_features = ['vendor_id']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])
        
        return Pipeline([
            ('preprocessor', preprocessor),
            ('model', Lasso())
        ])
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.pipeline.predict(X)
    
    def save(self, path: str):
        joblib.dump(self.pipeline, path)

    
    def grid(self, X_train, y_train):
        print("\nПодбор гиперпараметров:")
        param_grids = {
            "Lasso" : {
                'model__alpha': np.logspace(-4, 4, 25),          
                'model__max_iter': range(1000, 10000, 1000)       
            },
            "Ridge" : {
                'alpha': np.logspace(-4, 4, 5),
                'solver': ['auto', 'svd', 'cholesky', 'lsqr'], 
            },
            'KNeighborsRegressor': {
                'model__n_neighbors': [3, 5, 7, 9],
                'model__weights': ['uniform', 'distance'],
                'model__p': [1, 2]
            },
            'DecisionTreeRegressor': {
                'model__max_depth': [None, 3, 5, 7, 10],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            },
            'RandomForestRegressor': {
                'model__n_estimators': [50, 100, 200], 
                'model__max_depth': [None, 5, 10],
                'model__min_samples_split': [2, 5],
                'model__bootstrap': [True, False] 
            }
        }
        grid_search = GridSearchCV(estimator=self.pipeline, param_grid=param_grids["Lasso"], cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print(f"Модель: {self.pipeline}")
        print(f"Лучшие параметры: {grid_search.best_params_}")
        print(f"Лучший MAE: {(-grid_search.best_score_):.2f}")

        return grid_search

    
    @classmethod
    def load(cls, path: str) -> 'TaxiModel':
        model = cls()
        model.pipeline = joblib.load(path)
        return model