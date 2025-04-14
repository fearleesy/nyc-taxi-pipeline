import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class TaxiModel:
    def __init__(self, model_name):
        self.pipeline = self._build_pipeline(model_name)
    
    def _build_pipeline(self, model_name):
        numeric_features = ['passenger_count', 'pickup_longitude', 
                          'pickup_latitude', 'dropoff_longitude',
                          'dropoff_latitude']
        categorical_features = ['vendor_id']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])
        
        models = {
            "LR" : LinearRegression(),
            "KNN" : KNeighborsRegressor(),
            "DT" : DecisionTreeRegressor(),
            "RF" : RandomForestRegressor(),
            "Lasso" : Lasso(),
            "Ridge" : Ridge()
        }
        if model_name in models:
            return Pipeline([
                ('preprocessor', preprocessor),
                ('model', models[model_name])
            ])
        # print("Неправильное имя модели")
        raise Exception("Неправильное имя модели")
        # return None
    
    # def train(self, X: pd.DataFrame, y: pd.Series, is_warm_start: bool):

    #     self.pipeline.fit(X, y)
    
    def predict(self, X: pd.DataFrame, y: pd.Series, metric: str) -> pd.Series:
        y_pred = self.pipeline.predict(X)
        if metric.upper() == 'MAE':
            return mean_absolute_error(y, y_pred)
        elif metric.upper() == 'RMSE':
            return np.sqrt(mean_squared_error(y, y_pred))
        else:
            raise ValueError("Метрика должна быть 'MAE' или 'RMSE'")
    
    def save(self, path: str):
        joblib.dump(self.pipeline, path)

    
    def train(self, model_name: str, X_train : pd.DataFrame, y_train : pd.Series, is_warm_start: bool):
        if is_warm_start and model_name == "RF" and os.path.exists('./models/RF_model.pkl'):
            try:
                old_model = joblib.load("./models/RF_model.pkl")
                old_n_estimators = old_model.n_estimators
                print(f"Текущая модель загружена. n_estimators = {old_n_estimators}")
            except FileNotFoundError:
                raise Exception("Файл модели не найден!")
            print("Дообучение модели...")
        else:
            if is_warm_start:
                # print("Дообучение невозможно", is_warm_start)
                raise Exception("Дообучение невозможно")

        
        print("\nПодбор гиперпараметров...")

        param_grids = {
            "Lasso" : {
                'model__alpha': np.logspace(-4, 4, 25),          
                'model__max_iter': range(1000, 10000, 1000)       
            },
            "Ridge" : {
                'alpha': np.logspace(-4, 4, 5),
                'solver': ['auto', 'svd', 'cholesky', 'lsqr'], 
            },
            'KNN': {
                'model__n_neighbors': [3, 5, 7, 9],
                'model__weights': ['uniform', 'distance'],
                'model__p': [1, 2]
            },
            'DT': {
                'model__max_depth': [5, 7, 10],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            },
            'RF': {
                'model__n_estimators': [old_n_estimators, old_n_estimators + 10, old_n_estimators + 20] if is_warm_start else [10, 20, 40], 
                'model__max_depth': [None, 2, 4]
            }
        }
        if model_name == "LR":
            self.pipeline.fit(X_train, y_train)
        else:
            grid_search = GridSearchCV(estimator=self.pipeline, param_grid=param_grids[model_name], cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
            if is_warm_start:
                best_n_estimators = grid_search.best_params_['n_estimators']
                best_model = grid_search.best_estimator_
                if best_n_estimators > old_n_estimators:
                    print(f"\nДообучение модели: добавляем {best_n_estimators - old_n_estimators} деревьев")
                    best_model.n_estimators = best_n_estimators
                    best_model.fit(X_train, y_train)
                else:
                    print("\nЛучшая модель уже оптимальна (n_estimators не увеличилось)")
            else:
                grid_search.fit(X_train, y_train)

            print(f"Модель: {model_name}")
            print(f"Лучшие параметры: {grid_search.best_params_}")
            print(f"Лучший MAE: {(-grid_search.best_score_):.2f}")

            self.pipeline = best_model if is_warm_start else grid_search

    
    @classmethod
    def load(cls, path: str) -> 'TaxiModel':
        model = cls("LR")
        model.pipeline = joblib.load(path)
        return model