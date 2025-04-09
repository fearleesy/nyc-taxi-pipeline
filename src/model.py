import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np


def preprocess(df):
    """Переписать под настоящие значения!!!!!!"""
    X = df.drop(columns=['fare_amount'])
    y = df['fare_amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = ['payment_type', 'vendor_id']
    numeric_features = ['passenger_count', 'trip_distance', 'pickup_longitude', 'pickup_latitude']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
        ])
    return X_train, X_test, y_train, y_test, preprocessor


def create_model_lin_reg():
    return LinearRegression()


def create_model_knn(mode=False, n_neighbors=5, weights='distance'):
    return KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights) if mode else KNeighborsRegressor()


def create_model_dec_reg(mode=False, max_depth=5, min_samples_leaf=10, random_state=42):
    return DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state) if mode else DecisionTreeRegressor()


def create_model_rand_forest(mode=False, n_estimators=100, max_depth=10, random_state=42):
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state) if mode else RandomForestRegressor()


def train_models(models, preprocessor, X_train, y_train, X_test, y_test):
    print("Оценка базовых моделей:")
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"{name}: RMSE = {rmse:.2f}, R2 = {r2:.2f}")
        joblib.dump(pipeline, f'{name.lower().replace(" ", "_")}_pipeline.pkl')


def grid_model(models, preprocessor, X_train, y_train):
    for model_name, model in models.items():
        print("\nПодбор гиперпараметров:")
        param_grids = {
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
        rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                                    ('model', model)])
        grid_search = GridSearchCV(rf_pipeline, param_grids[model_name], cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        print(f"Модель: {model_name}")
        print(f"Лучшие параметры: {grid_search.best_params_}")
        print(f"Лучший RMSE: {np.sqrt(-grid_search.best_score_):.2f}")

def continue_training(model_path, X_new, y_new):
    pipeline = joblib.load(model_path)
    pipeline.named_steps['model'].fit(
        pipeline.named_steps['preprocessor'].transform(X_new),
        y_new
    )
    return pipeline