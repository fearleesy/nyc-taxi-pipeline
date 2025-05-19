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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib as plt


class TaxiModel:
    def __init__(self, model_name):
        numeric_preprocessors = {
            'impute_median_scaler': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]),
            'impute_mean_robust': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', RobustScaler())
            ]),
            'impute_zero_minmax': Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', MinMaxScaler())
            ])
        }

        # Варианты для категориальных признаков
        categorical_preprocessors = {
            'onehot': OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            'onehot_drop': OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
        }

        numeric_features = ['pickup_longitude', 
                          'pickup_latitude', 'dropoff_longitude',
                          'dropoff_latitude']
        categorical_features = ['passenger_count', 'vendor_id']
        
        # preprocessor = ColumnTransformer(
        #     transformers=[
        #         ('num', StandardScaler(), numeric_features),
        #         ('cat', OneHotEncoder(), categorical_features)
        #     ])
        
        self.preprocessor_options = {
            'option1': ColumnTransformer([
                ('num', numeric_preprocessors['impute_median_scaler'], numeric_features),
                ('cat', categorical_preprocessors['onehot'], categorical_features)
            ]),
            'option2': ColumnTransformer([
                ('num', numeric_preprocessors['impute_mean_robust'], numeric_features),
                ('cat', categorical_preprocessors['onehot_drop'], categorical_features)
            ]),
            'option3': ColumnTransformer([
                ('num', numeric_preprocessors['impute_zero_minmax'], numeric_features),
                ('cat', categorical_preprocessors['onehot'], categorical_features)
            ])
        }

        self.models = {
            "LR" : LinearRegression(),
            "KNN" : KNeighborsRegressor(),
            "DT" : DecisionTreeRegressor(),
            "RF" : RandomForestRegressor(),
            "Lasso" : Lasso(),
            "Ridge" : Ridge()
        }
        if model_name not in self.models:
            # print("Неправильное имя модели")
            raise Exception("Неправильное имя модели")
        # return None
    
    # def train(self, X: pd.DataFrame, y: pd.Series, is_warm_start: bool):

    #     self.pipeline.fit(X, y)
    def plot_coefficients_log_reg(model, feature_names):
        coefs = model.named_steps['model'].coef_[0]
        features = feature_names
        plt.figure(figsize=(10, 6))
        plt.barh(features, coefs)
        plt.title("Коэффициенты логистической регрессии")
        plt.xlabel("Значение коэффициента")
        plt.ylabel("Признак")
        plt.grid(True, axis='x')
        plt.show()

    
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
                'model__alpha': np.logspace(-4, 4, 5),
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
            # for prep_name, preprocessor in self.preprocessor_options.items():
            self.pipeline = Pipeline([
                ('preprocessor', self.preprocessor_options['option1']),
                ('model', self.models[model_name])
            ])
            self.pipeline.fit(X_train, y_train)
                # results.append({
                #     'Model': model_name,
                #     'Preprocessor': prep_name,
                #     'Best Score': grid_search.best_score_,
                #     'Best Params': grid_search.best_params_
                # })
            # cat_features = self.pipeline.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(['Pclass', 'Sex'])
            # feature_names = np.concatenate([['Age', 'SibSp', 'Parch', 'Fare'], cat_features])
            # plot_coefficients(lr, feature_names)
        else:
            results = []
            for prep_name, preprocessor in self.preprocessor_options.items():
                self.pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', self.models[model_name])
            ])
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
                results.append({
                    'Model': model_name,
                    'Preprocessor': prep_name,
                    'Best Score': grid_search.best_score_,
                    'Best Params': grid_search.best_params_
                })

            results_df = pd.DataFrame(results).sort_values('Best Score', ascending=False)
            print(results_df.head())

            self.pipeline = best_model if is_warm_start else grid_search

    
    @classmethod
    def load(cls, path: str) -> 'TaxiModel':
        model = cls("LR")
        model.pipeline = joblib.load(path)
        return model