import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

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
            ('regressor', SGDRegressor())
        ])
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.pipeline.predict(X)
    
    def save(self, path: str):
        joblib.dump(self.pipeline, path)
    
    @classmethod
    def load(cls, path: str) -> 'TaxiModel':
        model = cls()
        model.pipeline = joblib.load(path)
        return model