import lightgbm as lgb
from typing import Dict
from ..base_model import BaseModel
from joblib import dump, load

class LightGBMClassifier(BaseModel):

    def __init__(self, config: Dict):
        
        self.config = config
        self.model = lgb.LGBMClassifier(**config)

    def train(self, X_train, y_train) -> None:
        
        self.model = lgb.LGBMClassifier(**self.config)
        self.model.fit(X_train, y_train)

    def predict(self, X):
       
        if self.model is None:
            raise ValueError("Model has not been trained.")
        return self.model.predict(X)

    def save(self, path:str):

        with open(path, 'wb') as f:
            dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'LightGBMClassifier':
        
        with open(path, 'rb') as f:
            model = load(f)

        return model

    def predict_proba(self, X):
        
        if self.model is None:
            raise ValueError("Model has not been trained.")
        return self.model.predict(X)