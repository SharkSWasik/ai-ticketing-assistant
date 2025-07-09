import lightgbm as lgb
from typing import Dict
from ..base_model import BaseModel

class LightGBMClassifier(BaseModel):

    def __init__(self, config: Dict):
        
        self.config = config
        self.model: lgb.LGBMClassifier()

    def train(self, X_train, y_train) -> None:
        
        self.model = lgb.LGBMClassifier(**self.config)
        self.model.fit(X_train, y_train)

    def predict(self, X):
       
        if self.model is None:
            raise ValueError("Model has not been trained.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained.")
        return self.model.predict(X)