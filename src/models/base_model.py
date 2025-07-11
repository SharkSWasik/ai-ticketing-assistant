from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train: Any, y_train: Any) -> None:
        """
        """    
    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        """
    @abstractmethod
    def save(self, X: Any) -> Any:
        """
        """
    @abstractmethod
    def load(self, X: Any) -> Any:
        """
        """