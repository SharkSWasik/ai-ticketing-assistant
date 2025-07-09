from abc import ABC, abstractmethod
import pandas as pd
import os

class DataLoader(ABC):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

class SingleCSVDataLoader(DataLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"File not found : {self.file_path}")
        return pd.read_csv(self.file_path)