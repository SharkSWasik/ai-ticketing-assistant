from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, embedding_model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.label_encoders : Dict[str, LabelEncoder] = {}

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate dense vector embeddings for a list of input texts.

        Args:
            texts (List[str]): List of raw text strings to encode.

        Returns:
            np.ndarray: 2D array of shape (n_texts, embedding_dim) containing embeddings.
        """
        return self.embedding_model.encode(texts)
    
    def encode_labels_pred(self, df: pd.DataFrame, label_columns: List[str], ref_cols: List[str], pred_cols: List[str]) -> pd.DataFrame:
        """
        Encode reference and predicted label columns using LabelEncoder.

        For each logical label, fit a LabelEncoder on the
        reference column values and transform both reference and predicted
        columns to numeric encodings.

        Args:
            df (pd.DataFrame): DataFrame containing original and prediction columns.
            label_columns (List[str]): List of label names to encode.
            ref_cols (List[str]): List of DataFrame column names with ground truth.
            pred_cols (List[str]): List of DataFrame column names with predicted.

        Returns:
            pd.DataFrame: Copy of the input DataFrame with new encoded columns added.
        """
        df_encoded = df.copy()

        for col, ref_col, pred_col in zip(label_columns, ref_cols, pred_cols):

            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()

            le = self.label_encoders[col]
            
            df_encoded[f"{ref_col}_encoded"] = le.fit_transform(df_encoded[ref_col])
            
            df_encoded[f"{pred_col}_encoded"] = le.transform(df_encoded[pred_col])

        return df_encoded
    
    def encode_labels(self, df: pd.DataFrame, label_columns: List[str])-> pd.DataFrame:

        df_encoded = df.copy()

        for col in label_columns:

            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()

            le = self.label_encoders[col]
            
            df_encoded[f"{col}_encoded"] = le.fit_transform(df_encoded[col])

        return df_encoded
    
    def clean_df_inplace(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in the DataFrame with empty strings in place.

        Args:
            df (pd.DataFrame): Input DataFrame that may contain NaNs.

        Returns:
            pd.DataFrame: DataFrame with NaNs replaced by empty strings.
        """
        df = df.fillna('')
        return df