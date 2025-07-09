import pandas as pd
import numpy as np

from src.data.data_loader import SingleCSVDataLoader
from src.data.data_processor import DataProcessor
from src.visualization.plotter import DataPlotter
from src.models.baseline.top_k_recommender import TopKRecommender

from sklearn.model_selection import train_test_split


#Load tickets
loader = SingleCSVDataLoader(file_path="dataset/data/dataset-tickets-multi-lang3-4k.csv")
df = loader.load_data()

#Clean dataframe inplace
processor = DataProcessor()
df = processor.clean_df_inplace(df)

#Generate embeddings for text features
problem = df["body"].astype(str)
responses = df["answer"].astype(str)

embeddings = processor.generate_embeddings(problem)

train_idx, test_idx = train_test_split(
    np.arange(len(embeddings)), test_size=0.2, random_state=42
)

print("Creating Recommender")
recommender = TopKRecommender(k=5)

print("Train Recommender")
recommender.train(embeddings[train_idx], responses[train_idx].values)

print("Print the recommended answer for a problem")
for idx in test_idx[:1]:
    preds = recommender.predict([embeddings[idx]])
    print(problem[idx], preds[0][0])
