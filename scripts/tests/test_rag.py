import pandas as pd
import numpy as np
import os

from src.models.rag import SimpleGenerator, RAGModel
from src.data.data_loader import SingleCSVDataLoader
from src.data.data_processor import DataProcessor

from sklearn.model_selection import train_test_split

#Load tickets
loader = SingleCSVDataLoader(file_path="dataset/data/dataset-tickets-multi-lang3-4k.csv")
df = loader.load_data()

#Clean dataframe inplace
processor = DataProcessor()
df = processor.clean_df_inplace(df)

problem = df["body"].astype(str)
responses = df["answer"].astype(str)

train_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42
)

print("Create the Generator")
gen = SimpleGenerator()

print("Create the RAG")
rag = RAGModel(
    embedding_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    generator=gen,
    chunk_size=2048,
    k=1
)

print("Training the RAG")
#create the embedding and vector database
rag.train(df)

print("Testing the RAG")
#try gen on a test set sample
for idx in test_idx[:1]:
    print([problem.iloc[idx]])
    print("RAG Prediction")
    print(rag.predict([problem.iloc[idx]]))