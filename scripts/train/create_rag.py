import pandas as pd
import numpy as np
import os
import argparse

from src.models.rag import SimpleGenerator, RAGModel
from src.data.data_loader import SingleCSVDataLoader
from src.data.data_processor import DataProcessor

from sklearn.model_selection import train_test_split

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train LightGBM classifier for ticket priority prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="dataset/data/dataset-tickets-multi-lang3-4k.csv",
        help="Path to the dataset CSV file"
    )

    parser.add_argument(
        "--save-model", 
        action="store_true",
        default=True,
        help="Save the trained model to disk"
    )
    
    parser.add_argument(
        "--model-output", 
        type=str, 
        default="dataset/models/rag/",
        help="Output path for saving the model (without extension)"
    )
    
    parser.add_argument(
        "--plot-results", 
        action="store_true", 
        default=True,
        help="Generate and display performance plots"
    )
    
    return parser.parse_args()


def create_rag(args):
    #Load tickets
    loader = SingleCSVDataLoader(args.dataset)
    df = loader.load_data()

    #Clean dataframe inplace
    processor = DataProcessor()
    df = processor.clean_df_inplace(df)

    problem = df["body"].astype(str)

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
    rag.train(df.iloc[train_idx])

    if args.save_model:
        print("Save the RAG")
        rag.save(args.model_output)

if __name__ == "__main__":
    
    args = parse_arguments()

    create_rag(args)