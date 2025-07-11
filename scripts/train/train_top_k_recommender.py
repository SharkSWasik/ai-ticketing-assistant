import pandas as pd
import numpy as np
import argparse

from src.data.data_loader import SingleCSVDataLoader
from src.data.data_processor import DataProcessor
from src.models.baseline.top_k_recommender import TopKRecommender

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
        default="dataset/models/top_k_recommender.joblib",
        help="Output path for saving the model"
    )
    
    parser.add_argument(
        "--nb_test", 
        type=int, 
        default=1,
        help="Number of test sample where we test the topk"
    )

    return parser.parse_args()

def create_topk(args):

    #Load tickets
    loader = SingleCSVDataLoader(args.dataset)
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

    if args.save_model:
        print("Save Recommender")
        recommender.save(args.model_output)

    print("Print the recommended answer for a problem")

    for idx in test_idx[:args.nb_test]:
        preds = recommender.predict([embeddings[idx]])
        print(problem[idx])
        print(preds[0][0])

    
if __name__ == "__main__":
    
    args = parse_arguments()

    create_topk(args)