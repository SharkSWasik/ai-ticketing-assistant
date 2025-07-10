import pandas as pd
import os
import argparse

from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

from src.data.data_loader import SingleCSVDataLoader
from src.data.data_processor import DataProcessor
from src.visualization.plotter import DataPlotter
from src.models.baseline.lightgbm_classifier import LightGBMClassifier
from src.models import LLMClassifierInferencer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate LightGBM classifier and LLM Classifier for ticket priority prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--train_dataset_path", 
        type=str, 
        default="dataset/splits/train.csv",
        help="Path to the dataset train CSV file"
    )

    parser.add_argument(
        "--test_dataset_path", 
        type=str, 
        default="dataset/splits/test.csv",
        help="Path to the dataset test CSV file"
    )

    parser.add_argument(
        "--val_dataset_path", 
        type=str, 
        default="dataset/splits/val.csv",
        help="Path to the dataset val file"
    )

    parser.add_argument(
        "--lgb_model", 
        type=str, 
        default="dataset/models/lightgbm_priority.joblib",
        help="Path to the lgb model"
    )
    
    parser.add_argument(
        "--plot-results", 
        action="store_true", 
        default=True,
        help="Generate and display performance plots"
    )
    
    return parser.parse_args()

def compute_metrics(y_true, y_pred):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }
    return results

def split_dataset(args):

    #Load tickets
    loader = SingleCSVDataLoader(file_path="dataset/data/dataset-tickets-multi-lang3-4k.csv")
    df = loader.load_data()

    #Clean dataframe inplace
    processor = DataProcessor()
    df = processor.clean_df_inplace(df)

    if "combined_strata" not in df.columns:
        df["combined_strata"] = (
            df["priority"].astype(str) + "_" +
            df["language"].astype(str) + "_" +
            df["queue"].astype(str)
        )

    samples = []
    n_per = 20

    for _, group in df.groupby(["combined_strata"]):
        if len(group) >= n_per:
            sampled = group.sample(n=n_per, random_state=42, replace=False)
        else:
            sampled = group.sample(n=n_per, random_state=42, replace=True)
        samples.append(sampled)

    df_balanced = pd.concat(samples).reset_index(drop=True)

    train_df, rest_df = train_test_split(df_balanced, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=42)

    train_df.to_csv(args.train_dataset_path, index=False)
    val_df.to_csv(args.val_dataset_path, index=False)
    test_df.to_csv(args.test_dataset_path, index=False)

def evaluate_classifier(args):

    if (not os.path.exists(args.train_dataset_path) or
    not os.path.exists(args.val_dataset_path) or
    not os.path.exists(args.test_dataset_path)):
        split_dataset(args)
        
    processor = DataProcessor()

    test_df = pd.read_csv(args.test_dataset_path)

    print("Generate Embeddings")
    test_embeddings = processor.generate_embeddings(test_df["body"].astype(str))
    columns = [f"feature_{i+1}" for i in range(test_embeddings.shape[1])]

    print("Generate Labels")
    test_encoded_labels = processor.encode_labels(test_df, ["priority"])["priority_encoded"].values

    lightgbm_classifier = LightGBMClassifier.load(args.lgb_model)
    llm_classifier = LLMClassifierInferencer()

    baseline_preds = lightgbm_classifier.predict(pd.DataFrame(test_embeddings, columns=columns))
    new_preds = llm_classifier.predict(test_df["body"].values)
    new_priorities_str = [res['priority'] for res in new_preds if 'priority' in res]
    new_priorities = processor.encode_labels(pd.DataFrame(new_priorities_str, columns=['priority']), ["priority"])["priority_encoded"].values

    if args.plot_results:

        plotter = DataPlotter()

        plotter.create_model_comparison_plot({'priority' : compute_metrics(test_encoded_labels, baseline_preds)},
                                    {'priority' : compute_metrics(test_encoded_labels, new_priorities)},
                                    model_names=["Baseline Lightgbm", "LLM finetuned"])

        print(compute_metrics(test_encoded_labels, new_priorities))


if __name__ == "__main__":
    
    args = parse_arguments()

    evaluate_classifier(args)