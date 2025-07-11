import pandas as pd
import os
import argparse
import tqdm

from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.data.data_processor import DataProcessor
from src.visualization.plotter import DataPlotter
from src.models import LLMClassifierInferencer

from src.models.rag import RAGModel

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate LightGBM classifier and LLM Classifier for ticket priority prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--test_dataset_path", 
        type=str, 
        default="dataset/splits/test.csv",
        help="Path to the dataset test CSV file"
    )

    parser.add_argument(
        "--rag_model_path", 
        type=str, 
        default="dataset/models/rag/",
        help="Path to the lgb model"
    )

    parser.add_argument(
        "--llm_finetuned_model", 
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

    parser.add_argument(
        "--nb-to-test", 
        type=int, 
        default=150,
        help="Number of test set sample to evaluate"
    )
    
    return parser.parse_args()

def compute_metrics(y_true, y_pred):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
    }
    return results

def evaluate_classifier(args):
        
    processor = DataProcessor()

    test_df = pd.read_csv(args.test_dataset_path)

    print("RAG is loading")
    rag = RAGModel.load(args.rag_model_path)

    print("Finetuned Classifier LLM is loading")
    llm_clf_ft = LLMClassifierInferencer()

    rag_scores = []
    llm_ft_scores = []

    for problem in tqdm.tqdm(test_df["body"].iloc[:args.nb_to_test]):
        try:
            rag_result, _ = rag.predict([problem])
        except RuntimeError: #wrong rag output
            continue
        rag_scores.append(rag_result[0]["support_team"])

        llm_ft_result = llm_clf_ft.predict([problem])
        llm_ft_scores.append(llm_ft_result[0]["support_team"])

    plotter = DataPlotter()

    test_df.rename(columns={'queue': 'support_team'}, inplace=True)
    test_encoded_labels = processor.encode_labels(test_df, ["support_team"])["support_team"].values

    rag_preds = processor.encode_labels(pd.DataFrame(rag_scores, columns=["support_team"]), ["support_team"])["support_team"].values
    llm_ft_preds = processor.encode_labels(pd.DataFrame(llm_ft_scores, columns=["support_team"]), ["support_team"])["support_team"].values

    plotter.create_model_comparison_plot({'support_team' : compute_metrics(test_encoded_labels[:len(rag_preds)], rag_preds)},
                                {'support_team' : compute_metrics(test_encoded_labels[:len(llm_ft_preds)], llm_ft_preds)},
                                model_names=["RAG", "LLM finetuned"])
if __name__ == "__main__":
    
    args = parse_arguments()

    evaluate_classifier(args)