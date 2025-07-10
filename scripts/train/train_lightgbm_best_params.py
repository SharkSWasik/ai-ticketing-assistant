import pandas as pd
import argparse

from src.data.data_loader import SingleCSVDataLoader
from src.data.data_processor import DataProcessor
from src.models.baseline.lightgbm_classifier import LightGBMClassifier
from src.models.baseline.optimize_lightgbm import run_optimization

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from src.models.baseline.optimize_lightgbm import run_optimization
from sklearn.metrics import f1_score, accuracy_score, recall_score

from src.visualization.plotter import DataPlotter

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
        default="dataset/models/lightgbm_priority.joblib",
        help="Output path for saving the model (without extension)"
    )
    
    parser.add_argument(
        "--plot-results", 
        action="store_true", 
        default=True,
        help="Generate and display performance plots"
    )
    
    return parser.parse_args()

def train_lightgbm(args):

    #Load tickets
    loader = SingleCSVDataLoader(args.dataset)
    df = loader.load_data()

    #Clean dataframe inplace
    processor = DataProcessor()
    df = processor.clean_df_inplace(df)

    #Generate embeddings for text features
    texts = (df["body"].astype(str) + " " + df["queue"].astype(str)).tolist()
    embeddings = processor.generate_embeddings(texts)

    #Encode target labels
    encoded_labels = processor.encode_labels(df, ["priority"])["priority_encoded"].values

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, encoded_labels, stratify=encoded_labels, test_size=0.2, random_state=42
    )

    columns = [f"feature_{i+1}" for i in range(X_train.shape[1])]
    df_embeddings = pd.DataFrame(X_train, columns=columns)
    df_embeddings["y"] = y_train.tolist()

    #Run Optuna optimization to find best params
    print("Starting HPO")
    best_params = run_optimization(df_embeddings[columns], df_embeddings["y"], n_trials=5
    )

    print("Best parameters found:", best_params)

    #Best config dict for LightGBMClassifier
    lgb_config = {
        "n_estimators":        best_params["n_estimators"],
        "num_leaves":          best_params["num_leaves"],
        "learning_rate":       best_params["learning_rate"],
        "max_depth":           best_params["max_depth"],
        "min_data_in_leaf":    best_params["min_data_in_leaf"],
        "reg_alpha":           best_params["reg_alpha"],
        "reg_lambda":          best_params["reg_lambda"],
        "class_weight":        "balanced",
        "random_state":        42,
        "verbose":             -1,
    }

    clf = LightGBMClassifier(config=lgb_config)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = cross_validate(
        clf.model, df_embeddings[columns], df_embeddings["y"],
        cv=cv, 
        scoring='f1_weighted',
        return_estimator=True
    )

    best_fold_idx = results['test_score'].argmax()
    clf.model = results['estimator'][best_fold_idx]

    #Evaluate on test set
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    rec = recall_score(y_test, preds, average="weighted")

    print("Test Accuracy :", acc)
    print("Test F1 Score :", f1)

    plotter = DataPlotter()

    df_test_scores = pd.DataFrame({
        "class_type": ["priority"],
        "f1 score": [f1],
        "accuracy": [acc],
        "recall": [rec]
    }) 

    if args.save_model:
        clf.save(args.model_output)

    if args.plot_results:
        plotter.plot_metric_scores(df_test_scores)

    return df_test_scores

if __name__ == "__main__":
    
    args = parse_arguments()

    train_lightgbm(args)