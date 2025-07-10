import pandas as pd
import numpy as np
import os

from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

from src.models.baseline.optimize_lightgbm import run_optimization
from src.data.data_loader import SingleCSVDataLoader
from src.data.data_processor import DataProcessor
from src.visualization.plotter import DataPlotter
from src.models.baseline.lightgbm_classifier import LightGBMClassifier
from src.models import LLMClassifierInferencer

def evaluate_classifier(y_true, y_pred):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
    }
    return results

train_dataset_path = "dataset/splits/train.csv"
val_dataset_path = "dataset/splits/val.csv"
test_dataset_path = "dataset/splits/test.csv"

if not os.path.exists(train_dataset_path) or not os.path.exists(val_dataset_path) or not os.path.exists(test_dataset_path):
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

    train_df.to_csv(train_dataset_path, index=False)
    val_df.to_csv(val_dataset_path, index=False)
    test_df.to_csv(test_dataset_path, index=False)

processor = DataProcessor()

train_df = pd.read_csv(train_dataset_path)
test_df = pd.read_csv(test_dataset_path)

print("Generate Embeddings")
train_embeddings = processor.generate_embeddings(train_df["body"].astype(str))
test_embeddings = processor.generate_embeddings(test_df["body"].astype(str))

print("Generate Labels")
train_encoded_labels = processor.encode_labels(train_df, ["priority"])["priority_encoded"].values
test_encoded_labels = processor.encode_labels(test_df, ["priority"])["priority_encoded"].values

columns = [f"feature_{i+1}" for i in range(train_embeddings.shape[1])]
df_embeddings = pd.DataFrame(train_embeddings, columns=columns)
df_embeddings["y"] = train_encoded_labels.tolist()

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
best_model = results['estimator'][best_fold_idx]

llm_classifier = LLMClassifierInferencer()

baseline_preds = best_model.predict(test_embeddings)
new_preds = llm_classifier.predict(test_df["body"].values)
new_priorities_str = [res['priority'] for res in new_preds if 'priority' in res]
new_priorities = processor.encode_labels(pd.DataFrame(new_priorities_str, columns=['priority']), ["priority"])["priority_encoded"].values

plotter = DataPlotter()

plotter.create_model_comparison_plot({'priority' : evaluate_classifier(test_encoded_labels, baseline_preds)},
                             {'priority' : evaluate_classifier(test_encoded_labels, new_priorities)},
                             model_names=["Baseline Lightgbm", "LLM finetuned"])

print(evaluate_classifier(test_encoded_labels, new_priorities))