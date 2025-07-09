import pandas as pd

from src.data.data_loader import SingleCSVDataLoader
from src.data.data_processor import DataProcessor
from src.models.baseline.lightgbm_classifier import LightGBMClassifier
from src.models.baseline.optimize_lightgbm import run_optimization

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score

from src.visualization.plotter import DataPlotter

#Load tickets
loader = SingleCSVDataLoader(file_path="dataset/data/dataset-tickets-multi-lang3-4k.csv")
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

#Run Optuna optimization to find best params
print("Starting HPO")
best_params = run_optimization(
    pd.DataFrame(X_train), pd.Series(y_train), n_trials=20
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

#Train LightGBMClassifier with best params
clf = LightGBMClassifier(config=lgb_config)
clf.train(X_train, y_train)

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

plotter.plot_metric_scores(df_test_scores)