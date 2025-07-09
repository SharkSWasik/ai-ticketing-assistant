import pandas as pd
from src.visualization.plotter import DataPlotter
import os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

df = pd.read_csv(BASE + "/dataset/data/dataset-tickets-multi-lang3-4k.csv")

df = df.fillna('')

plotter = DataPlotter()

plotter.plot_ticket_overview(df)

df_test_scores = pd.DataFrame({
    "class_type": ["support_team", "priority"],
    "f1 score": [0.75, 0.85],
    "accuracy": [0.80, 0.88],
    "recall": [0.70, 0.82]
})

plotter.plot_metric_scores(df_test_scores)