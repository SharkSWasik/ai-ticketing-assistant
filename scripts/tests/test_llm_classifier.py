import pandas as pd
import numpy as np
import os

from src.data.data_loader import SingleCSVDataLoader
from src.data.data_processor import DataProcessor
from src.models import  ClassificationFineTuner

from mistralai import Mistral

#Load tickets
loader = SingleCSVDataLoader(file_path="dataset/data/dataset-tickets-multi-lang3-4k.csv")
df = loader.load_data()

#Clean dataframe inplace
processor = DataProcessor()
df = processor.clean_df_inplace(df)

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

tuner = ClassificationFineTuner(
    df,
    strata_col="combined_strata",
    text_col="body",
    label_cols=["queue", "priority", "language"],
    client=client,
    dataset_dir="dataset/data/",
)

job = tuner.run(
    wandb_proj="ticket_classifier",
    wandb_key=os.environ["WANDB_KEY"],
)

print("Job created:", job)