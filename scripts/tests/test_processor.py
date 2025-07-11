import pandas as pd

from src.data.data_processor import DataProcessor

df_test = pd.DataFrame({
    "description": [
        "Mon imprimante est en panne",
        "Écran bleu au démarrage"
    ],
    "priority": ["high", "medium"],
    "support_team": ["Technical Support", "IT Support"],
    "priority_pred": ["high", "medium"],
    "support_team_pred": ["Technical Support", "IT Support"]
})

print("Creation of Data Processor OBJ")
processor = DataProcessor()

df_clean = processor.clean_df_inplace(df_test)

embeddings = processor.generate_embeddings(df_clean["description"].tolist())
print("Embeddings shape:", embeddings.shape)

df_encoded = processor.encode_labels(
    df_clean,
    label_columns=["priority", "support_team"],
    ref_cols=["priority", "support_team"],
    pred_cols=["priority_pred", "support_team_pred"]
)
print(df_encoded.columns)

print(df_encoded[[
    "priority_encoded", "support_team_encoded",
    "priority_pred_encoded", "support_team_pred_encoded"
]])