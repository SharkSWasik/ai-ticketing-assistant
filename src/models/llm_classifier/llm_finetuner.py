import os
import json
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split

class ClassificationFineTuner:

    def __init__(
        self,
        df: pd.DataFrame,
        strata_col: str,
        text_col: str,
        label_cols: List[str],
        client,
        dataset_dir,
    ):
        self.df = df.copy()
        self.strata = strata_col
        self.text = text_col
        self.labels = label_cols
        self.dataset_dir = dataset_dir
        self.client = client
        self.n_per = 20
        self.rs = 42

    def create_strata(self) -> None:
        #we do all class combination to balance each of them next
        self.df['combined_strata'] = self.df['priority'] + '_' + self.df['language'] + '_' + self.df['queue']

    def balance_strata(self, df) -> pd.DataFrame:

        samples = []
        for _, group in df.groupby(self.strata):
            if len(group) >= self.n_per:
                sampled = group.sample(n=self.n_per, random_state=self.rs, replace=False)
            else:
                sampled = group.sample(n=self.n_per, random_state=self.rs, replace=True)
            samples.append(sampled)
        return pd.concat(samples).reset_index(drop=True)

    def split_indices(self, balanced_df: pd.DataFrame):

        idx = np.arange(len(balanced_df))
        train_idx, rest = train_test_split(idx, random_state=self.rs)
        val_idx, test_idx = train_test_split(rest, test_size=0.5, random_state=self.rs)
        return train_idx, val_idx, test_idx

    def write_jsonl(self, path: str, df: pd.DataFrame, idxs: np.ndarray):

        with open(path, "w", encoding="utf-8") as f:
            for i in idxs:
                problem, priority, support_team, lg = df.iloc[i][["body", "priority", "queue", "language"]]

                data = {
                    "text": f"DESCRIPTION: {problem}",
                    "labels": {
                        "support_team": support_team,
                        "priority": priority,
                        "lg": lg}
                }

                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def upload_file(self, path: str, remote_name: str) -> str:

        responses = self.client.files.upload(
            file={"file_name": f"{remote_name}", "content": open(f"{path}", "rb")}
        )
        
        files = self.client.files.list()
        for file in files.data:
            if file.filename == remote_name:
                return file.id
            
        return responses

    def create_job(
        self,
        train_id: str,
        val_id: str,
        project: str,
        wandb_key:str,
        model: str = "ministral-3b-latest",
        steps: int = 150,
        lr: float = 0.001,
        auto_start: bool = False
    ) -> dict:
        
        return self.client.fine_tuning.jobs.create(
            model=model,
            job_type="classifier",
            training_files=[{"file_id": train_id, "weight": 1}],
            validation_files=[val_id],
            hyperparameters={
                "training_steps": steps,
                "learning_rate":  lr
            },
            integrations=[{"project": project, "api_key": wandb_key}],
            auto_start=auto_start,
        )

    def run(self, wandb_proj: str, wandb_key: str) -> dict:
        
        self.create_strata()
        
        #balance class distributions
        train_i, val_i, _ = self.split_indices(self.df)
        balanced = self.balance_strata(self.df.iloc[train_i])

        train_path = os.path.join(self.dataset_dir, "train_data_classification.jsonl")
        val_path = os.path.join(self.dataset_dir, "validation_file_classification.jsonl")
        self.write_jsonl(train_path, balanced, np.arange(len(balanced)))
        self.write_jsonl(val_path, self.df, val_i)
        
        train_id = self.upload_file(train_path, "train_data_classification.jsonl")
        val_id = self.upload_file(val_path, "validation_file_classification.jsonl")
        
        job = self.create_job(
            train_id, val_id,
            project=wandb_proj,
            wandb_key=wandb_key,
        )
        
        return job