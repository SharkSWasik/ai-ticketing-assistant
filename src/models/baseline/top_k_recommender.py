from typing import List, Optional
from ..base_model import BaseModel

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

class TopKRecommender(BaseModel):
    def __init__(
        self,
        k: int = 5,
    ):
        self.k = k
        self.knowledge_problem: List[float] = None
        self.knowledge_responses: List[str] = None

    def train(self, problem: List[float], responses: List[str]) -> None:
        
        self.knowledge_problem = problem
        self.knowledge_responses = responses

    def predict(self, queries: List[float]) -> List[List[str]]:
        
        sims = cosine_similarity(self.knowledge_problem, queries)
        recommendations = []

        for j in range(sims.shape[1]):
            
            top_k_idx = np.argsort(-sims[:, j])[:self.k]
            top_k_responses = [self.knowledge_responses[i] for i in top_k_idx]
            recommendations.append(top_k_responses)

        return recommendations
