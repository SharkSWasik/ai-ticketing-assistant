import os
import json
from typing import List
from mistralai import Mistral

class LLMClassifierInferencer:

    def __init__(self, api_key: str = None, client=None, model_id: str = "ft:classifier:ministral-3b-latest:7e530b29:20250709:e3ee779b"):

        self.model_id = model_id
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.client = client or Mistral(api_key=self.api_key)

    def predict(self, texts: List[str]) -> List[dict]:
        
        results = []
        for problem in texts:

            classifier_response = self.client.classifiers.classify(
            model=self.model_id,
            inputs=[f"DESCRIPTION: {problem}"])

            content = classifier_response.results[0]
            
            try:
                res = {}
                res['priority'] = max(content['priority'].scores.items(), key=lambda x:x[1])[0]
                res['support_team'] = max(content['support_team'].scores.items(), key=lambda x:x[1])[0]

                results.append(res)
                
            except Exception as e:
                results.append({"error": str(e), "raw": content})

        return results