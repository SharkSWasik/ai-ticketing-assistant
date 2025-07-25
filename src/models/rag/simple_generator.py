import os
import json

from mistralai import Mistral
from joblib import dump, load

from typing import Dict

class SimpleGenerator:

    def __init__(
        self,
        model_name: str = "mistral-small-latest",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ):
       
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        self.client = Mistral(api_key=self.api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, problem: str, context: str) -> Dict:
        
        prompt = f"""
        Context information is below.
        ---------------------
        {context}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {problem}
        
        CRITICAL : You MUST detect the language of the Query, and respond in the exact same language.

        CRITICAL: You MUST return ONLY a valid JSON object with exactly this structure. 
        Do not include any explanatory text, markdown formatting, or additional content.
        
        Return only the JSON object with these exact field names and value types:

        {{
            "answer": "your response text here",
            "priority": "low",
            "support_team": "Technical Support", 
            "language": "en"
        }}

        Rules:
        - priority must be exactly one of: "low", "medium", "high"
        - support_team must be exactly one of: "Technical Support", "Customer Service", "IT Support", "Product Support", "Billing and Payments", "Service Outages and Maintenance", "General Inquiry", "Returns and Exchanges", "Sales and Pre-Sales", "Human Resources"
        - language must be exactly one of: "es", "de", "pt", "en", "fr"
        - All values must be simple strings, not objects
        - Return ONLY the JSON object, nothing else

        JSON:
        """
        response = self.client.chat.complete(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON returned by generator: {content}")
    
    def __getstate__(self):
        
        state = self.__dict__.copy()
        state.pop('client', None) #no client
        return state

    def __setstate__(self, state):

        self.__dict__.update(state)
        self.client = Mistral(api_key=self.api_key)

    def save(self, path: str):
        
        with open(path, 'wb') as f:
            dump(self, f)

    @classmethod
    def load(self, path: str) -> 'SimpleGenerator':

        with open(path, 'wb') as f:
            model = load(path)

        return model