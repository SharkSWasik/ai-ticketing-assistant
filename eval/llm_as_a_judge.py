from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict
from mistralai import Mistral
import os

# Define Enum for scores
class Score(str, Enum):
    no_relevance = "0"
    low_relevance = "1"
    medium_relevance = "2"
    high_relevance = "3"

# Define a constant for the score description
SCORE_DESCRIPTION = (
    "Score as a string between '0' and '3'. "
    "0: No relevance/Not grounded/Irrelevant - The context/answer is completely unrelated or not based on the context. "
    "1: Low relevance/Low groundedness/Somewhat relevant - The context/answer has minimal relevance or grounding. "
    "2: Medium relevance/Medium groundedness/Mostly relevant - The context/answer is somewhat relevant or grounded. "
    "3: High relevance/High groundedness/Fully relevant - The context/answer is highly relevant or grounded."
)

# Define separate classes for each criterion with detailed descriptions
class ContextRelevance(BaseModel):
    explanation: str = Field(..., description=("Step-by-step reasoning explaining how the retrieved context aligns with the user's query. "
                    "Consider the relevance of the information to the query's intent and the appropriateness of the context "
                    "in providing a coherent and useful response."))
    score: Score = Field(..., description=SCORE_DESCRIPTION)

class LanguageConsistency(BaseModel):
    explanation: str = Field(..., description="Step-by-step reasoning explaining whether the generated answer is in the same language as the query.")
    is_same_language: bool = Field(..., description="Return true if the answer is in the same language as the query, false otherwise.")

class AnswerRelevance(BaseModel):
    explanation: str = Field(..., description=("Step-by-step reasoning explaining how well the generated answer addresses the user's original query. "
                    "Consider the helpfulness and on-point nature of the answer, aligning with the user's intent and providing valuable insights."))
    score: Score = Field(..., description=SCORE_DESCRIPTION)

class Groundedness(BaseModel):
    explanation: str = Field(..., description=("Step-by-step reasoning explaining how faithful the generated answer is to the retrieved context. "
                    "Consider the factual accuracy and reliability of the answer, ensuring it is grounded in the retrieved information."))
    score: Score = Field(..., description=SCORE_DESCRIPTION)

class RAGEvaluation(BaseModel):
    context_relevance: ContextRelevance = Field(..., description="Evaluation of the context relevance to the query, considering how well the retrieved context aligns with the user's intent." )
    answer_relevance: AnswerRelevance = Field(..., description="Evaluation of the answer relevance to the query, assessing how well the generated answer addresses the user's original query." )
    groundedness: Groundedness = Field(..., description="Evaluation of the groundedness of the generated answer, ensuring it is faithful to the retrieved context." )
    language_consistency: LanguageConsistency = Field(..., description="Evaluation of whether the answer is in the same language as the query.")


# Function to evaluate RAG metrics
def evaluate_rag(query: str, retrieved_context: str, generated_answer: str, model : str = "mistral-medium-latest", api_key=os.environ.get("MISTRAL_API_KEY")):
    client = Mistral(api_key=api_key)
    chat_response = client.chat.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a judge for evaluating a Retrieval-Augmented Generation (RAG) system. "
                    "Evaluate the context relevance, answer relevance, and groundedness based on the following criteria: "
                    "Provide a reasoning and a score as a string between '0' and '3' for each criterion. "
                    "Context Relevance: How relevant is the retrieved context to the query? "
                    "Answer Relevance: How relevant is the generated answer to the query? "
                    "Groundedness: How faithful is the generated answer to the retrieved context?"
                )
            },
            {
                "role": "user",
                "content": f"Query: {query}\nRetrieved Context: {retrieved_context}\nGenerated Answer: {generated_answer}"
            },
        ],
        response_format=RAGEvaluation,
        temperature=0
    )
    return chat_response.choices[0].message.parsed