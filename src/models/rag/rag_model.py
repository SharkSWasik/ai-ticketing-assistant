from typing import List, Optional
from ..base_model import BaseModel
from .simple_generator import SimpleGenerator

import faiss
import pandas as pd
import re
from joblib import dump, load

from sentence_transformers import SentenceTransformer
from typing import Dict

class RAGModel(BaseModel):

    def __init__(
        self,
        generator: SimpleGenerator,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        chunk_size: int = 2048,
        k: int = 5
    ):

        self.encoder = SentenceTransformer(embedding_model_name)
        self.generator = generator
        
        self.chunk_size = chunk_size
        self.k = k
        
        self.chunks: List[str] = []
        self.index: Optional[faiss.IndexFlatL2] = None

    def format_ticket(self, row: pd.Series) -> str:
        return (
            f"SUBJECT: {row['subject']} "
            f"PROBLEM: {row['body']} "
            f"SOLUTION: {row['answer']} "
            f"PRIORITY: {row['priority']} "
            f"LANGUAGE: {row['language']} "
            f"SUPPORT TEAM: {row['queue']}"
        )

    def make_chunks(self, texts: List[str]) -> None:

        for text in texts:
            #if less than 500 length we keep the entire string as a chunk
            if len(text) <= self.chunk_size:
                self.chunks.append(text)
            else:
                #using regex to cut the text into sections
                sections = re.split(
                    r"(SUBJECT:|PROBLEM:|SOLUTION:|PRIORITY:|LANGUAGE:|SUPPORT TEAM:)",
                    text
                )
                sections = [s.strip() for s in sections if s.strip()]
                it = iter(sections)
                sections_list = [f"{k} {v}" for k, v in zip(it, it)]
                temp_chunk = ""
                for sec in sections_list:
                    #if temp_chunk + new sec is greater than the chunk size, we save the last temp chunk
                    if len(temp_chunk) + len(sec) > self.chunk_size:
                        if temp_chunk: #no empty chunk
                            self.chunks.append(temp_chunk)
                        temp_chunk = sec
                    else:
                        temp_chunk = temp_chunk + " " + sec if temp_chunk else sec
                #adding last chunk
                if temp_chunk:
                    self.chunks.append(temp_chunk.strip())

    def train(self, df: pd.DataFrame, _: Optional[List[str]] = None) -> None:

        texts = df.apply(self.format_ticket, axis=1).tolist()
        
        self.make_chunks(texts)
        
        embeddings = self.encoder.encode(self.chunks, convert_to_numpy=True)
        self.d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.d)
        self.index.add(embeddings)

    def predict(self, queries: List[str]) -> tuple[List[Dict], List[str]]:
        
        results = []
        q_emb = self.encoder.encode(queries)
        _, I = self.index.search(q_emb, self.k)

        for idx in range(len(I)):
            contexts = [self.chunks[i] for i in I.tolist()[idx]]
            gen = self.generator.generate(contexts, queries[idx])
            results.append(gen)

        return results, contexts
    
    def save(self, dir_path:str):
    
        with open(dir_path + "recommender.joblib", 'wb') as f:
            dump(self.encoder, f)

        self.generator.save(dir_path + "generator.joblib")

        with open(dir_path + "index.joblib", 'wb') as f:
            dump(self.index, f)

        with open(dir_path + "chunks.joblib", "wb") as f:
            dump(self.chunks, f)


    @classmethod
    def load(cls, dir_path: str) -> 'RAGModel':
    
        with open(dir_path + "recommender.joblib", 'rb') as f:
            encoder = load(f)

        with open(dir_path + "generator.joblib", 'rb') as f:
            generator = load(f)

        with open(dir_path + "index.joblib", 'rb') as f:
            index = load(f)

        with open(dir_path + "chunks.joblib", 'rb') as f:
            chunks = load(f)

        model = RAGModel(generator=generator)
        model.chunks = chunks
        model.index = index
        model.encoder = encoder

        return model