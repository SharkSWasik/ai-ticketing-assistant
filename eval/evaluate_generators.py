from src.models.rag import RAGModel
from llm_as_a_judge import evaluate_rag

import argparse
import numpy as np
import tqdm

from src.models.rag import SimpleGenerator, RAGModel
from src.data.data_loader import SingleCSVDataLoader
from src.data.data_processor import DataProcessor
from src.visualization.plotter import DataPlotter
from src.models.baseline.top_k_recommender import TopKRecommender

from sklearn.model_selection import train_test_split

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate LightGBM classifier and LLM Classifier for ticket priority prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--test_dataset_path", 
        type=str, 
        default="dataset/splits/test.csv",
        help="Path to the dataset test CSV file"
    )

    parser.add_argument(
        "--rag_model_path", 
        type=str, 
        default="dataset/models/rag.joblib",
        help="Path to the rag model"
    )

    parser.add_argument(
        "--topk_model_path", 
        type=str, 
        default="dataset/models/top_k_recommender.joblib",
        help="Path to the rag model"
    )
    
    parser.add_argument(
        "--plot-results", 
        action="store_true", 
        default=True,
        help="Generate and display performance plots"
    )
    
    return parser.parse_args()

def create_rag(df_train):

    print("Create the Generator")
    gen = SimpleGenerator()

    print("Create the RAG")
    rag = RAGModel(
        embedding_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        generator=gen,
        chunk_size=2048,
        k=1
    )

    print("Training the RAG")
    #create the embedding and vector database
    rag.train(df_train)
    return rag

def evaluate_generators(args):

    #Load tickets
    loader = SingleCSVDataLoader(file_path="dataset/data/dataset-tickets-multi-lang3-4k.csv")
    df = loader.load_data()

    #Clean dataframe inplace
    processor = DataProcessor()
    df = processor.clean_df_inplace(df)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42
    )

    rag = create_rag(df.iloc[train_idx])
    top_k_recommender = TopKRecommender.load(args.topk_model_path)
    processor = DataProcessor()

    rag_scores = {
       "context_relevance": [],
        "answer_relevance": [],
        "groundedness": []
    }

    topk_scores = {
       "context_relevance": [],
        "answer_relevance": [],
        "groundedness": []
    }

    for idx in tqdm.tqdm(test_idx[:10]):

        result, context = rag.predict([df.iloc[idx]["body"]])
        evaluations = evaluate_rag(df.iloc[idx]["body"], context, result)
        rag_scores["context_relevance"].append(evaluations.context_relevance.score)
        rag_scores["answer_relevance"].append(evaluations.answer_relevance.score)
        rag_scores["groundedness"].append(evaluations.groundedness.score)

        reco = top_k_recommender.predict(processor.generate_embeddings([df.iloc[idx]["body"]]))
        evaluations = evaluate_rag(df.iloc[idx]["body"], reco[0], reco[0][0])
        topk_scores["context_relevance"].append(evaluations.context_relevance.score)
        topk_scores["answer_relevance"].append(evaluations.answer_relevance.score)
        topk_scores["groundedness"].append(evaluations.groundedness.score)

    plotter = DataPlotter()

    if args.plot_results:
        plotter.plot_llm_judge_score(rag_scores["context_relevance"], rag_scores["answer_relevance"], rag_scores["groundedness"], title="RAG score with LLM as a Judge")
        plotter.plot_llm_judge_score(topk_scores["context_relevance"], topk_scores["answer_relevance"], topk_scores["groundedness"], title="TopK score with LLM as a Judge")

if __name__ == "__main__":
    
    args = parse_arguments()

    evaluate_generators(args)