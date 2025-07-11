# Ai-ticketing-assistant
Automated multilingual IT ticket triage and prioritization using fine-tuned LLMs. Classifies, routes, and prioritizes IT support tickets across languages, and automatically suggests solutions for recurring issues.

Built on the Customer IT Support dataset (Kaggle)

# Setup environment

```
conda env create --file environment.yml
conda activate ai_ticket_env
poetry install
```

# Notebooks

I first used notebooks to visualize and experiment with different approaches:

- [priority_classification_notebook](notebooks/priority_classification.ipynb): used LightGBM to classify tickets by priority (baseline).
- [similar_tickets_notebook](notebooks/similar_tickets.ipynb): created a top-k retrieval system based on embeddings to recommend answers (baseline).
- [rag_notebook](notebooks/rag.ipynb): implemented a RAG approach to improve ticket answer quality and evaluated its impact on classification tasks (priority / support team routing).
- [llm_fine_tuning](notebooks/fine-tuning.ipynb): aimed to improve classification results (especially for support team routing) using fine-tuned LLMs.

# Packaging

The solution was then packaged into different modules. You can explore them to find:

- Baseline models
- RAG implementation
- Fine-tuned LLM

# Quick Start

To train the lightgbm classifier
```
poetry run python scripts/train/train_lightgbm_best_params.py 
```

To create the RAG
```
poetry run python scripts/train/create_rag.py 
```

To train the top k recommender
```
poetry run python scripts/train/train_top_k_recommender.py
```

# Evaluation

To evaluate both classification and generation quality on baseline and LLM-based models, I used:

### Classification Metrics

- **F1 Score**
- **Recall**
- **Accuracy**

```
poetry run python eval/evaluate_classifiers.py
```

### Generation Metrics

- **AnswerRelevance**
- **Groundedness**
- **ContextRelevance**
- **LanguageConsistency**


```
poetry run python eval/evaluate_generators.py
```

# Demo

To provide a customer-facing application, I used **Streamlit**.  
Users can interact with the system and ask questions through a simple UI.

### Run the demo with:

```
poetry run streamlit run demo/rag_app.py
```