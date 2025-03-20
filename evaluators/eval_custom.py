import pandas as pd
import torch
import numpy as np
import spacy
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download NLTK dependencies
nltk.download("punkt")

# Load pre-trained models
sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")
consistency_checker = pipeline("text-classification", model="facebook/bart-large-mnli")
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# Step 1: Load dataset
df = pd.read_csv("output_answered_updated2.csv" )

# Step 2: Compute Retrieval Relevance
def compute_retrieval_relevance(query, retrieved_chunks):
    if not isinstance(retrieved_chunks, str) or not retrieved_chunks.strip():
        return 0.0 

    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    context_embedding = sentence_model.encode(retrieved_chunks, convert_to_tensor=True)

    similarity = torch.nn.functional.cosine_similarity(query_embedding, context_embedding, dim=0)
    return similarity.item()

# Step 3: Compute Mean Reciprocal Rank (MRR)
def mean_reciprocal_rank(relevant_indices):
    if not relevant_indices:
        return 0.0
    return sum(1.0 / (rank + 1) for rank in relevant_indices) / len(relevant_indices)

# Step 4: Compute Recall@K
def recall_at_k(relevant_docs, retrieved_docs, k=5):
    retrieved_top_k = retrieved_docs[:k]
    return len(set(relevant_docs) & set(retrieved_top_k)) / len(relevant_docs) if relevant_docs else 0.0

# Step 5: Compute Answer Correctness (ROUGE-L)
def compute_answer_correctness(llm_answer, expected_answer):
    if not isinstance(llm_answer, str) or not isinstance(expected_answer, str):
        return 0.0
    scores = scorer.score(expected_answer, llm_answer)
    return scores["rougeL"].fmeasure

# Step 6: Compute BLEU Score
def compute_bleu_score(llm_answer, expected_answer):
    if not isinstance(llm_answer, str) or not isinstance(expected_answer, str):
        return 0.0
    reference = [expected_answer.split()]
    hypothesis = llm_answer.split()
    smoothing = SmoothingFunction().method1
    return sentence_bleu(reference, hypothesis, smoothing_function=smoothing)

# Step 7: Detect Hallucinations (Named Entity Overlap)
def detect_hallucination(llm_answer, retrieved_chunks):
    answer_entities = {ent.text.lower() for ent in nlp(llm_answer).ents}
    context_entities = {ent.text.lower() for ent in nlp(retrieved_chunks).ents}
    hallucinated_entities = answer_entities - context_entities
    return len(hallucinated_entities) > 0, hallucinated_entities

# Step 8: Detect Hallucinations using LLM Consistency Check
def detect_hallucination_with_llm(llm_answer, retrieved_chunks):
    result = consistency_checker(f"Context: {retrieved_chunks}\nAnswer: {llm_answer}")
    return result[0]["label"] == "contradiction"

# Step 9: Evaluate the RAG pipeline and store results
results = []

for _, row in df.iterrows():
    question = str(row["question"]) if pd.notna(row["question"]) else ""
    retrieved_chunks = str(row["retrieved_chunks"]) if pd.notna(row["retrieved_chunks"]) else ""
    llm_answer = str(row["llm_answer"]) if pd.notna(row["llm_answer"]) else ""
    expected_answer = str(row["expected_answer"]) if pd.notna(row["expected_answer"]) else ""

    # Compute retrieval relevance
    retrieval_score = compute_retrieval_relevance(question, retrieved_chunks)

    # Compute answer correctness
    correctness_score = compute_answer_correctness(llm_answer, expected_answer)

    # Compute BLEU Score
    bleu_score = compute_bleu_score(llm_answer, expected_answer)

    # Compute Recall@K (for now, we assume relevant_docs are expected_answers)
    relevant_docs = [expected_answer] if expected_answer else []
    retrieved_docs = [retrieved_chunks] if retrieved_chunks else []

    # Detect hallucinations (NER-based)
    hallucination_flag, hallucinated_entities = detect_hallucination(llm_answer, retrieved_chunks)

    # Detect hallucinations (LLM-based)
    hallucination_llm_flag = detect_hallucination_with_llm(llm_answer, retrieved_chunks)

    # Append results to list
    results.append({
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "llm_answer": llm_answer,
        "expected_answer": expected_answer,
        "retrieval_relevance_score": retrieval_score,
        "answer_correctness_score": correctness_score,
        "bleu_score": bleu_score,
        "hallucination_flag": hallucination_flag,
        "hallucinated_entities": ", ".join(hallucinated_entities),
        "hallucination_llm_flag": hallucination_llm_flag,
    })

# Step 10: Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("evaluated_output.csv", index=False)

# Step 11: Aggregate results for final scores
avg_retrieval_score = results_df["retrieval_relevance_score"].mean()
avg_correctness_score = results_df["answer_correctness_score"].mean()
avg_bleu_score = results_df["bleu_score"].mean()
hallucination_rate = results_df["hallucination_flag"].mean()
hallucination_llm_rate = results_df["hallucination_llm_flag"].mean()

# Print Final Scores
print(f"🔹 Average Retrieval Relevance Score: {avg_retrieval_score:.2f}")
print(f"🔹 Average Answer Correctness Score (ROUGE-L): {avg_correctness_score:.2f}")
print(f"🔹 Average BLEU Score: {avg_bleu_score:.2f}")
print(f"🔹 Hallucination Rate (NER-based): {hallucination_rate:.2f}")
print(f"🔹 Hallucination Rate (LLM-based): {hallucination_llm_rate:.2f}")
