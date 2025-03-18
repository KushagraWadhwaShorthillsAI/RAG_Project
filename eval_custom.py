import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics import f1_score

# Step 1: Load your Excel sheet
df = pd.read_csv("output_answers_updated.csv")

# Step 2: Define a function to compute retrieval relevance using Cosine Similarity
def compute_retrieval_relevance(query, retrieved_chunks):
    """
    Compute the relevance between the query and retrieved context using Cosine Similarity.
    """
    # Check if retrieved_chunks is a valid string
    if not isinstance(retrieved_chunks, str) or not retrieved_chunks.strip():
        return 0.0  # Return 0 relevance for invalid or empty context

    # Load a pre-trained sentence transformer model (e.g., BERT)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    # Tokenize and encode the query and context
    # Tokenize and encode the query and context with truncation enabled
    query_embedding = model(**tokenizer(query, return_tensors="pt", truncation=True, max_length=512)).last_hidden_state.mean(dim=1)
    context_embedding = model(**tokenizer(retrieved_chunks, return_tensors="pt", truncation=True, max_length=512)).last_hidden_state.mean(dim=1)


    # Compute cosine similarity
    similarity = cosine_similarity(query_embedding.detach().numpy(), context_embedding.detach().numpy())
    return similarity[0][0]

# Step 3: Define a function to compute answer correctness using F1 Score
def compute_answer_correctness(llm_answer, expected_answer):
    """
    Compute the F1 score between the LLM's answer and the expected answer.
    """
    # Check if llm_answer and expected_answer are valid strings
    if not isinstance(llm_answer, str) or not isinstance(expected_answer, str):
        return 0.0  # Return 0 correctness for invalid or empty answers

    # Tokenize the answers
    llm_tokens = set(llm_answer.lower().split())
    expected_tokens = set(expected_answer.lower().split())

    # Compute F1 score
    common_tokens = llm_tokens.intersection(expected_tokens)
    precision = len(common_tokens) / len(llm_tokens) if len(llm_tokens) > 0 else 0
    recall = len(common_tokens) / len(expected_tokens) if len(expected_tokens) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

# Step 4: Define a function to detect hallucinations
def detect_hallucination(llm_answer, retrieved_chunks):
    """
    Detect if the LLM's answer contains information not supported by the retrieved context.
    """
    # Check if llm_answer and retrieved_chunks are valid strings
    if not isinstance(llm_answer, str) or not isinstance(retrieved_chunks, str):
        return False  # Return False for invalid or empty inputs

    # Tokenize the LLM answer and context
    llm_tokens = set(llm_answer.lower().split())
    context_tokens = set(retrieved_chunks.lower().split())

    # Check if any tokens in the LLM answer are not in the context
    hallucinated_tokens = llm_tokens - context_tokens
    return len(hallucinated_tokens) > 0  # True if hallucination is detected

# Step 5: Evaluate the RAG pipeline and store results
results = []

for _, row in df.iterrows():
    question = row["question"]
    retrieved_chunks = row["retrieved_chunks"]
    llm_answer = row["llm_answer"]
    expected_answer = row["expected_answer"]

    # Compute retrieval relevance
    retrieval_score = compute_retrieval_relevance(question, retrieved_chunks)

    # Compute answer correctness
    correctness_score = compute_answer_correctness(llm_answer, expected_answer)

    # Detect hallucinations
    hallucination_flag = detect_hallucination(llm_answer, retrieved_chunks)

    # Append results to the list
    results.append({
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "llm_answer": llm_answer,
        "expected_answer": expected_answer,
        "retrieval_relevance_score": retrieval_score,
        "answer_correctness_score": correctness_score,
        "hallucination_flag": hallucination_flag,
    })

# Step 6: Convert results to a DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("evaluated_output.csv", index=False)

# Step 7: Aggregate results for final scores
avg_retrieval_score = results_df["retrieval_relevance_score"].mean()
avg_correctness_score = results_df["answer_correctness_score"].mean()
hallucination_rate = results_df["hallucination_flag"].mean()

print(f"Average Retrieval Relevance Score: {avg_retrieval_score:.2f}")
print(f"Average Answer Correctness Score: {avg_correctness_score:.2f}")
print(f"Hallucination Rate: {hallucination_rate:.2f}")