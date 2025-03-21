import pandas as pd
import nltk
import numpy as np
import difflib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.corpus import stopwords
from rouge_score import rouge_scorer
import spacy

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")

stop_words = set(stopwords.words("english"))

# Load spaCy model for POS and noun phrase extraction
nlp = spacy.load("en_core_web_sm")

# Load pre-trained ROUGE scorer
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# Load dataset (CSV file should have columns: question, expected_answer, llm_answer)
df = pd.read_csv("golden_answered.csv")

# ------------------ Existing Metrics ------------------

def compute_rouge_l(llm_answer, expected_answer):
    if not isinstance(llm_answer, str) or not isinstance(expected_answer, str):
        return 0.0
    scores = scorer.score(expected_answer, llm_answer)
    return scores["rougeL"].fmeasure

def compute_bleu_score(llm_answer, expected_answer):
    if not isinstance(llm_answer, str) or not isinstance(expected_answer, str):
        return 0.0
    reference = [expected_answer.split()]
    hypothesis = llm_answer.split()
    smoothing = SmoothingFunction().method1
    return sentence_bleu(reference, hypothesis, smoothing_function=smoothing)

def compute_edit_distance_ratio(llm_answer, expected_answer):
    if not isinstance(llm_answer, str) or not isinstance(expected_answer, str):
        return 0.0
    return difflib.SequenceMatcher(None, llm_answer, expected_answer).ratio()

# ------------------ New Custom Metrics ------------------

# 1. Keyword Coverage Score (existing idea)
def compute_keyword_coverage(expected_answer, llm_answer):
    if not isinstance(expected_answer, str) or not isinstance(llm_answer, str):
        return 0.0
    tokens_expected = [word.lower() for word in nltk.word_tokenize(expected_answer)
                       if word.isalnum() and word.lower() not in stop_words]
    tokens_llm = [word.lower() for word in nltk.word_tokenize(llm_answer) if word.isalnum()]
    if not tokens_expected:
        return 0.0
    common_tokens = set(tokens_expected) & set(tokens_llm)
    return len(common_tokens) / len(set(tokens_expected))

# 2. TF-IDF Cosine Similarity (existing idea)
def compute_tfidf_cosine_similarity(expected_answer, llm_answer):
    if not isinstance(expected_answer, str) or not isinstance(llm_answer, str):
        return 0.0
    vectorizer = TfidfVectorizer().fit([expected_answer, llm_answer])
    vectors = vectorizer.transform([expected_answer, llm_answer])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

# 3. Lexical Diversity Similarity (existing idea)
def compute_lexical_diversity(answer):
    if not isinstance(answer, str):
        return 0.0
    tokens = [word.lower() for word in nltk.word_tokenize(answer) if word.isalnum()]
    return len(set(tokens)) / len(tokens) if tokens else 0.0

def compute_lexical_diversity_similarity(expected_answer, llm_answer):
    div_expected = compute_lexical_diversity(expected_answer)
    div_llm = compute_lexical_diversity(llm_answer)
    return 1 - abs(div_expected - div_llm)

# 4. Readability Similarity (existing idea using approximate Flesch Reading Ease)
def count_syllables(word):
    word = word.lower()
    syllables = re.findall(r'[aeiouy]+', word)
    count = len(syllables)
    if word.endswith("e") and count > 1:
        count -= 1
    return count if count > 0 else 1

def flesch_reading_ease(text):
    if not isinstance(text, str):
        return 0.0
    sentences = nltk.sent_tokenize(text)
    words = [word for word in nltk.word_tokenize(text) if word.isalnum()]
    if len(sentences) == 0 or len(words) == 0:
        return 0.0
    syllable_count = sum(count_syllables(word) for word in words)
    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = syllable_count / len(words)
    score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
    return score

def compute_readability_similarity(expected_answer, llm_answer):
    score_expected = flesch_reading_ease(expected_answer)
    score_llm = flesch_reading_ease(llm_answer)
    diff = abs(score_expected - score_llm)
    norm_diff = min(diff / 100, 1.0)
    return 1 - norm_diff

# ------------------ New Custom Metrics: Syntactic & Structural ------------------

# 5. POS Distribution Similarity
def compute_pos_distribution_similarity(expected_answer, llm_answer):
    if not isinstance(expected_answer, str) or not isinstance(llm_answer, str):
        return 0.0

    def pos_vector(text):
        doc = nlp(text)
        pos_counts = {}
        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        # Normalize counts to frequencies
        total = sum(pos_counts.values())
        for pos in pos_counts:
            pos_counts[pos] /= total
        return pos_counts

    vec1 = pos_vector(expected_answer)
    vec2 = pos_vector(llm_answer)
    # Create a combined list of POS tags
    pos_tags = set(vec1.keys()) | set(vec2.keys())
    vec1_list = [vec1.get(pos, 0) for pos in pos_tags]
    vec2_list = [vec2.get(pos, 0) for pos in pos_tags]
    # Compute cosine similarity between the frequency vectors
    numerator = sum(a*b for a, b in zip(vec1_list, vec2_list))
    denom1 = np.sqrt(sum(a*a for a in vec1_list))
    denom2 = np.sqrt(sum(b*b for b in vec2_list))
    if denom1 == 0 or denom2 == 0:
        return 0.0
    return numerator / (denom1 * denom2)

# 6. Noun Phrase Overlap Score
def compute_noun_phrase_overlap(expected_answer, llm_answer):
    if not isinstance(expected_answer, str) or not isinstance(llm_answer, str):
        return 0.0

    def extract_noun_phrases(text):
        doc = nlp(text)
        return {chunk.text.lower() for chunk in doc.noun_chunks}

    np_expected = extract_noun_phrases(expected_answer)
    np_llm = extract_noun_phrases(llm_answer)
    if not np_expected:
        return 0.0
    overlap = np_expected.intersection(np_llm)
    return len(overlap) / len(np_expected)

# ------------------ Evaluation Loop ------------------

results = []

for _, row in df.iterrows():
    question = str(row["question"]) if pd.notna(row["question"]) else ""
    expected_answer = str(row["expected_answer"]) if pd.notna(row["expected_answer"]) else ""
    llm_answer = str(row["llm_answer"]) if pd.notna(row["llm_answer"]) else ""

    # Existing metrics
    rouge_score_val = compute_rouge_l(llm_answer, expected_answer)
    bleu_score_val = compute_bleu_score(llm_answer, expected_answer)
    edit_distance_ratio = compute_edit_distance_ratio(llm_answer, expected_answer)

    # Previously defined custom metrics
    keyword_coverage = compute_keyword_coverage(expected_answer, llm_answer)
    tfidf_similarity = compute_tfidf_cosine_similarity(expected_answer, llm_answer)
    lexical_diversity_sim = compute_lexical_diversity_similarity(expected_answer, llm_answer)
    readability_sim = compute_readability_similarity(expected_answer, llm_answer)

    # New syntactic & structural metrics
    pos_distribution_sim = compute_pos_distribution_similarity(expected_answer, llm_answer)
    noun_phrase_overlap = compute_noun_phrase_overlap(expected_answer, llm_answer)

    results.append({
        "question": question,
        "expected_answer": expected_answer,
        "llm_answer": llm_answer,
        "rouge_l_score": rouge_score_val,
        "bleu_score": bleu_score_val,
        "normalized_edit_distance": edit_distance_ratio,
        "keyword_coverage": keyword_coverage,
        "tfidf_cosine_similarity": tfidf_similarity,
        "lexical_diversity_similarity": lexical_diversity_sim,
        "readability_similarity": readability_sim,
        "pos_distribution_similarity": pos_distribution_sim,
        "noun_phrase_overlap": noun_phrase_overlap
    })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("evaluated_output3.csv", index=False)

# Print aggregated final scores for a quick overview
print("🔹 Average ROUGE-L Score:", results_df["rouge_l_score"].mean())
print("🔹 Average BLEU Score:", results_df["bleu_score"].mean())
print("🔹 Average Normalized Edit Distance:", results_df["normalized_edit_distance"].mean())
print("🔹 Average Keyword Coverage:", results_df["keyword_coverage"].mean())
print("🔹 Average TF-IDF Cosine Similarity:", results_df["tfidf_cosine_similarity"].mean())
print("🔹 Average Lexical Diversity Similarity:", results_df["lexical_diversity_similarity"].mean())
print("🔹 Average Readability Similarity:", results_df["readability_similarity"].mean())
print("🔹 Average POS Distribution Similarity:", results_df["pos_distribution_similarity"].mean())
print("🔹 Average Noun Phrase Overlap:", results_df["noun_phrase_overlap"].mean())
