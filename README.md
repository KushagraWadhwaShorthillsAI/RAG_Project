# Retrieval-Augmented Generation (RAG) Pipeline

## 1. Overview  
This RAG pipeline integrates document retrieval and generative language models to answer user queries using a knowledge base. The system:
- Ingests and indexes documents (stored in JSON format).
- Retrieves relevant text chunks using semantic and keyword search.
- Generates answers using a large language model (LLM) grounded in retrieved context.

---

## 2. Web Scraper for Python Documentation  

### Overview  
A custom web scraper is used to collect content from Python's official documentation (https://docs.python.org/3/). This scraper:
- Crawls through documentation sections and sub-pages.
- Extracts text content from HTML pages.
- Automatically follows nested links in documentation.
- Saves output to `python_docs.txt`.

### Key Components  

| File/Function        | Description |
|----------------------|-------------|
| `scrape_page()`     | Extracts text content from a single documentation page |
| `get_all_links()`   | Finds all internal documentation links from the table-of-contents (TOC) |
| `scrape_section()`  | Recursively scrapes a section and all its nested pages |
| `scrape_python_docs()` | Main function coordinating the scraping process |

### Technical Stack  

| Technology      | Purpose |
|---------------|---------|
| BeautifulSoup | HTML parsing and content extraction |
| Requests      | HTTP requests to fetch documentation pages |
| URLJoin       | Handles relative/absolute URL resolution |

### Workflow  
1. **Entry Point**: Starts from Python docs homepage (`BASE_URL = "https://docs.python.org/3/"`).
2. **Section Discovery**: Locates main documentation sections from the content table.
3. **Recursive Scraping**: Extracts content from main section pages and follows links.
4. **Content Filtering**: Skips irrelevant sections such as "What's New".

### Output Structure  
```
[Section Title 1]
[Content from main section page]
[Content from linked subpage 1]
[Content from linked subpage 2]

[Section Title 2]
[Content from main section page]
...
```

### Integration with RAG Pipeline  
The raw `python_docs.txt` is converted to structured JSON format for ingestion:
```json
{
  "Tutorial": {
    "introduction": {
      "text": "Python is an easy to learn...",
      "metadata": {
        "source": "Python Documentation",
        "url": "https://docs.python.org/3/tutorial/introduction.html"
      }
    }
  }
}
```

---

## 3. Technologies Used for RAG Pipeline  

| Technology        | Role |
|-----------------|------|
| ChromaDB        | Vector database for storing and querying document embeddings. |
| Google Gemini   | Generates text embeddings and powers the LLM (`gemini-1.5-flash`). |
| LlamaIndex      | Orchestrates document indexing, retrieval, and response synthesis. |
| Streamlit       | Web UI for user interaction and query input. |
| Python Libraries | `chromadb`, `llama-index-core`, `google-generativeai`, `streamlit`, `tqdm`, `sklearn`. |

---

## 4. Project Structure  
```
.
├── core/                 # Pipeline logic
│   ├── config.py        🛠️ Model/API configurations
│   ├── indexer.py       📚 Document processing
│   ├── retriever.py     🔍 Hybrid search logic
│   ├── engine.py        🤖 Response generation
│   └── __init__.py      📦 Package definition
├── main.py              🖥️ Streamlit UI
├── scraper/             🕸️ Web scraping utilities
│   └── docs_scraper.py  🕷️ Python docs crawler
├── data_main/           📁 Input documents
│   └── python_docs.json 🗄️ Processed documentation
└── chroma_db/           🧠 Vector database storage
└── eval_custom.py       📊 Evaluates RAG pipeline output
└── chatbot_logs.txt       Log files
└── evaluated_output.csv   Final evaluated csv sheet
```

---

## 5. Workflow Pipeline  
```
+---------------+
|  Python Docs  |
+-------+-------+
        |
+-------v-------+
|  Web Scraper  |  (BeautifulSoup/Requests)
+-------+-------+
        |
+-------v-------+
|  JSON Storage |  (data_main/)
+-------+-------+
        |
+-------v-------+
|  Index Builder|  (ChromaDB + Gemini)
+-------+-------+
        |
+-------v-------+
| Hybrid Search |  (LlamaIndex)
+-------+-------+
        |
+-------v-------+
|  Query Engine |  (Gemini-1.5-Flash)
+-------+-------+
        |
+-------v-------+
| Streamlit UI  |  (User Interface)
+---------------+
```

---

## 6. Evaluation  

### Synthetic Data Generation Using Mistral-Instruct LLM  
- **Objective**: Generate synthetic questions for evaluating RAG.
- **Methodology**:
  - Generate factoid questions from documentation.
  - Evaluate groundedness and relevance.
  - Filter out low-quality questions.

#

**Prompt for QA Generation**:

```
QA_generation_prompt = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""

```

### Scoring Mechanism

| Metric             | Description                                                                   |
| ------------------ | ----------------------------------------------------------------------------- |
| Groundedness Score | Measures how well the generated question is grounded in the provided context. |
| Relevance Score    | Assesses how relevant the question is to the topic.                           |

**Evaluation Prompts for Scoring**:

```
question_groundedness_critique_prompt = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """

question_relevance_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to python developers building python applications.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """

```


### Performance Metrics  
1. **Retrieval Relevance Score (Cosine Similarity)** – Measures how relevant the retrieved chunks are to the query.
2. **Answer Correctness Score (F1 Score)** – Compares the LLM's answer with the expected answer.
3. **Hallucination Detection** – Flags if the LLM generates information not found in the retrieved chunks.

Example Calculation:
```python
def compute_retrieval_relevance(query, retrieved_chunks):
    # Uses BERT to compute embeddings
    # Applies cosine similarity
```
#

### Evaluation Metrics

1. **Retrieval Relevance Score (Cosine Similarity)** – Measures how relevant the retrieved chunks are to the query.
2. **Answer Correctness Score (F1 Score)** – Compares the LLM's answer with the expected answer.
3. **Hallucination Detection** – Flags if the LLM generates information not found in the retrieved chunks.

### Code Implementation

- **Retrieval Relevance**: Uses BERT embeddings and cosine similarity to compare user query with retrieved context.
- **Answer Correctness**: Calculates F1-score based on word overlap between the expected and generated answer.
- **Hallucination Detection**: Compares generated answers with retrieved chunks, flagging extra words.

```python
def compute_retrieval_relevance(query, retrieved_chunks):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    query_embedding = model(**tokenizer(query, return_tensors="pt")).last_hidden_state.mean(dim=1)
    context_embedding = model(**tokenizer(retrieved_chunks, return_tensors="pt")).last_hidden_state.mean(dim=1)
    return cosine_similarity(query_embedding.detach().numpy(), context_embedding.detach().numpy())[0][0]
```

```python
def compute_answer_correctness(llm_answer, expected_answer):
    llm_tokens = set(llm_answer.lower().split())
    expected_tokens = set(expected_answer.lower().split())
    common_tokens = llm_tokens.intersection(expected_tokens)
    precision = len(common_tokens) / len(llm_tokens) if llm_tokens else 0
    recall = len(common_tokens) / len(expected_tokens) if expected_tokens else 0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
```

```python
def detect_hallucination(llm_answer, retrieved_chunks):
    return len(set(llm_answer.lower().split()) - set(retrieved_chunks.lower().split())) > 0
```


## 7. Configuration Details  
```python
class Config:
    GEMINI_API_KEY = "your-api-key"  # From .env
    DATA_DIR = "data_main"  # Input JSON directory
    CHUNK_SIZE = 1000  # Tokens per chunk
    SIMILARITY_TOP_K = 3  # Top chunks for vector search
    MODEL_NAME = "models/gemini-1.5-flash"
```

---

## 8. Future Enhancements  
- Support for more document sources.
- Advanced filtering techniques for improved retrieval.
- Optimized embeddings using fine-tuned Gemini models.

---
