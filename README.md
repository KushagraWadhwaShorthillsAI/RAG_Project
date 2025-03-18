**Retrieval-Augmented Generation (RAG) Pipeline**

1. Overview 
This RAG pipeline integrates document retrieval and generative language models to answer user queries using a knowledge base. The system:
Ingests and indexes documents (stored in JSON format).
Retrieves relevant text chunks using semantic and keyword search.
Generates answers using a large language model (LLM) grounded in retrieved context.



2. Web Scraper for Python Documentation 

Overview
A custom web scraper is used to collect content from Python's official documentation (https://docs.python.org/3/). This scraper:
Crawls through documentation sections and sub-pages
Extracts text content from HTML pages
Automatically follows nested links in documentation
Saves output to python_docs.txt




Key Components


File/Function	Description	
`scrape_page()`	Extracts text content from a single documentation page	
`get_all_links()`	Finds all internal documentation links from table-of-contents (TOC)	
`scrape_section()`	Recursively scrapes a section and all its nested pages	
`scrape_python_docs()`	Main function coordinating the scraping process	




Technical Stack


Technology	Purpose	
BeautifulSoup	HTML parsing and content extraction	
Requests	HTTP requests to fetch documentation pages	
URLJoin	Handles relative/absolute URL resolution	




Workflow
Entry Point 
Starts from Python docs homepage (`BASE_URL = "https://docs.python.org/3/"`)

Section Discovery
   table = soup.find('table', {'class': 'contentstable'})


Locates main documentation sections from content table

Recursive Scraping 
For each section:
Scrapes main section page
Follows all toctree-wrapper links to sub-pages
Combines content hierarchically

Content Filtering
   if "What's new" in a_tag.get_text(): continue


Skips volatile "What's New" sections



Output Structure
[Section Title 1]
[Content from main section page]
[Content from linked subpage 1]
[Content from linked subpage 2]

[Section Title 2]
[Content from main section page]
...




Integration with RAG Pipeline
Text Conversion 
The raw python_docs.txt is converted to structured JSON format for ingestion:
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



2. Technologies Used for RAG pipeline

Technology	Role	
ChromaDB	Vector database for storing and querying document embeddings.	
Google Gemini	Generates text embeddings (via text-embedding-004) and powers the LLM (`gemini-1.5-pro`).	
LlamaIndex	Orchestrates document indexing, retrieval, and response synthesis.	
Streamlit	Web UI for user interaction and query input.	
Python Libraries	chromadb, llama-index-core, google-generativeai, streamlit, tqdm, sklearn.	



3. Project Structure
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
└── eval_custom.py        To evaluate the final LLM answers and retrieved chunks based on                                      evaluation metrics defined on the Goldent dataset



1. config.py 
Purpose: Central configuration management  
Technologies:
python-dotenv: Loads environment variables
llama-index-core: For global LLM/embedding settings

Key Components:
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Load from .env
    DATA_DIR = "data_main"  # Directory for input JSON
    CHUNK_SIZE = 1000  # Character count per chunk
    EMBED_MODEL_NAME = "models/text-embedding-004"  # Gemini embedding model




2. indexer.py 
Purpose: Document processing and vector storage  
Technologies:
chromadb: Vector database operations
google.generativeai: Text embedding generation
hashlib: File versioning via MD5 hashes
tqdm: Progress tracking 

Workflow:
load_documents(): Reads JSON input with metadata
split_text_into_chunks(): Creates 1000-char chunks with 50-char overlap
build_indexes(): Stores embeddings in ChromaDB
collection.add(
    documents=[chunk_text],
    embeddings=[embedding],
    metadatas=[metadata],
    ids=[unique_id]
)





3. retriever.py 
Purpose: Hybrid document retrieval  
Technologies:
llama-index-core: Base retriever classes
streamlit: UI status updates 

Key Classes:
class HybridRetriever(BaseRetriever):
    def _retrieve():
        # Combines vector + keyword results
        vector_nodes = vector_retriever.retrieve()
        keyword_nodes = keyword_retriever.retrieve()




4. engine.py 
Purpose: Response generation engine  
Technologies:
llama-index-core: Query engine orchestration
llama-index-postprocessor: Similarity filtering 

Configuration:
return RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=get_response_synthesizer(
        response_mode="tree_summarize"
    ),
    node_postprocessors=[SimilarityPostprocessor(0.7)]
)




5. main.py 
Purpose: Streamlit UI and workflow control  
Technologies:
streamlit: Web interface
session_state: Preserve query engine between reloads 

Key Features:
# Initialization
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = initialize_system()

# Query handling
response = st.session_state.query_engine.query(user_input)
st.write(response.response)




6. Web Scraper Script 
Purpose: Documentation content extraction  
Technologies:
BeautifulSoup: HTML parsing
requests: Page fetching
urllib.parse: URL resolution 

Key Functions:
def scrape_page(url):  # Extracts text from <div class="body">
def get_all_links():  # Finds TOC links in <div class="toctree-wrapper">
def scrape_python_docs():  # Coordinates full scraping workflow




7. init.py 
Purpose: Package initialization  
Content:
from .indexer import IndexBuilder
from .retriever import RetrieverFactory
from .engine import QueryEngineBuilder
from .config import Config




4. Workflow Pipeline
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
          |  Query Engine |  (Gemini-1.5-Pro)
          +-------+-------+
                  |
          +-------v-------+
          | Streamlit UI  |  (User Interface)
          +---------------+




Key Data Flow
Scraping: HTML → Clean Text → JSON
Indexing: JSON → Chunks → Gemini Embeddings → ChromaDB
Query: User Input → Hybrid Retrieval → Context → LLM → Response



6. Configuration Details 
`config.py`  
class Config:
    GEMINI_API_KEY = "your-api-key"     # From .env
    DATA_DIR = "data_main"              # Input JSON directory
    CHUNK_SIZE = 1000                   # Tokens per chunk
    CHUNK_OVERLAP = 50                  # Overlap between chunks
    SIMILARITY_TOP_K = 3                # Top chunks for vector search
    KEYWORD_TOP_K = 3                   # Top chunks for keyword search
    MODEL_NAME = "models/gemini-1.5-pro"
    EMBED_MODEL_NAME = "models/text-embedding-004"




7. Evaluation 

-> Synthetic Data Generation
Generating and Evaluating Synthetic Data Using Mistral-Instruct LLM
Introduction
The generation of high-quality synthetic data is a crucial step in improving the performance of Retrieval-Augmented Generation (RAG) systems. In this study, we utilized the Mistral-Instruct Large Language Model (LLM) to generate synthetic questions based on a given set of prompts. The generated questions were then evaluated for their groundedness and relevance, ensuring that only high-quality questions were retained for further use in our system.
Methodology
1. Data Generation Process
We prompted the Mistral-Instruct LLM with a set of structured prompts designed to produce high-quality questions that align with the context of our dataset. The prompts encouraged the model to generate diverse and meaningful questions that could be used for retrieval and evaluation purposes. Along with question and expected_answers context was also stored which the LLM used to create the q/a pair.
This was the prompt used to generate the question answer pairs from the extracted content from Python.org documentation.
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




2. Scoring Mechanism
To filter out low-quality questions, we implemented a scoring mechanism based on two key metrics:
Groundedness Score: Measures how well the generated question is grounded in the provided context. 
Relevance Score: Assesses how relevant the question is to the overall topic. 
These prompts were used to judge the questions created by "Mistral Instruct" for Synthetic Dataset.
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




The LLM was further prompted to assign scores to each generated question on a scale from 1 to 5 for both groundedness and relevance. A higher score indicates stronger alignment with the context and higher quality.
3. Filtering Criteria
To ensure that only the most relevant and contextually accurate questions were retained, we applied the following filtering criteria:
Questions with a groundedness score > 4 were removed. 
Questions with a relevance score > 4 were also filtered out. 
This threshold was set to eliminate questions that might be too generic, ungrounded, or lacking direct relevance to our dataset. By enforcing this filtering step, we ensured that the final set of questions used for evaluation would be meaningful and well-aligned with the underlying content.

A script was created to generate retrieved_chunks and llm_answer for each of the question present in Synthetic golden dataset and then they were evaluated on 3 metrics that are as follows: -

Retrieval Relevance Score (Cosine Similarity) – Measures how relevant the retrieved_chunks are to the query. 
Answer Correctness Score (F1 Score) – Compares the LLM's answer with the expected answer. 
Hallucination Detection – Flags if the LLM generates information not found in the retrieved_chunks. 
Breakdown of the Code:
Step 1: Load Data
df = pd.read_csv("output_answers.csv")
df = df.head(10)


Reads an evaluation dataset from output_answers.csv. 
Keeps only the first 10 rows for testing. 
The dataset likely contains columns: 
"question" (user query) 
"retrieved_chunks" (retrieved context from RAG) 
"llm_answer" (LLM's response) 
"expected_answer" (ground truth answer)   
Step 2: Compute Retrieval Relevance using Cosine Similarity
def compute_retrieval_relevance(query, retrieved_chunks):


Uses BERT to compute embeddings for the query and retrieved chunks. 
Applies Cosine Similarity to measure how well the retrieved context matches the query. 
Returns a similarity score between 0 and 1 (higher is better). 
Step 3: Compute Answer Correctness using F1 Score
def compute_answer_correctness(llm_answer, expected_answer):


Tokenizes the LLM answer and expected answer into sets of words. 
Computes Precision, Recall, and F1 Score to measure answer correctness. 
Returns an F1 score between 0 and 1 (higher is better). 
Step 4: Detect Hallucinations
def detect_hallucination(llm_answer, retrieved_chunks):


Compares the LLM answer and retrieved context at the token level. 
Flags True if the LLM generates words not found in the retrieved context. 
Step 5: Process Each Question and Store Results
for _, row in df.iterrows():


Loops through each row in the dataset. 
Calls the three evaluation functions for each query. 
Stores the results in a list. 
Step 6: Save Evaluations to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("evaluated_output.csv", index=False)


Converts results into a DataFrame. 
Saves the evaluation metrics to evaluated_output.csv. 
Step 7: Aggregate Final Scores
avg_retrieval_score = results_df["retrieval_relevance_score"].mean()
avg_correctness_score = results_df["answer_correctness_score"].mean()
hallucination_rate = results_df["hallucination_flag"].mean()


Computes the average retrieval relevance and correctness scores. 
Calculates the hallucination rate (percentage of flagged answers). 
This is how the final csv sheet looks like: -


Final Output:
Average Retrieval Relevance Score: 0.xx
Average Answer Correctness Score: 0.xx
Hallucination Rate: 0.xx




Potential Issues & Improvements
✅ Optimization – Avoid loading the model inside the function (move it outside for efficiency).
✅ Hallucination Detection – Could use NER matching or TF-IDF instead of simple token matching.
✅ Cosine Similarity – Consider using SBERT (better for sentence embeddings than BERT).
This script is useful for evaluating the quality of a RAG pipeline, ensuring it retrieves relevant data, generates accurate answers, and avoids hallucinations. 🚀


