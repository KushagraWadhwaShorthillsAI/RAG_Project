import chromadb
from chromadb.config import Settings
import hashlib
from tqdm import tqdm
from core.config import Config
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from chromadb import PersistentClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json
# ChromaDB settings
chroma_settings = Settings(
    persist_directory="chroma_db",
    anonymized_telemetry=False
)
client = PersistentClient(path="chroma_db")

# Create or get the collection with the correct dimensionality (384 for MiniLM-L6)
collection = client.get_or_create_collection(
    name="rag_store",
    metadata={"hnsw:space": "cosine"},  # Optional: Set similarity metric
    embedding_function=None  # ChromaDB will infer dimensionality from embeddings
)

class IndexBuilder:
    def __init__(self, config):
        self.config = config

    def load_documents(self):
        """Loads documents from file and generates a single file hash."""
        input_file_path = f"{self.config.DATA_DIR}/python_docs.json"
        print(f"📂 Reading file: {input_file_path}")
        
        try:
            with open(input_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                print(f"✅ File read successfully.")
        except FileNotFoundError:
            print(f"❌ Error: The file '{input_file_path}' does not exist.")
            return [], ""
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return [], ""
        
        if not data:
            print("❌ Error: The file is empty.")
            return [], ""
        
        # Extract text and metadata from the JSON structure
        documents = []
        for section, content in data.items():
            for subsection, details in content.items():
                text = details.get("text", "")
                metadata = details.get("metadata", {})
                if text:
                    documents.append({
                        "text": text,
                        "metadata": metadata
                    })
        
        # Generate a file hash based on the JSON content
        file_hash = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        
        print(f"✅ Extracted {len(documents)} documents from JSON.")
        return documents, file_hash
    
   
    def build_indexes(self, documents, file_hash):
        """Indexes documents in ChromaDB and returns a ChromaVectorStore."""
        results = collection.get()
        stored_chunk_ids = set(results["ids"]) if "ids" in results else set()
        
        # Filter out chunks that are already indexed
        missing_chunks = [
            (i, doc) for i, doc in enumerate(documents) 
            if f"{file_hash}_{i}" not in stored_chunk_ids
        ]
        
        print(f"📊 Found {len(stored_chunk_ids)} stored chunks.")
        print(f"💾 {len(missing_chunks)} missing chunks will be processed.")
        
        if not missing_chunks:
            print("✅ All chunks are already stored. Proceeding to the next step.")
            return ChromaVectorStore(chroma_collection=collection)

        print("🧠 Generating embeddings and indexing...")
        for i, chunk in tqdm(missing_chunks, desc="Embedding Chunks", unit="chunk"):
            if not chunk["text"].strip():
                print(f"⚠️ Skipping empty chunk {i}.")
                continue
            
            try:
                # Generate embeddings using Hugging Face model (MiniLM-L6)
                embed_model = HuggingFaceEmbedding(model_name=self.config.EMBED_MODEL_NAME)
                embedding = embed_model.get_text_embedding(chunk["text"])
            except Exception as e:
                print(f"❌ Error generating embedding for chunk {i}: {e}")
                continue
            
            # Use a unique ID combining file_hash and chunk index
            chunk_id = f"{file_hash}_{i}"
            collection.add(
                documents=[chunk["text"]],
                embeddings=[embedding],
                metadatas=[{"chunk_id": i, "file_hash": file_hash, **chunk["metadata"]}],
                ids=[chunk_id]  # Unique ID for each chunk
            )

        print(f"✅ Indexed {len(missing_chunks)} new chunks into ChromaDB.")
        return ChromaVectorStore(chroma_collection=collection)