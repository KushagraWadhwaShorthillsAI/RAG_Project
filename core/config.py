import os
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini  # Keep Gemini for LLM
from llama_index.core import Settings

load_dotenv()

class Config:
    # Store multiple API keys from the .env file
    GEMINI_API_KEYS = [

        os.getenv("GEMINI_API_KEY3"),
    ]
    DATA_DIR = "data_main"
    CHUNK_SIZE = 512  # Optimized chunk size for better context preservation
    CHUNK_OVERLAP = 80
    SIMILARITY_TOP_K = 3  # Increased for better recall
    KEYWORD_TOP_K = 3
    MODEL_NAME = "models/gemini-1.5-flash"  # Keep Gemini for LLM
    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Hugging Face embedding model

    @classmethod
    def initialize_models(cls):
        """Initialize and return the LLM and embedding models using available API keys."""
        embed_model = HuggingFaceEmbedding(model_name=cls.EMBED_MODEL_NAME)
        
        # Iterate through available API keys
        for api_key in cls.GEMINI_API_KEYS:
            if not api_key:
                continue  # Skip if the key is not set
            try:
                # Attempt to initialize Gemini LLM with the current API key
                llm = Gemini(model_name=cls.MODEL_NAME, api_key=api_key)
                # If initialization is successful, set the global settings and return
                Settings.llm = llm
                Settings.embed_model = embed_model
                print(f"✅ Initialized Gemini LLM with API key: {api_key[:4]}...")  # Log partial key
                return {"llm": llm, "embed_model": embed_model}
            except Exception as e:
                error_message = str(e).lower()
                if "429" in error_message or "rate limit" in error_message:
                    print(f"⚠️ 429 Rate limit hit with API key: {api_key[:4]}... Trying next key...")
                else:
                    print(f"❌ Error initializing Gemini with API key: {api_key[:4]}... Error: {e}")
        
        print("❌ ERROR: All Gemini API keys failed. Please check your keys and rate limits.")
        return None
