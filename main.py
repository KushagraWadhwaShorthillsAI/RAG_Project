import streamlit as st
import logging
from core.config import Config
from core.indexer import IndexBuilder
from core.retriever import RetrieverFactory
from core.engine import QueryEngineBuilder
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(filename="chatbot_logs.txt", level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

def initialize_system():
    """Initialize the system by setting up models, indexes, and the query engine."""
    print("\n🚀 Initializing System...\n")
    config = Config()
    models = config.initialize_models()
    
    if not models:
        st.error("❌ Failed to initialize models. Check API keys and settings.")
        return None

    with st.spinner("📄 Loading and indexing documents..."):
        index_builder = IndexBuilder(config)
        documents, file_hash = index_builder.load_documents()
        index = index_builder.build_indexes(documents, file_hash)

        if not index:
            st.error("❌ Failed to load embeddings from ChromaDB. Please index the documents first.")
            return None

    with st.spinner("🔍 Setting up Hybrid Retriever..."):
        retriever_factory = RetrieverFactory(config, index)
        retriever = retriever_factory.create_retrievers(keyword_index=None)

    with st.spinner("⚙️ Initializing Query Engine..."):
        engine_builder = QueryEngineBuilder(config)
        query_engine = engine_builder.build_engine(retriever)

    print("✅ System Ready!\n")
    return query_engine

def safe_query(query):
    """Wrapper for query execution to catch 429 errors and retry with a fallback API key."""
    try:
        response = st.session_state.query_engine.query(query)
        return response
    except Exception as e:
        error_message = str(e).lower()
        if "429" in error_message or "rate limit" in error_message:
            print("⚠️ 429 error encountered during query. Reinitializing with fallback API key...")
            st.session_state.query_engine = initialize_system()
            if st.session_state.query_engine:
                return st.session_state.query_engine.query(query)
        raise e

def log_interaction(query, response):
    """Logs chatbot interactions to a file."""
    logging.info(f"User Query: {query}")
    logging.info(f"Chatbot Response: {response}")
    logging.info("-" * 50)

def main():
    st.title("Chatbot")
    st.write("Ask me questions!")

    if 'query_engine' not in st.session_state:
        with st.spinner("⚡ Initializing system... Please wait..."):
            st.session_state.query_engine = initialize_system()
        
        if not st.session_state.query_engine:
            st.error("❌ Initialization failed. Please check logs.")
            return
        
        st.success("✅ System ready!")

    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("🔎 Processing your query..."):
            try:
                response = safe_query(query)
                
                if not response.source_nodes:
                    st.warning("⚠️ No relevant information found.")
                    return
                
                st.markdown("### 📖 Relevant Chunks Used:")
                for i, node in enumerate(response.source_nodes):
                    st.markdown(f"**Chunk {i+1}:**\n> {node.text[:150]}...")
                
                response_text = response.response
                st.write("📝 **Response:**")
                st.write(response_text)
                
                # Log the interaction
                log_interaction(query, response_text)
                
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")

if __name__ == "__main__":
    main()
