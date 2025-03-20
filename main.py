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
    st.toast("🚀 Initializing System...")
    config = Config()
    models = config.initialize_models()
    
    if not models:
        st.error("❌ Failed to initialize models. Check API keys and settings.")
        return None

    with st.spinner("📄 Loading and indexing documents... This may take a while."):
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

    st.toast("✅ System Ready!")
    return query_engine

def safe_query(query):
    try:
        response = st.session_state.query_engine.query(query)
        return response
    except Exception as e:
        error_message = str(e).lower()
        if "429" in error_message or "rate limit" in error_message:
            st.warning("⚠️ Rate limit hit. Retrying with fallback API key...")
            st.session_state.query_engine = initialize_system()
            if st.session_state.query_engine:
                return st.session_state.query_engine.query(query)
        raise e

def log_interaction(query, response):
    logging.info(f"User Query: {query}")
    logging.info(f"Chatbot Response: {response}")
    logging.info("-" * 50)
    
    # Store interactions in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"question": query, "answer": response})

def main():
    st.set_page_config(page_title="Smart Chatbot", page_icon="🤖")
    st.title("💬 AI-Powered Chatbot")
    st.write("Ask me anything! I'm here to help.")

    if 'query_engine' not in st.session_state:
        with st.spinner("⚡ Initializing system... Please wait."):
            st.session_state.query_engine = initialize_system()
        
        if not st.session_state.query_engine:
            st.error("❌ Initialization failed. Please check logs.")
            return
        
        st.success("✅ System ready!")

    st.divider()
    query = st.text_input("💡 Enter your question:")
    if query:
        with st.spinner("🔎 Processing your query..."):
            try:
                response = safe_query(query)

                if not response.source_nodes:
                    st.warning("⚠️ No relevant information found.")
                    return

                response_text = response.response
                st.write("📝 **Response:**")
                st.success(response_text)
                
                log_interaction(query, response_text)

            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
    
    # Display chat history
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.divider()
        st.write("🗂️ **Chat History:**")
        for chat in st.session_state.chat_history[::-1]:
            st.write(f"**Q:** {chat['question']}")
            st.write(f"**A:** {chat['answer']}")
            st.divider()

if __name__ == "__main__":
    main()
