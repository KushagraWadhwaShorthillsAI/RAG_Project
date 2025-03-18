from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
import streamlit as st

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, keyword_retriever=None):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever

    def _retrieve(self, query_bundle):
        print(f"\n🔎 Running Hybrid Retrieval for query: '{query_bundle.query_str}'")

        # Retrieve from ChromaDB (Semantic Retrieval)
        with st.spinner("🔍 Retrieving results from ChromaDB..."):
            vector_nodes = self.vector_retriever.retrieve(query_bundle) if self.vector_retriever else []

        # Retrieve from Keyword Table (Keyword Retrieval)
        keyword_nodes = []
        if self.keyword_retriever:
            with st.spinner("🔍 Retrieving results from Keyword Search..."):
                keyword_nodes = self.keyword_retriever.retrieve(query_bundle)

        # Log retrieved chunks
        print("\n📄 Retrieved Chunks from ChromaDB:")
        for i, node in enumerate(vector_nodes):
            print(f"🔹 Chunk {i+1}: {node.node.text[:300]}...")  # Log first 300 characters

        if self.keyword_retriever:
            print("\n📄 Retrieved Chunks from Keyword Search:")
            for i, node in enumerate(keyword_nodes):
                print(f"🔹 Chunk {i+1}: {node.node.text[:300]}...")  # Log first 300 characters

        # Merge results
        combined_results = {node.node.node_id: node for node in vector_nodes}
        for node in keyword_nodes:
            if node.node.node_id not in combined_results:
                combined_results[node.node.node_id] = node

        return list(combined_results.values())

class RetrieverFactory:
    def __init__(self, config, index_store):
        self.config = config
        self.index_store = index_store  

    def create_retrievers(self, keyword_index=None):
        """Creates vector and keyword-based retrievers"""
        storage_context = StorageContext.from_defaults(vector_store=self.index_store)
        vector_index = VectorStoreIndex.from_vector_store(storage_context.vector_store)

        vector_retriever = VectorIndexRetriever(
            index=vector_index,  
            similarity_top_k=self.config.SIMILARITY_TOP_K,
            verbose=True
        )
        
        keyword_retriever = None
        if keyword_index:
            keyword_retriever = KeywordTableSimpleRetriever(
                index=keyword_index,
                top_k=self.config.KEYWORD_TOP_K
            )
        
        return HybridRetriever(vector_retriever, keyword_retriever)
