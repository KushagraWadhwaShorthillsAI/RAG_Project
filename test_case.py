import unittest
from unittest.mock import MagicMock, patch
from core.retriever import HybridRetriever, RetrieverFactory
from core.indexer import IndexBuilder
from core.engine import QueryEngineBuilder
from core.config import Config
from llama_index.core.retrievers import VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
import json
import os

class TestRetriever(unittest.TestCase):
    @patch("streamlit.spinner")
    def test_hybrid_retriever(self, mock_spinner):
        vector_retriever = MagicMock(VectorIndexRetriever)
        keyword_retriever = MagicMock(KeywordTableSimpleRetriever)
        retriever = HybridRetriever(vector_retriever, keyword_retriever)

        query_bundle = MagicMock()
        query_bundle.query_str = "Test Query"

        vector_retriever.retrieve.return_value = [MagicMock()]
        keyword_retriever.retrieve.return_value = [MagicMock()]

        results = retriever._retrieve(query_bundle)
        self.assertGreater(len(results), 0)

class TestIndexer(unittest.TestCase):
    @patch("core.indexer.collection")
    def test_index_builder_load_documents(self, mock_collection):
        config = Config()
        index_builder = IndexBuilder(config)
        
        with patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps({"Section": {"Sub": {"text": "Test"}}}))):
            documents, file_hash = index_builder.load_documents()
        
        self.assertEqual(len(documents), 1)
        self.assertIsInstance(file_hash, str)

class TestQueryEngine(unittest.TestCase):
    def test_query_engine_builder(self):
        config = Config()
        retriever = MagicMock()
        engine_builder = QueryEngineBuilder(config)
        query_engine = engine_builder.build_engine(retriever)

        self.assertIsNotNone(query_engine)

class TestConfig(unittest.TestCase):
    @patch("core.config.Gemini")
    @patch("core.config.HuggingFaceEmbedding")
    def test_initialize_models(self, mock_hf_embedding, mock_gemini):
        mock_hf_embedding.return_value = MagicMock()
        mock_gemini.return_value = MagicMock()
        models = Config.initialize_models()
        self.assertIsNotNone(models)

if __name__ == "__main__":
    unittest.main()
