from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

class QueryEngineBuilder:
    def __init__(self, config):
        self.config = config
        
    def build_engine(self, retriever):
        """Build the query engine with a response synthesizer."""
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",  
            verbose=True  # Enable verbose logging for debugging
        )
        
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],  
        )
