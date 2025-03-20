from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate  # <-- Add this import

class QueryEngineBuilder:
    def __init__(self, config):
        self.config = config
        
    def build_engine(self, retriever):
        """Build the query engine with a custom prompt and response synthesizer."""
        # Define your custom prompt template
        custom_prompt = PromptTemplate("""\
        You are a helpful AI assistant. Answer the question using ONLY the provided context.
        Keep responses concise and factual. If unsure, say "I don't know".
        
        Context: {context_str}
        Question: {query_str}
        
        Answer:""")

        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            verbose=True,
            text_qa_template=custom_prompt  # <-- Add the custom prompt
        )
        
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
        )