import argparse
import pandas as pd
import time
import logging
from tenacity import retry, wait_exponential, stop_after_attempt
from core.config import Config
from core.indexer import IndexBuilder
from core.retriever import RetrieverFactory
from core.engine import QueryEngineBuilder

# Configure logging
logging.basicConfig(
    filename="batch_processor.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class BatchProcessor:
    def __init__(self):
        self.config = Config()
        self.query_engine = None
        self.last_query_time = 0
        self.request_delay = 1.2  # 1.2 seconds between requests
        self.current_key_index = 0  # Track current API key

    def initialize_system(self):
        """Initialize RAG system with API key rotation"""
        try:
            # Rotate to the next API key
            self.current_key_index = (self.current_key_index + 1) % len(self.config.GEMINI_API_KEYS)
            api_key = self.config.GEMINI_API_KEYS[self.current_key_index]
            
            if not api_key:
                logging.error("No valid API key found")
                return False

            # Reinitialize models with the new API key
            models = self.config.initialize_models()
            if not models:
                raise ValueError("Failed to initialize models")

            index_builder = IndexBuilder(self.config)
            documents, file_hash = index_builder.load_documents()
            index = index_builder.build_indexes(documents, file_hash)

            retriever_factory = RetrieverFactory(self.config, index)
            retriever = retriever_factory.create_retrievers(keyword_index=None)

            engine_builder = QueryEngineBuilder(self.config)
            self.query_engine = engine_builder.build_engine(retriever)
            
            logging.info(f"System initialized with API key: {api_key[:4]}...")
            return True
            
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            return False

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), 
           stop=stop_after_attempt(3))
    def process_question(self, question):
        """Process a single question with retry logic"""
        try:
            # Rate limiting
            elapsed = time.time() - self.last_query_time
            if elapsed < self.request_delay:
                time.sleep(self.request_delay - elapsed)
            
            response = self.query_engine.query(question)
            self.last_query_time = time.time()
            
            return response.response
            
        except Exception as e:
            if "429" in str(e):  # API quota exceeded
                logging.warning("API quota exceeded - rotating keys")
                if not self.initialize_system():  # Rotate API key and reinitialize
                    raise RuntimeError("All API keys exhausted")
            raise e

    def process_csv(self, input_file, output_file,start_row=0):
        """Process CSV file and generate answers"""
        try:
            df = pd.read_csv(input_file,delimiter="|")
            
            # Add LLM_answer column if not exists
            if 'LLM_answer' not in df.columns:
                df['LLM_answer'] = None
                
            total = len(df)
            processed = 0
            
            for index, row in df.iterrows():
                if index<start_row:
                    continue
                if pd.isnull(row['LLM_answer']) and pd.notna(row['question']):
                    try:
                        answer = self.process_question(row['question'])
                        df.at[index, 'LLM_answer'] = answer
                        processed += 1
                        
                        # Save progress every 10 rows
                        if processed % 10 == 0:
                            df.to_csv(output_file, index=False)
                            logging.info(f"Progress: {processed}/{total} rows processed")
                            
                    except Exception as e:
                        df.at[index, 'LLM_answer'] = f"Error: {str(e)[:200]}"
                        logging.error(f"Failed on row {index}: {str(e)}")
                        
            # Final save
            df.to_csv(output_file, index=False)
            logging.info("Processing completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Fatal error: {str(e)}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV with RAG answers')
    parser.add_argument('-i', '--input', required=True, help='Input CSV file path')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file path')
    
    args = parser.parse_args()
    
    processor = BatchProcessor()
    
    if processor.initialize_system():
        print(f"Starting processing on {args.input}")
        success = processor.process_csv(args.input, args.output)
        
        if success:
            print(f"Processing completed. Output saved to {args.output}")
            logging.info("Batch processing completed successfully")
        else:
            print("Processing failed with errors. Check batch_processor.log")
    else:
        print("System initialization failed. Check logs and API keys")