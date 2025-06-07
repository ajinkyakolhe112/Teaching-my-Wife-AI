"""
Simple RAG Implementation using LlamaIndex
"""

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    HuggingFaceEmbedding,
    SimpleNodeParser
)
from datasets import load_dataset
import os
import shutil

class LlamaIndexRAG:
    def __init__(self, num_books=100):
        # Initialize components
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,
            node_parser=self.node_parser
        )
        
        # Load and process data
        self.dataset = load_dataset("manu/project_gutenberg", split="train")
        self._create_index(num_books)
    
    def _create_index(self, num_books):
        """Create vector index from books"""
        # Save books temporarily
        os.makedirs("temp_docs", exist_ok=True)
        for i, book in enumerate(self.dataset[:num_books]):
            with open(f"temp_docs/book_{i}.txt", "w") as f:
                f.write(book['text'])
        
        # Create and store index
        documents = SimpleDirectoryReader("temp_docs").load_data()
        self.index = VectorStoreIndex.from_documents(
            documents,
            service_context=self.service_context
        )
        self.query_engine = self.index.as_query_engine(similarity_top_k=3)
        
        # Cleanup
        shutil.rmtree("temp_docs")
    
    def answer_question(self, question):
        """Answer a question using RAG"""
        response = self.query_engine.query(question)
        
        # Print relevant passages
        print("\nMost relevant passages:")
        for i, node in enumerate(response.source_nodes, 1):
            print(f"\n--- Passage {i} ---")
            print(node.node.text[:500] + "...")
        
        return response

def main():
    rag = LlamaIndexRAG(num_books=100)
    
    questions = [
        "What is the meaning of life?",
        "Tell me about love and romance",
        "What is the role of nature in human existence?",
        "How do people deal with loss and grief?"
    ]
    
    for question in questions:
        rag.answer_question(question)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 