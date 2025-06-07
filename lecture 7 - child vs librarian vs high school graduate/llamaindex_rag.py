"""
RAG Implementation using LlamaIndex
This demonstrates how to build a RAG system using LlamaIndex's components
"""

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.node_parser import SimpleNodeParser
from datasets import load_dataset
import os
import json

class LlamaIndexRAG:
    def __init__(self):
        print("Initializing LlamaIndex RAG system...")
        
        # Initialize the embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize node parser
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Create service context
        self.service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,
            node_parser=self.node_parser
        )
        
        # Load dataset
        print("Loading Project Gutenberg dataset...")
        self.dataset = load_dataset("manu/project_gutenberg", split="train")
        
        # Process and index the documents
        self._process_documents()
        
    def _process_documents(self):
        """Process documents and create index"""
        print("Processing documents and creating index...")
        
        # Create a temporary directory for documents
        os.makedirs("temp_docs", exist_ok=True)
        
        # Save first 100 books as text files
        print("Saving books as text files...")
        for i, book in enumerate(self.dataset[:100]):
            with open(f"temp_docs/book_{i}.txt", "w") as f:
                f.write(book['text'])
        
        # Load documents
        print("Loading documents into LlamaIndex...")
        documents = SimpleDirectoryReader("temp_docs").load_data()
        
        # Create index
        print("Creating vector index...")
        self.index = VectorStoreIndex.from_documents(
            documents,
            service_context=self.service_context
        )
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3
        )
        
        print("Index created successfully!")
        
        # Clean up temporary files
        import shutil
        shutil.rmtree("temp_docs")
    
    def answer_question(self, question):
        """
        Answer a question using LlamaIndex's RAG
        
        Args:
            question (str): The question to answer
            
        Returns:
            str: Answer to the question
        """
        print(f"\nQuestion: {question}")
        
        # Get response
        response = self.query_engine.query(question)
        
        # Print source nodes
        print("\nMost relevant passages:")
        for i, node in enumerate(response.source_nodes, 1):
            print(f"\n--- Passage {i} ---")
            print(node.node.text[:500] + "...")
        
        return response

def main():
    # Initialize RAG system
    rag = LlamaIndexRAG()
    
    # Example questions
    questions = [
        "What is the meaning of life?",
        "Tell me about love and romance",
        "What is the role of nature in human existence?",
        "How do people deal with loss and grief?"
    ]
    
    # Answer each question
    for question in questions:
        rag.answer_question(question)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 