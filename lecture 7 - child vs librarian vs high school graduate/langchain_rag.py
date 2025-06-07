"""
RAG Implementation using LangChain
This demonstrates how to build a RAG system using LangChain's components
"""

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from datasets import load_dataset
import os

class LangChainRAG:
    def __init__(self):
        print("Initializing LangChain RAG system...")
        
        # Initialize the embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Load dataset
        print("Loading Project Gutenberg dataset...")
        self.dataset = load_dataset("manu/project_gutenberg", split="train")
        
        # Process and index the documents
        self._process_documents()
        
    def _process_documents(self):
        """Process documents and create vector store"""
        print("Processing documents and creating vector store...")
        
        # Process first 100 books
        texts = []
        for book in self.dataset[:100]:
            texts.append(book['text'])
        
        # Split texts into chunks
        print("Splitting texts into chunks...")
        chunks = self.text_splitter.create_documents(texts)
        
        # Create vector store
        print("Creating vector store...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        print("Vector store created successfully!")
    
    def answer_question(self, question):
        """
        Answer a question using LangChain's RAG
        
        Args:
            question (str): The question to answer
            
        Returns:
            str: Answer to the question
        """
        print(f"\nQuestion: {question}")
        
        # Get relevant documents
        docs = self.retriever.get_relevant_documents(question)
        
        print("\nMost relevant passages:")
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Passage {i} ---")
            print(doc.page_content[:500] + "...")
        
        return docs

def main():
    # Initialize RAG system
    rag = LangChainRAG()
    
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