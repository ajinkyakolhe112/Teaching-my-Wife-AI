"""
Simple RAG Implementation using Project Gutenberg Dataset
This demonstrates the library analogy where we:
1. Load books (our library)
2. Create embeddings (index the library)
3. Answer questions by retrieving relevant passages
"""

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleRAG:
    def __init__(self):
        print("Initializing RAG system (like setting up a library)...")
        # Load the pre-trained model for embeddings (like having a librarian who can understand books)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load Project Gutenberg dataset (like getting all the books for our library)
        print("Loading Project Gutenberg dataset (like stocking our library with books)...")
        self.dataset = load_dataset("manu/project_gutenberg", split="train")
        
        # Store book passages and their embeddings
        self.passages = []
        self.embeddings = []
        
        # Process books in chunks (like organizing books on shelves)
        self._process_books()
        
    def _process_books(self):
        """Process books and create embeddings (like indexing the library)"""
        print("Processing books and creating embeddings (like indexing the library)...")
        
        # Process first 100 books for demonstration
        for book in self.dataset[:100]:
            # Split book into passages (like breaking books into manageable sections)
            text = book['text']
            # Split into passages of roughly 200 words
            words = text.split()
            passages = [' '.join(words[i:i+200]) for i in range(0, len(words), 200)]
            
            # Create embeddings for each passage
            passage_embeddings = self.encoder.encode(passages, show_progress_bar=False)
            
            # Store passages and their embeddings
            self.passages.extend(passages)
            self.embeddings.extend(passage_embeddings)
            
        # Convert to numpy array for easier computation
        self.embeddings = np.array(self.embeddings)
        print(f"Library ready! Indexed {len(self.passages)} passages from 100 books.")
    
    def answer_question(self, question, top_k=3):
        """
        Answer a question using RAG (like a librarian finding relevant information)
        
        Args:
            question (str): The question to answer
            top_k (int): Number of most relevant passages to retrieve
            
        Returns:
            list: Most relevant passages that might contain the answer
        """
        print(f"\nQuestion: {question}")
        
        # Create embedding for the question
        question_embedding = self.encoder.encode([question])[0]
        
        # Calculate similarity between question and all passages
        similarities = cosine_similarity([question_embedding], self.embeddings)[0]
        
        # Get top-k most similar passages
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        print("\nMost relevant passages from our library:")
        for idx in top_indices:
            print(f"\n--- Passage (Similarity: {similarities[idx]:.3f}) ---")
            print(self.passages[idx][:500] + "...")  # Show first 500 chars of each passage
            
        return [self.passages[idx] for idx in top_indices]

def main():
    # Initialize our RAG system
    rag = SimpleRAG()
    
    # Example questions to demonstrate the system
    example_questions = [
        "What is the meaning of life?",
        "Tell me about love and romance",
        "What is the role of nature in human existence?",
        "How do people deal with loss and grief?"
    ]
    
    # Answer each question
    for question in example_questions:
        rag.answer_question(question)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 