"""
Simple RAG Implementation using Project Gutenberg Dataset
Demonstrates basic RAG functionality with book passages and semantic search
"""

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleRAG:
    def __init__(self, num_books=100, passage_size=200):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dataset = load_dataset("manu/project_gutenberg", split="train")
        self.passages = []
        self.embeddings = []
        self._process_books(num_books, passage_size)
        
    def _process_books(self, num_books, passage_size):
        for book in self.dataset[:num_books]:
            words = book['text'].split()
            passages = [' '.join(words[i:i+passage_size]) for i in range(0, len(words), passage_size)]
            passage_embeddings = self.encoder.encode(passages, show_progress_bar=False)
            
            self.passages.extend(passages)
            self.embeddings.extend(passage_embeddings)
            
        self.embeddings = np.array(self.embeddings)
        print(f"Indexed {len(self.passages)} passages from {num_books} books")
    
    def answer_question(self, question, top_k=3):
        question_embedding = self.encoder.encode([question])[0]
        similarities = cosine_similarity([question_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        print(f"\nQuestion: {question}")
        for idx in top_indices:
            print(f"\n--- Passage (Similarity: {similarities[idx]:.3f}) ---")
            print(self.passages[idx][:500] + "...")
            
        return [self.passages[idx] for idx in top_indices]

def main():
    rag = SimpleRAG()
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