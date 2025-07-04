"""
Ultra-Minimal IFT Dataset Creator - ~40 lines
Uses modern libraries + LLM for realistic responses
"""

import json
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline

def create_ift_dataset(text_file="datasets/pride_prejudice.txt", n=10):
    # Load & chunk text
    with open(text_file, 'r') as f:
        chunks = [f.read()[i:i+500] for i in range(0, len(f.read()), 500)]
    
    # Instructions & embeddings
    instructions = [
        "What is the entail on Longbourn?", "Analyze Elizabeth's character", 
        "How does Austen explore marriage?", "Significance of opening sentence?",
        "Compare Darcy and Bingley", "How does social class affect relationships?",
        "What motivates Mr. Collins?", "Analyze pride and prejudice themes",
        "How does the novel critique society?", "What if Elizabeth accepted Collins?"
    ][:n]
    
    # Semantic search with embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_emb = model.encode(chunks)
    
    # Load LLM for response generation
    llm = pipeline("text-generation", model="microsoft/DialoGPT-medium", max_length=200)
    
    # Create dataset
    dataset = []
    for inst in instructions:
        query_emb = model.encode([inst])
        best_chunk = chunks[np.argmax(np.dot(chunk_emb, query_emb.T))]
        
        # Generate response using LLM
        prompt = f"Context: {best_chunk[:300]}\nQuestion: {inst}\nAnswer:"
        response = llm(prompt)[0]['generated_text'].split("Answer:")[-1].strip()
        
        dataset.append({
            "instruction": inst,
            "response": response,
            "context": best_chunk[:300]
        })
    
    # Save
    with open("ultra_minimal_ift.json", 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Created {len(dataset)} examples with LLM responses")
    return "ultra_minimal_ift.json"

if __name__ == "__main__":
    create_ift_dataset() 