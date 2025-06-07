"""
Generate Q&A Dataset using LLM
This script:
1. Downloads Pride and Prejudice
2. Splits it into manageable chunks
3. Uses an LLM to generate Q&A pairs
4. Saves the dataset for fine-tuning
"""

import requests
import json
import re
from typing import List, Dict
import openai
from tqdm import tqdm
import os
from dotenv import load_dotenv

class QADatasetGenerator:
    def __init__(self):
        print("Initializing QA Dataset Generator...")
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Please set OPENAI_API_KEY in .env file")
        
        openai.api_key = self.api_key
        
    def download_book(self) -> str:
        """Download Pride and Prejudice from Project Gutenberg"""
        print("Downloading Pride and Prejudice...")
        url = "https://www.gutenberg.org/files/1342/1342-0.txt"
        response = requests.get(url)
        
        # Save the book
        with open("pride_prejudice.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
        
        print("Book downloaded successfully!")
        return response.text
    
    def split_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks of approximately chunk_size words"""
        print("Splitting text into chunks...")
        
        # Split into chapters first
        chapters = re.split(r'Chapter \d+', text)[1:]  # Skip the header
        
        chunks = []
        for chapter in chapters:
            # Split chapter into sentences
            sentences = re.split(r'[.!?]+', chapter)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            # Combine sentences into chunks
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                sentence_words = len(sentence.split())
                if current_size + sentence_words > chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sentence_words
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_words
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def generate_qa_pairs(self, chunk: str) -> List[Dict]:
        """Generate Q&A pairs for a text chunk using GPT-3.5"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a helpful assistant that generates question-answer pairs from text.
                    For each text chunk, generate 3-5 high-quality question-answer pairs.
                    Each pair should include:
                    1. A clear, specific question
                    2. The exact answer from the text
                    3. The relevant context from the text
                    
                    Format your response as a JSON array of objects with 'question', 'answer', and 'context' fields."""},
                    {"role": "user", "content": f"Generate question-answer pairs from this text:\n\n{chunk}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse the response
            content = response.choices[0].message.content
            qa_pairs = json.loads(content)
            
            # Validate the format
            for pair in qa_pairs:
                assert all(key in pair for key in ['question', 'answer', 'context'])
            
            return qa_pairs
            
        except Exception as e:
            print(f"Error generating QA pairs: {e}")
            return []
    
    def create_dataset(self, chunks: List[str]) -> List[Dict]:
        """Create the full dataset by generating QA pairs for each chunk"""
        print("Generating QA pairs for each chunk...")
        
        all_qa_pairs = []
        for chunk in tqdm(chunks):
            qa_pairs = self.generate_qa_pairs(chunk)
            all_qa_pairs.extend(qa_pairs)
        
        return all_qa_pairs
    
    def save_dataset(self, qa_pairs: List[Dict], filename: str = "pride_prejudice_qa.json"):
        """Save the dataset to a JSON file"""
        print(f"Saving dataset to {filename}...")
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2)
        
        print(f"Saved {len(qa_pairs)} QA pairs")
    
    def validate_dataset(self, qa_pairs: List[Dict]) -> bool:
        """Validate the dataset format and quality"""
        print("Validating dataset...")
        
        required_fields = ['question', 'answer', 'context']
        
        for pair in qa_pairs:
            # Check required fields
            if not all(field in pair for field in required_fields):
                print(f"Missing required fields in pair: {pair}")
                return False
            
            # Check answer is in context
            if pair['answer'] not in pair['context']:
                print(f"Answer not found in context: {pair}")
                return False
            
            # Check question is not empty
            if not pair['question'].strip():
                print(f"Empty question found: {pair}")
                return False
        
        return True

def main():
    # Initialize the generator
    generator = QADatasetGenerator()
    
    # Download the book
    text = generator.download_book()
    
    # Split into chunks
    chunks = generator.split_into_chunks(text)
    
    # Generate QA pairs
    qa_pairs = generator.create_dataset(chunks)
    
    # Validate the dataset
    if generator.validate_dataset(qa_pairs):
        # Save the dataset
        generator.save_dataset(qa_pairs)
        
        # Print some examples
        print("\nExample QA pairs:")
        for i, pair in enumerate(qa_pairs[:3]):
            print(f"\nExample {i+1}:")
            print(f"Question: {pair['question']}")
            print(f"Answer: {pair['answer']}")
            print(f"Context: {pair['context'][:200]}...")
    else:
        print("Dataset validation failed!")

if __name__ == "__main__":
    main() 