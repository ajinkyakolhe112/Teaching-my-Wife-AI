"""
Fine-tuning a Language Model for Question Answering on Pride and Prejudice
This demonstrates the graduate course analogy where we:
1. Take a pre-trained model (like a graduate student)
2. Teach it to answer questions about the book (like studying for an exam)
3. Make it learn so well it doesn't need to look things up
"""

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from datasets import Dataset
import requests
import json
import re

class PridePrejudiceQAFineTuner:
    def __init__(self):
        print("Initializing QA fine-tuning process...")
        
        # Use a model pre-trained for QA
        self.model_name = "distilbert-base-cased-distilled-squad"  # Smaller model good for QA
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
    
    def download_book(self):
        """Download Pride and Prejudice from Project Gutenberg"""
        print("Downloading Pride and Prejudice...")
        url = "https://www.gutenberg.org/files/1342/1342-0.txt"
        response = requests.get(url)
        
        # Save the book
        with open("datasets/pride_prejudice.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
        
        print("Book downloaded successfully!")
        return response.text
    
    def create_qa_dataset(self, text):
        """Create a QA dataset from the book"""
        print("Creating QA dataset...")
        
        # Split text into chapters
        chapters = re.split(r'Chapter \d+', text)[1:]  # Skip the header
        
        # Create some example QA pairs
        qa_pairs = [
            {
                "question": "Who is the main character of Pride and Prejudice?",
                "context": "The story follows the main character Elizabeth Bennet as she deals with issues of manners, upbringing, morality, education, and marriage in the society of the landed gentry of the British Regency.",
                "answer": "Elizabeth Bennet"
            },
            {
                "question": "What is Mr. Darcy's first name?",
                "context": "Mr. Fitzwilliam Darcy is a wealthy gentleman who has an income of £10,000 a year. He is the master of Pemberley, a large estate in Derbyshire.",
                "answer": "Fitzwilliam"
            },
            {
                "question": "How many sisters does Elizabeth Bennet have?",
                "context": "The Bennet family consists of Mr. and Mrs. Bennet and their five daughters: Jane, Elizabeth, Mary, Kitty, and Lydia.",
                "answer": "four"
            },
            {
                "question": "What is the name of Mr. Bingley's estate?",
                "context": "Mr. Bingley has taken Netherfield Park, a large estate near the Bennet family home.",
                "answer": "Netherfield Park"
            },
            {
                "question": "Who is Mr. Collins?",
                "context": "Mr. Collins is a clergyman and a cousin of Mr. Bennet. He is the heir to the Longbourn estate.",
                "answer": "a clergyman and cousin of Mr. Bennet"
            }
        ]
        
        # Add more QA pairs from the text
        for chapter in chapters[:5]:  # Use first 5 chapters for demonstration
            # Extract some sentences for context
            sentences = re.split(r'[.!?]+', chapter)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 50]
            
            # Create some QA pairs from these sentences
            for i in range(0, len(sentences)-1, 2):
                if i+1 < len(sentences):
                    context = sentences[i] + " " + sentences[i+1]
                    # Create a simple question
                    words = sentences[i].split()
                    if len(words) > 5:
                        answer = " ".join(words[2:4])  # Take some words as answer
                        question = f"What is mentioned about {' '.join(words[2:4])}?"
                        qa_pairs.append({
                            "question": question,
                            "context": context,
                            "answer": answer
                        })
        
        # Convert to HuggingFace dataset format
        dataset_dict = {
            "question": [pair["question"] for pair in qa_pairs],
            "context": [pair["context"] for pair in qa_pairs],
            "answer": [pair["answer"] for pair in qa_pairs]
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def prepare_dataset(self, dataset):
        """Prepare the dataset for training"""
        print("Preparing dataset for training...")
        
        def preprocess_function(examples):
            questions = examples["question"]
            contexts = examples["context"]
            answers = examples["answer"]
            
            # Tokenize the inputs
            inputs = self.tokenizer(
                questions,
                contexts,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True
            )
            
            # Find the start and end positions of the answers
            offset_mapping = inputs.pop("offset_mapping")
            start_positions = []
            end_positions = []
            
            for i, answer in enumerate(answers):
                # Find the answer in the context
                start_char = contexts[i].find(answer)
                end_char = start_char + len(answer)
                
                # Find the token indices
                token_start_index = 0
                token_end_index = 0
                
                for idx, (start, end) in enumerate(offset_mapping[i]):
                    if start <= start_char < end:
                        token_start_index = idx
                    if start < end_char <= end:
                        token_end_index = idx
                        break
                
                start_positions.append(token_start_index)
                end_positions.append(token_end_index)
            
            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            
            return inputs
        
        # Apply preprocessing
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return processed_dataset
    
    def fine_tune(self, train_dataset):
        """Fine-tune the model"""
        print("Starting fine-tuning process...")
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./pride_prejudice_qa_model",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=100,
            save_total_limit=2,
            logging_steps=10,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DefaultDataCollator(),
        )
        
        # Train the model
        print("Training the model (this might take a while)...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained("./pride_prejudice_qa_model")
        print("Model saved successfully!")
    
    def answer_question(self, question, context):
        """Answer a question using the fine-tuned model"""
        # Load the fine-tuned model
        model = AutoModelForQuestionAnswering.from_pretrained("./pride_prejudice_qa_model")
        tokenizer = AutoTokenizer.from_pretrained("./pride_prejudice_qa_model")
        
        # Prepare inputs
        inputs = tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        
        # Get model output
        outputs = model(**inputs)
        
        # Get the answer span
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)
        
        # Convert tokens to text
        answer = tokenizer.decode(
            inputs["input_ids"][0][start_idx:end_idx+1],
            skip_special_tokens=True
        )
        
        return answer

def main():
    # Initialize the fine-tuner
    fine_tuner = PridePrejudiceQAFineTuner()
    
    # Download the book
    text = fine_tuner.download_book()
    
    # Create and prepare dataset
    dataset = fine_tuner.create_qa_dataset(text)
    processed_dataset = fine_tuner.prepare_dataset(dataset)
    
    # Fine-tune the model
    fine_tuner.fine_tune(processed_dataset)
    
    # Test the model with some questions
    test_questions = [
        {
            "question": "Who is the main character of Pride and Prejudice?",
            "context": "The story follows the main character Elizabeth Bennet as she deals with issues of manners, upbringing, morality, education, and marriage in the society of the landed gentry of the British Regency."
        },
        {
            "question": "What is Mr. Darcy's first name?",
            "context": "Mr. Fitzwilliam Darcy is a wealthy gentleman who has an income of £10,000 a year. He is the master of Pemberley, a large estate in Derbyshire."
        },
        {
            "question": "How many sisters does Elizabeth Bennet have?",
            "context": "The Bennet family consists of Mr. and Mrs. Bennet and their five daughters: Jane, Elizabeth, Mary, Kitty, and Lydia."
        }
    ]
    
    print("\nTesting the fine-tuned model:")
    for qa in test_questions:
        print(f"\nQuestion: {qa['question']}")
        print(f"Context: {qa['context']}")
        answer = fine_tuner.answer_question(qa['question'], qa['context'])
        print(f"Answer: {answer}")
        print("-" * 80)

if __name__ == "__main__":
    main() 