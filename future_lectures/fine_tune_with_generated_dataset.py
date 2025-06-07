"""
Fine-tune a model using the generated Q&A dataset
This script:
1. Loads the generated Q&A dataset
2. Prepares it for fine-tuning
3. Fine-tunes a model for question answering
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
import json
from typing import List, Dict
import os

class QAFineTuner:
    def __init__(self):
        print("Initializing QA Fine-tuner...")
        
        # Use a model pre-trained for QA
        self.model_name = "distilbert-base-cased-distilled-squad"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
    
    def load_dataset(self, filename: str = "pride_prejudice_qa.json") -> Dataset:
        """Load the generated Q&A dataset"""
        print(f"Loading dataset from {filename}...")
        
        with open(filename, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)
        
        # Convert to HuggingFace dataset format
        dataset_dict = {
            "question": [pair["question"] for pair in qa_pairs],
            "context": [pair["context"] for pair in qa_pairs],
            "answer": [pair["answer"] for pair in qa_pairs]
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
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
    
    def fine_tune(self, train_dataset: Dataset):
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
            evaluation_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="loss"
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
    
    def answer_question(self, question: str, context: str) -> str:
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
    fine_tuner = QAFineTuner()
    
    # Load and prepare dataset
    dataset = fine_tuner.load_dataset()
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