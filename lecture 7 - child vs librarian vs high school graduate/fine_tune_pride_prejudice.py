"""
Simple Fine-tuning of a Language Model on Pride and Prejudice
This demonstrates the graduate course analogy where we:
1. Take a pre-trained model (like a graduate student)
2. Teach it specific knowledge (like studying a subject in depth)
3. Make it learn so well it doesn't need to look things up
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import requests
import os
from datasets import load_dataset

class PridePrejudiceFineTuner:
    def __init__(self):
        print("Initializing fine-tuning process...")
        
        # Use a small model for demonstration
        self.model_name = "gpt2"  # Small model, good for demonstration
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
    
    def download_book(self):
        """Download Pride and Prejudice from Project Gutenberg"""
        print("Downloading Pride and Prejudice...")
        url = "https://www.gutenberg.org/files/1342/1342-0.txt"
        response = requests.get(url)
        
        # Save the book
        with open("pride_prejudice.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
        
        print("Book downloaded successfully!")
    
    def prepare_dataset(self):
        """Prepare the dataset for training using Hugging Face Datasets."""
        print("Preparing dataset...")

        # Load the text file as a dataset
        datasets = load_dataset("text", data_files={"train": "pride_prejudice.txt"})

        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

        tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["text"])

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        return tokenized_datasets["train"], data_collator
    
    def fine_tune(self, train_dataset, data_collator):
        """Fine-tune the model"""
        print("Starting fine-tuning process...")
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./pride_prejudice_model",
            overwrite_output_dir=True,
            num_train_epochs=3,  # Small number for demonstration
            per_device_train_batch_size=4,
            save_steps=1000,
            save_total_limit=2,
            logging_steps=100,
            # Add these new arguments
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=500,
            gradient_accumulation_steps=4,
            fp16=False,  # Disable mixed precision training
            optim="adamw_torch",  # Use PyTorch's AdamW implementation
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        
        # Train the model
        print("Training the model (this might take a while)...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained("./pride_prejudice_model")
        print("Model saved successfully!")
    
    def generate_text(self, prompt, max_length=100):
        """Generate text using the fine-tuned model"""
        # Load the fine-tuned model
        model = AutoModelForCausalLM.from_pretrained("./pride_prejudice_model")
        tokenizer = AutoTokenizer.from_pretrained("./pride_prejudice_model")
        
        # Generate text
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # Initialize the fine-tuner
    fine_tuner = PridePrejudiceFineTuner()
    
    # Download the book
    fine_tuner.download_book()
    
    # Prepare dataset
    train_dataset, data_collator = fine_tuner.prepare_dataset()
    
    # Fine-tune the model
    fine_tuner.fine_tune(train_dataset, data_collator)
    
    # Test the model with some prompts
    test_prompts = [
        "It is a truth universally acknowledged that",
        "Mr. Darcy was",
        "Elizabeth Bennet thought that",
        "The ball at Netherfield was"
    ]
    
    print("\nTesting the fine-tuned model:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {fine_tuner.generate_text(prompt)}")
        print("-" * 80)

if __name__ == "__main__":
    main() 