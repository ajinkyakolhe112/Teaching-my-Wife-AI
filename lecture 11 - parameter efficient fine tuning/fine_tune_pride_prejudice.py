#!/usr/bin/env python3
"""
Pride & Prejudice Fine-tuning Implementation
Lecture 11 - Parameter Efficient Fine-tuning

This script demonstrates how to fine-tune language models on Pride & Prejudice
instruction datasets using both full fine-tuning and PEFT methods.
"""

import json
import torch
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

# Import required libraries
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
except ImportError as e:
    print(f"Missing required libraries: {e}")
    print("Please install: pip install transformers datasets peft torch")
    exit(1)

@dataclass
class TrainingConfig:
    """Configuration for fine-tuning"""
    model_name: str = "microsoft/DialoGPT-medium"
    output_dir: str = "./pride_prejudice_model"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    max_length: int = 512
    use_lora: bool = True
    use_qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    save_steps: int = 500
    eval_steps: int = 500

class PridePrejudiceFineTuner:
    """Main class for fine-tuning Pride & Prejudice models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_dataset(self, file_path: str) -> Dataset:
        """Load the Pride & Prejudice instruction dataset"""
        print(f"Loading dataset from {file_path}...")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to Hugging Face Dataset format
        dataset = Dataset.from_list(data)
        
        # Split into train and validation
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        print(f"Dataset loaded: {len(dataset['train'])} training samples, {len(dataset['test'])} validation samples")
        return dataset
    
    def setup_model_and_tokenizer(self):
        """Set up model and tokenizer based on configuration"""
        print(f"Setting up model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model based on configuration
        if self.config.use_qlora:
            self.model = self._setup_qlora_model()
        elif self.config.use_lora:
            self.model = self._setup_lora_model()
        else:
            self.model = self._setup_full_model()
        
        print("Model and tokenizer setup complete")
    
    def _setup_full_model(self):
        """Set up model for full fine-tuning"""
        return AutoModelForCausalLM.from_pretrained(self.config.model_name)
    
    def _setup_lora_model(self):
        """Set up model with LoRA"""
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, lora_config)
        return model
    
    def _setup_qlora_model(self):
        """Set up model with QLoRA"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, lora_config)
        return model
    
    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Preprocess the dataset for training"""
        print("Preprocessing dataset...")
        
        def tokenize_function(examples):
            texts = []
            for instruction, response in zip(examples['instruction'], examples['response']):
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}\n\n"
                texts.append(text)
            
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # Set labels to input_ids for causal language modeling
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Apply preprocessing to both train and test sets
        train_dataset = dataset["train"].map(tokenize_function, batched=True)
        test_dataset = dataset["test"].map(tokenize_function, batched=True)
        
        print("Dataset preprocessing complete")
        return {"train": train_dataset, "test": test_dataset}
    
    def setup_training(self, train_dataset: Dataset, eval_dataset: Dataset):
        """Set up training configuration"""
        print("Setting up training configuration...")
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=10,
            save_steps=self.config.save_steps,
            save_total_limit=2,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        print("Training setup complete")
    
    def train(self):
        """Start the training process"""
        print("Starting training...")
        
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_training() first.")
        
        # Start training
        train_result = self.trainer.train()
        
        # Save the model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training results
        with open(f"{self.config.output_dir}/training_results.json", 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        print(f"Training complete. Model saved to {self.config.output_dir}")
        return train_result
    
    def evaluate(self) -> Dict:
        """Evaluate the trained model"""
        print("Evaluating model...")
        
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_training() first.")
        
        eval_results = self.trainer.evaluate()
        
        # Save evaluation results
        with open(f"{self.config.output_dir}/evaluation_results.json", 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"Evaluation complete. Results: {eval_results}")
        return eval_results
    
    def generate_response(self, instruction: str, max_length: int = 200) -> str:
        """Generate a response using the fine-tuned model"""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded. Call setup_model_and_tokenizer() first.")
        
        # Format input
        input_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        # Move to device if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        self.model.to(device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response part
        response = response.split("### Response:\n")[-1].strip()
        
        return response
    
    def save_model_info(self):
        """Save model information for deployment"""
        model_info = {
            "model_type": "pride_prejudice_assistant",
            "base_model": self.config.model_name,
            "fine_tuning_method": "QLoRA" if self.config.use_qlora else "LoRA" if self.config.use_lora else "Full",
            "training_config": {
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "max_length": self.config.max_length
            },
            "lora_config": {
                "r": self.config.lora_r,
                "alpha": self.config.lora_alpha
            } if self.config.use_lora or self.config.use_qlora else None,
            "description": "Fine-tuned model for Pride & Prejudice literary analysis"
        }
        
        with open(f"{self.config.output_dir}/model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model info saved to {self.config.output_dir}/model_info.json")

def interactive_testing(fine_tuner: PridePrejudiceFineTuner):
    """Interactive testing of the fine-tuned model"""
    
    print("\n" + "="*50)
    print("Pride & Prejudice AI Assistant")
    print("Type 'quit' to exit")
    print("="*50)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            break
        
        try:
            response = fine_tuner.generate_response(user_input)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"\nError: {e}")

def run_evaluation_tests(fine_tuner: PridePrejudiceFineTuner):
    """Run predefined evaluation tests"""
    
    test_questions = [
        {
            "instruction": "What is Elizabeth Bennet's personality like?",
            "expected": "Elizabeth Bennet is intelligent, witty, and possesses a strong sense of self-respect."
        },
        {
            "instruction": "How does Mr. Darcy change throughout the novel?",
            "expected": "Mr. Darcy learns to be more open and less judgmental of others."
        },
        {
            "instruction": "What is the significance of the first proposal scene?",
            "expected": "The first proposal scene reveals both characters' true feelings and misunderstandings."
        },
        {
            "instruction": "Describe the relationship between Jane and Mr. Bingley.",
            "expected": "Jane and Mr. Bingley have a sweet, straightforward romance."
        },
        {
            "instruction": "What role does Mr. Wickham play in the story?",
            "expected": "Mr. Wickham is a charming but deceitful character who causes conflict."
        }
    ]
    
    print("\nRunning evaluation tests...")
    results = []
    
    for i, test in enumerate(test_questions, 1):
        print(f"\nTest {i}: {test['instruction']}")
        
        try:
            response = fine_tuner.generate_response(test['instruction'])
            print(f"Generated: {response}")
            print(f"Expected: {test['expected']}")
            
            results.append({
                "test_id": i,
                "instruction": test['instruction'],
                "expected": test['expected'],
                "generated": response,
                "success": True
            })
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "test_id": i,
                "instruction": test['instruction'],
                "error": str(e),
                "success": False
            })
    
    # Save test results
    with open(f"{fine_tuner.config.output_dir}/test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to {fine_tuner.config.output_dir}/test_results.json")
    return results

def main():
    """Main function to run the complete fine-tuning pipeline"""
    
    print("Pride & Prejudice Fine-tuning Pipeline")
    print("="*50)
    
    # Configuration
    config = TrainingConfig(
        model_name="microsoft/DialoGPT-medium",
        output_dir="./pride_prejudice_model",
        num_epochs=3,
        batch_size=4,
        use_lora=True,  # Use LoRA for efficiency
        save_steps=100,  # Save more frequently for demo
        eval_steps=100
    )
    
    # Create fine-tuner
    fine_tuner = PridePrejudiceFineTuner(config)
    
    try:
        # 1. Load dataset
        dataset = fine_tuner.load_dataset("pride_prejudice_instruction_dataset.json")
        
        # 2. Set up model and tokenizer
        fine_tuner.setup_model_and_tokenizer()
        
        # 3. Preprocess dataset
        processed_dataset = fine_tuner.preprocess_dataset(dataset)
        
        # 4. Set up training
        fine_tuner.setup_training(processed_dataset["train"], processed_dataset["test"])
        
        # 5. Train model
        train_result = fine_tuner.train()
        
        # 6. Evaluate model
        eval_results = fine_tuner.evaluate()
        
        # 7. Save model info
        fine_tuner.save_model_info()
        
        # 8. Run evaluation tests
        test_results = run_evaluation_tests(fine_tuner)
        
        # 9. Interactive testing
        print("\nStarting interactive testing...")
        interactive_testing(fine_tuner)
        
        print("\nFine-tuning pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 