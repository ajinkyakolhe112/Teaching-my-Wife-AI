#!/usr/bin/env python3
"""
LoRA Fine-tuning with PEFT - Simplified Version
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset

def main():
    print("LoRA Fine-tuning with PEFT - Simplified")
    
    # Step 1: Simple dataset
    print("1. Loading dataset...")
    with open('pride_prejudice_ift.json', 'r') as f:
        data = json.load(f)
    dataset = Dataset.from_list([{"text": f"### Instruction:\n{item['question']}\n\n### Response:\n{item['answer']}"} for item in data])
    print(f"Loaded {len(dataset)} examples")
    
    # Step 2: Load model & tokenizer
    print("2. Loading model & tokenizer...")
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto")
    
    # Step 3: Simple LoRA config (like file 3)
    print("3. Creating LoRA config...")
    lora_config = LoraConfig(r=16)
    
    # Step 4: Create LoRA model
    print("4. Creating LoRA model...")
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    # Step 5: Training
    print("5. Training...")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    # Training config
    training_args = TrainingArguments(
        output_dir="./lora_model",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        fp16=False,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Train
    trainer.train()
    trainer.save_model()
    print("Training completed!")
    
    # Save LoRA adapters only
    print("Saving LoRA adapters...")
    peft_model.save_pretrained("./lora_adapters_only")
    
    # Load and test
    print("Loading and testing...")
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto")
    loaded_peft_model = PeftModel.from_pretrained(base_model, "./lora_adapters_only")
    
    # Simple test
    test_prompt = "### Instruction:\nWho is Mr. Darcy?\n\n### Response:\n"
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512).to(loaded_peft_model.device)
    
    with torch.no_grad():
        outputs = loaded_peft_model.generate(**inputs, max_length=100, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test response: {response[:100]}...")
    
    print("Done!")

if __name__ == "__main__":
    main()