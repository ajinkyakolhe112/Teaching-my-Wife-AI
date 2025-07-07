#!/usr/bin/env python3
"""
Simple PEFT Tutorial - Teaching Essential Concepts
Fine-tune Llama 1 on Pride and Prejudice IFT dataset
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

def load_dataset():
    """Load the IFT dataset from Lecture 10."""
    print("📚 Loading Pride and Prejudice IFT dataset...")

    # Modified to load from the current directory
    with open('pride_prejudice_ift.json', 'r') as f:
        data = json.load(f)

    # Format as instruction-response pairs
    formatted_data = []
    for item in data:
        formatted_data.append({
            "text": f"### Instruction:\n{item['question']}\n\n### Response:\n{item['answer']}"
        })

    dataset = Dataset.from_list(formatted_data)
    print(f"✅ Loaded {len(dataset)} examples")
    return dataset

def load_model():
    """Load Llama model and tokenizer."""
    print("🤖 Loading Llama model...")

    model_name = "openai-community/gpt2"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        # Removed load_in_4bit=True as it requires GPU support with bitsandbytes
    )

    print("✅ Model loaded successfully!")
    return model, tokenizer

def setup_lora(model):
    """Set up LoRA configuration."""
    print("🔧 Setting up LoRA...")

    lora_config = LoraConfig(
        r=16,                    # Rank of LoRA matrices
        lora_alpha=32,           # Scaling factor
        target_modules=["c_attn", "c_proj"],  # Which layers to adapt for GPT-2
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to model
    peft_model = get_peft_model(model, lora_config)

    # Show parameter efficiency
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())

    print(f"📊 Trainable parameters: {trainable_params:,}")
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Efficiency: {trainable_params/total_params*100:.2f}%")

    return peft_model

def prepare_data(dataset, tokenizer):
    """Prepare dataset for training."""
    print("📝 Preparing dataset...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    print(f"✅ Dataset prepared: {len(tokenized_dataset)} examples")
    return tokenized_dataset

def train_model(peft_model, tokenizer, tokenized_dataset):
    """Train the model."""
    print("🚀 Starting training...")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./simple_peft_model",
        num_train_epochs=2,           # Keep it short for demo
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        fp16=True,                    # Mixed precision
    )

    # Set up trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train
    trainer.train()
    trainer.save_model()
    print("✅ Training completed!")

    return trainer

def test_model(peft_model, tokenizer):
    """Test the fine-tuned model."""
    print("🧪 Testing the model...")

    test_questions = [
        "Who is Mr. Darcy?",
        "What is the main theme of Pride and Prejudice?",
        "How does Elizabeth change in the story?"
    ]

    for question in test_questions:
        print(f"\n❓ Question: {question}")

        # Format prompt
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        # Generate
        with torch.no_grad():
            outputs = peft_model.generate(
                input_ids=inputs["input_ids"],
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Response:" in response:
            response = response.split("### Response:")[1].strip()

        print(f"💬 Answer: {response}")
        print("-" * 50)

def main():
    """Main function - simple PEFT tutorial."""
    print("🎓 Simple PEFT Tutorial")
    print("=" * 50)
    print("Teaching essential PEFT concepts with Llama 1")
    print()

    try:
        # Step 1: Load dataset
        dataset = load_dataset()

        # Step 2: Load model
        model, tokenizer = load_model()

        # Step 3: Set up LoRA
        peft_model = setup_lora(model)

        # Step 4: Prepare data
        tokenized_dataset = prepare_data(dataset, tokenizer)

        # Step 5: Train
        trainer = train_model(peft_model, tokenizer, tokenized_dataset)

        # Step 6: Test
        test_model(peft_model, tokenizer)

        print("\n🎉 Tutorial completed successfully!")
        print("💡 Key concepts learned:")
        print("   - PEFT reduces trainable parameters")
        print("   - LoRA adds small adapters to existing layers")
        print("   - 4-bit quantization saves memory")
        print("   - Model maintains general capabilities")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have:")
        print("   - Hugging Face token set")
        print("   - GPU with sufficient memory")
        print("   - IFT dataset from Lecture 10")

if __name__ == "__main__":
    main()