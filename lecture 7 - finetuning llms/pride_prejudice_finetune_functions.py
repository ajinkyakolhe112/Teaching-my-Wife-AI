
# CODE is restructured to use functions instead of class
import torch
import requests
from datasets import Dataset
import transformers

# Configuration
TRAINING_CONFIG = {
    "model_name": "openai-community/gpt2",
    "max_length": 128,
    "num_train_epochs": 5,
    "batch_size": 4,
    "learning_rate": 5e-5,
    "output_dir": "./pride_prejudice_model"
}

# 1. Get the model
def get_model_and_tokenizer():
    print("Loading model and tokenizer...")
    
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(TRAINING_CONFIG["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = transformers.AutoModelForCausalLM.from_pretrained(TRAINING_CONFIG["model_name"]).to(device)
    
    return model, tokenizer, device

# 2. Download the dataset
def download_dataset():
    print("Downloading Pride and Prejudice...")
    url = "https://www.gutenberg.org/files/1342/1342-0.txt"
    response = requests.get(url)
    
    # Save the book
    with open("pride_prejudice.txt", "w", encoding="utf-8") as f:
        f.write(response.text)
    
    print("Book downloaded successfully!")
    
    # Read the text file
    with open("pride_prejudice.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    return text

# 3. Process the dataset
def process_dataset(text, tokenizer):
    print("Processing dataset...")
    
    # Split text into chunks of max_length tokens
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Tokenize the text
    tokens = tokenizer.encode(text)
    
    for token in tokens:
        current_chunk.append(token)
        current_length += 1
        
        if current_length >= TRAINING_CONFIG["max_length"]:
            chunks.append(tokenizer.decode(current_chunk))
            current_chunk = []
            current_length = 0
    
    if current_chunk:  # Add the last chunk if it's not empty
        chunks.append(tokenizer.decode(current_chunk))
    
    # Create dataset from chunks
    dataset = Dataset.from_dict({"text": chunks})
    print(f"Total dataset size: {len(dataset)} chunks")
    
    # Split into train and validation
    train_val_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    
    print(f"Training set size: {len(train_dataset)} chunks")
    print(f"Validation set size: {len(val_dataset)} chunks")
    
    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=TRAINING_CONFIG["max_length"],
            return_tensors="pt"
        )
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return train_dataset, val_dataset

# 4. Initialize the trainer
def initialize_trainer(model, tokenizer, train_dataset, val_dataset):
    print("Initializing trainer...")
    
    # Training arguments
    training_args = transformers.TrainingArguments(
        output_dir=TRAINING_CONFIG["output_dir"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        logging_steps=5,
        save_strategy="epoch",
    )
    
    # Create trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    return trainer

# 5. Train the model
def train_model(trainer):
    print("Starting training...")
    trainer.train()
    print("Saving model...")
    trainer.save_model(TRAINING_CONFIG["output_dir"])

# Example inference function
def generate_text(model, tokenizer, device, prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main execution
if __name__ == "__main__":
    # 1. Get the model
    model, tokenizer, device = get_model_and_tokenizer()
    
    # 2. Download the dataset
    text = download_dataset()
    
    # 3. Process the dataset
    train_dataset, val_dataset = process_dataset(text, tokenizer)
    
    # 4. Initialize the trainer
    trainer = initialize_trainer(model, tokenizer, train_dataset, val_dataset)
    
    # 5. Train the model
    train_model(trainer)
    
    # Test the model
    print("\nTesting model...")
    test_prompts = [
        "It is a truth universally acknowledged, that a single man in possession of a good fortune,",
        "Mr. Darcy was",
        "Elizabeth Bennet thought that",
        "The ball at Netherfield was"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generate_text(model, tokenizer, device, prompt)}")
        print("-" * 80) 