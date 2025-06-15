import torch
from datasets import load_dataset
import transformers

# Configuration
TRAINING_CONFIG = {
    "data_percentage": 0.1,  # Use 10% of the data by default
    "model_name": "facebook/opt-350m",
    "max_length": 256,  # Increased for diverse tasks
    "num_train_epochs": 1,
    "batch_size": 4,
    "learning_rate": 2e-4,
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
    print("Loading FLAN dataset...")
    dataset = load_dataset("google/flan_v2", split="train")
    return dataset

# 3. Process the dataset
def process_dataset(dataset, tokenizer):
    print("Processing dataset...")
    print(f"Total dataset size: {len(dataset)} examples")
    
    # Format the data
    def format_instruction(example):
        prompt = f"""
            Task: {example['task_name']}
            Input: {example['inputs']}
            Output: {example['targets']}
        """
        return {"text": prompt}
    
    # Process dataset
    processed_dataset = dataset.map(
        format_instruction,
        remove_columns=dataset.column_names
    )
    
    # Select percentage of data
    num_examples = int(len(processed_dataset) * TRAINING_CONFIG["data_percentage"])
    print(f"Using {TRAINING_CONFIG['data_percentage']*100}% of data: {num_examples} examples")
    small_dataset = processed_dataset.select(range(num_examples))
    
    # Split into train and validation
    train_val_split = small_dataset.train_test_split(test_size=0.1)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    
    print(f"Training set size: {len(train_dataset)} examples")
    print(f"Validation set size: {len(val_dataset)} examples")
    
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
        output_dir="./flan_model",
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        gradient_accumulation_steps=2,
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
    trainer.save_model("./flan_model_final")

# Example inference function
def generate_response(model, tokenizer, device, task, input_text):
    prompt = f"""
        Task: {task}
        Input: {input_text}
        Output:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main execution
if __name__ == "__main__":
    # 1. Get the model
    model, tokenizer, device = get_model_and_tokenizer()
    
    # 2. Download the dataset
    dataset = download_dataset()
    
    # 3. Process the dataset
    train_dataset, val_dataset = process_dataset(dataset, tokenizer)
    
    # 4. Initialize the trainer
    trainer = initialize_trainer(model, tokenizer, train_dataset, val_dataset)
    
    # 5. Train the model
    train_model(trainer)
    
    # Test the model
    print("\nTesting model...")
    test_task = "Summarize the following text"
    test_input = "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals."
    print(f"Task: {test_task}")
    print(f"Input: {test_input}")
    print(f"Output: {generate_response(model, tokenizer, device, test_task, test_input)}") 