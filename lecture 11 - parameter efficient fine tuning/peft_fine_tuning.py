import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

def load_and_prepare_dataset(file_path):
    """Loads the instruction dataset and prepares it for training."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # We need to format the data into a single string for the model
    for item in data:
        item['text'] = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}"
        
    return Dataset.from_list(data)

def main():
    # Load the dataset
    dataset = load_and_prepare_dataset('pride_and_prejudice_instructions.json')

    # Load the Model and Tokenizer
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set the padding token
    tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,  # The rank of the LoRA matrices
        lora_alpha=32,  # A scaling factor for the LoRA updates
        target_modules=["c_attn", "c_proj"],  # The layers to apply LoRA to
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Create the PEFT model
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Set Up the Trainer
    training_args = TrainingArguments(
        output_dir="./pride_prejudice_peft_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Train the Model
    trainer.train()

    # Save the model
    peft_model.save_pretrained("./pride_prejudice_peft_model")
    tokenizer.save_pretrained("./pride_prejudice_peft_model")

    # Testing the Fine-Tuned Model
    def generate_response(instruction):
        """Generates a response from the fine-tuned model."""
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate the response
        outputs = peft_model.generate(
            input_ids=inputs["input_ids"], 
            max_length=150, 
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("### Response:\n")[1]

    # Test with a question
    instruction = "Who is Mr. Collins?"
    print(f"Instruction: {instruction}")
    print(f"Response: {generate_response(instruction)}")

if __name__ == "__main__":
    main()
