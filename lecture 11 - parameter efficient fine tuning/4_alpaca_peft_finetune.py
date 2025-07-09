#!/usr/bin/env python3
"""
Alpaca Fine-tuning with PEFT - Simplified Version
"""

import torch
from datasets import load_dataset
import transformers, peft

#%
# Step 1: Load Alpaca dataset
print("2. Loading Alpaca dataset...")
dataset = load_dataset("tatsu-lab/alpaca", split="train")
print(f"Loaded {len(dataset)} examples")

#%
# Step 2: Process dataset
print("2. Processing dataset...")
def format_instruction(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return {"text": prompt}

small_dataset     = dataset.select(range(min(1000, len(dataset))))
processed_dataset = small_dataset.map(format_instruction, remove_columns=small_dataset.column_names)
print(f"Using {len(processed_dataset)} examples")

#%
# Step 3: Tokenize dataset
print("3. Tokenizing dataset...")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256, return_tensors="pt")
tokenized_dataset = processed_dataset.map(tokenize_function, batched=True, remove_columns=processed_dataset.column_names)

#%
# Step 4: Load model & tokenizer
print("4. Loading model & tokenizer...")
model_name = "openai-community/gpt2"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto")

#%
# Step 5: Create LoRA config
print("5. Creating LoRA config & creating LORA model")
lora_config = peft.LoraConfig(r=16)
peft_model  = peft.get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

#%
# Step 6: Training
print("6. Training...")
training_args = transformers.TrainingArguments(
    output_dir="./alpaca_lora_model",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    fp16=False,
    report_to="none",
)

trainer = transformers.Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.save_model()
print("Training completed!")

#%
# Step 7: Save LoRA adapters
print("7. Saving LoRA adapters...")
peft_model.save_pretrained("./alpaca_lora_adapters")

#%
# Step 9: Test the model
print("9. Testing model...")
test_prompt = "### Instruction:\nWrite a short poem about artificial intelligence.\n\n### Input:\n\n### Response:\n"
inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=256).to(peft_model.device)

with torch.no_grad():
    outputs = peft_model.generate(**inputs, max_length=150, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Test response: {response}")

print("Done!")