
# Lecture 11: Simple PEFT Fine-Tuning on an Instruction Dataset

## 1. Introduction

Welcome to Lecture 11! In our last session, we created an instruction dataset from "Pride and Prejudice." Today, we'll use that dataset to fine-tune a pre-trained language model. We'll focus on a technique called **Parameter-Efficient Fine-Tuning (PEFT)**, which allows us to achieve great results without the massive computational resources typically required for full model fine-tuning.

### Learning Objectives:

*   Understand the concept of Parameter-Efficient Fine-Tuning (PEFT).
*   Learn about LoRA (Low-Rank Adaptation), a popular PEFT method.
*   Prepare our "Pride and Prejudice" instruction dataset for training.
*   Fine-tune a model using the Hugging Face `transformers` and `peft` libraries.
*   Test the newly fine-tuned model to see how well it understands "Pride and Prejudice."

## 2. What is PEFT and Why Do We Need It?

Fine-tuning a large language model (LLM) like GPT-2 or Llama can be incredibly resource-intensive. A full fine-tune requires updating all of the model's billions of parameters, which demands powerful GPUs and a lot of time.

**PEFT** offers a solution. Instead of updating all the model's parameters, we only update a small, strategically chosen subset. This has several advantages:

*   **Reduced Computational Cost:** Requires significantly less GPU memory and processing power.
*   **Faster Training:** The training process is much quicker.
*   **Smaller Model Footprint:** Since we're only training a small number of parameters, the resulting model files are much smaller.

### LoRA: A Key PEFT Technique

**LoRA (Low-Rank Adaptation)** is one of the most popular PEFT methods. It works by injecting small, trainable "adapter" layers into the model. We freeze the original model weights and only train these new adapters. This way, we can adapt the model to our specific task (in this case, answering questions about "Pride and Prejudice") without the cost of a full fine-tune.

## 3. Setting Up Our Environment

First, let's make sure we have the necessary libraries installed. We'll need `transformers` for the models, `datasets` to handle our data, and `peft` for the LoRA implementation.

```bash
pip install transformers datasets peft torch
```

## 4. Preparing the Data

We'll use the instruction dataset we created in Lecture 10. Let's assume it's saved in a JSON file named `pride_and_prejudice_instructions.json`. Each entry in the JSON file should have an "instruction" and a "response" field.

Our first step is to load this data and format it into a prompt that the model can understand. A good prompt structure helps the model distinguish between the instruction and the desired response.

```python
import json
from datasets import Dataset

def load_and_prepare_dataset(file_path):
    """Loads the instruction dataset and prepares it for training."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # We need to format the data into a single string for the model
    for item in data:
        item['text'] = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}"
        
    return Dataset.from_list(data)

# Load the dataset
dataset = load_and_prepare_dataset('pride_and_prejudice_instructions.json')
```

## 5. Fine-Tuning with PEFT and LoRA

Now for the exciting part! We'll choose a base model to fine-tune. A smaller model like `distilgpt2` is a good starting point for experimentation.

### Step 1: Load the Model and Tokenizer

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token
```

### Step 2: Configure LoRA

Next, we'll set up our LoRA configuration. This tells the `peft` library how to apply LoRA to our model.

```python
from peft import LoraConfig, get_peft_model

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
```

You'll see that we're only training a very small percentage of the total parameters!

### Step 3: Set Up the Trainer

We'll use the `Trainer` class from the `transformers` library to handle the training loop.

```python
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

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
```

### Step 4: Train the Model

Now, we can start the training process.

```python
trainer.train()
```

## 6. Testing Our Fine-Tuned Model

Once the training is complete, we can test our model to see if it has learned to answer questions about "Pride and Prejudice."

```python
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
```

You should see a response that is relevant to "Pride and Prejudice"!

## 7. Conclusion

Congratulations! You've successfully fine-tuned a language model on a custom instruction dataset using PEFT. This is a powerful technique that you can apply to many different domains and tasks.

In our next lecture, we'll explore how to evaluate our fine-tuned model more rigorously and how to deploy it as a simple application.

### Homework:

1.  Experiment with different `LoraConfig` parameters (e.g., `r`, `lora_alpha`). How do they affect the results?
2.  Try fine-tuning a different base model. How does the choice of base model impact performance?
3.  Create a few more test questions and evaluate your model's responses.

