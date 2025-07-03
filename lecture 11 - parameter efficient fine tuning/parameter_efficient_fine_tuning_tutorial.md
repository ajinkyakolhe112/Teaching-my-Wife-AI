# Parameter Efficient Fine-Tuning (PEFT) Tutorial

## Introduction

Large Language Models (LLMs) are powerful but expensive to fine-tune due to their size. Parameter Efficient Fine-Tuning (PEFT) methods allow us to adapt these models to new tasks by updating only a small subset of parameters, making the process faster and less resource-intensive.

## Why PEFT?
- **Lower compute and memory requirements**
- **Faster training**
- **Smaller storage footprint for each task**

## Common PEFT Techniques
- **Adapters**: Small neural network modules inserted into each layer of the model. Only the adapters are trained.
- **LoRA (Low-Rank Adaptation)**: Injects trainable low-rank matrices into the model's weights. Only these matrices are updated.
- **Prompt Tuning**: Learns a small set of task-specific prompt embeddings.

## Example: LoRA with Hugging Face PEFT

We'll use the `peft` library to fine-tune a model using LoRA.

### 1. Install Required Libraries
```bash
pip install transformers datasets peft
```

### 2. Basic LoRA Fine-Tuning Example
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# Load model and tokenizer
model_name = "tiiuae/falcon-7b-instruct"  # Or any other supported model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("yelp_review_full", split="train[:1000]")

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(preprocess, batched=True)

# LoRA config
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_steps=10,
    output_dir="./lora-finetuned-model"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
```

## Summary
- PEFT allows efficient adaptation of LLMs to new tasks.
- LoRA and adapters are popular PEFT methods.
- Hugging Face's `peft` library makes it easy to use these techniques.

---
**Further Reading:**
- [PEFT Library Documentation](https://huggingface.co/docs/peft/index)
- [LoRA: Low-Rank Adaptation of Large Language Models (paper)](https://arxiv.org/abs/2106.09685) 