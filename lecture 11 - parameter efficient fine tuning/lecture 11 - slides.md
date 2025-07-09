## 1. The Challenge of Fine-Tuning Large Language Models

- **Full Fine-Tuning is Expensive:**
    - Requires updating all model parameters.
    - Computationally intensive (requires a lot of VRAM).
    - Time-consuming.
    - Results in a new, large model for each task.

## 2. What is Parameter Efficient Fine-Tuning (PEFT)?

- **Fine-tune only a small subset of parameters.**
- **Keep the original pre-trained weights frozen.**
- **Significantly reduces computational and storage costs.**
- **Makes fine-tuning accessible on consumer hardware.**

## The Problem

**Full Fine-Tuning:**
- Llama 1 has 7 billion parameters
- Need 40GB+ GPU memory
- Takes hours to train
- Model file is 13GB

**PEFT Solution:**
- Only train ~0.1% of parameters
- Need 8-16GB GPU memory
- Takes minutes to train
- Model file is ~50MB

## What is PEFT?

**PEFT** = Parameter-Efficient Fine-Tuning


Instead of updating ALL parameters in a large model, we only update a small subset. This saves:
- Memory (90% less)
- Time (10x faster)
- Storage (tiny model files)

1. Parameter Efficiency
- **Before PEFT**: Train 7 billion parameters
- **After PEFT**: Train ~10 million parameters
- **Efficiency**: 99.9% reduction!


```python
# PYTORCH MODEL
import transformers

model_name = "openai-community/gpt2"
tokenizer  = transformers.AutoTokenizer.from_pretrained(model_name)
model      = transformers.AutoModelForCausalLM.from_pretrained(model_name) 
```

```python
# PEFT MODEL
import peft

lora_config = LoraConfig(r=16)

peft_model = peft.get_peft_model(model, lora_config)

peft_model.print_trainable_parameters()
```

```python
print("7. Training...")
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


```