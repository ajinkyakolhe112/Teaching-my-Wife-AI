# Lecture 11: Simple PEFT Fine-Tuning

## What is PEFT?

**PEFT** = Parameter-Efficient Fine-Tuning

Instead of updating ALL parameters in a large model, we only update a small subset. This saves:
- Memory (90% less)
- Time (10x faster)
- Storage (tiny model files)

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

## How LoRA Works

**LoRA** = Low-Rank Adaptation

1. **Freeze** the original model
2. **Add** small trainable matrices to key layers
3. **Train** only these small matrices

```
Original: W (frozen)
LoRA:    W + α/r * (A × B)  (trainable)
```

Where:
- A, B are small matrices (rank r=16)
- α is a scaling factor (32)
- Only A and B are trained!

## Simple Implementation

### Step 1: Load Dataset
```python
# Load our Pride and Prejudice Q&A data
with open('pride_prejudice_ift.json', 'r') as f:
    data = json.load(f)

# Format as instruction-response pairs
formatted_data = []
for item in data:
    formatted_data.append({
        "text": f"### Instruction:\n{item['question']}\n\n### Response:\n{item['answer']}"
    })
```

### Step 2: Load Model
```python
# Load Llama with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,  # Saves memory
    torch_dtype=torch.float16
)
```

### Step 3: Set Up LoRA
```python
lora_config = LoraConfig(
    r=16,                    # Rank (smaller = less memory)
    lora_alpha=32,           # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)
```

### Step 4: Train
```python
trainer = Trainer(
    model=peft_model,
    train_dataset=tokenized_dataset,
    args=TrainingArguments(
        num_train_epochs=2,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        fp16=True,  # Mixed precision
    )
)

trainer.train()
```

### Step 5: Test
```python
# Ask questions about Pride and Prejudice
question = "Who is Mr. Darcy?"
prompt = f"### Instruction:\n{question}\n\n### Response:\n"

outputs = peft_model.generate(
    input_ids=tokenizer(prompt, return_tensors="pt")["input_ids"],
    max_length=512,
    temperature=0.7
)
```

## Key Concepts

### 1. Parameter Efficiency
- **Before PEFT**: Train 7 billion parameters
- **After PEFT**: Train ~10 million parameters
- **Efficiency**: 99.9% reduction!

### 2. Memory Savings
- **4-bit quantization**: Reduces memory by 75%
- **LoRA**: Only stores small adapters
- **Result**: 8GB instead of 40GB

### 3. Speed
- **Fewer parameters** = faster training
- **Smaller gradients** = faster updates
- **Result**: Minutes instead of hours

### 4. Quality
- **Base model** keeps general knowledge
- **LoRA adapters** add domain-specific knowledge
- **Result**: Best of both worlds!

## What You Learned

✅ **PEFT reduces computational cost dramatically**
✅ **LoRA adds small trainable matrices to frozen layers**
✅ **4-bit quantization saves memory**
✅ **Model maintains general capabilities while gaining specific knowledge**

## Try It Yourself

```bash
# Install requirements
pip install transformers peft torch datasets

# Set your Hugging Face token
export HUGGING_FACE_HUB_TOKEN="your_token_here"

# Run the tutorial
python simple_peft_tutorial.py
```

## Next Steps

1. **Experiment with different ranks** (r=8, 16, 32)
2. **Try different target modules** (attention, MLP layers)
3. **Use your own dataset** (any Q&A format)
4. **Deploy the model** (tiny file size!)

---

**Remember**: PEFT makes fine-tuning accessible to everyone, not just those with massive GPU clusters! 