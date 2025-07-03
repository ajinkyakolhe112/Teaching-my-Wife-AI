# Lecture 11: Fine-tuning Language Models for Pride & Prejudice

## Overview
In this lecture, we'll learn how to fine-tune language models using the instruction dataset we created in Lecture 10. We'll explore both full fine-tuning and parameter-efficient fine-tuning (PEFT) methods.

## Learning Objectives
- Understand the fine-tuning process for instruction datasets
- Learn parameter-efficient fine-tuning techniques (LoRA, QLoRA)
- Implement fine-tuning using Hugging Face Transformers and PEFT
- Evaluate and test the fine-tuned model

## Part 1: Understanding Fine-tuning for Instruction Datasets

### What is Instruction Fine-tuning?
Instruction fine-tuning adapts a pre-trained language model to follow specific instructions by training it on instruction-response pairs. This enables the model to:
- Answer domain-specific questions accurately
- Follow specific response formats
- Maintain consistency with the source material

### Why Fine-tune for Pride & Prejudice?
- **Domain Expertise**: Create an AI that understands Jane Austen's work deeply
- **Consistent Responses**: Ensure accurate information about characters, plot, and themes
- **Educational Tool**: Help students and readers understand the novel better

## Part 2: Preparing for Fine-tuning

### Data Format Requirements
```python
# For causal language modeling
def format_for_causal_lm(instruction, response):
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}\n\n"

# For sequence-to-sequence models
def format_for_seq2seq(instruction, response):
    return {"input": instruction, "output": response}
```

### Loading and Preprocessing
```python
import json
from datasets import Dataset
from transformers import AutoTokenizer

def load_pride_prejudice_dataset(file_path: str):
    """Load the Pride & Prejudice instruction dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

def preprocess_dataset(dataset, tokenizer, max_length=512):
    """Preprocess the dataset for training"""
    
    def tokenize_function(examples):
        texts = []
        for instruction, response in zip(examples['instruction'], examples['response']):
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}\n\n"
            texts.append(text)
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    return dataset.map(tokenize_function, batched=True)
```

## Part 3: Full Fine-tuning Approach

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

def setup_full_finetuning():
    """Set up full fine-tuning configuration"""
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def train_full_model(model, tokenizer, train_dataset, output_dir="./pride_prejudice_model"):
    """Perform full fine-tuning"""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return trainer
```

## Part 4: Parameter-Efficient Fine-tuning (PEFT)

### LoRA (Low-Rank Adaptation)
```python
from peft import LoraConfig, get_peft_model, TaskType

def setup_lora_finetuning():
    """Set up LoRA fine-tuning configuration"""
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def train_lora_model(model, tokenizer, train_dataset, output_dir="./pride_prejudice_lora"):
    """Train model using LoRA"""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return trainer
```

### QLoRA (Quantized LoRA)
```python
from transformers import BitsAndBytesConfig

def setup_qlora_finetuning():
    """Set up QLoRA fine-tuning configuration"""
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    return model, tokenizer
```

## Part 5: Complete Fine-tuning Pipeline

```python
def complete_finetuning_pipeline():
    """Complete pipeline for fine-tuning Pride & Prejudice model"""
    
    print("Starting Pride & Prejudice fine-tuning pipeline...")
    
    # 1. Load dataset
    dataset = load_pride_prejudice_dataset("pride_prejudice_instruction_dataset.json")
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    # 2. Set up model and tokenizer
    model, tokenizer = setup_lora_finetuning()
    
    # 3. Preprocess dataset
    train_dataset = preprocess_dataset(train_dataset, tokenizer)
    eval_dataset = preprocess_dataset(eval_dataset, tokenizer)
    
    # 4. Train model
    trainer = train_lora_model(model, tokenizer, train_dataset)
    
    # 5. Evaluate model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    return model, tokenizer, trainer

def generate_response(model, tokenizer, instruction, max_length=200):
    """Generate a response using the fine-tuned model"""
    
    input_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:\n")[-1].strip()
    
    return response
```

## Part 6: Model Evaluation and Testing

```python
def evaluate_model_performance(model, tokenizer, test_questions):
    """Evaluate model performance on test questions"""
    
    results = []
    
    for question in test_questions:
        instruction = question["instruction"]
        expected_response = question["response"]
        
        generated_response = generate_response(model, tokenizer, instruction)
        
        result = {
            "instruction": instruction,
            "expected": expected_response,
            "generated": generated_response,
            "length_ratio": len(generated_response) / len(expected_response) if expected_response else 0
        }
        
        results.append(result)
    
    return results

def interactive_testing(model, tokenizer):
    """Interactive testing of the fine-tuned model"""
    
    print("Pride & Prejudice AI Assistant")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            break
        
        try:
            response = generate_response(model, tokenizer, user_input)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"\nError: {e}")

# Test questions for evaluation
test_questions = [
    {
        "instruction": "What is Elizabeth Bennet's personality like?",
        "response": "Elizabeth Bennet is intelligent, witty, and possesses a strong sense of self-respect."
    },
    {
        "instruction": "How does Mr. Darcy change throughout the novel?",
        "response": "Mr. Darcy learns to be more open and less judgmental of others."
    },
    {
        "instruction": "What is the significance of the first proposal scene?",
        "response": "The first proposal scene reveals both characters' true feelings and misunderstandings."
    }
]
```

## Part 7: Model Deployment

```python
def save_model_for_deployment(model, tokenizer, output_dir="./pride_prejudice_deployed"):
    """Save model for deployment"""
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    model_info = {
        "model_type": "pride_prejudice_assistant",
        "base_model": "microsoft/DialoGPT-medium",
        "fine_tuning_method": "LoRA",
        "description": "Fine-tuned model for Pride & Prejudice literary analysis"
    }
    
    with open(f"{output_dir}/model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)

def load_deployed_model(model_path):
    """Load deployed model"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer
```

## Part 8: Simple Web API

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model globally
model, tokenizer = load_deployed_model("./pride_prejudice_deployed")

@app.route('/ask', methods=['POST'])
def ask_question():
    """API endpoint for asking questions"""
    
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        response = generate_response(model, tokenizer, question)
        return jsonify({
            "question": question,
            "response": response
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

## Part 9: Advanced Techniques

### Multi-turn Conversations
```python
def handle_conversation(model, tokenizer, conversation_history):
    """Handle multi-turn conversations"""
    
    formatted_history = ""
    for turn in conversation_history:
        formatted_history += f"### Instruction:\n{turn['user']}\n\n### Response:\n{turn['assistant']}\n\n"
    
    current_instruction = conversation_history[-1]['user']
    formatted_history += f"### Instruction:\n{current_instruction}\n\n### Response:\n"
    
    response = generate_response(model, tokenizer, formatted_history)
    return response
```

### Controlled Generation
```python
def generate_with_controls(model, tokenizer, instruction, temperature=0.7, top_p=0.9, top_k=50):
    """Generate response with controlled parameters"""
    
    input_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=200,
            num_return_sequences=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:\n")[-1].strip()
    
    return response
```

## Part 10: Hands-on Exercise

### Exercise: Fine-tune Your Own Model

1. **Prepare Your Dataset**
   - Use the dataset you created in Lecture 10
   - Or create a smaller subset for testing

2. **Choose Your Approach**
   - Full fine-tuning (if you have sufficient resources)
   - LoRA fine-tuning (recommended for most cases)
   - QLoRA fine-tuning (for very limited resources)

3. **Train and Evaluate**
   ```python
   # Run the complete pipeline
   model, tokenizer, trainer = complete_finetuning_pipeline()
   
   # Test your model
   interactive_testing(model, tokenizer)
   ```

4. **Compare Results**
   - Test with different questions
   - Compare with base model responses
   - Evaluate consistency and accuracy

## Part 11: Best Practices and Tips

### Training Best Practices
1. **Start Small**: Begin with a subset of your dataset
2. **Monitor Training**: Use wandb or similar tools to track progress
3. **Validate Early**: Check model performance during training
4. **Save Checkpoints**: Save models at different stages
5. **Test Thoroughly**: Evaluate on diverse questions

### Common Issues and Solutions
1. **Overfitting**: Reduce training epochs or increase dataset size
2. **Poor Responses**: Check data quality and formatting
3. **Memory Issues**: Use PEFT methods or smaller models
4. **Inconsistent Outputs**: Adjust temperature and sampling parameters

### Optimization Tips
1. **Batch Size**: Start small and increase if memory allows
2. **Learning Rate**: Use learning rate scheduling
3. **Gradient Accumulation**: For larger effective batch sizes
4. **Mixed Precision**: Use fp16 or bf16 for faster training

## Conclusion

In this lecture, we've learned:
1. **Complete fine-tuning pipeline** for instruction datasets
2. **Parameter-efficient methods** (LoRA, QLoRA) for resource-constrained environments
3. **Model evaluation and testing** techniques
4. **Deployment strategies** for practical use
5. **Advanced techniques** for better performance

### Next Steps
- Experiment with different base models
- Try different PEFT configurations
- Explore ensemble methods
- Consider domain adaptation techniques

### Resources
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

### Assignment
1. Fine-tune a model using your instruction dataset
2. Evaluate the model on 10 test questions
3. Create a simple web interface for the model
4. Write a report comparing different fine-tuning approaches 