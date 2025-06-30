# Building Instruction Datasets and Fine-tuning: Harry Potter Example

## Overview
In this lecture, we'll learn how to:
1. Create an instruction dataset from domain-specific content (Harry Potter books)
2. Fine-tune a language model on this dataset

## Part 1: Creating Instruction Dataset

### What is an Instruction Dataset?
An instruction dataset consists of pairs of:
- Instructions/Questions
- Expected responses/answers

### Steps to Create Harry Potter Instruction Dataset

1. **Data Collection**
   - Source: All 7 Harry Potter books
   - Format: Text files or PDFs
   - Size: Approximately 1.1 million words across all books

2. **Types of Instructions to Create**
   ```python
   instruction_types = [
       "Character Analysis",
       "Plot Explanation",
       "Spell Information",
       "Location Description",
       "Magical Object Details",
       "Timeline Questions",
       "Relationship Analysis"
   ]
   ```

3. **Example Instruction-Response Pairs**
   ```json
   {
       "instruction": "Explain the significance of the Deathly Hallows in the Harry Potter series.",
       "response": "The Deathly Hallows are three magical objects: the Elder Wand, the Resurrection Stone, and the Cloak of Invisibility. According to the tale of the Three Brothers, possessing all three would make one the Master of Death. The symbol represents the three objects: a vertical line for the wand, a circle for the stone, and a triangle for the cloak. Harry Potter unknowingly possessed all three at different points, making him the true Master of Death.",
       "type": "Magical Object Details"
   }
   ```

4. **Dataset Creation Process**
   ```python
   # Example code structure
   def create_instruction_dataset(book_text):
       dataset = []
       
       # Extract character information
       characters = extract_characters(book_text)
       for character in characters:
           instruction = f"Describe the character {character['name']} and their role in the story."
           response = character['description']
           dataset.append({
               "instruction": instruction,
               "response": response,
               "type": "Character Analysis"
           })
       
       # Extract spell information
       spells = extract_spells(book_text)
       for spell in spells:
           instruction = f"What is the spell {spell['name']} and what does it do?"
           response = spell['description']
           dataset.append({
               "instruction": instruction,
               "response": response,
               "type": "Spell Information"
           })
       
       return dataset
   ```

## Part 2: Fine-tuning the Model

### Prerequisites
- Base model (e.g., GPT-2, LLaMA)
- Instruction dataset
- Fine-tuning framework (e.g., Hugging Face Transformers)

### Fine-tuning Process

1. **Data Preparation**
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   import torch
   
   # Load tokenizer and model
   tokenizer = AutoTokenizer.from_pretrained("gpt2")
   model = AutoModelForCausalLM.from_pretrained("gpt2")
   
   # Prepare dataset
   def prepare_dataset(instructions, responses):
       formatted_data = []
       for instruction, response in zip(instructions, responses):
           text = f"Instruction: {instruction}\nResponse: {response}\n"
           formatted_data.append(text)
       return formatted_data
   ```

2. **Training Configuration**
   ```python
   from transformers import TrainingArguments, Trainer
   
   training_args = TrainingArguments(
       output_dir="./harry_potter_model",
       num_train_epochs=3,
       per_device_train_batch_size=4,
       save_steps=1000,
       save_total_limit=2,
   )
   
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_dataset,
   )
   ```

3. **Fine-tuning Execution**
   ```python
   # Start training
   trainer.train()
   
   # Save the model
   model.save_pretrained("./harry_potter_model")
   tokenizer.save_pretrained("./harry_potter_model")
   ```

### Best Practices
1. **Data Quality**
   - Ensure diverse instruction types
   - Maintain consistent response format
   - Include both simple and complex questions

2. **Training Considerations**
   - Start with a small dataset for testing
   - Monitor for overfitting
   - Use validation split
   - Implement early stopping

3. **Evaluation**
   - Test on unseen instructions
   - Compare responses with original text
   - Check for factual accuracy

## Example Usage

```python
# Load fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./harry_potter_model")
tokenizer = AutoTokenizer.from_pretrained("./harry_potter_model")

# Generate response
def generate_response(instruction):
    input_text = f"Instruction: {instruction}\nResponse:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=200,
        num_return_sequences=1,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Response:")[1].strip()

# Test the model
instruction = "What is the significance of the Sorting Hat?"
response = generate_response(instruction)
print(f"Instruction: {instruction}")
print(f"Response: {response}")
```

## Conclusion
This lecture demonstrated how to:
1. Create a structured instruction dataset from Harry Potter books
2. Fine-tune a language model on this domain-specific dataset
3. Use the fine-tuned model for generating responses

The same process can be applied to any domain-specific content, making it a powerful tool for creating specialized AI assistants. 