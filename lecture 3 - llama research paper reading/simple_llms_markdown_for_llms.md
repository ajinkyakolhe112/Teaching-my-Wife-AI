# Converted from simple_llms.ipynb - Markdown format optimized for LLM readability

```
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Llama 7B Chat model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def chat_with_llama(prompt, max_length=200):
    # Format the prompt for chat
    chat_prompt = f"<s>[INST] {prompt} [/INST]"
    
    # Tokenize and generate
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # Decode and clean up the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the input prompt from the response
    response = response.replace(chat_prompt, "").strip()
    return response

# Test the chat model
test_prompt = "What is the capital of France?"
print("Llama 7B Chat Response:")
print(chat_with_llama(test_prompt))
```

```
# Load TinyLlama Chat model and tokenizer
tiny_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tiny_tokenizer = AutoTokenizer.from_pretrained(tiny_model_name)
tiny_model = AutoModelForCausalLM.from_pretrained(tiny_model_name, torch_dtype=torch.float16, device_map="auto")

def chat_with_tinyllama(prompt, max_length=200):
    # Format the prompt for chat
    chat_prompt = f"<|system|>\nYou are a helpful AI assistant.\n<|user|>\n{prompt}\n<|assistant|>\n"
    
    # Tokenize and generate
    inputs = tiny_tokenizer(chat_prompt, return_tensors="pt").to(tiny_model.device)
    outputs = tiny_model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # Decode and clean up the response
    response = tiny_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the input prompt from the response
    response = response.replace(chat_prompt, "").strip()
    return response

# Test the chat model
test_prompt = "What is the capital of France?"
print("TinyLlama Chat Response:")
print(chat_with_tinyllama(test_prompt))
```
