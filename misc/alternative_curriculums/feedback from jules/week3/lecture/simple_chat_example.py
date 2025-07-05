from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
tiny_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(tiny_model_name)
model = AutoModelForCausalLM.from_pretrained(tiny_model_name, torch_dtype=torch.float16, device_map="auto")

# Step 1: Tokenizer converts text to model inputs
prompt = "What is the capital of France?"
inputs = tokenizer(f"<|system|>\nYou are a helpful AI assistant.\n<|user|>\n{prompt}\n<|assistant|>\n", return_tensors="pt").to(model.device)

# Step 2: Model generates output tokens
output_tokens = model.generate(**inputs, max_length=200)

# Step 3: Tokenizer converts output tokens back to text
response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# Step 4: Print the response
print(response) 