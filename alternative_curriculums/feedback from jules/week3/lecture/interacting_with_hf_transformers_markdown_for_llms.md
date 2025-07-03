# Converted from interacting_with_hf_transformers.ipynb - Markdown format optimized for LLM readability

# Interacting with LLMs using Hugging Face `transformers`

This notebook demonstrates how to load and interact with pre-trained Large Language Models (LLMs) using the Hugging Face `transformers` library. We will explore two different chat models: `meta-llama/Llama-2-7b-chat-hf` and `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

**Key Steps:**
1. **Installation:** Ensure you have the necessary libraries installed. You'll primarily need `transformers` and `torch`.
   ```bash
   pip install transformers torch
   # For GPU support, ensure PyTorch is installed with CUDA compatibility.
   # For Apple Silicon (MPS), ensure you have a recent PyTorch version.
   ```
2. **Model Loading:** Choose a model from the Hugging Face Hub and load it using `AutoModelForCausalLM` and its corresponding tokenizer using `AutoTokenizer`.
3. **Prompt Formatting:** Different models require different prompt formats, especially for chat applications. We'll see examples for Llama 2 and TinyLlama.
4. **Tokenization:** Convert the formatted prompt into token IDs that the model can understand.
5. **Generation:** Use the model's `generate()` method to produce text based on the input tokens. This method has various parameters to control the output (e.g., `max_length`, `temperature`, `top_p`).
6. **Decoding:** Convert the generated token IDs back into human-readable text.

## 1. Interacting with `meta-llama/Llama-2-7b-chat-hf`

Llama 2 is a family of pre-trained and fine-tuned LLMs released by Meta. The `7b-chat-hf` version is a 7 billion parameter model specifically fine-tuned for dialogue use cases and made available in the Hugging Face `transformers` format.

**Note on Access:** To use Meta's Llama 2 models, you typically need to request access through Meta's official channels and agree to their terms of use. You also need to be authenticated with Hugging Face Hub (`huggingface-cli login`) if you haven't already, and have accepted the model's terms on its Hugging Face page.

```python
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define the model name from Hugging Face Hub
model_name = "meta-llama/Llama-2-7b-chat-hf"

# Load the tokenizer
# The tokenizer converts text into a sequence of numbers (tokens) that the model can understand.
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True) # Add use_auth_token=True if required

# Load the pre-trained model
# AutoModelForCausalLM is suitable for text generation tasks.
# torch_dtype=torch.float16 uses half-precision floating points to save memory and potentially speed up inference.
# device_map="auto" automatically distributes the model across available hardware (GPU, CPU, MPS).
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    use_auth_token=True # Add use_auth_token=True if required
)

def chat_with_llama(prompt, max_length=200):
    # Format the prompt according to Llama 2's chat template.
    # The template often involves special tokens like <s> (start of sequence), [INST] (user instruction), and [/INST] (end of instruction).
    # This specific format is crucial for getting good responses from the chat model.
    chat_prompt = f"<s>[INST] {prompt} [/INST]"
    
    # Tokenize the formatted prompt and move tensors to the model's device (e.g., GPU).
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    # Generate text using the model.
    outputs = model.generate(
        **inputs, # Pass the tokenized inputs to the model.
        max_length=max_length, # Maximum length of the generated sequence (prompt + response).
        num_return_sequences=1, # Number of different sequences to generate.
        temperature=0.7, # Controls randomness. Lower values make the output more deterministic, higher values make it more random.
        top_p=0.9,       # Nucleus sampling: considers the smallest set of tokens whose cumulative probability exceeds top_p.
        do_sample=True   # Whether to use sampling (temperature, top_p) or greedy decoding.
    )
    
    # Decode the generated tokens back into a string.
    # skip_special_tokens=True removes tokens like <s> and [INST] from the output.
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the response: Remove the input prompt part to get only the model's answer.
    # The raw output often includes the prompt, so we strip it.
    # Note: Llama 2's specific prompt format `<s>[INST] {prompt} [/INST]` is what we are removing here.
    # The model's actual response starts after `[/INST]`.
    # A more robust way might be to find the end of `[/INST]` and take the substring after it.
    # For this example, simple replacement is used if the prompt is exactly matched.
    if response.startswith(prompt): # A simple check
        response = response[len(prompt):].strip()
    elif f"[INST] {prompt} [/INST]" in response: # More specific to Llama2 format
        response = response.split(f"[/INST]")[-1].strip()
        
    return response

# Test the Llama 2 chat model
test_prompt_llama = "What is the capital of France? And what are 3 things to do there?"
print("Llama 2 7B Chat Response:")
print(chat_with_llama(test_prompt_llama))
```

### Explanation of `generate()` parameters:

*   `**inputs`: This unpacks the dictionary returned by the tokenizer (containing `input_ids` and `attention_mask`) as arguments to the `generate` method.
*   `max_length`: Sets the maximum number of tokens for the generated output (including the prompt). If the model generates a response that, combined with the prompt, exceeds this length, generation will stop.
*   `num_return_sequences`: Specifies how many different sequences to generate. Typically set to 1 for chat.
*   `temperature`: A float value (e.g., 0.7). It controls the randomness of the model's output. 
    *   Lower temperatures (e.g., 0.2) make the output more deterministic and focused (good for factual answers).
    *   Higher temperatures (e.g., 0.9, 1.0) make the output more random and creative (good for story generation).
*   `top_p` (Nucleus Sampling): A float value (e.g., 0.9). The model considers only the most probable tokens whose cumulative probability mass exceeds `top_p`. This can lead to more diverse and interesting outputs compared to greedy decoding, while avoiding very low-probability tokens.
*   `do_sample=True`: This flag must be set to `True` to enable sampling-based generation strategies (using `temperature` and `top_p`). If `False`, the model uses greedy decoding (always picks the most probable next token), which can be repetitive.

## 2. Interacting with `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

TinyLlama is a smaller, more compact language model designed to be efficient. The `1.1B-Chat-v1.0` version has 1.1 billion parameters and is fine-tuned for chat. Smaller models like this are easier to run on consumer hardware.

The prompt format for TinyLlama's chat model is different from Llama 2's. It typically uses a system prompt and user/assistant roles.

```python
# Define the model name for TinyLlama
tiny_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load the TinyLlama tokenizer
tiny_tokenizer = AutoTokenizer.from_pretrained(tiny_model_name)

# Load the TinyLlama model
tiny_model = AutoModelForCausalLM.from_pretrained(
    tiny_model_name, 
    torch_dtype=torch.float16, # Use float16 for efficiency
    device_map="auto" # Automatically use available hardware
)

def chat_with_tinyllama(prompt, max_length=200):
    # Format the prompt according to TinyLlama's chat template.
    # This format includes roles like <|system|>, <|user|>, and <|assistant|>.
    # The system prompt can be used to set the context or behavior of the AI assistant.
    chat_prompt = f"<|system|>\nYou are a helpful AI assistant who provides concise answers.\n<|user|>\n{prompt}\n<|assistant|>\n"
    
    # Tokenize the prompt and move to the model's device.
    inputs = tiny_tokenizer(chat_prompt, return_tensors="pt").to(tiny_model.device)
    
    # Generate response using the same parameters as before for consistency, 
    # but these can be tuned per model.
    outputs = tiny_model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # Decode the generated tokens.
    response = tiny_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the response: Remove the input prompt part.
    # The model's response starts after the "<|assistant|>\n" part of the prompt.
    assistant_marker = "<|assistant|>\n"
    marker_position = response.find(assistant_marker)
    if marker_position != -1:
        response = response[marker_position + len(assistant_marker):].strip()
        
    return response

# Test the TinyLlama chat model
test_prompt_tiny = "What is the capital of France? And what are 3 things to do there?"
print("TinyLlama 1.1B Chat Response:")
print(chat_with_tinyllama(test_prompt_tiny))
```

## Conclusion

This notebook provided a hands-on look at interacting with two different open-source chat models using the Hugging Face `transformers` library. Key takeaways include:

1.  **Library Power:** The `transformers` library simplifies loading models and tokenizers with just a few lines of code.
2.  **Model Variety:** Many different models are available on the Hugging Face Hub, each with its own characteristics (size, training data, performance).
3.  **Prompt Engineering is Key:** The format of your prompt, including any special tokens or structures (like Llama 2's `[INST]` tags or TinyLlama's role markers), significantly impacts the quality and relevance of the model's response.
4.  **Generation Parameters:** Parameters like `temperature` and `top_p` allow you to control the creativity and determinism of the output.
5.  **Resource Management:** Using `torch_dtype=torch.float16` and `device_map="auto"` helps in managing memory and utilizing available hardware efficiently, which is especially important for larger models.

Experimenting with different models and prompt engineering techniques is crucial for getting the best results from LLMs.
