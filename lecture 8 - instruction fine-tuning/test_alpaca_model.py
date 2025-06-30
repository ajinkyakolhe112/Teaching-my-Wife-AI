import torch
import transformers

def load_model_and_tokenizer():
    print("Loading model and tokenizer...")
    
    # Load tokenizer & model
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("./alpaca_model_final")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device

def generate_response(model, tokenizer, device, instruction, input_text=""):
    prompt = f"""
        Instruction: {instruction}
        Input: {input_text}
        Output:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Load the model
    model, tokenizer, device = load_model_and_tokenizer()
    
    # Test the model with some examples
    test_cases = [
        {
            "instruction": "Write a short poem about artificial intelligence.",
            "input": ""
        },
        {
            "instruction": "Explain the concept of machine learning to a 10-year-old.",
            "input": ""
        },
        {
            "instruction": "What are the three laws of robotics?",
            "input": ""
        }
    ]
    
    print("\nTesting model with different prompts...")
    print("="*50)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Instruction: {test['instruction']}")
        print(f"Input: {test['input']}")
        response = generate_response(model, tokenizer, device, test['instruction'], test['input'])
        print(f"Output: {response}")
        print("-"*50) 