# Converted from interacting_with_ollama_sdk.ipynb - Markdown format optimized for LLM readability

# Interacting with Local LLMs using the Ollama Python SDK

This notebook demonstrates how to use the Ollama Python SDK to interact with Large Language Models (LLMs) running locally via the Ollama service.

**Key Benefits of using Ollama and its SDK:**
*   **Privacy:** Your data stays on your local machine.
*   **Simplicity:** Ollama makes it easy to download and run various open-source LLMs.
*   **Control:** Full control over the models you use.
*   **Python Integration:** The SDK allows seamless integration of local LLMs into your Python applications.

**Prerequisites:**
1.  **Ollama Installed and Running:** Ensure you have Ollama installed and the Ollama application/service is running in the background. You can download it from [ollama.com](https://ollama.com/).
2.  **Ollama Python SDK Installed:**
    ```bash
    pip install ollama
    ```
3.  **Models Downloaded:** You need to have at least one model downloaded through Ollama. For example:
    ```bash
    ollama pull llama3 # Or any other model like phi3, mistral, etc.
    ```

## 1. Listing Available Local Models

You can list all the models you have downloaded locally using `ollama.list()`.

```python
import ollama

try:
    local_models = ollama.list()
    print("Available local models:")
    if local_models['models']:
        for model_info in local_models['models']:
            print(f"- Name: {model_info['name']}, Size: {model_info['size']//(1024**3):.2f} GB, Modified: {model_info['modified_at']}")
    else:
        print("No models found. Please pull a model using 'ollama pull <model_name>'")
except Exception as e:
    print(f"Error connecting to Ollama: {e}")
    print("Please ensure the Ollama application/service is running.")
```

## 2. Basic Chat Interaction (`ollama.chat()`)

The `ollama.chat()` function is the primary way to have conversations with a model. It takes the model name and a list of messages as input.

The `messages` list should contain dictionaries, each with a `role` (`user`, `assistant`, or `system`) and `content`.

```python
# Ensure you have a model listed from the previous step. Change 'llama3' if needed.
# If local_models list is empty or Ollama is not running, this will fail.
MODEL_TO_TEST = 'llama3' # Or 'phi3', 'mistral', etc.
if not local_models or not any(m['name'].startswith(MODEL_TO_TEST) for m in local_models.get('models',[])):
    print(f"Model '{MODEL_TO_TEST}' not found locally. Please run 'ollama pull {MODEL_TO_TEST}' or choose an available model.")
else:
    messages = [
        {
            'role': 'user',
            'content': 'Why is the sky blue?',
        },
    ]

    try:
        # The response is a dictionary-like object (ChatResponse)
        response = ollama.chat(model=MODEL_TO_TEST, messages=messages)
        
        # Accessing the content of the assistant's message
        print(f"Assistant's response (dict access): {response['message']['content']}")
        
        # You can also access fields directly from the response object if using newer ollama versions
        # print(f"Assistant's response (object access): {response.message.content}")
        
        # The full response object contains other useful information
        print(f"\nFull response object:\n{response}")
        
    except Exception as e:
        print(f"Error during chat: {e}")
        print(f"Ensure model '{MODEL_TO_TEST}' is available and Ollama is running.")
```

## 3. Multi-turn Conversation

To have a conversation, you append the assistant's response and the new user message to the `messages` list.

```python
if not local_models or not any(m['name'].startswith(MODEL_TO_TEST) for m in local_models.get('models',[])):
    print(f"Model '{MODEL_TO_TEST}' not found locally. Skipping multi-turn conversation.")
else:
    conversation_messages = [
        {
            'role': 'user',
            'content': 'Hi! My name is Alex. What are three interesting facts about Jupiter?'
        }
    ]
    
    try:
        print(f"User: {conversation_messages[0]['content']}")
        response1 = ollama.chat(model=MODEL_TO_TEST, messages=conversation_messages)
        assistant_response1 = response1['message']['content']
        print(f"Assistant: {assistant_response1}")
        
        # Add assistant's response to the conversation history
        conversation_messages.append({'role': 'assistant', 'content': assistant_response1})
        
        # Follow-up question
        follow_up_question = 'Of those facts, which one is most surprising to humans and why?'
        conversation_messages.append({'role': 'user', 'content': follow_up_question})
        print(f"User: {follow_up_question}")
        
        response2 = ollama.chat(model=MODEL_TO_TEST, messages=conversation_messages)
        assistant_response2 = response2['message']['content']
        print(f"Assistant: {assistant_response2}")
        
    except Exception as e:
        print(f"Error during multi-turn chat: {e}")
```

## 4. Streaming Responses

For longer generations, you might want to stream the response token by token as it's generated. This is done by setting `stream=True`.
Each part of the stream provides a chunk of the response.

```python
if not local_models or not any(m['name'].startswith(MODEL_TO_TEST) for m in local_models.get('models',[])):
    print(f"Model '{MODEL_TO_TEST}' not found locally. Skipping streaming example.")
else:
    stream_messages = [
        {
            'role': 'user',
            'content': 'Write a short poem about a cat observing a rainy day.'
        }
    ]
    
    try:
        print(f"User: {stream_messages[0]['content']}")
        print("Assistant (streaming): ", end="")
        
        # stream=True returns a generator
        stream_response = ollama.chat(
            model=MODEL_TO_TEST,
            messages=stream_messages,
            stream=True
        )
        
        for chunk in stream_response:
            # Each chunk is a dictionary, similar to the non-streaming response,
            # but 'content' will be a part of the full message.
            print(chunk['message']['content'], end='', flush=True)
        print() # For a newline at the end of the streamed response
            
    except Exception as e:
        print(f"\nError during streaming chat: {e}")
```

## 5. Simple Generation (`ollama.generate()`)

For tasks that don't require chat history or complex role structures, `ollama.generate()` is a simpler alternative. It takes a model name and a prompt string.

You can also provide `system`, `template` (if the model uses one), and `context` (for stateless history) as optional parameters.

```python
if not local_models or not any(m['name'].startswith(MODEL_TO_TEST) for m in local_models.get('models',[])):
    print(f"Model '{MODEL_TO_TEST}' not found locally. Skipping generate example.")
else:
    prompt_text = "Translate this sentence to French: 'Hello, how are you today?'"
    
    try:
        print(f"Prompt: {prompt_text}")
        generate_response = ollama.generate(model=MODEL_TO_TEST, prompt=prompt_text)
        print(f"Response: {generate_response['response']}")
        
        # Example with streaming for generate
        print("\nStreaming generate response: ", end="")
        stream_gen_response = ollama.generate(model=MODEL_TO_TEST, prompt=prompt_text, stream=True)
        for chunk in stream_gen_response:
            print(chunk['response'], end='', flush=True)
        print()
            
    except Exception as e:
        print(f"\nError during generation: {e}")
```

## 6. Getting Model Information (`ollama.show()`)

The `ollama.show()` command provides detailed information about a specific model, including its parameters, template, and system prompt if defined in its Modelfile.

```python
if not local_models or not any(m['name'].startswith(MODEL_TO_TEST) for m in local_models.get('models',[])):
    print(f"Model '{MODEL_TO_TEST}' not found locally. Skipping show model example.")
else:
    try:
        print(f"--- Information for model: {MODEL_TO_TEST} ---")
        model_details = ollama.show(MODEL_TO_TEST)
        # Print some key details
        if 'license' in model_details:
            print(f"License: {model_details['license']}")
        if 'modelfile' in model_details:
            print(f"Modelfile (first 150 chars): {model_details['modelfile'][:150]}...")
        if 'parameters' in model_details:
            print(f"Parameters: {model_details['parameters']}")
        if 'template' in model_details: # Deprecated, use format
            print(f"Template: {model_details['template']}")
        if 'details' in model_details and 'format' in model_details['details']:
             print(f"Format: {model_details['details']['format']}")
        if 'details' in model_details and 'parameter_size' in model_details['details']:
             print(f"Parameter Size: {model_details['details']['parameter_size']}")
        if 'details' in model_details and 'quantization_level' in model_details['details']:
             print(f"Quantization Level: {model_details['details']['quantization_level']}")
            
    except Exception as e:
        print(f"Error showing model details: {e}")
```

## Conclusion

The Ollama Python SDK provides a convenient and powerful way to integrate local LLMs into your Python projects. Key functions like `ollama.list()`, `ollama.chat()`, `ollama.generate()`, and `ollama.show()` cover most common interaction needs, from listing models and having conversations to simple text generation and model inspection.

Remember that the Ollama application must be running for the SDK to function. Experiment with different models and their capabilities to find the best fit for your tasks.
