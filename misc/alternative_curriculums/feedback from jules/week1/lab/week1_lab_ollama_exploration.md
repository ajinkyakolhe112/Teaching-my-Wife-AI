# Lab 1: Getting Started with Ollama

**Objective:** To give users hands-on experience with installing Ollama, downloading models, and interacting with them through both the command line and Python.

**Prerequisites:**
*   Ollama installed (refer to `week1_lecture2_ollama_local_llms.md` from the lecture materials if you haven't installed it yet).
*   Python 3.x installed on your system.
*   The `ollama` Python library installed. If not, open your terminal and run:
    ```bash
    pip install ollama
    ```

---

## Tasks:

### 1. Verify Ollama Installation

*   Open your terminal (Terminal on macOS/Linux, PowerShell or Command Prompt on Windows).
*   Run the following command to check if Ollama is installed and to see its version:
    ```bash
    ollama --version
    ```
*   You should see output similar to `ollama version is <version_number>`. If you encounter an error, please revisit the installation guide in the lecture notes or the official Ollama documentation.

### 2. Pull Different Models

We will download a few models of varying sizes and specializations to get a feel for what's available. Depending on your internet speed and computer resources (especially RAM), some downloads might take time, and some larger models might run slowly.

*   **Pull a small, fast model:**
    ```bash
    ollama pull phi3:mini
    ```
    *(This is a relatively small model, good for quick tests. Typically requires ~4GB of RAM)*

*   **Pull a medium-sized, capable model:**
    ```bash
    ollama pull llama3:8b
    ```
    *(This is a more powerful model. Typically requires ~8-16GB of RAM)*

*   **Pull a code-specific model:**
    ```bash
    ollama pull codellama:7b
    ```
    *(This model is fine-tuned for code generation tasks. Typically requires ~8-16GB of RAM)*

*   **List all pulled models:**
    After the downloads are complete, list all models you have locally:
    ```bash
    ollama list
    ```
    You should see `phi3:mini`, `llama3:8b`, `codellama:7b`, and any other models you might have pulled previously in the output.

### 3. CLI Interaction

Let's interact with one of the models directly from the command line.

*   **Choose a model:** For this exercise, let's start with `phi3:mini` as it's generally faster.
*   **Run the model:**
    ```bash
    ollama run phi3:mini
    ```
    You should see a prompt like `>>> Send a message (/? for help)`.

*   **Experiment with Prompts:**
    Try at least 3-5 different types of prompts. Here are some ideas (feel free to come up with your own!):
    1.  **Factual Question:** `Why is the sky blue?`
    2.  **Creative Writing:** `Tell me a short story about a curious cat who discovers a hidden library.`
    3.  **Simple Code Generation:** `Write a short Python function that takes two numbers and returns their sum.`
    4.  **Summarization:** Provide a short paragraph of text and ask the model to summarize it. For example:
        `Summarize this text: "The quick brown fox jumps over the lazy dog. This sentence is famous because it contains all the letters of the English alphabet. It's often used for testing typewriters or keyboards."`
    5.  **Problem Solving/Brainstorming:** `What are some good names for a new coffee shop?`

*   **Observations:**
    *   How quickly does the model respond?
    *   Are the answers coherent?
    *   Are there any surprising or incorrect answers?
    *   If you have time, try one or two of the same prompts with `llama3:8b` (exit `phi3:mini` with `/bye` then run `ollama run llama3:8b`). Do you notice a difference in the quality or style of the responses? Note down any interesting observations.
    *   To exit the Ollama chat session, type `/bye` and press Enter.

### 4. Python Interaction

Now, let's interact with Ollama models using Python.

*   **Create a Python script:**
    Create a new Python file named `ollama_test.py` and open it in your favorite text editor or IDE.

*   **Write the script:**
    Copy and paste the boilerplate code below into your `ollama_test.py` file. Then, fill in the sections marked with `# TODO`.

    ```python
    import ollama

    def main():
        # --- 1. List Available Models ---
        print("--- Available Local Models ---")
        try:
            models_info = ollama.list()
            if models_info['models']:
                for model in models_info['models']:
                    print(f"- {model['name']}")
            else:
                print("No models found. Make sure you have pulled some models using 'ollama pull'.")
        except Exception as e:
            print(f"Error listing models: {e}")
            print("Ensure the Ollama application/service is running.")
        print("-" * 30)
        print()

        # --- 2. Chat with a Model ---
        chosen_model = 'phi3:mini' # You can change this to another model you have, e.g., 'llama3:8b'
        print(f"--- Chatting with {chosen_model} ---")

        messages = [
            {
                'role': 'user',
                'content': '', # TODO 1: Write your first question for the LLM here
            }
        ]

        try:
            # First question
            print(f"You: {messages[0]['content']}")
            response = ollama.chat(
                model=chosen_model,
                messages=messages
            )
            ai_response_content = response['message']['content']
            print(f"AI: {ai_response_content}\n")

            # Add AI's response to messages for context
            messages.append({'role': 'assistant', 'content': ai_response_content})

            # Second question (utilizing conversation history)
            messages.append({
                'role': 'user',
                'content': '', # TODO 2: Write your second question, which can be a follow-up to the first
            })
            
            print(f"You: {messages[-1]['content']}") # Print the latest user message
            follow_up_response = ollama.chat(
                model=chosen_model,
                messages=messages
            )
            ai_follow_up_content = follow_up_response['message']['content']
            print(f"AI: {ai_follow_up_content}\n")

        except Exception as e:
            print(f"Error during chat with {chosen_model}: {e}")
            print("Ensure the Ollama application/service is running and the model is available.")
        print("-" * 30)

    if __name__ == '__main__':
        main()
    ```

*   **Instructions for completing the script:**
    1.  Replace the empty string in `# TODO 1: Write your first question for the LLM here` with a question of your choice (e.g., "What is the capital of France?").
    2.  Replace the empty string in `# TODO 2: Write your second question, which can be a follow-up to the first` with another question. This question can be a follow-up to the first one, as the script maintains conversation history (e.g., if your first question was about France, you could ask "And what is its population?").

*   **Run the script:**
    Save your `ollama_test.py` file and run it from your terminal:
    ```bash
    python ollama_test.py
    ```
    Observe the output. It should first list your models and then print the conversation with the LLM.

---

## Submission (Optional/Self-check)

To help solidify your learning, consider gathering the following:

*   **Screenshots:** Take a screenshot of your CLI interaction when you were experimenting with prompts (Task 3).
*   **Python Script:** Save your completed `ollama_test.py` script.
*   **Brief Summary of Observations:** Write a short paragraph (3-5 sentences) summarizing:
    *   Your experience interacting with the models via CLI.
    *   Any differences you noticed between models (if you tried more than one for the same prompt).
    *   Your experience using the Ollama Python library.

This lab is primarily for your own learning and experimentation. Have fun exploring!
