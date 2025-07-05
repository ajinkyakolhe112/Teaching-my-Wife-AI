# Week 1, Lecture 2: Ollama and Local LLMs

## 1. Introduction to Ollama

*   **What is Ollama?**
    *   Ollama is a powerful and user-friendly tool that simplifies the process of downloading, setting up, and running Large Language Models (LLMs) locally on your own computer.
    *   It provides a command-line interface (CLI) and an API to interact with these models.
    *   It bundles model weights, configurations, and a way to serve the model into a single, easy-to-manage package.
    *   Think of it like Docker for LLMs – it makes complex setups much more accessible.

*   **Why Use Local LLMs?**
    *   **Privacy:** Your data (prompts and responses) stays on your machine. This is crucial for sensitive information or proprietary code that you don't want to send to third-party cloud services.
    *   **Offline Access:** Once a model is downloaded, you can use it without an internet connection.
    *   **Cost:** While there's an initial investment in hardware (if needed), running models locally avoids per-token or subscription fees associated with many cloud-based LLM APIs.
    *   **Experimentation & Customization:** Local LLMs offer greater flexibility for experimentation with different models, parameters, and even fine-tuning (though fine-tuning itself is a more advanced topic).
    *   **Control:** You have full control over the model and its usage.
    *   **Learning:** Running models locally provides a deeper understanding of how they work and their resource requirements.

## 2. Installation Guide

Ollama supports macOS, Linux, and Windows. The easiest way to install Ollama is by downloading the official application from the Ollama website.

*   **Official Download Page:** [https://ollama.com/download](https://ollama.com/download)

Follow the instructions for your specific operating system:

*   **macOS:**
    1.  Download the `Ollama-darwin.zip` file from the website.
    2.  Unzip the file.
    3.  Move the `Ollama` application to your Applications folder.
    4.  Run the Ollama application. It will install the CLI tools and start the Ollama server in the background (you'll see an icon in your menu bar).

*   **Linux:**
    1.  The recommended way is to use the install script:
        ```bash
        curl -fsSL https://ollama.com/install.sh | sh
        ```
    2.  This script will download the Ollama binary and set up the `ollama` command for your user.
    3.  Ollama typically runs as a systemd service (`ollama.service`). After installation, the server should start automatically. You can check its status with `systemctl status ollama`.

*   **Windows (Preview):**
    1.  Download the `OllamaSetup.exe` installer from the website.
    2.  Run the installer and follow the on-screen prompts.
    3.  Ollama will be installed, and the server will run in the background.

**Verification:**
After installation, open your terminal (Terminal on macOS/Linux, PowerShell or Command Prompt on Windows) and type:
```bash
ollama --version
```
This should display the installed Ollama version.

## 3. Pulling Models

Ollama hosts a library of popular open-source models that you can easily download.

*   **Command:** `ollama pull <model_name>:<tag>`
    *   `<model_name>`: The name of the model (e.g., `llama3`, `phi3`, `codellama`, `mistral`).
    *   `<tag>` (optional): Specific version or variant of the model (e.g., `latest`, `7b`, `8x7b`, `instruct`). If omitted, `latest` is usually assumed.

*   **Examples:**
    *   To pull the latest Llama 3 model (usually the smallest instruction-tuned version):
        ```bash
        ollama pull llama3
        ```
    *   To pull the Phi-3 mini model:
        ```bash
        ollama pull phi3
        ```
    *   To pull a specific version of Code Llama (e.g., the 7 billion parameter base model):
        ```bash
        ollama pull codellama:7b-code
        ```
    *   To see all available tags for a model, you can check the Ollama library online: [https://ollama.com/library](https://ollama.com/library)

*   **Model Sizes and Resource Considerations:**
    *   LLMs come in various sizes, indicated by their parameter count (e.g., 3B, 7B, 8x7B, 70B).
    *   **Larger models** generally offer better performance and coherence but require more RAM and disk space, and are slower to run.
    *   **Smaller models** are faster, use less RAM, but might not be as capable for complex tasks.
    *   **RAM is key:**
        *   ~3B parameter models: Need at least 8GB RAM (though 16GB is more comfortable).
        *   ~7B parameter models: Typically need 16GB RAM.
        *   ~13B parameter models: Typically need 32GB RAM.
        *   Larger models (30B+): Often require 64GB RAM or more, and powerful GPUs for reasonable speed.
    *   Ollama will download model weights to a local directory (usually `~/.ollama/models` on macOS/Linux and `C:\Users\<username>\.ollama\models` on Windows). Ensure you have sufficient disk space (models can range from a few GBs to tens of GBs).

*   **Listing Downloaded Models:**
    To see which models you have downloaded locally:
    ```bash
    ollama list
    ```

## 4. Basic CLI Interaction

The simplest way to interact with a downloaded model is using the `ollama run` command.

*   **Starting a Chat Session:**
    ```bash
    ollama run <model_name>
    ```
    For example, to chat with Llama 3:
    ```bash
    ollama run llama3
    ```
    This will load the model and present you with a `>>>` prompt.

*   **Interacting with the Model:**
    *   Simply type your prompt (question, instruction, or start of a story) and press Enter.
    *   The model will generate a response.
    *   The conversation maintains context, so you can ask follow-up questions.

    Example:
    ```
    >>> Tell me a short story about a brave robot.
    Once upon a time, in a bustling city of whirring gears and flashing lights, lived a small sanitation bot named Bolt. Unlike his fellow bots who diligently swept streets and collected refuse, Bolt dreamed of adventure... (model continues)

    >>> What was the robot's name?
    The robot's name was Bolt.
    ```

*   **Useful Commands within Ollama Chat:**
    While in an `ollama run` session, you can use special commands:
    *   `/bye`: Exit the current chat session and return to your terminal.
    *   `/list`: Show all models you have downloaded locally.
    *   `/show info`: Display information about the current model being used (parameters, family, etc.).
    *   `/show license`: Display the license for the current model.
    *   `/?` or `/help`: Show available commands.

## 5. Using the Ollama Python Library

Ollama also provides a Python library to interact with your local models programmatically.

*   **Installation:**
    Make sure you have Python installed (preferably version 3.7+).
    ```bash
    pip install ollama
    ```

*   **Basic Usage:**

    *   **Ensure Ollama Server is Running:** The Ollama application (or `ollama serve` command / `ollama.service`) must be running in the background for the Python client to connect.

    *   **Listing Local Models:**
        ```python
        import ollama

        try:
            models = ollama.list()
            print("Available models:")
            for model in models['models']:
                print(f"- {model['name']} (Size: {model['size'] // (1024**3)}GB)")
        except Exception as e:
            print(f"Error connecting to Ollama or listing models: {e}")
            print("Ensure the Ollama application is running.")

        ```

    *   **Generating a Simple Completion (`ollama.generate()`):**
        This is for a single turn, stateless generation.
        ```python
        import ollama

        try:
            response = ollama.generate(
                model='llama3', # Or any other model you have downloaded
                prompt='Why is the sky blue?'
            )
            print(response['response'])
        except Exception as e:
            print(f"Error during generation: {e}")
        ```

    *   **Chatting with a Model (`ollama.chat()`):**
        This method is stateful and allows for conversational history.
        ```python
        import ollama

        messages = [
            {
                'role': 'user',
                'content': 'Hello! Can you tell me a fun fact about programming?',
            },
        ]

        try:
            response = ollama.chat(
                model='llama3', # Or any other model
                messages=messages
            )
            print(f"AI: {response['message']['content']}")

            # Add AI's response to messages for context
            messages.append(response['message'])

            # Follow-up question
            messages.append({
                'role': 'user',
                'content': 'What language was that fun fact related to?'
            })
            
            follow_up_response = ollama.chat(model='llama3', messages=messages)
            print(f"AI: {follow_up_response['message']['content']}")

        except Exception as e:
            print(f"Error during chat: {e}")
        ```

    *   **Streaming Responses:**
        For longer generations, you might want to stream the response token by token as it's generated, rather than waiting for the entire response. Both `ollama.generate()` and `ollama.chat()` support streaming by adding `stream=True`.

        ```python
        import ollama

        try:
            stream = ollama.generate(
                model='llama3',
                prompt='Write a short poem about a cat watching rain.',
                stream=True
            )

            print("Streaming response:")
            for chunk in stream:
                print(chunk['response'], end='', flush=True)
            print() # Newline at the end
        except Exception as e:
            print(f"Error during streaming generation: {e}")
        ```

        When streaming with `ollama.chat()`, each `chunk` will contain a `message` dictionary similar to the non-streaming response, but the `content` will be partial.

This covers the basics of getting started with Ollama, both from the command line and using its Python library. The lab session will provide hands-on practice with these concepts.
---

This concludes the lecture material for Week 1. The next step will be the lab session where participants will install Ollama and interact with their first local LLM.
