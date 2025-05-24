# Lab 4: Exploring Fine-tuning Setups and Pre-tuned Models

**Objective:** To understand the components of a fine-tuning process by examining configuration files and to analyze the behavior of pre-fine-tuned models.

**Prerequisites:**
*   Web browser for accessing Hugging Face Model Hub, GitHub, and documentation.
*   (Optional) Python environment with `transformers` and `ollama` for interacting with models. If you plan to do Task 3, ensure Ollama is running and you have pulled relevant models.

---

## Tasks:

### 1. Analyzing a Fine-tuning Configuration (Axolotl)

Axolotl is a popular tool that simplifies fine-tuning LLMs using YAML configuration files. We will examine an example configuration to understand its key components.

*   **Example Axolotl Configuration:**
    Below is a simplified example of an Axolotl YAML configuration file for fine-tuning a model like Phi-3 or Llama 3 using LoRA. You can also find many real-world examples in the [Axolotl GitHub repository's examples folder](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/examples/).

    ```yaml
    # Example Axolotl Configuration for LoRA Fine-tuning

    base_model: teknium/OpenHermes-2.5-Mistral-7B # Base model from Hugging Face Hub
    model_type: MistralForCausalLM # Type of the model class
    tokenizer_type: AutoTokenizer # Type of the tokenizer class

    datasets:
      - path: teknium/openhermes # Path to dataset on Hugging Face Hub or local path
        type: openhermes # Dataset type/format identifier (Axolotl has many predefined types)
        shards: 10 # Optional: use a fraction of dataset shards for faster runs

    dataset_prepared_path: last_run_prepared # Where to cache preprocessed data
    val_set_size: 0.05 # Percentage of data to use for validation (e.g., 5%)
    output_dir: ./qlora-out-hermes # Where to save fine-tuned model adapters and checkpoints

    sequence_len: 2048 # Maximum sequence length the model can handle
    sample_packing: true # Combines multiple short examples into one sequence for efficiency
    pad_to_sequence_len: true # Pad all sequences to `sequence_len`

    adapter: lora # Specify LoRA as the adapter type
    lora_r: 32 # Rank for LoRA matrices (common values: 8, 16, 32, 64)
    lora_alpha: 64 # Alpha for LoRA scaling (often 2*r)
    lora_dropout: 0.05 # Dropout probability for LoRA layers
    lora_target_modules: # Modules to apply LoRA to (varies by model architecture)
      - q_proj
      - v_proj
      # - k_proj # Sometimes k_proj is also included or other linear layers
      # - o_proj
      # - gate_proj
      # - up_proj
      # - down_proj

    gradient_accumulation_steps: 2
    micro_batch_size: 1 # Batch size per GPU
    num_epochs: 3 # Number of times to iterate over the dataset
    optimizer: paged_adamw_8bit # Memory-efficient AdamW optimizer
    learning_rate: 0.0002 # Peak learning rate for the optimizer
    lr_scheduler: cosine # Learning rate scheduler type (e.g., linear, cosine)
    weight_decay: 0.01

    fp16: true # Use mixed-precision training (reduces memory, speeds up training on compatible GPUs)
    # bf16: false # Alternative mixed-precision, requires newer GPUs

    gradient_checkpointing: true # Saves memory by recomputing some activations during backward pass
    # deepspeed: deepspeed_configs/zero2.json # Optional: for advanced distributed training

    # save_strategy: "steps" # How often to save checkpoints
    # save_steps: 200
    # logging_steps: 10
    ```

*   **Identify Key Sections & Parameters:**
    Examine the YAML configuration above (or one from the Axolotl examples link). Identify the following parameters in the configuration:
    *   `base_model`
    *   `datasets` (specifically the `path` and `type`)
    *   `tokenizer_type`
    *   `sequence_len`
    *   `sample_packing`
    *   `lora_r`
    *   `lora_alpha`
    *   `lora_dropout`
    *   `lora_target_modules`
    *   `gradient_accumulation_steps`
    *   `micro_batch_size`
    *   `num_epochs`
    *   `optimizer`
    *   `learning_rate`
    *   `output_dir`

*   **Explain Selected Parameters:**
    Choose 4 of the following parameters from the list above and briefly explain their role in the fine-tuning process in your own words:
    1.  `base_model`: *[Your explanation here]*
    2.  `datasets` (path and type): *[Your explanation here]*
    3.  `lora_r`: *[Your explanation here]*
    4.  `num_epochs`: *[Your explanation here]*
    5.  `learning_rate`: *[Your explanation here]*
    6.  `output_dir`: *[Your explanation here]*

    *(Example explanation for `base_model`: This parameter specifies the identifier of the pre-trained model from the Hugging Face Hub that will be used as the starting point for fine-tuning. All the general knowledge of this model will be adapted to the new task.)*

### 2. Exploring Pre-fine-tuned Models on Hugging Face Hub

The Hugging Face Model Hub hosts thousands of models, including many that have been pre-fine-tuned for specific tasks.

*   **Instructions:**
    1.  Go to the [Hugging Face Model Hub](https://huggingface.co/models).
    2.  Use the search bar and filters (e.g., sort by "Most Downloads", filter by "Task") to find **2-3 examples of models** that have been fine-tuned for specific tasks. Look for models with clear descriptions in their model cards.
        *   **Examples of tasks to look for:**
            *   Code generation (e.g., fine-tuned on Python, JavaScript).
            *   Instruction following / Chat (e.g., fine-tuned on datasets like Dolly, OpenHermes, Alpaca).
            *   Summarization.
            *   Sentiment analysis.
            *   Translation.

*   **For each model you find, note down the following:**

    *   **Model 1:**
        *   **Model Name/Link:**
        *   **Base Model (if specified):**
        *   **Task it was fine-tuned for:**
        *   **Dataset used (if specified or inferable):**
        *   **Example Prompt & Output (from model card, if available):**

    *   **Model 2:**
        *   **Model Name/Link:**
        *   **Base Model (if specified):**
        *   **Task it was fine-tuned for:**
        *   **Dataset used (if specified or inferable):**
        *   **Example Prompt & Output (from model card, if available):**

    *   **(Optional) Model 3:**
        *   **Model Name/Link:**
        *   **Base Model (if specified):**
        *   **Task it was fine-tuned for:**
        *   **Dataset used (if specified or inferable):**
        *   **Example Prompt & Output (from model card, if available):**

    *(Hint: Look for model names that include terms like "fine-tuned", "ft", "chat", "instruct", or task-specific keywords like "code", "summary", "sentiment". Read the model cards carefully!)*

### 3. (Optional) Interacting with a Pre-fine-tuned Model (e.g., via Ollama)

This task allows you to qualitatively compare a base model with a version fine-tuned for a specific task.

*   **Instructions:**
    1.  **Choose Models:**
        *   Identify a base model available on Ollama (e.g., `phi3:mini`, `llama3:8b`).
        *   Find a fine-tuned version of that (or a similar) base model also available on Ollama that is specialized for a particular task (e.g., code generation, instruction following).
            *   Example: `phi3:mini` (general) vs. a hypothetical `phi3:mini-code` (if available and fine-tuned for code). Or, compare `llama3:8b` (general) with a model known to be fine-tuned for better instruction following if you can find one on Ollama.
            *   *(Note: Specific fine-tuned versions readily available on Ollama can change. You might need to search the Ollama library or community resources for suitable pairs.)*
            *   If a direct fine-tune isn't available, you can compare two different models where one is known to be specialized (e.g., `phi3:mini` vs. `codellama:7b-instruct` for a coding task).

    2.  **Craft a Task-Specific Prompt:**
        *   Create a prompt that is specific to the task the fine-tuned model is designed for (e.g., if comparing a code model, give it a simple coding problem).
        *   Example coding prompt: `"Write a Python function that takes a list of strings and returns a new list containing only the strings that have more than 5 characters."`

    3.  **Interact with Both Models:**
        *   Using `ollama run <model_name>` or the Ollama Python SDK, provide the exact same prompt to both the base model and the fine-tuned model.
        *   Record the full responses from both.

    4.  **Compare and Observe:**
        *   **Base Model Response:** `[Paste response here]`
        *   **Fine-tuned Model Response:** `[Paste response here]`
        *   **Observations:**
            *   Did the fine-tuned model perform better on the specific task? How? (e.g., more accurate code, better instruction following, more relevant information).
            *   Were there differences in the style, formatting, or completeness of the responses?
            *   Did the base model struggle with any aspects of the task that the fine-tuned model handled well?

---

**Lab Conclusion:**
This lab provided a glimpse into the configuration aspects of fine-tuning and the results of such processes by exploring pre-fine-tuned models. Understanding these concepts helps in appreciating how LLMs can be specialized and in making informed decisions when considering using or building fine-tuned models. Even without running a full fine-tuning pipeline yourself, analyzing these components is a valuable learning step.
