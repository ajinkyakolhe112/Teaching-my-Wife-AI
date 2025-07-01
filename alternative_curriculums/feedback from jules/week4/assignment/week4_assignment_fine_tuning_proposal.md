# Assignment 4: Fine-tuning Strategy Proposal

**Objective:** To research and propose a conceptual fine-tuning strategy for a given problem and dataset, demonstrating understanding of the concepts learned in Week 4.

**Prerequisites:**
*   Understanding of concepts covered in Week 4 Lecture: Fine-tuning LLMs for Specific Tasks (Conceptual).
*   Familiarity with terms like base model, dataset, LoRA, QLoRA, evaluation metrics, etc.

---

## Instructions:

Develop a conceptual fine-tuning strategy by addressing each of the sections below. You can choose one of the provided scenarios or define your own specific problem where fine-tuning might be beneficial.

---

### 1. Problem Scenario

*   **Choose ONE of the following scenarios OR define your own simple, specific problem.**
    *   **Scenario A:** Adapting an LLM to generate polite and empathetic customer service email replies for a small e-commerce company that sells handmade crafts. Replies should address common issues like shipping delays or product inquiries.
    *   **Scenario B:** Making an LLM better at generating concise Python function docstrings in a specific format (e.g., Google Python Style Guide) for a library of utility functions.
    *   **Scenario C:** Fine-tuning an LLM to answer questions based *only* on a company's internal FAQ document (assume this document is about 50-100 question-answer pairs). The LLM should decline to answer if the information is not in the FAQ.
    *   **Scenario D (Define Your Own):** Clearly describe a simple, specific problem where fine-tuning an LLM could be beneficial. (e.g., "Fine-tuning an LLM to classify short news headlines into categories like 'Sports', 'Technology', 'Business'").

*   **Your Chosen Scenario (or Custom Definition):**
    *   *(Clearly state and describe the scenario you've chosen or defined here.)*

---

### 2. Base Model Selection

*   **Propose a suitable open-source pre-trained LLM** to use as a base for your fine-tuning project.
    *   Examples: `phi3:mini` (various versions), `meta-llama/Llama-3-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.2`, `gemma-7b-it`.
*   **Justify your choice (2-3 sentences).** Consider factors like:
    *   **Size:** Is it manageable for a conceptual project or for someone with limited resources?
    *   **Existing Capabilities:** Is the model generally good at language understanding or the type of generation needed (e.g., chat/instruction-tuned models are often good starting points)?
    *   **License:** Is it permissive for experimentation or potential deployment (if relevant)?
    *   **Community Support/Availability:** Is it a well-known model that's easy to access and find information about?

*   **Your Proposed Base Model and Justification:**
    *   Base Model: *[e.g., `microsoft/phi-3-mini-4k-instruct`]*
    *   Justification: *[Your reasoning here]*

---

### 3. Dataset Considerations

*   **Describe the ideal characteristics of a dataset** for fine-tuning for your chosen scenario.
    *   What kind of data would be needed? (e.g., for Scenario A: pairs of customer queries and ideal empathetic replies; for Scenario B: pairs of Python functions and their correctly formatted docstrings).
    *   What specific information should each data sample contain?

*   **Suggest a hypothetical format for the dataset.**
    *   How would you structure each data entry? (e.g., JSONL with specific keys like `"prompt"` and `"completion"`, or a chat format like `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`). Provide a small example.

*   **Estimate a reasonable (even if small) dataset size** you think would be a good starting point for a conceptual fine-tuning attempt for this specific problem. Justify briefly.

*   **Your Dataset Considerations:**
    *   Data Description: *[Describe the data needed]*
    *   Data Format (with example):
        ```json
        // Example for Scenario A:
        // {"messages": [{"role": "user", "content": "Hi, my order #123 hasn't arrived yet, it's been 2 weeks!"}, {"role": "assistant", "content": "Dear Customer, I'm so sorry to hear your order #123 is delayed! Could you please provide your tracking number so I can investigate this for you immediately? We'll do our best to resolve this quickly."}]}
        ```
        *(Provide your own example relevant to your chosen scenario)*
    *   Estimated Dataset Size & Justification: *[e.g., "Around 200-500 high-quality examples. This might be enough to teach the model the specific style and common issues for this task without requiring massive data collection efforts."]*

---

### 4. Fine-tuning Method

*   **Recommend a fine-tuning method** (e.g., LoRA, QLoRA).
*   **Justify why this method is appropriate** for your scenario and chosen base model. Consider factors like:
    *   Resource constraints (assuming limited GPU availability, as is common).
    *   Effectiveness for adapting the model.
    *   Ease of implementation with common tools.

*   **Briefly mention key parameters** you would consider setting or experimenting with for this method (e.g., for LoRA: `r` (rank), `lora_alpha`, `target_modules`). You don't need to specify exact values, just identify them.

*   **Your Recommended Fine-tuning Method and Justification:**
    *   Method: *[e.g., QLoRA]*
    *   Justification: *[Your reasoning here, e.g., "QLoRA is suitable because it offers significant memory savings by quantizing the base model, allowing fine-tuning of relatively large models like Llama 3 8B even on a single consumer GPU. It's effective in adapting models while preserving most of their original knowledge."]*
    *   Key Parameters to Consider: *[e.g., "For QLoRA, I would consider the LoRA parameters `r` and `lora_alpha`, the `lora_dropout`, which layers to target (`target_modules`), and the quantization settings like `bits` (e.g., 4-bit)."]*

---

### 5. Evaluation Strategy

*   **How would you evaluate if the fine-tuned model is performing well** on your specific task?
*   Suggest **1-2 qualitative methods** (e.g., manual review of generated outputs against a checklist, comparison with human-written examples) and/or **1-2 quantitative methods** (e.g., accuracy on a held-out test set, BLEU/ROUGE scores if applicable for generation, specific checks for code correctness). Be specific to your chosen scenario.

*   **Your Evaluation Strategy:**
    *   Qualitative Method(s): *[Describe your qualitative evaluation approach. e.g., "For Scenario A, I would manually review 50-100 generated email replies for empathy, politeness, and whether they correctly address the (simulated) customer issue. I'd also compare them against a set of ideal human-written replies."]*
    *   Quantitative Method(s) (if applicable): *[Describe your quantitative evaluation approach. e.g., "For Scenario B, if I had a test set of functions with gold-standard docstrings, I could calculate a BLEU score to measure n-gram overlap, or develop a script to check for adherence to specific formatting rules (e.g., presence of Args, Returns sections)."]*

---

### 6. Potential Challenges

*   **Identify 1-2 potential challenges** you might face in this fine-tuning process (beyond just "not having enough compute"). Think about data, overfitting, evaluation difficulties, or ethical concerns specific to your scenario.

*   **Your Identified Potential Challenges:**
    1.  Challenge 1: *[Describe a potential challenge, e.g., "For Scenario C (FAQ bot), a key challenge would be ensuring the model *only* uses the FAQ data and reliably says 'I don't know' for out-of-scope questions. Preventing hallucination of answers not in the provided text would be difficult."]*
    2.  Challenge 2: *[Describe another potential challenge, e.g., "For Scenario A (customer service), collecting a diverse enough set of realistic customer queries and crafting genuinely empathetic, non-canned replies for the fine-tuning dataset could be time-consuming and require careful quality control to avoid training the model on suboptimal examples."]*

---

## Submission Guidelines:

*   Compile your proposal, addressing all sections above, into a single document.
*   Ensure your answers are specific to the scenario you chose or defined.
*   You can use Markdown (preferred, save as a `.md` file) or create a PDF.
*   Name your file clearly, e.g., `week4_assignment_yourname.md`.

This assignment is designed to assess your conceptual understanding of the fine-tuning landscape. Focus on clear reasoning and practical considerations for your proposal. Good luck!
