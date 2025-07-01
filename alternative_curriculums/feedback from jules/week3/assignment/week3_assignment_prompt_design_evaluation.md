# Assignment 3: Designing and Evaluating Prompts

**Objective:** To apply prompt engineering principles to a specific task, design multiple prompts, and evaluate their effectiveness using a local LLM.

**Prerequisites:**
*   Completion of Lab 3: Experimenting with Prompt Engineering (or equivalent understanding).
*   Ollama installed and a model like `phi3:mini` (recommended for speed and resource constraints) or `llama3:8b` pulled.
    *   To pull `phi3:mini`: `ollama pull phi3:mini`
*   Python environment with the `ollama` library installed (if using Ollama SDK for programmatic access).
*   OR, ability to run the Hugging Face `transformers` notebooks/scripts from the lecture.
*   **Recommendation:** Use a consistent LLM (e.g., `phi3:mini` via Ollama) for all prompts in this assignment for fair comparison.

---

## Tasks:

### 1. Define Your Task

*   **Choose a specific task** that an LLM can perform. Be precise.
*   **Provide any necessary input data** for the task (e.g., a text to summarize, product details for a description, a problem for a code solution, context for Q&A).

*   **Examples of Tasks (Choose ONE or define your own of similar complexity):**

    *   **A. Summarize a News Article:**
        *   **Task Statement:** "Summarize the provided news article in 2-3 concise sentences, capturing the main points."
        *   **Input Data (Example Article - Replace with your own if you choose this task, approx. 200-300 words):**
            ```text
            (Paste your chosen news article here. For instance, find a short, recent news brief from a reputable source. 
            Example placeholder: 
            "Global tech leaders convened this week for the annual Innovate Summit, focusing on advancements in artificial intelligence and sustainable technology. Keynotes highlighted the potential of AI to address climate change and the importance of ethical guidelines in AI development. Several major companies announced new AI-powered tools aimed at reducing carbon footprints in various industries. The summit also featured discussions on the need for international collaboration to ensure equitable access to emerging technologies. Experts predict that innovations showcased will significantly impact global markets within the next five years.")
            ```

    *   **B. Generate a Product Description:**
        *   **Task Statement:** "Generate an engaging and persuasive product description (approx. 50-75 words) for a fictional product."
        *   **Input Data (Fictional Product Details - Be creative!):**
            *   Product Name: "ChronoCube"
            *   Product Type: "A small, glowing cube that displays personalized holographic motivational quotes that change daily."
            *   Key Features: "Pocket-sized, wireless charging, customizable glow colors, syncs with a mobile app for quote selection."
            *   Target Audience: "Students, young professionals, anyone needing a daily dose of inspiration."

    *   **C. Write a Short Python Function:**
        *   **Task Statement:** "Write a Python function that takes a list of integers as input and returns a new list containing only the even numbers from the input list, preserving their order."
        *   **Input Data (Implicit):** The function should be named, e.g., `get_even_numbers`.

    *   **D. Answer Questions Based on a Provided Text:**
        *   **Task Statement:** "Answer the following questions based *only* on the information available in the provided text."
        *   **Input Data (Example Text and Questions - Replace with your own):**
            ```text
            (Paste your chosen short text here. For example, a paragraph from a Wikipedia article on a specific animal or historical event.)
            Example Text: "The Arctic fox (Vulpes lagopus) is a small fox native to the Arctic regions of the Northern Hemisphere and common throughout the Arctic tundra biome. It is well adapted to living in cold environments and is best known for its thick, warm fur that is also used as camouflage. It has a body length ranging from 46 to 68 cm (18 to 27 in), with a generally rounded body shape to minimize the escape of body heat."

            Questions:
            1. What is the scientific name of the Arctic fox?
            2. Where is the Arctic fox primarily found?
            3. Name one adaptation of the Arctic fox to cold environments.
            ```

*   **Your Chosen Task and Input Data:**
    *   Clearly state your chosen task here.
    *   Include all necessary input data (text, product details, etc.).

---

### 2. Design Three Different Prompts

*   For your chosen task and input data, design **three distinct prompts**.
*   These prompts should vary in their approach. Consider using techniques learned in the lecture and lab, such as:
    *   Zero-shot vs. Few-shot (if applicable).
    *   Role prompting.
    *   Explicit instructions for output format or length.
    *   Varying levels of detail or context provided in the prompt.
    *   Chain-of-Thought hints (e.g., "explain your reasoning").

*   **List Your Three Prompts:**

    *   **Prompt 1:**
        ```
        (Your Prompt 1 Text Here)
        ```

    *   **Prompt 2:**
        ```
        (Your Prompt 2 Text Here)
        ```

    *   **Prompt 3:**
        ```
        (Your Prompt 3 Text Here)
        ```

---

### 3. Execute and Record Results

*   For each of your three designed prompts:
    1.  Run it through your chosen LLM (e.g., `phi3:mini` via `ollama run phi3:mini` or using the Python SDK).
    2.  Record the **full, verbatim output (response)** from the LLM.

*   **LLM Responses:**

    *   **Response to Prompt 1:**
        ```
        (Full LLM Output for Prompt 1 Here)
        ```

    *   **Response to Prompt 2:**
        ```
        (Full LLM Output for Prompt 2 Here)
        ```

    *   **Response to Prompt 3:**
        ```
        (Full LLM Output for Prompt 3 Here)
        ```

---

### 4. Evaluate and Compare

*   **A. Define Evaluation Criteria:**
    *   List 2-3 specific criteria you will use to evaluate the quality and effectiveness of the LLM's output for YOUR chosen task.
    *   Examples:
        *   Accuracy (Is the information correct?)
        *   Completeness (Does it fulfill all parts of the prompt?)
        *   Conciseness (Is it to the point, especially if length was specified?)
        *   Adherence to Format (Did it follow formatting instructions?)
        *   Clarity (Is the output easy to understand?)
        *   Creativity/Engagement (For tasks like product descriptions).
        *   Correctness of Code (For coding tasks - does it run and work?)

    *   **Your Criteria:**
        1.  Criterion 1: [Description]
        2.  Criterion 2: [Description]
        3.  (Optional) Criterion 3: [Description]

*   **B. Evaluate Each Prompt's Output:**
    *   For each prompt, assess its output against your defined criteria. You can use a simple rating system (e.g., Good/Fair/Poor) or a brief textual evaluation for each criterion.

    *   **Evaluation of Prompt 1 Output:**
        *   Criterion 1: [Your evaluation]
        *   Criterion 2: [Your evaluation]
        *   (Optional) Criterion 3: [Your evaluation]
        *   Overall comments on Prompt 1's effectiveness:

    *   **Evaluation of Prompt 2 Output:**
        *   Criterion 1: [Your evaluation]
        *   Criterion 2: [Your evaluation]
        *   (Optional) Criterion 3: [Your evaluation]
        *   Overall comments on Prompt 2's effectiveness:

    *   **Evaluation of Prompt 3 Output:**
        *   Criterion 1: [Your evaluation]
        *   Criterion 2: [Your evaluation]
        *   (Optional) Criterion 3: [Your evaluation]
        *   Overall comments on Prompt 3's effectiveness:

*   **C. Comparative Analysis and Learning:**
    *   Write a brief analysis (2-3 paragraphs) comparing the three prompts.
        *   Which prompt was the most effective in achieving the desired output for your task, based on your criteria? Why do you think it performed better?
        *   Which prompt was least effective, and why?
        *   What did you learn about prompt engineering from this experiment? Were there any surprising results?
        *   How might you further refine your best prompt if you had more attempts?

---

## Submission Guidelines:

*   Compile all sections (Task Definition, Prompts, LLM Responses, Evaluation) into a single document.
*   You can use Markdown (preferred, save as a `.md` file) or create a PDF.
*   Ensure all LLM outputs are recorded verbatim.
*   Name your file clearly, e.g., `week3_assignment_yourname.md`.

This assignment encourages you to think critically about how to communicate effectively with LLMs and to systematically evaluate the results of your prompting strategies. Good luck!
