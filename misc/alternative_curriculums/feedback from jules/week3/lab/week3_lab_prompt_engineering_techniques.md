# Lab 3: Experimenting with Prompt Engineering

**Objective:** To provide hands-on practice with various prompt engineering techniques using a local LLM via Ollama or the Hugging Face examples from the lecture.

**Prerequisites:**
*   Ollama installed and a model like `phi3:mini` (recommended for speed) or `llama3:8b` pulled.
    *   To pull `phi3:mini`: `ollama pull phi3:mini`
*   Python environment with the `ollama` library installed (if using Ollama SDK).
    *   `pip install ollama`
*   OR, ability to run the Hugging Face `transformers` notebooks/scripts from the lecture (e.g., using `TinyLlama/TinyLlama-1.1B-Chat-v1.0` or `meta-llama/Llama-2-7b-chat-hf` if you have access).
*   **Recommendation:** For consistency and easier comparison across tasks in this lab, try to use the **same model** (e.g., `phi3:mini` via Ollama) for all experiments.

---

## General Instructions for Recording:

For each task below:
1.  Clearly write down the **full prompt** you used.
2.  Record the **full, verbatim response** from the LLM.
3.  Briefly note any observations or comparisons as requested.

You can perform these tasks directly in your terminal using `ollama run <model_name>` or by adapting the Python scripts from the Week 3 lecture materials (either `interacting_with_ollama_sdk.ipynb` or `interacting_with_hf_transformers.ipynb`).

---

## Tasks:

### 1. Zero-Shot vs. Few-Shot Prompting

*   **Objective:** To observe the difference in LLM performance when provided with examples versus no examples.
*   **Choose a Task:** Sentiment Classification.
    *   Example input review: "The service at the restaurant was incredibly slow, and the food was cold."

*   **A. Zero-Shot Prompting:**
    1.  **Prompt Design:** Craft a prompt that asks the LLM to classify the sentiment of the review without any examples.
        *   *Example Zero-Shot Prompt Idea:*
            ```
            Classify the sentiment of the following customer review as Positive, Negative, or Neutral.

            Review: "The service at the restaurant was incredibly slow, and the food was cold."

            Sentiment:
            ```
    2.  **Execute:** Run this prompt with your chosen LLM.
    3.  **Record:**
        *   Your full prompt.
        *   The LLM's full response.

*   **B. Few-Shot Prompting:**
    1.  **Prompt Design:** Craft a prompt that includes 2-3 examples of reviews and their corresponding sentiments before asking the LLM to classify the new review.
        *   *Example Few-Shot Prompt Idea:*
            ```
            Classify the sentiment of customer reviews as Positive, Negative, or Neutral.

            Review: "I loved the new movie, it was so exciting!"
            Sentiment: Positive

            Review: "The product broke after only one day of use. Very disappointing."
            Sentiment: Negative

            Review: "The weather today is average, neither sunny nor rainy."
            Sentiment: Neutral

            Review: "The service at the restaurant was incredibly slow, and the food was cold."
            Sentiment:
            ```
    2.  **Execute:** Run this prompt with your chosen LLM.
    3.  **Record:**
        *   Your full prompt.
        *   The LLM's full response.

*   **C. Compare Results:**
    *   Did both prompts yield the correct sentiment?
    *   Was there any difference in the confidence, style, or formatting of the LLM's response between the zero-shot and few-shot attempts?
    *   Which approach seemed more reliable for this task?

### 2. Role Prompting (Persona Prompting)

*   **Objective:** To observe how instructing an LLM to adopt a specific persona influences its output.
*   **Base Question:** "Explain in a short paragraph why the ocean is salty."

*   **Experiment with 2-3 Different Roles:**
    *   **Role 1 Example:** A knowledgeable Marine Biologist.
        *   *Prompt Idea:* `You are Dr. Marina Current, a leading marine biologist with 20 years of experience. Explain in a short paragraph why the ocean is salty, using scientifically accurate but accessible language.`
    *   **Role 2 Example:** A five-year-old child.
        *   *Prompt Idea:* `Pretend you are a 5-year-old kid named Timmy. Explain in a short paragraph why the ocean is salty, using simple words like a kid would use.`
    *   **Role 3 Example (Choose your own!):** A grumpy pirate, a Shakespearean poet, an alien visiting Earth for the first time, etc.
        *   *Prompt Idea:* `[Your persona instruction here]. Explain in a short paragraph why the ocean is salty.`

*   **For each role:**
    1.  **Execute:** Run your designed prompt.
    2.  **Record:**
        *   Your full prompt (including the persona instruction).
        *   The LLM's full response.
    3.  **Observe:** How did the persona affect the response's:
        *   Tone (e.g., formal, informal, enthusiastic, grumpy)?
        *   Style (e.g., vocabulary, sentence structure)?
        *   Content focus (did it emphasize different aspects)?

### 3. Chain-of-Thought (CoT) Prompting (Simple)

*   **Objective:** To see if encouraging step-by-step reasoning improves clarity or accuracy for a simple multi-step problem.
*   **Problem:** "A bakery made 3 batches of cookies. Each batch contained 12 cookies. If they sold 20 cookies, how many cookies are left? Explain your reasoning."

*   **A. Direct Prompt (No CoT):**
    1.  **Prompt Design:** Ask the question directly without explicitly asking for step-by-step reasoning.
        *   *Prompt Idea:* `A bakery made 3 batches of cookies. Each batch contained 12 cookies. If they sold 20 cookies, how many cookies are left?`
    2.  **Execute:** Run this prompt.
    3.  **Record:**
        *   Your full prompt.
        *   The LLM's full response.

*   **B. CoT Prompt:**
    1.  **Prompt Design:** Modify the prompt to encourage the LLM to "think step by step" or "show its work."
        *   *Prompt Idea 1 (Explicit request):* `A bakery made 3 batches of cookies. Each batch contained 12 cookies. If they sold 20 cookies, how many cookies are left? Let's think step by step to solve this.`
        *   *Prompt Idea 2 (Few-shot CoT, more advanced but optional for this lab):*
            ```
            Q: If a farmer has 10 apples and sells 3, then buys 5 more, how many does he have?
            A: The farmer starts with 10 apples. Sells 3, so 10 - 3 = 7 apples. Buys 5 more, so 7 + 5 = 12 apples. The answer is 12.

            Q: A bakery made 3 batches of cookies. Each batch contained 12 cookies. If they sold 20 cookies, how many cookies are left?
            A:
            ```
            *(For this lab, Prompt Idea 1 is sufficient to explore basic zero-shot CoT.)*
    2.  **Execute:** Run your chosen CoT prompt.
    3.  **Record:**
        *   Your full prompt.
        *   The LLM's full response.

*   **C. Compare Results:**
    *   Did both prompts lead to the correct final answer?
    *   Was the reasoning process (if any) clearer in the CoT response?
    *   Did the CoT prompt help the LLM break down the problem effectively?

### 4. Experiment with Output Formatting

*   **Objective:** To test the LLM's ability to adhere to specific output formatting instructions.
*   **Base Task:** "List 5 common household pets."

*   **Experiment with Different Formatting Requests:**

    *   **A. Comma-Separated List:**
        1.  **Prompt Design:** Ask for the list as a single line, with items separated by commas.
            *   *Prompt Idea:* `List 5 common household pets, separated by commas on a single line.`
        2.  **Execute & Record:** Prompt and LLM response.

    *   **B. Numbered List:**
        1.  **Prompt Design:** Ask for the list with each item numbered.
            *   *Prompt Idea:* `List 5 common household pets as a numbered list.`
        2.  **Execute & Record:** Prompt and LLM response.

    *   **C. JSON Array:**
        1.  **Prompt Design:** Ask for the list formatted as a JSON array of strings.
            *   *Prompt Idea:* `List 5 common household pets. Present the list as a JSON array of strings.`
        2.  **Execute & Record:** Prompt and LLM response.

*   **D. Observe and Compare:**
    *   How well did the LLM adhere to each formatting request?
    *   Were there any errors or inconsistencies in the formatting?
    *   Which format seemed easiest or hardest for the LLM to produce correctly?

---

**Lab Conclusion:**
This lab provided a quick tour of several fundamental prompt engineering techniques. You should have observed that how you ask the LLM to perform a task can significantly alter the output you receive. Iteration and clear instruction are key to effective prompting.
