# Week 3, Lecture 1: LLaMA Insights & Prompt Engineering

## Part 1: LLaMA Research Paper Insights (Conceptual)

**Objective:** To provide key takeaways from the LLaMA (Large Language Model Meta AI) research paper without requiring a full read-through, focusing on its impact and high-level concepts.

---

### 1. Overview of the LLaMA Family

*   **Introduction:** LLaMA refers to a family of Large Language Models developed and released by Meta AI in early 2023.
*   **Model Sizes:** The initial LLaMA release included models of various sizes: 7B, 13B, 33B, and 65B parameters.
    *   *(B denotes Billions. So, 7B means 7 billion parameters.)*
*   **General Capabilities:** LLaMA models are decoder-only Transformer architectures, similar to GPT-3, designed for a wide range of natural language understanding and generation tasks. They are known for their strong performance across various benchmarks.
*   **Open Approach (for Research):** While not fully open-source in the traditional sense initially (weights were released to researchers under a non-commercial license), LLaMA's availability significantly spurred research and development in the open LLM community, leading to many derivative works and fine-tuned versions. Subsequent releases like Llama 2 and Llama 3 have adopted more open licensing for commercial use as well.

### 2. Key Architectural Choices & Training Aspects

The LLaMA paper highlighted several design choices and training methodologies that contributed to its efficiency and performance. While the deep technical details are extensive, here are some high-level takeaways:

*   **Standard Transformer Enhancements:**
    *   **Pre-normalization (RMSNorm):** Instead of normalizing the output of a layer (Post-LN), LLaMA normalizes the input to each Transformer sub-layer (using RMSNorm, a variant of Layer Normalization).
        *   **Benefit:** Contributes to improved training stability and performance, especially for very deep networks.
    *   **SwiGLU Activation Function:** Replaced the standard ReLU activation function in the feed-forward network (FFN) sub-layer with SwiGLU (Swish Gated Linear Unit).
        *   **Benefit:** Often leads to better performance compared to ReLU and other activation functions in Transformers.
    *   **Rotary Positional Embeddings (RoPE):** Applied rotary embeddings at each layer of the network, instead of absolute or learned positional embeddings at the input layer only.
        *   **Benefit:** RoPE is known for its ability to effectively encode relative positional information and potentially better extrapolate to longer sequence lengths.

*   **Training Data:**
    *   **Massive and Diverse Dataset:** LLaMA was trained on a very large corpus of publicly available text data, emphasizing diversity. The dataset sources included:
        *   Common Crawl (filtered)
        *   C4 (Colossal Clean Crawled Corpus)
        *   Wikipedia (multiple languages)
        *   Books (Gutenberg, Books3)
        *   ArXiv papers
        *   Stack Exchange
    *   **Scale:** The total amount of training data was in the order of **1.4 trillion tokens**. This massive scale is a key factor in the performance of modern LLMs.
    *   **Tokenization:** Used a Byte Pair Encoding (BPE) algorithm.

*   **Efficient Training:**
    *   The paper emphasized strategies to train these models efficiently, leveraging optimizations in parallelization and model architecture.
    *   One of the key findings was that **smaller models trained for longer on more data** could achieve performance comparable to or better than much larger models trained for shorter periods. This was a significant insight.

### 3. Performance Highlights

*   **Competitive Performance:** LLaMA models, particularly the larger ones (LLaMA-33B and LLaMA-65B), demonstrated performance competitive with or exceeding that of other large-scale models like OpenAI's GPT-3 (specifically Davinci, the 175B parameter version at the time) on many NLP benchmarks.
*   **Efficiency:** Crucially, LLaMA-13B was shown to outperform GPT-3 (175B) on most benchmarks despite being significantly smaller (more than 10x smaller). This highlighted the possibility of achieving strong results with more accessible model sizes, given sufficient training data and efficient architectures.
*   **Versatility:** The models performed well across a range of tasks including question answering, common sense reasoning, code generation (though not its primary focus), and reading comprehension.

### 4. Impact of LLaMA

*   **Democratizing Access:** The release of LLaMA (even initially to researchers) had a profound impact on the AI research community. It provided a highly capable, relatively accessible base model that researchers could build upon, fine-tune, and experiment with.
*   **Catalyst for Open LLM Development:** LLaMA became the foundation for numerous open-source projects and fine-tuned models (e.g., Alpaca, Vicuna, and many others). This spurred innovation and made powerful LLM technology more widely available beyond large corporate labs.
*   **Shift in Focus:** It reinforced the idea that model scale isn't the only factor; data quality, quantity, and training efficiency are also paramount.
*   **Paving the Way for Llama 2 & 3:** The success and learnings from the original LLaMA project directly led to the development and more open release of Llama 2 and Llama 3, which are now widely used for both research and commercial applications.

---

## Part 2: Introduction to Prompt Engineering

**Objective:** To introduce basic prompt engineering techniques to effectively communicate with and guide Large Language Models.

---

### 1. What is Prompt Engineering?

*   **Definition:** Prompt Engineering is the art and science of designing effective inputs (prompts) to guide Large Language Models (LLMs) towards generating desired outputs.
*   **Why is it important?**
    *   LLMs are powerful but their behavior is highly sensitive to the input prompt.
    *   Well-crafted prompts can significantly improve the accuracy, relevance, and quality of the model's responses.
    *   It allows users to control the LLM's output for specific tasks without needing to retrain or fine-tune the model itself.
    *   It's like giving clear and precise instructions to a very capable but literal-minded assistant.

### 2. Basic Prompting Techniques

*   **A. Zero-Shot Prompting:**
    *   **Explanation:** You ask the model to perform a task directly without providing any prior examples of how to do it. The model relies on its pre-existing knowledge and understanding.
    *   **Example:**
        ```
        User Prompt:
        Translate this English sentence to French: "Hello, how are you today?"
        ```
        ```
        User Prompt:
        Summarize the following text:
        [Long text paragraph here...]
        ```
    *   **When to use:** For straightforward tasks where the model is likely to have been trained on similar types_of requests (e.g., simple translation, summarization of common text, direct questions).

*   **B. Few-Shot Prompting:**
    *   **Explanation:** You provide a few examples (typically 1 to 5, hence "few-shot") of the task you want the model to perform within the prompt itself. These examples demonstrate the desired input-output format or style.
    *   **Example (Sentiment Classification):**
        ```
        User Prompt:
        This is awesome! // Positive
        This is bad. // Negative
        Wow, I love this movie! // Positive
        What a terrible experience. // Negative
        The food was okay, not great but not bad either. // Neutral
        That was a great flight. // 
        ```
        *(Expected LLM completion: Positive)*
    *   **Example (Simple Q&A format):**
        ```
        User Prompt:
        Q: What is the capital of France?
        A: Paris.

        Q: Who wrote "Romeo and Juliet"?
        A: William Shakespeare.

        Q: What is the main ingredient in guacamole?
        A: 
        ```
        *(Expected LLM completion: Avocado)*
    *   **When to use:** When the task is more nuanced, requires a specific output format, or when zero-shot prompting doesn't yield good enough results. It helps the model understand the expected pattern.

*   **C. Role Prompting (Persona Prompting):**
    *   **Explanation:** You instruct the LLM to adopt a specific role, persona, or character before performing a task. This influences its tone, style, and the type of information it might focus on.
    *   **Example:**
        ```
        User Prompt:
        You are a friendly and helpful AI assistant. Explain quantum computing in simple terms.
        ```
        ```
        User Prompt:
        Act as a Shakespearean poet. Write a short verse about a modern smartphone.
        ```
        ```
        User Prompt:
        You are a master chef. Suggest a recipe for a vegan pasta dish using ingredients commonly found in a pantry.
        ```
    *   **When to use:** To control the style, tone, expertise, or perspective of the generated text. Useful for creative writing, specialized explanations, or generating content for a specific audience.

*   **D. Clear Instructions & Formatting:**
    *   **Specificity:** Be as specific and unambiguous as possible in your instructions.
    *   **Delimiters:** Use delimiters like triple backticks (```), quotes ("""), XML tags (`<tag></tag>`), or hashes (###) to clearly separate different parts of your prompt, such as instructions, context, examples, and the input query.
    *   **Output Format Definition:** If you need the output in a particular format (e.g., JSON, list, specific heading structure), explicitly ask for it.
    *   **Example (Using Delimiters and Requesting Format):**
        ```
        User Prompt:
        You will be provided with a piece of text delimited by triple backticks.
        Your task is to summarize this text in exactly three bullet points.
        Then, identify the main sentiment of the text (Positive, Negative, or Neutral).

        Text:
        ```
        The new cafe on Main Street has quickly become my favorite spot. 
        The coffee is excellent, the pastries are always fresh, and the staff are incredibly welcoming. 
        I've been there three times this week! The ambiance is cozy, though it can get a bit noisy during peak hours.
        ```

        Output format:
        Summary:
        * Bullet point 1
        * Bullet point 2
        * Bullet point 3
        Sentiment: [Positive/Negative/Neutral]
        ```

### 3. Chain-of-Thought (CoT) Prompting (Basic Introduction)

*   **Explanation:** CoT prompting encourages the LLM to generate intermediate reasoning steps before giving a final answer. By asking the model to "think step by step" or showing it examples that include reasoning, you can often improve its performance on tasks that require logical deduction, arithmetic, or multi-step problem-solving.
*   **Core Idea:** The model learns to "show its work," and this process of articulating the reasoning helps it arrive at more accurate conclusions.
*   **Simple Example (Few-Shot CoT for Arithmetic):**
    ```
    User Prompt:
    Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
    A: Roger started with 5 balls. He bought 2 cans, and each can has 3 balls, so that's 2 * 3 = 6 more balls. In total, he now has 5 + 6 = 11 tennis balls. The final answer is 11.

    Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
    A: The cafeteria started with 23 apples. They used 20, so they had 23 - 20 = 3 apples left. Then they bought 6 more, so they now have 3 + 6 = 9 apples. The final answer is 9.

    Q: Natalia sold clips to fill her register. She sold 6 boxes of paper clips and 4 boxes of binder clips. If each box of paper clips contained 10 clips and each box of binder clips contained 5 clips, how many clips did she sell in total?
    A:
    ```
    *(Expected LLM completion would show steps: 6 boxes * 10 clips/box = 60 paper clips. 4 boxes * 5 clips/box = 20 binder clips. Total = 60 + 20 = 80 clips. The final answer is 80.)*

*   **Zero-Shot CoT (Simpler version):** You can sometimes trigger step-by-step thinking by simply adding "Let's think step by step" or "Show your work" to your prompt for a complex question.
    ```
    User Prompt:
    Solve this word problem: John has twice as many apples as Mary. Mary has 5 apples. If John gives 2 apples to Mary, how many apples will John have left? Let's think step by step.
    ```

### 4. General Best Practices for Prompt Engineering

*   **Iteration and Experimentation:** Prompt engineering is often an iterative process. Try different phrasings, examples, and techniques until you get the desired output. What works for one model might not work as well for another.
*   **Keep it Simple (KISS):** Start with simple prompts and gradually add complexity if needed. Overly complex prompts can confuse the model.
*   **Understand Model Limitations:** Be aware of the model's knowledge cutoff date, potential biases, and tendency for hallucinations. Don't assume it "knows" everything or is always correct.
*   **Clarity is Key:** The clearer your prompt, the better the LLM can understand your intent.
*   **Temperature and Other Parameters:** For more creative or diverse outputs, experiment with generation parameters like temperature and top_p (if available in your LLM interface). For factual tasks, lower temperatures are often better.

Prompt engineering is a rapidly evolving field, but these fundamental techniques provide a strong starting point for effectively interacting with LLMs.
