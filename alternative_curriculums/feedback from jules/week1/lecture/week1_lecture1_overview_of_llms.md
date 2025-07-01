# Week 1, Lecture 1: Overview of Large Language Models (LLMs)

## 1. What are Large Language Models (LLMs)?

*   **Definition:** Large Language Models (LLMs) are advanced artificial intelligence (AI) models specifically designed to understand, generate, and manipulate human language. They are "large" because they are trained on massive amounts of text data and typically have billions of parameters (learnable variables).
*   **Core Idea: Predicting the Next Word:** At their heart, most LLMs are trained to predict the next word in a sequence of words. Given an input sequence (a "prompt"), the model calculates the probability distribution of what the next word (or "token") should be. This seemingly simple task, when performed at scale and with sophisticated architectures, enables complex language capabilities.
    *   Example: Given "The quick brown fox jumps over the...", the model predicts "lazy" or "fence".
*   **Tokens:** LLMs don't see words directly but "tokens". Tokens can be words, sub-words, or characters, depending on the tokenization method used.

## 2. Brief History: Key Milestones

*   **Early Concepts:** While language modeling has roots in statistical methods from decades ago, modern LLMs were enabled by advances in deep learning.
*   **2017 - The Transformer Architecture ("Attention Is All You Need"):** This paper by Google researchers introduced the Transformer architecture, which uses "attention mechanisms" to weigh the importance of different words in a sequence. This was a pivotal moment, as Transformers allowed for more parallelizable training and better handling of long-range dependencies in text compared to previous recurrent neural network (RNN) based models.
*   **2018 onwards - GPT Series (OpenAI):**
    *   **GPT (Generative Pre-trained Transformer):** Demonstrated the effectiveness of generative pre-training on diverse internet text for downstream NLP tasks.
    *   **GPT-2:** Showed remarkable text generation capabilities and sparked discussions about potential misuse. Initially, the full model was not released due to these concerns.
    *   **GPT-3:** A significantly larger model (175 billion parameters) that exhibited impressive few-shot and zero-shot learning abilities, meaning it could perform tasks it wasn't explicitly trained for with minimal or no examples.
    *   **ChatGPT & GPT-4:** Introduced conversational AI capabilities refined through Reinforcement Learning from Human Feedback (RLHF), making interaction more natural and intuitive. GPT-4 further improved performance and multimodal capabilities.
*   **BERT (Bidirectional Encoder Representations from Transformers) (Google, 2018):** Focused on understanding context by looking at words before and after a given token (bidirectional context), making it very powerful for tasks like question answering and sentiment analysis. Unlike GPT, BERT is primarily an encoder model.
*   **LLaMA Series (Meta AI, 2023 onwards):**
    *   **LLaMA (Large Language Model Meta AI):** A family of open-source (weights released to researchers) models ranging from 7B to 65B parameters, demonstrating that smaller models trained on more data could achieve performance competitive with larger models.
    *   **LLaMA 2:** Improved version, also open-sourced for research and commercial use, with a focus on safety.
    *   **LLaMA 3:** Further iteration with enhanced performance and larger model sizes, continuing the trend of powerful open models.
*   **Other Notable Models:** PaLM (Google), Claude (Anthropic), Mistral (Mistral AI), Phi (Microsoft).

## 3. Common Architectures

LLMs primarily use the Transformer architecture, but variations exist:

*   **Encoder-Decoder (e.g., T5, BART):**
    *   Consist of two main parts: an encoder that processes the input text and creates a contextual representation, and a decoder that generates the output text based on this representation.
    *   Well-suited for tasks that involve transforming input text to output text, such as translation (input: French sentence, output: English sentence) or summarization (input: long document, output: short summary).
    *   Example: T5 (Text-to-Text Transfer Transformer) frames all NLP tasks as a text-to-text problem.

*   **Decoder-Only (e.g., GPT series, LLaMA, PaLM, Mistral, Phi):**
    *   Consist only of a decoder stack from the Transformer architecture.
    *   These models are autoregressive, meaning they generate text one token at a time, feeding the output back into the input for the next token prediction.
    *   Dominant architecture for generative tasks like text completion, creative writing, and chatbots.
    *   Their training objective is typically next-token prediction.
    *   The focus of this workshop will primarily be on decoder-only models as they are prevalent for local LLM experimentation.

## 4. Key Capabilities

LLMs have demonstrated a wide array of capabilities, including:

*   **Text Generation:** Creating coherent and contextually relevant text (stories, articles, poems).
*   **Translation:** Translating text between different languages.
*   **Summarization:** Condensing long documents into shorter summaries while retaining key information.
*   **Question Answering (Q&A):** Answering questions based on provided context or their internal knowledge.
*   **Code Generation:** Generating code snippets in various programming languages based on natural language descriptions.
*   **Conversational AI / Chatbots:** Engaging in interactive dialogues with users.
*   **Sentiment Analysis:** Identifying the sentiment (positive, negative, neutral) expressed in a piece of text.
*   **Text Classification:** Categorizing text into predefined classes (e.g., spam detection, topic labeling).
*   **Reasoning (emergent capability):** Performing simple reasoning tasks, though this is an area of active research and development.

## 5. Limitations

Despite their power, LLMs have significant limitations:

*   **Hallucinations/Fabrication:** LLMs can generate text that sounds plausible but is factually incorrect or nonsensical. They don't "know" things in a human sense but predict likely sequences.
*   **Bias:** LLMs are trained on vast datasets from the internet, which can contain societal biases related to race, gender, religion, etc. The models can learn and perpetuate these biases.
*   **Computational Cost:** Training very large LLMs requires massive computational resources (specialized hardware like GPUs/TPUs, significant energy consumption). Running larger models locally can also be demanding.
*   **Data Privacy:** Using cloud-based LLMs involves sending data to third-party servers, raising privacy concerns for sensitive information. Training data might also inadvertently contain private information.
*   **Lack of True Understanding/Common Sense:** While LLMs can process and generate language effectively, they lack genuine understanding and common sense reasoning in the way humans do.
*   **Knowledge Cutoff:** Models have knowledge only up to the point their training data was collected. They are not aware of events or information that occurred after their training. (RAG can help mitigate this).
*   **Vulnerability to Adversarial Prompts:** Prompts can be crafted to elicit undesirable, biased, or harmful outputs.

## 6. Ethical Considerations

The development and deployment of LLMs raise several ethical concerns:

*   **Misinformation and Disinformation:** LLMs can be used to generate realistic-sounding fake news or propaganda at scale, making it harder to distinguish truth from falsehood.
*   **Job Displacement:** Automation of tasks previously done by humans (e.g., content creation, customer service) could lead to job losses or shifts in the job market.
*   **Environmental Impact:** The energy consumption for training and running large-scale LLMs contributes to carbon emissions.
*   **Fairness and Bias:** As mentioned, biases in training data can lead to discriminatory or unfair outputs, reinforcing societal inequalities.
*   **Accountability and Transparency:** Determining who is responsible when an LLM produces harmful or incorrect output can be challenging. The "black box" nature of some models makes them difficult to interpret.
*   **Intellectual Property:** LLMs trained on copyrighted material raise questions about fair use and ownership of generated content.
*   **Security Risks:** LLMs can be used for malicious purposes like generating phishing emails, malware, or impersonating individuals.

Addressing these ethical considerations is crucial for the responsible development and deployment of LLM technology. This involves ongoing research, development of safety protocols, and public discourse.
