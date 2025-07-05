# Week 2: Core Concepts - Embeddings, Attention, and Transformers

## 1. Introduction

*   **Brief Recap of Week 1:**
    In Week 1, we introduced the concept of Large Language Models (LLMs), discussed their capabilities, limitations, and ethical considerations. We also got hands-on experience with **Ollama**, setting up and running LLMs locally. This provided a practical foundation for interacting with these powerful tools.

*   **Importance of Understanding Core Concepts:**
    While it's possible to use LLMs as black boxes, understanding their underlying mechanisms is crucial for several reasons:
    *   **Effective Use:** Knowing how LLMs "think" helps in crafting better prompts and interpreting their outputs more critically.
    *   **Advanced Topics:** Core concepts are the building blocks for more advanced topics like fine-tuning, understanding model behavior, and developing LLM-powered applications (e.g., Retrieval Augmented Generation - RAG).
    *   **Troubleshooting:** When things go wrong, a conceptual understanding can guide you in diagnosing issues.
    *   **Appreciation:** It unveils the elegance and innovation behind these transformative technologies.

    This week, we will delve into three fundamental pillars of modern LLMs: **Embeddings**, the **Attention Mechanism**, and the **Transformer Architecture**.

## 2. Word Embeddings

*   **What are Word Embeddings?**
    *   **Word Embeddings** are numerical representations of words as **dense vectors** (lists of numbers, e.g., `[0.23, -0.45, 0.67, ..., 0.09]`) in a multi-dimensional space.
    *   The key idea is that these vectors capture the **semantic meaning** and **contextual relationships** of words. Words with similar meanings or that appear in similar contexts will have vectors that are "close" to each other in this vector space.
    *   For example, the vectors for "king" and "queen" would be closer than the vectors for "king" and "apple".
    *   The dimensionality of these vectors can range from tens to thousands, depending on the model.

*   **Why are they needed?**
    *   Computers don't understand text directly; they need numerical input.
    *   **Limitations of One-Hot Encoding:** A naive approach would be **one-hot encoding**, where each word is a sparse vector with a single '1' at the index corresponding to the word in a vocabulary and '0's everywhere else.
        *   Example: If vocabulary is ["apple", "ball", "cat"], then "ball" = `[0, 1, 0]`.
        *   **Problems with one-hot encoding:**
            1.  **High Dimensionality:** For large vocabularies (e.g., 50,000 words), these vectors become extremely large and sparse.
            2.  **No Semantic Relationship:** One-hot vectors are orthogonal (`[1,0,0]` vs `[0,1,0]`). The dot product is 0, implying no similarity, even for synonyms. This doesn't capture that "cat" and "kitten" are related.
    *   **Dense Embeddings solve these issues:**
        1.  **Lower Dimensionality:** They are "dense" (most values are non-zero) and typically have much lower dimensions (e.g., 100-300 for classic embeddings, 768-4096+ for Transformer-based models) than one-hot vectors.
        2.  **Capture Similarity:** The learning process positions word vectors such that similar words are close in the vector space, allowing for mathematical operations to reveal semantic relationships (e.g., "king" - "man" + "woman" ≈ "queen").

*   **Evolution/Examples (Conceptual Ideas):**
    These methods learn embeddings by analyzing vast amounts of text and observing word co-occurrence patterns.
    *   **Word2Vec (Google):**
        *   **CBOW (Continuous Bag of Words):** Predicts a target word based on its context words (surrounding words). *Learns by "filling in the blank."*
        *   **Skip-gram:** Predicts context words given a target word. *Learns by "predicting the neighbors."*
    *   **GloVe (Global Vectors for Word Representation, Stanford):**
        *   Learns by factorizing a global word-word co-occurrence matrix, essentially capturing statistics of how often words appear together in a large corpus.
    *   **FastText (Facebook):**
        *   An extension of Word2Vec that represents each word as a bag of character n-grams (e.g., "apple" as "ap", "app", "ppl", "ple", "le", plus the whole word "<apple>").
        *   This allows it to generate embeddings for **out-of-vocabulary (OOV) words** and often captures morphological information better (e.g., "teach", "teacher", "teaching").

*   **Pre-trained Embeddings:**
    *   Training word embeddings from scratch requires a large corpus and significant computation.
    *   **Pre-trained embeddings** are word vectors made available by researchers/organizations, already trained on massive text datasets (e.g., Wikipedia, Google News).
    *   These can be directly incorporated into models, providing a strong initialization of word meanings, saving time and resources.

*   **Visualizing Embeddings (Conceptual):**
    *   Since embedding vectors can have hundreds of dimensions, we can't directly visualize them.
    *   Techniques like **t-SNE (t-distributed Stochastic Neighbor Embedding)** or **PCA (Principal Component Analysis)** can reduce these high-dimensional vectors to 2D or 3D.
    *   When plotted, words with similar meanings often form distinct clusters. For example, names of cities might cluster together, animals in another cluster, and verbs related to movement in yet another.
        *   *(Conceptual Diagram: Imagine a 2D scatter plot with clusters of related words like "dog, cat, hamster" in one area, and "run, jump, walk" in another.)*

*   **How Embeddings are used in LLMs:**
    *   In LLMs, the **embedding layer** is typically the very first layer.
    *   When you input text, it's first tokenized (broken into words or sub-word units).
    *   Each token is then mapped to its corresponding pre-trained or learned embedding vector.
    *   This sequence of vectors becomes the initial numerical representation of your input text that is fed into the subsequent layers of the Transformer.

## 3. The Attention Mechanism

*   **Intuition:**
    Think about how humans process language. When you read a sentence, especially a long or complex one, you don't weigh every word equally. You intuitively focus on certain words that are more relevant to understanding the meaning of other words or the sentence as a whole.
    *   Example: "The **cat**, which was fluffy and playful, chased the **mouse**." To understand what "chased" refers to, you pay more attention to "cat" and "mouse".
    *   Similarly, in translation: "Je suis étudiant" -> "I **am** a student". "suis" needs to be translated to "am", and its meaning is tied to "Je".
    *   The **Attention Mechanism** in neural networks tries to mimic this by allowing the model to dynamically weigh the importance of different parts of the input sequence when processing information.

*   **Self-Attention:**
    *   **Core Idea:** In **self-attention**, words (or tokens) in a single sequence "pay attention" to other words *within the same sequence* to build a more contextually rich representation of each word. Each token looks at its neighbors (and itself) to better understand its own meaning in that specific context.
    *   **Analogy:** Imagine a group of people in a room trying to define themselves. Each person looks at others in the room to understand their own role or identity *within that group*. The "self-attention" helps each word understand itself better by considering its relationship with all other words in the sentence.

    *   **Queries, Keys, and Values (QKV):**
        For each input token (embedding), three distinct vectors are created:
        1.  **Query (Q):** Represents the current token's "question" or what it's looking for. "I am this word, what other words are relevant to me?"
        2.  **Key (K):** Represents a characteristic or "label" of other tokens in the sequence. "I am this other word, this is what I'm about."
        3.  **Value (V):** Represents the actual content or meaning of other tokens. "I am this other word, this is the information I carry."

    *   **Scaled Dot-Product Attention (Conceptual Overview):**
        This is the most common type of self-attention. Here's the process for a single token:
        1.  **Calculate Similarity (Scores):** The **Query (Q)** vector of the current token is compared with the **Key (K)** vectors of all other tokens (including itself) in the sequence. This comparison is typically done using a **dot product**. A higher dot product means higher similarity or relevance.
            *   *Why dot product? It measures how much two vectors point in the same direction.*
        2.  **Scale:** The scores are scaled down by dividing by the square root of the dimension of the key vectors. This helps stabilize gradients during training, preventing the dot products from becoming too large.
        3.  **Softmax for Weights:** A **softmax** function is applied to the scaled scores. This converts the scores into probabilities (positive numbers that sum to 1). These probabilities are the **attention weights**. Each weight indicates how much attention the current token should pay to every other token in the sequence.
            *   *A token with a high attention weight contributes more to the current token's updated representation.*
        4.  **Weighted Sum of Values:** The attention weights are then used to compute a weighted sum of the **Value (V)** vectors of all tokens in the sequence. Tokens with higher attention weights contribute more of their "value" to the output.
        5.  **Output:** The result of this weighted sum is the new, contextually enriched representation for the current token. This process is repeated for every token in the sequence, typically in parallel.

        *   *(Conceptual Diagram: A token's Query vector interacting with Key vectors of all tokens, producing scores. Scores -> Softmax -> Weights. Weights multiply Value vectors, which are then summed up to form the output vector for that token.)*

    *   **Advantage:**
        *   **Capturing Long-Range Dependencies:** Unlike RNNs where information from distant words can get diluted, attention can directly connect words far apart in a sequence if they are semantically relevant. For "The cat ... chased the mouse," "cat" can directly attend to "mouse" and vice-versa, regardless of the intervening words.
        *   **Parallelization:** Attention scores between all pairs of tokens can be computed simultaneously, making it highly efficient on modern hardware like GPUs.

*   **Cross-Attention (Briefly):**
    *   Used in Encoder-Decoder architectures (discussed next).
    *   The **Query (Q)** vectors come from the decoder (e.g., the word being generated).
    *   The **Key (K)** and **Value (V)** vectors come from the encoder's output (representing the input sentence).
    *   This allows the decoder, while generating an output sequence (e.g., a translation), to look back and pay attention to relevant parts of the *entire input sequence*.

## 4. The Transformer Architecture

*   **Recap: Limitations of RNNs/LSTMs:**
    *   **Sequential Processing:** RNNs (Recurrent Neural Networks) and LSTMs (Long Short-Term Memory networks) process words one by one, sequentially. This makes it difficult to parallelize computations.
    *   **Vanishing/Exploding Gradients:** For long sequences, gradients (signals used for learning) can become too small (vanish) or too large (explode), making it hard to learn long-range dependencies. LSTMs mitigate this but don't fully solve it for very long sequences.

*   **The "Attention Is All You Need" Paper (Vaswani et al., 2017):**
    *   This seminal paper from Google introduced the **Transformer** architecture.
    *   Its significance lies in demonstrating that a model architecture based *solely on attention mechanisms*, without any recurrent layers, could achieve state-of-the-art results in machine translation and other NLP tasks.
    *   It paved the way for the development of modern LLMs like GPT and BERT.

*   **High-Level Components:**

    *(Conceptual Diagram: A high-level block diagram showing the Encoder stack on the left feeding into the Decoder stack on the right. Arrows indicate data flow. Maybe another simpler one for Decoder-only.)*
    [Link to a good Transformer diagram: e.g., Jay Alammar's "The Illustrated Transformer" blog post has excellent visualizations: http://jalammar.github.io/illustrated-transformer/]

    *   **A. Encoder-Decoder Architecture (Original Transformer, e.g., for Neural Machine Translation - NMT):**
        This architecture is designed for sequence-to-sequence tasks like translating "Hello world" (input sequence) to "Bonjour le monde" (output sequence).

        *   **Encoder Stack:** Processes the entire input sequence and creates a rich contextual representation. It consists of multiple identical layers (e.g., 6 layers). Each layer has two main sub-layers:
            1.  **Input Embedding + Positional Encoding:**
                *   **Input Embedding:** As discussed, input tokens are converted into dense vectors.
                *   **Positional Encoding:** Since self-attention itself doesn't inherently know word order (it treats input as a "bag of words" with attention scores), we need to inject information about the position of each token in the sequence. This is done by adding a **positional encoding vector** to each input embedding. These vectors are pre-calculated (e.g., using sine and cosine functions of different frequencies) or learned.
            2.  **Multi-Head Self-Attention:** This is the core self-attention mechanism. "Multi-Head" means the attention mechanism is run multiple times in parallel with different, learned linear projections of Q, K, and V. This allows the model to jointly attend to information from different representational subspaces at different positions. The outputs are concatenated and linearly transformed.
            3.  **Add & Norm (Residual Connection and Layer Normalization):**
                *   **Residual Connection:** The output of the multi-head attention sub-layer is added to its input (the output from the layer below). This helps with deeper networks by allowing gradients to flow more easily.
                *   **Layer Normalization:** Normalizes the activations within a layer to stabilize training.
            4.  **Position-wise Feed-Forward Network (FFN):** A simple fully connected neural network applied independently to each position (token representation). It consists of two linear transformations with a ReLU activation in between. This adds further processing capacity.
                *   (Another Add & Norm layer follows the FFN).

        *   **Decoder Stack:** Takes the encoder's output and generates the output sequence token by token. It also consists of multiple identical layers. Each layer has three main sub-layers:
            1.  **Output Embedding + Positional Encoding:** The previously generated output tokens are embedded and positional encodings are added.
            2.  **Masked Multi-Head Self-Attention:** Similar to the encoder's self-attention, but with a crucial difference: **masking**. When predicting the *i*-th word, the decoder should only attend to previously generated words (positions 1 to *i*-1) and not "see" future words. The mask prevents attention to subsequent positions.
            3.  **(Add & Norm)**
            4.  **Multi-Head Cross-Attention:**
                *   **Queries (Q)** come from the output of the previous masked multi-head attention layer in the decoder.
                *   **Keys (K)** and **Values (V)** come from the *output of the entire encoder stack*.
                *   This allows the decoder to look at all parts of the input sentence to decide which parts are most relevant for generating the current output token.
            5.  **(Add & Norm)**
            6.  **Position-wise Feed-Forward Network (FFN):** Same as in the encoder.
            7.  **(Add & Norm)**
        *   **Final Linear + Softmax Layer:** After the decoder stack, a linear layer projects the final decoder output into logits over the entire output vocabulary. A softmax function then converts these logits into probabilities, and the token with the highest probability is chosen as the next output token.

    *   **B. Decoder-Only Architecture (e.g., GPT-style models, LLaMA, most generative LLMs):**
        *   **Simplification:** These models essentially use only the **decoder** part of the original Transformer architecture. They are well-suited for tasks where the goal is to generate text based on an initial prompt (text completion, creative writing, chatbots).
        *   **Architecture Components:**
            1.  **Input Embedding + Positional Encoding:** The input prompt tokens are embedded and positional encodings added.
            2.  **Masked Multi-Head Self-Attention:** Each token in the prompt (and tokens generated so far) attends to all other tokens up to its own position in the sequence. The "masking" is crucial because these models are **autoregressive** – they generate one token at a time, and the prediction of the next token should only depend on the preceding tokens.
            3.  **Add & Norm**
            4.  **Position-wise Feed-Forward Network (FFN)**
            5.  **Add & Norm**
        *   This block (Masked Multi-Head Attention -> Add & Norm -> FFN -> Add & Norm) is repeated multiple times (many layers).
        *   **Focus:** This architecture is the foundation for most of the generative LLMs we interact with today, including those run with Ollama. The primary task is often "next token prediction."

*   **Key Innovations (Recap and Emphasis):**
    *   **Multi-Head Attention:** Instead of one set of Q, K, V, the model learns multiple sets in parallel. Each "head" can focus on different types of relationships or aspects of the input. For example, one head might focus on syntactic relationships, another on semantic similarity over long distances. This provides a richer, more nuanced understanding.
    *   **Positional Encoding:** Crucial for injecting word order information because the self-attention mechanism itself is permutation-invariant (shuffling words wouldn't change the raw attention scores without positional info).
    *   **Layer Normalization and Residual Connections:** These are standard neural network techniques that are vital for successfully training deep Transformer models. Residual connections help prevent vanishing gradients by providing "shortcuts" for the gradient signal, while layer normalization helps stabilize the learning process by keeping activations in a consistent range.

## 5. Conclusion

*   **Building Blocks of Modern LLMs:**
    The concepts of **embeddings** (representing words numerically), **attention** (dynamically weighing word importance), and the **Transformer architecture** (which effectively combines these ideas) are the fundamental building blocks of the powerful LLMs we see today.
    *   Embeddings provide the initial meaningful input.
    *   Self-attention allows models to understand context within the input.
    *   The Transformer provides a scalable and parallelizable framework to train very deep networks using these mechanisms.

*   **Preview of Upcoming Topics:**
    Understanding these core concepts will be essential as we move forward:
    *   **Fine-tuning (Week 4):** When we discuss fine-tuning, we're essentially taking a pre-trained Transformer model and further training its weights (including embeddings and attention parameters) on a more specific dataset.
    *   **Retrieval Augmented Generation (RAG) (Week 6):** In RAG, the LLM (a Transformer) uses its attention mechanism to process both the user's query and the retrieved documents to generate an informed answer. The documents themselves are often found by comparing the query embedding with embeddings of document chunks.
    *   **Sentiment Analysis (Week 5):** The Transformer's ability to create context-rich representations is key to understanding the sentiment expressed in text.

By grasping these fundamentals, you're better equipped to understand not just *what* LLMs can do, but also *how* they achieve their remarkable linguistic capabilities.
