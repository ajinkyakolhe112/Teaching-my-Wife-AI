# Lab 2: Exploring Word Embeddings and Attention Concepts

**Objective:** To provide hands-on experience with loading and visualizing pre-trained word embeddings, and to conceptually understand attention scores.

**Prerequisites:**
*   Python 3.x installed.
*   Familiarity with basic Python programming.
*   Libraries: `gensim`, `numpy`, `matplotlib`, `scikit-learn` (sklearn).

If you don't have these libraries installed, please install them using pip:
```bash
pip install gensim numpy matplotlib scikit-learn
# You might also need to download a model for gensim, see Task 1.
```

---

## Tasks:

### 1. Loading Pre-trained Word Embeddings (GloVe)

For this lab, we'll use a relatively small set of GloVe embeddings. GloVe (Global Vectors for Word Representation) is another popular word embedding technique. The `glove-wiki-gigaword-50` embeddings are trained on Wikipedia and Gigaword data, with each word represented by a 50-dimensional vector.

*   **Downloading the Gensim Data:**
    The `gensim` library provides a convenient way to download and use common datasets and models.
    ```python
    import gensim.downloader as api

    # This will download the model (around 69MB).
    # It might take a few minutes depending on your internet connection.
    # The model will be saved to a local cache directory (usually ~/gensim-data).
    try:
        model = api.load("glove-wiki-gigaword-50")
        print("Model 'glove-wiki-gigaword-50' loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an active internet connection for the first download.")
        print("You can find more models here: https://radimrehurek.com/gensim/downloader.html")

    # If the above fails, or you prefer manual download:
    # 1. Go to https://nlp.stanford.edu/projects/glove/
    # 2. Download 'glove.6B.zip' (822MB)
    # 3. Extract it. You'll find 'glove.6B.50d.txt', 'glove.6B.100d.txt', etc.
    # 4. To load 'glove.6B.50d.txt' (you'll need to convert it to word2vec format or use a different loading method for raw GloVe files):
    # from gensim.scripts.glove2word2vec import glove2word2vec
    # glove_input_file = 'path/to/your/glove.6B.50d.txt'
    # word2vec_output_file = 'glove.6B.50d.txt.word2vec'
    # glove2word2vec(glove_input_file, word2vec_output_file)
    # model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    ```

*   **Basic Operations with Word Embeddings:**
    Once the model is loaded, you can perform various operations:

    ```python
    # Ensure the model is loaded from the previous step before running this
    if 'model' not in globals():
        print("Model not loaded. Please run the download cell first.")
    else:
        # Get the vector for a word
        try:
            king_vector = model['king']
            print(f"Vector for 'king' (first 5 dimensions): {king_vector[:5]}")
            print(f"Dimension of word vectors: {len(king_vector)}")
        except KeyError:
            print("'king' not in vocabulary.")

        # Find the most similar words
        try:
            similar_to_king = model.most_similar('king', topn=5)
            print(f"\nMost similar to 'king': {similar_to_king}")

            similar_to_woman = model.most_similar('woman', topn=5)
            print(f"\nMost similar to 'woman': {similar_to_woman}")

            similar_to_apple = model.most_similar('apple', topn=5)
            print(f"\nMost similar to 'apple': {similar_to_apple}")
        except KeyError as e:
            print(f"Word {e} not in vocabulary for similarity search.")

        # Vector arithmetic (Analogy: "king" - "man" + "woman" should be close to "queen")
        try:
            # Ensure all words are in the vocabulary
            words_for_analogy = ['king', 'man', 'woman']
            if all(word in model for word in words_for_analogy):
                result_vector = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=3)
                print(f"\n'king' - 'man' + 'woman' = {result_vector}") # Expected: close to 'queen'
            else:
                print(f"\nOne or more words for analogy ({words_for_analogy}) not in vocabulary.")

            # Another example: "france" - "paris" + "london"
            words_for_analogy_2 = ['france', 'paris', 'london']
            if all(word in model for word in words_for_analogy_2):
                result_vector_2 = model.most_similar(positive=['france', 'london'], negative=['paris'], topn=3)
                print(f"\n'france' - 'paris' + 'london' = {result_vector_2}") # Expected: close to 'england' or 'britain'
            else:
                print(f"\nOne or more words for analogy ({words_for_analogy_2}) not in vocabulary.")

        except KeyError as e:
            print(f"Word {e} not in vocabulary for vector arithmetic.")
    ```

    **Questions to Ponder:**
    *   Are the "most similar" words what you expected?
    *   How well does the vector arithmetic work? Does "king - man + woman" result in "queen" or something very similar?

### 2. Visualizing Word Embeddings with t-SNE

t-SNE (t-distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique that is particularly well-suited for visualizing high-dimensional datasets in 2D or 3D.

*   **Code for Visualization:**

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Ensure the model is loaded from Task 1
    if 'model' not in globals():
        print("Model not loaded. Please run the download cell in Task 1 first.")
    else:
        # Select a vocabulary of 20-30 words
        vocab = [
            'king', 'queen', 'man', 'woman', 'prince', 'princess', # Royalty/People
            'apple', 'orange', 'banana', 'grape', 'mango',      # Fruits
            'dog', 'cat', 'horse', 'lion', 'tiger',             # Animals
            'france', 'germany', 'china', 'japan', 'india',     # Countries
            'teacher', 'doctor', 'engineer', 'artist'           # Professions
        ]

        # Filter out words not in the model's vocabulary
        words_in_model = [word for word in vocab if word in model]
        if len(words_in_model) < 5: # Need at least a few words to visualize
             print("Not enough words from the vocab are in the model. Please check your vocab or model.")
        else:
            word_vectors = np.array([model[word] for word in words_in_model])

            # Reduce dimensionality using t-SNE
            # n_components=2 means we want to reduce to 2 dimensions
            # perplexity is roughly the number of close neighbors
            # random_state ensures reproducibility
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(words_in_model)-1))
            vectors_2d = tsne.fit_transform(word_vectors)

            # Create a scatter plot
            plt.figure(figsize=(14, 10))
            plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], edgecolors='k', c='r', s=100, alpha=0.7)

            # Annotate points with the words
            for i, word in enumerate(words_in_model):
                plt.annotate(word,
                             xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
                             xytext=(5, 2), # Small offset for text
                             textcoords='offset points',
                             ha='right',
                             va='bottom')

            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.title("Visualizing Word Embeddings using t-SNE")
            plt.grid(True)
            plt.show()

    ```

*   **Observations:**
    *   Examine the resulting plot. Do words with similar meanings (e.g., fruits, countries, professions) cluster together?
    *   Are there any surprising placements?
    *   What does this tell you about how word embeddings capture semantic relationships?

### 3. Conceptual Attention Score Calculation (Simplified)

This task is designed to give you a conceptual feel for how attention scores might be calculated. We are **not** building a Transformer here, just simulating a tiny part of the attention mechanism with hypothetical, simplified vectors.

Imagine we have a **Query** word and want to see which **Key** words are most relevant to it.

*   **Scenario:**
    *   Query word: "fruit"
    *   Key words: "apple", "car", "banana"

*   **Hypothetical Vectors (Keep them simple, e.g., 3-4 dimensions):**
    Let's manually define some very simple vectors for these words. In a real model, these would be learned embeddings.
    *   `vec_fruit = np.array([0.9, 0.1, 0.8])`  (Represents "fruit-ness", "vehicle-ness", "sweetness")
    *   `vec_apple = np.array([0.8, 0.2, 0.7])`  (High fruit-ness, low vehicle-ness, high sweetness)
    *   `vec_car   = np.array([0.1, 0.9, 0.1])`  (Low fruit-ness, high vehicle-ness, low sweetness)
    *   `vec_banana= np.array([0.7, 0.1, 0.9])`  (High fruit-ness, low vehicle-ness, very high sweetness)

*   **Calculate Dot Product Similarity (Scores):**
    The dot product between the query vector and each key vector gives us a similarity score. A higher score means more similarity.

    ```python
    import numpy as np

    # Hypothetical vectors
    vec_query_fruit = np.array([0.9, 0.1, 0.8])
    key_vectors = {
        "apple": np.array([0.8, 0.2, 0.7]),
        "car":   np.array([0.1, 0.9, 0.1]),
        "banana":np.array([0.7, 0.1, 0.9])
    }

    attention_scores = {}
    print("--- Attention Scores (Dot Product) ---")
    for key_word, key_vec in key_vectors.items():
        score = np.dot(vec_query_fruit, key_vec)
        attention_scores[key_word] = score
        print(f"Score(fruit, {key_word}): {score:.2f}")
    ```

*   **Apply Softmax to Get Attention Weights:**
    The softmax function converts these raw scores into probabilities (values between 0 and 1 that sum to 1). These are the "attention weights".

    ```python
    def softmax(scores_dict):
        """Applies softmax to a dictionary of scores."""
        scores_values = np.array(list(scores_dict.values()))
        exp_scores = np.exp(scores_values - np.max(scores_values)) # Subtract max for numerical stability
        probabilities = exp_scores / np.sum(exp_scores)
        
        weighted_dict = {}
        for i, key in enumerate(scores_dict.keys()):
            weighted_dict[key] = probabilities[i]
        return weighted_dict

    attention_weights = softmax(attention_scores)
    print("\n--- Attention Weights (Softmax of Scores) ---")
    for word, weight in attention_weights.items():
        print(f"Weight(fruit, {word}): {weight:.3f}")

    # Find the key with the highest attention weight
    most_attended_word = max(attention_weights, key=attention_weights.get)
    print(f"\nFor the query 'fruit', the key word with the highest attention is: '{most_attended_word}' with a weight of {attention_weights[most_attended_word]:.3f}")
    ```

*   **Interpretation:**
    *   Which key word(s) received the highest attention weight for the query "fruit"? Does this make intuitive sense based on our hypothetical vectors?
    *   How do these weights indicate which keys are most "important" or "attended to" for the given query?
    *   This simplified example demonstrates the core idea: calculating relevance scores and normalizing them to distribute "attention" across different items. In a full Transformer, this happens with learned Q, K, and V vectors derived from input embeddings, and it's done across many "heads" in parallel.

---

This lab provides a glimpse into how words are represented numerically and how attention mechanisms can selectively focus on relevant information. These are foundational concepts for understanding how LLMs process and generate language.
