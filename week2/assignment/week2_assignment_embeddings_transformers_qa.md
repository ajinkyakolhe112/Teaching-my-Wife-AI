# Assignment 2: Understanding Embeddings and Transformer Concepts

**Objective:** To test understanding of word embeddings, attention, and the Transformer architecture through conceptual questions and a small coding task.

**Prerequisites:**
*   Understanding of concepts covered in Week 2 Lecture and Lab.
*   Python 3.x installed.
*   Access to `gensim` library and a pre-trained word embedding model (e.g., `glove-wiki-gigaword-50` used in the lab, or you can use others like `glove-twitter-25`).

If you need to install `gensim` or download models:
```bash
pip install gensim
# In Python, to download a model (e.g., glove-twitter-25, ~45MB):
# import gensim.downloader as api
# model = api.load("glove-twitter-25")
```

---

## Tasks:

### 1. Word Embeddings Questions

Please answer the following questions concisely, in your own words.

1.  **One-Hot Encoding vs. Embeddings:**
    Explain why word embeddings are generally preferred over one-hot encoding for representing words in Natural Language Processing (NLP) models (2-3 sentences).

2.  **Vector Relationships:**
    If you have word embeddings for "cat", "dog", and "animal", how might their vectors be related to each other in the embedding space? (e.g., consider proximity, direction).

3.  **Limitation of Pre-trained Embeddings:**
    What is one potential limitation or drawback of using pre-trained word embeddings directly without fine-tuning them on a specific task's dataset?

### 2. Attention and Transformer Questions

Please answer the following questions concisely, in your own words.

1.  **Purpose of Self-Attention:**
    What is the main purpose or benefit of the self-attention mechanism in a Transformer model?

2.  **Need for Positional Encoding:**
    Why is positional encoding necessary in Transformer models, given that they use self-attention?

3.  **Masked Multi-Head Attention:**
    What is "masked" multi-head attention? In which part of the standard Transformer architecture (encoder or decoder) is it typically used, and why is the masking important in that context?

4.  **Transformer vs. RNNs/LSTMs:**
    Briefly describe one significant advantage of the Transformer architecture over traditional Recurrent Neural Networks (RNNs) or Long Short-Term Memory (LSTM) networks for processing sequences.

### 3. Small Coding Task: Finding the "Odd One Out"

*   **Objective:** Using a pre-trained word embedding model, write a Python function to identify the word that is semantically least similar to the others in a given list.

*   **Methodology:**
    1.  Calculate the average vector of all words in the list.
    2.  For each word in the list, calculate its cosine similarity to this average vector.
    3.  The word with the lowest cosine similarity to the average vector is considered the "odd one out."

*   **Instructions:**
    1.  Load a pre-trained word embedding model (e.g., `glove-wiki-gigaword-50` from the lab, or `glove-twitter-25`, or any other `gensim` compatible model you prefer).
    2.  Implement the Python function `find_odd_one_out(word_list, model)`.
    3.  Test your function with the following word lists:
        *   `list1 = ["apple", "banana", "car", "orange"]`
        *   `list2 = ["python", "java", "cat", "javascript"]`
        *   `list3 = ["paris", "london", "berlin", "banana"]`
        *   `list4 = ["happy", "joyful", "sad", "ecstatic"]` (Note: this one might be trickier or more subjective based on the embeddings)

*   **Code Snippet to Start (using `gensim`):**

    ```python
    import numpy as np
    from gensim.models import KeyedVectors # Or use gensim.downloader
    import gensim.downloader as api
    from sklearn.metrics.pairwise import cosine_similarity # For similarity calculation

    # --- Load your chosen model ---
    # Example: model = api.load("glove-wiki-gigaword-50")
    # Or: model = api.load("glove-twitter-25")
    # Ensure the model is loaded before calling the function.
    # For this assignment, you can load it globally or pass the loaded model object.
    
    # Example of loading (uncomment and choose one, or use your preferred method)
    try:
        # model_name = "glove-wiki-gigaword-50" # ~69MB
        model_name = "glove-twitter-25" # ~45MB, trained on tweets, might give interesting results
        model = api.load(model_name)
        print(f"Model '{model_name}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}. Please ensure you have an internet connection or the model is cached.")
        model = None # Set model to None if loading fails

    def find_odd_one_out(word_list, embedding_model):
        if embedding_model is None:
            return "Model not loaded."

        # Filter out words not in the model's vocabulary
        valid_words = [word for word in word_list if word in embedding_model.key_to_index]
        if len(valid_words) < 2: # Need at least two words to compare
            return "Not enough valid words from the list are in the model's vocabulary."

        # Get the vectors for the valid words
        word_vectors = [embedding_model[word] for word in valid_words]

        # Calculate the average vector
        average_vector = np.mean(word_vectors, axis=0).reshape(1, -1) # Reshape for cosine_similarity

        min_similarity = float('inf')
        odd_one_out = None

        print(f"\nProcessing list: {word_list}")
        print(f"Valid words in model: {valid_words}")

        for i, word in enumerate(valid_words):
            current_vector = word_vectors[i].reshape(1, -1) # Reshape for cosine_similarity
            similarity = cosine_similarity(current_vector, average_vector)[0][0]
            print(f"- Similarity of '{word}' to average: {similarity:.4f}")
            
            if similarity < min_similarity:
                min_similarity = similarity
                odd_one_out = word
        
        return odd_one_out

    # --- Test your function with the provided lists ---
    if model: # Only run if the model loaded successfully
        list1 = ["apple", "banana", "car", "orange"]
        odd1 = find_odd_one_out(list1, model)
        print(f"The odd one out in {list1} is: {odd1}")

        list2 = ["python", "java", "cat", "javascript"]
        odd2 = find_odd_one_out(list2, model)
        print(f"The odd one out in {list2} is: {odd2}")

        list3 = ["paris", "london", "berlin", "banana"]
        odd3 = find_odd_one_out(list3, model)
        print(f"The odd one out in {list3} is: {odd3}")
        
        list4 = ["happy", "joyful", "sad", "ecstatic"]
        odd4 = find_odd_one_out(list4, model)
        print(f"The odd one out in {list4} is: {odd4}")
    else:
        print("\nCannot run coding task as the embedding model failed to load.")

    ```

*   **Submission Content for this Task:**
    *   Your complete Python code for the `find_odd_one_out` function and the script to test it.
    *   The identified "odd one out" for each of the provided lists, along with the similarity scores printed by your script for each word in the lists.

---

## Submission Guidelines:

*   Compile your answers to the conceptual questions (Part 1 and 2) and your Python code and results for the coding task (Part 3) into a single document.
*   Clearly label each part and question.
*   You can use Markdown (preferred, save as a `.md` file) or create a PDF.
*   Name your file clearly, e.g., `week2_assignment_yourname.md`.

Good luck! This assignment will help solidify your understanding of the foundational concepts behind modern LLMs.
