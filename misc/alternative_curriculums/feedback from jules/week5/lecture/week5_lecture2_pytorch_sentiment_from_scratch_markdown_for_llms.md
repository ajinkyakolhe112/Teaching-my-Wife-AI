# Converted from week5_lecture2_pytorch_sentiment_from_scratch.ipynb - Markdown format optimized for LLM readability

# Lecture 5.2: Building a Simple Sentiment Analysis Model from Scratch with PyTorch

## Introduction

In this notebook, we'll walk through the process of building, training, and evaluating a simple sentiment analysis model using PyTorch. This approach gives us more control over the model architecture and training process compared to using pre-built pipelines. We'll be examining the components from two Python scripts:

1.  `simple_sentiment_model_pytorch.py`: Defines the tokenizer, the PyTorch model class, and data loading/preparation functions.
2.  `train_sentiment_model_pytorch.py`: Contains the training loop, evaluation logic, and utilities for saving the model.

While the scripts are designed to be run directly, we will replicate parts of their logic here to explain each component step-by-step.

**Learning Objectives:**
*   Understand how to create a custom tokenizer.
*   Learn to define a simple neural network for text classification in PyTorch.
*   See how to prepare a dataset and use PyTorch `DataLoader`.
*   Grasp the fundamentals of a PyTorch training loop.

**Note:** For this notebook, we'll focus on explaining the code. To run the full training, you would typically execute the `train_sentiment_model_pytorch.py` script in an environment with PyTorch, datasets, and tqdm installed (`pip install torch datasets tqdm scikit-learn pandas`). Weights and Biases (`wandb`) is also used for logging in the script, which is optional but good practice.

## 1. The `SimpleTokenizer` Class

Source: `simple_sentiment_model_pytorch.py`

The `SimpleTokenizer` is responsible for converting raw text into a sequence of numerical tokens that our model can understand. It performs several key steps:

1.  **Vocabulary Building (`build_vocab`):**
    *   Takes a list of texts as input.
    *   Cleans the text: converts to lowercase, removes special characters (keeps alphanumeric and spaces).
    *   Counts the frequency of all words in the corpus.
    *   Builds a vocabulary (`word2idx` mapping words to integer indices) using the most common words, up to `max_vocab_size`.
    *   Includes special tokens: `<pad>` (for padding shorter sequences) and `<unk>` (for unknown words not in the vocabulary).

2.  **Tokenization (`tokenize`):**
    *   Takes a single text string as input.
    *   Cleans the text (lowercase, remove special characters).
    *   Splits the text into words.
    *   Converts words to their corresponding integer indices using the built vocabulary. Uses `<unk>` for out-of-vocabulary words.
    *   Pads or truncates the sequence of indices to a fixed `max_length`.
    *   Returns a PyTorch tensor of these indices.

```python
import torch
import torch.nn as nn
import re
from collections import Counter

class SimpleTokenizer:
    def __init__(self, max_length=250):
        self.max_length = max_length
        self.word2idx = {} # Dictionary to map words to unique integer IDs
        self.idx2word = {} # Dictionary to map IDs back to words
        
    def build_vocab(self, texts, max_vocab_size=10000):
        """Builds a vocabulary from a list of texts."""
        word_counts = Counter()
        for text in texts:
            # Clean and tokenize text
            text = text.lower() # Convert to lowercase
            text = re.sub(r'[^\w\s]', ' ', text) # Remove punctuation, keep words and spaces
            words = text.split() # Split by whitespace
            word_counts.update(words)
        
        # Get the most common words, reserving space for <pad> and <unk> tokens
        most_common = word_counts.most_common(max_vocab_size - 2)
        
        # Initialize vocabulary with special tokens
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        for word, _ in most_common:
            if word not in self.word2idx: # Ensure no duplicates if a common word is somehow a special token name
                self.word2idx[word] = len(self.word2idx)
        
        # Create reverse mapping from index to word
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
    def tokenize(self, text):
        """Converts a single text string to a tensor of token indices."""
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        
        # Convert words to indices, using <unk> for words not in vocab
        indices = [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
        
        # Pad or truncate to max_length
        if len(indices) > self.max_length:
            indices = indices[:self.max_length] # Truncate
        else:
            indices = indices + [self.word2idx['<pad>']] * (self.max_length - len(indices)) # Pad
            
        return torch.tensor(indices, dtype=torch.long)

# Example Usage of SimpleTokenizer
sample_texts = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document? Yes, it is."
]

tokenizer = SimpleTokenizer(max_length=10)
tokenizer.build_vocab(sample_texts, max_vocab_size=10) # Small vocab for demo

print(f"Vocabulary (word2idx): {tokenizer.word2idx}")
print(f"Vocabulary size: {len(tokenizer.word2idx)}")

example_sentence = "This is a new document, is it not?"
tokenized_sentence = tokenizer.tokenize(example_sentence)

print(f"\nOriginal sentence: {example_sentence}")
print(f"Tokenized indices: {tokenized_sentence}")
print(f"Reconstructed tokens: {[tokenizer.idx2word[idx.item()] for idx in tokenized_sentence]}")
```

## 2. The `SimpleSentimentModel` Class

Source: `simple_sentiment_model_pytorch.py`

This PyTorch `nn.Module` defines a simple neural network for sentiment classification. Its architecture is:

1.  **Embedding Layer (`nn.Embedding`):**
    *   Takes the vocabulary size and an embedding dimension as input.
    *   Maps each token index to a dense vector representation (embedding).
    *   These embeddings are learned during training.
    *   Output shape: `(batch_size, sequence_length, embedding_dim)`.

2.  **Average Pooling:**
    *   The model uses `torch.mean(x, dim=1)` to average the embeddings across the `sequence_length` dimension.
    *   This creates a single fixed-size vector representing the entire input sequence.
    *   Output shape: `(batch_size, embedding_dim)`.
    *   *Note: More complex models might use LSTMs, GRUs, or Transformer encoders here instead of simple averaging.*

3.  **First Fully Connected Layer (`nn.Linear`):**
    *   Takes the `embedding_dim` as input and outputs 32 features.
    *   Followed by a ReLU activation function (`nn.ReLU`) to introduce non-linearity.

4.  **Second Fully Connected Layer (`nn.Linear`):**
    *   Takes the 32 features from the previous layer and outputs a single logit.
    *   This logit represents the model's prediction for the sentiment (e.g., a higher value might indicate positive sentiment, a lower value negative, before applying a sigmoid for binary classification).

The `forward` method defines how input `x` (a batch of tokenized sequences) flows through these layers.

```python
class SimpleSentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=32):
        super(SimpleSentimentModel, self).__init__()
        # Embedding layer: Converts token IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # padding_idx=0 tells the model to ignore <pad> tokens for learning embeddings
        
        # First fully connected layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        
        # ReLU activation function
        self.relu = nn.ReLU()
        
        # Output layer: A single neuron for binary sentiment (positive/negative)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        
        embedded = self.embedding(x)  # Shape: (batch_size, sequence_length, embedding_dim)
        
        # Average pooling: Take the mean of word embeddings across the sequence length dimension
        # This creates a single vector representation for the entire sequence.
        pooled = torch.mean(embedded, dim=1)  # Shape: (batch_size, embedding_dim)
        
        # Pass through the first fully connected layer and ReLU activation
        out_fc1 = self.relu(self.fc1(pooled)) # Shape: (batch_size, hidden_dim)
        
        # Pass through the output layer to get logits
        logits = self.fc2(out_fc1) # Shape: (batch_size, 1)
        
        return logits

# Example instantiation of the model
VOCAB_SIZE_EXAMPLE = len(tokenizer.word2idx) # From our previous tokenizer example
EMBEDDING_DIM_EXAMPLE = 50
HIDDEN_DIM_EXAMPLE = 16

example_model = SimpleSentimentModel(VOCAB_SIZE_EXAMPLE, EMBEDDING_DIM_EXAMPLE, HIDDEN_DIM_EXAMPLE)
print(example_model)

# Test with a dummy batch of tokenized sentences
dummy_batch_size = 2
dummy_input_indices = torch.stack([
    tokenizer.tokenize("this is a good movie"), 
    tokenizer.tokenize("a bad film experience")
]) # Shape: (2, max_length)

print(f"\nDummy input shape: {dummy_input_indices.shape}")
with torch.no_grad(): # No need to calculate gradients for this example
    dummy_output = example_model(dummy_input_indices)
print(f"Dummy output logits shape: {dummy_output.shape}") # Expected: (batch_size, 1)
print(f"Dummy output logits: \n{dummy_output}")
```

## 3. The `load_and_prepare_data` Function

Source: `simple_sentiment_model_pytorch.py`

This function handles loading the IMDB dataset (a common benchmark for sentiment analysis) and preparing it for training.

1.  **Load Dataset:** Uses the `datasets` library from Hugging Face to download and load the `stanfordnlp/imdb` dataset. This dataset contains movie reviews labeled as positive (1) or negative (0).
2.  **Initialize Tokenizer & Build Vocabulary:** Creates an instance of `SimpleTokenizer` and builds its vocabulary using the training texts from the IMDB dataset.
3.  **Convert Texts to Tensors:** Tokenizes all training texts and converts their labels into PyTorch tensors.
4.  **Create DataLoader:** Uses `torch.utils.data.DataLoader` to create an iterable that provides batches of data (tokenized texts and labels) during training. This helps manage memory and provides options for shuffling and batching.

```python
from torch.utils.data import DataLoader
from datasets import load_dataset # Hugging Face datasets library

def load_and_prepare_data(batch_size=32, max_vocab_size=10000, max_seq_length=250, subset_size=None):
    """Loads the IMDB dataset, tokenizes, and prepares DataLoader."""
    print("Loading IMDB dataset...")
    # Load IMDB dataset (train split only for this example)
    dataset = load_dataset("stanfordnlp/imdb", split='train')
    
    if subset_size:
        print(f"Using a subset of {subset_size} examples for faster demonstration.")
        dataset = dataset.select(range(subset_size))
        
    train_texts = dataset['text']
    train_labels = dataset['label']
    
    print("Building tokenizer vocabulary...")
    # Initialize tokenizer and build vocabulary
    local_tokenizer = SimpleTokenizer(max_length=max_seq_length)
    local_tokenizer.build_vocab(train_texts, max_vocab_size=max_vocab_size)
    
    print("Tokenizing texts...")
    # Convert texts to tensors
    # This can be memory intensive for large datasets if done all at once
    train_data_tensors = torch.stack([local_tokenizer.tokenize(text) for text in train_texts])
    train_labels_tensors = torch.tensor(train_labels, dtype=torch.float) # BCEWithLogitsLoss expects float labels
    
    print("Creating DataLoader...")
    # Create DataLoader directly from tensors
    # In PyTorch, a Dataset object is often used here, but for simplicity, we use a list of tuples.
    train_dataset_for_loader = list(zip(train_data_tensors, train_labels_tensors))
    
    train_loader = DataLoader(
        train_dataset_for_loader,
        batch_size=batch_size,
        shuffle=True
    )
    
    print("Data loading and preparation complete.")
    return train_loader, local_tokenizer

# Example of using load_and_prepare_data (with a small subset for speed)
# Note: Running this will download the IMDB dataset if you haven't already (approx. 84MB)
try:
    # Using a small subset to speed up the demonstration in the notebook
    # For actual training, you'd use a larger portion or the whole dataset.
    example_train_loader, example_tokenizer_for_data = load_and_prepare_data(batch_size=4, subset_size=20) 
    
    print(f"\nVocabulary size from data: {len(example_tokenizer_for_data.word2idx)}")
    print(f"Number of training examples (subset): {len(example_train_loader.dataset)}")
    print(f"Number of batches: {len(example_train_loader)}")
    
    # Inspect a batch
    for inputs, labels in example_train_loader:
        print("\nSample batch:")
        print(f"Inputs shape: {inputs.shape}") # (batch_size, max_seq_length)
        print(f"Labels shape: {labels.shape}") # (batch_size)
        print(f"First input sequence: {inputs[0]}")
        print(f"First label: {labels[0]}")
        break # Only show one batch
except Exception as e:
    print(f"Could not load data (อาจจะต้องต่อเน็ต or check dataset availability): {e}")
    print("Skipping data loading demonstration.")
```

## 4. The `train_model` Function

Source: `train_sentiment_model_pytorch.py`

This function orchestrates the training process for our `SimpleSentimentModel`.

1.  **Set Model to Train Mode:** `model.train()` tells PyTorch that the model is in training mode (this enables features like dropout if used).
2.  **Epoch Loop:** Iterates for a specified number of `num_epochs`.
3.  **Batch Loop:** Uses `tqdm` for a progress bar while iterating through batches from the `train_loader`.
    *   **Move Data to Device:** Moves input tensors and labels to the specified `device` (CPU or GPU).
    *   **Zero Gradients:** `optimizer.zero_grad()` clears old gradients before calculating new ones.
    *   **Forward Pass:** `outputs = model(inputs)` gets the model's predictions (logits).
    *   **Calculate Loss:** `loss = criterion(outputs.squeeze(), labels)` computes the loss. `nn.BCEWithLogitsLoss` is used, which combines a Sigmoid layer and Binary Cross Entropy loss, suitable for binary classification. It expects raw logits.
    *   **Backward Pass:** `loss.backward()` computes the gradients of the loss with respect to model parameters.
    *   **Optimizer Step:** `optimizer.step()` updates the model parameters using the calculated gradients.
    *   **Track Statistics:** Accumulates loss and calculates accuracy for the batch and epoch.
    *   **Logging (W&B):** If Weights & Biases (`wandb`) is used (as in the script), it logs batch and epoch metrics.

The `main()` function in `train_sentiment_model_pytorch.py` handles:
*   Initializing `wandb` (optional).
*   Setting the device.
*   Calling `load_and_prepare_data`.
*   Initializing the `SimpleSentimentModel` and moving it to the device.
*   Defining the loss function (`nn.BCEWithLogitsLoss`) and optimizer (`torch.optim.Adam`).
*   Calling `train_model`.
*   Saving the trained model's state dictionary and the tokenizer.

### Conceptual Training Snippet (Not for full execution here)

Below is a simplified conceptual representation of the training loop logic. For full execution, refer to `train_sentiment_model_pytorch.py` as it includes `wandb` integration and more complete setup.

```python
# --- Conceptual Training Snippet ---
import torch.optim as optim
from tqdm import tqdm # For progress bars

# Assume: 
# example_train_loader, example_tokenizer_for_data are loaded from previous cell
# SimpleSentimentModel class is defined

if 'example_train_loader' in globals() and 'example_tokenizer_for_data' in globals():
    # Hyperparameters (example values)
    VOCAB_SIZE_TRAIN = len(example_tokenizer_for_data.word2idx)
    EMBEDDING_DIM_TRAIN = 64
    HIDDEN_DIM_TRAIN = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 1 # Keep epochs low for notebook demo

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model, loss, optimizer
    training_model = SimpleSentimentModel(VOCAB_SIZE_TRAIN, EMBEDDING_DIM_TRAIN, HIDDEN_DIM_TRAIN).to(device)
    criterion = nn.BCEWithLogitsLoss() # Suitable for binary classification with logits output
    optimizer = optim.Adam(training_model.parameters(), lr=LEARNING_RATE)

    training_model.train() # Set model to training mode
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(example_train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1) # Ensure labels are float and match output shape
            
            optimizer.zero_grad()
            outputs = training_model(inputs) # Forward pass (get logits)
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward() # Backward pass (calculate gradients)
            optimizer.step() # Update weights
            
            total_loss += loss.item()
            # Convert logits to probabilities (via sigmoid) and then to binary predictions (0 or 1)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            progress_bar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
        
        epoch_loss = total_loss / len(example_train_loader)
        epoch_accuracy = (correct_predictions / total_samples) * 100
        print(f"Epoch {epoch+1} Completed: Avg Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
else:
    print("Skipping conceptual training snippet as data was not loaded.")
# --- End Conceptual Training Snippet ---
```

To perform actual training, you would typically run the `train_sentiment_model_pytorch.py` script. It includes important details like proper dataset handling, Weights & Biases integration for experiment tracking, and saving the final model and tokenizer.

Example command to run the script (from your terminal, in the `week5/lecture/` directory):
```bash
python train_sentiment_model_pytorch.py
```
This will train the model on the IMDB dataset and save `sentiment_model.pth` containing the model state and tokenizer.

## 5. Conclusion

This notebook has dissected the components of a simple sentiment analysis model built from scratch using PyTorch.

Key Takeaways:
*   **Custom Tokenization:** We saw how to build a vocabulary and tokenize text, providing control over text preprocessing.
*   **PyTorch Model Definition:** Defining a neural network involves creating a class that inherits from `nn.Module`, defining layers in `__init__`, and specifying the data flow in `forward`.
*   **Data Handling:** The `datasets` library simplifies loading standard datasets, and PyTorch's `DataLoader` is essential for efficient batching and shuffling during training.
*   **Training Loop:** The core PyTorch training loop involves iterating through epochs and batches, performing forward and backward passes, calculating loss, and updating model parameters.

Building models from scratch offers a deeper understanding and more flexibility, though it requires more effort than using pre-trained models via pipelines. This foundational knowledge is crucial for more advanced topics like fine-tuning larger pre-trained transformers or implementing novel architectures.
