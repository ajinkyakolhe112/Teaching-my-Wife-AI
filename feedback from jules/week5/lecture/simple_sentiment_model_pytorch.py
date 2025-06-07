import torch
import torch.nn as nn
import re
from collections import Counter
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader

class SimpleTokenizer:
    def __init__(self, max_length=250):
        self.max_length = max_length
        self.word2idx = {}
        self.idx2word = {}
        
    def build_vocab(self, texts, max_vocab_size=10000):
        # Count all words
        word_counts = Counter()
        for text in texts:
            # Clean and tokenize text
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            words = text.split()
            word_counts.update(words)
        
        # Get most common words
        most_common = word_counts.most_common(max_vocab_size - 2)  # -2 for <pad> and <unk>
        
        # Create vocabulary
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        for word, _ in most_common:
            self.word2idx[word] = len(self.word2idx)
        
        # Create reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
    def tokenize(self, text):
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        # Split on whitespace
        tokens = text.split()
        
        # Convert words to indices
        indices = [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
        
        # Truncate or pad to max_length
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [self.word2idx['<pad>']] * (self.max_length - len(indices))
            
        return torch.tensor(indices)

class SimpleSentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64):
        super(SimpleSentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        x = torch.mean(x, dim=1)  # Average pooling over sequence length
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_and_prepare_data(batch_size=32):
    # Load IMDB dataset
    dataset = load_dataset("stanfordnlp/imdb")
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    
    # Initialize tokenizer and build vocabulary
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(train_texts)
    
    # Convert texts to tensors
    train_data = torch.stack([tokenizer.tokenize(text) for text in train_texts])
    train_labels = torch.tensor(train_labels, dtype=torch.float)
    
    # Create DataLoader directly from tensors
    train_loader = DataLoader(
        list(zip(train_data, train_labels)),
        batch_size=batch_size,
        shuffle=True
    )
    
    return train_loader, tokenizer

# Example usage:
if __name__ == "__main__":
    # Load and prepare data
    train_loader, tokenizer = load_and_prepare_data()
    
    # Print some statistics
    print(f"Vocabulary size: {len(tokenizer.word2idx)}")
    print(f"Number of training examples: {len(train_loader.dataset)}")
    
    # Example text processing
    example_text = "This movie was really great! I loved it."
    tokens = tokenizer.tokenize(example_text)
    print("\nExample tokenization:")
    print(f"Original text: {example_text}")
    print(f"Tokenized indices: {tokens}")
    print(f"Tokenized words: {[tokenizer.idx2word[idx.item()] for idx in tokens]}")
    
    # Initialize model
    model = SimpleSentimentModel(len(tokenizer.word2idx))
    print("\nModel architecture:")
    print(model) 