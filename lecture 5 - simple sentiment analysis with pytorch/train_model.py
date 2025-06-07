import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from simple_sentiment_model import SimpleSentimentModel, load_and_prepare_data
import numpy as np
from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss/len(train_loader)
        epoch_accuracy = 100.*correct/total
        
        # Print epoch statistics
        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Average Loss: {epoch_loss:.4f}')
        print(f'Accuracy: {epoch_accuracy:.2f}%')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load and prepare data
    train_loader, tokenizer = load_and_prepare_data(batch_size=32)
    
    # Initialize model
    model = SimpleSentimentModel(len(tokenizer.word2idx))
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, device)
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
    }, 'sentiment_model.pth')
    print('\nModel saved to sentiment_model.pth')

if __name__ == "__main__":
    main() 