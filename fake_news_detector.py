import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Assuming the columns are 'headline' and 'label' (0 for real, 1 for fake)
    return df['headline'].values, df['label'].values

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare the data
def prepare_data(headlines, labels, max_length=64):
    # Tokenize the headlines
    encodings = tokenizer(
        headlines.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Create dataset
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels)
    )
    return dataset

# Training function
def train_model(model, train_loader, val_loader, device, epochs=3):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Training loop
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_acc = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                val_acc += (predictions == labels).sum().item()
                val_steps += len(labels)
        
        val_acc = val_acc / val_steps
        print(f'Epoch {epoch + 1}:')
        print(f'Average training loss: {total_loss / len(train_loader):.4f}')
        print(f'Validation accuracy: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_fake_news_model.pt')
            print(f'New best model saved with validation accuracy: {best_val_acc:.4f}')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    headlines, labels = load_data('C:/Users/Shrey/OneDrive/Desktop/los_altos/losaltos/WELFake_Dataset.csv')
    
    # Split data
    train_headlines, val_headlines, train_labels, val_labels = train_test_split(
        headlines, labels, test_size=0.2, random_state=42
    )
    
    # Prepare datasets
    print('Preparing datasets...')
    train_dataset = prepare_data(train_headlines, train_labels)
    val_dataset = prepare_data(val_headlines, val_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize model
    print('Initializing model...')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2  # Binary classification (real/fake)
    ).to(device)
    
    # Train model
    print('Starting training...')
    train_model(model, train_loader, val_loader, device)
    
    print('Training completed!')

if __name__ == '__main__':
    main() 
