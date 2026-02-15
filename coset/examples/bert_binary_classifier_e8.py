#!/usr/bin/env python3
"""
Pre-trained BERT Binary Classifier with Quantized Head Example

This example demonstrates how to use a pre-trained BERT model with a quantized
binary classification head using the new constructor-based QAT layers:
- Load pre-trained BERT from transformers library
- Add quantized MLP classification head using create_e8_hnlq_linear()
- Train only the classification head with QAT for binary classification
- Keep BERT frozen during training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import time
import sys
import os

# Add the parent directory to the path to import coset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from coset.core.layers import HNLQLinearQAT
    from coset.core.base import LatticeConfig
    from coset.core.e8.lattice import E8Lattice
    from coset.core.e8.layers import create_e8_hnlq_linear
    print("✓ Successfully imported coset core modules")
except ImportError as e:
    print(f"✗ Failed to import coset core modules: {e}")
    sys.exit(1)

# Check if transformers is available
try:
    from transformers import AutoTokenizer, AutoModel
    print("✓ Successfully imported transformers")
except ImportError as e:
    print(f"✗ Failed to import transformers: {e}")
    print("Please install transformers: pip install transformers")
    sys.exit(1)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class QuantizedBERTBinaryClassifier(nn.Module):
    """BERT binary classifier with quantized classification head."""
    
    def __init__(self, model_name='bert-base-uncased', warmup_epochs=2, 
                 weight_clip_value=2.0, freeze_bert=True):
        super().__init__()
        
        # Load pre-trained BERT
        print(f"Loading pre-trained BERT model: {model_name}")
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze BERT parameters if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("✓ BERT parameters frozen")
        else:
            print("✓ BERT parameters trainable")
        
        # Get BERT hidden size
        bert_hidden_size = self.bert.config.hidden_size
        print(f"BERT hidden size: {bert_hidden_size}")
        
        # Ensure hidden size is divisible by 8 for E8 lattice
        if bert_hidden_size % 8 != 0:
            # Add a projection layer to make it divisible by 8
            self.projection = nn.Linear(bert_hidden_size, ((bert_hidden_size + 7) // 8) * 8)
            classifier_input_size = ((bert_hidden_size + 7) // 8) * 8
            print(f"Added projection layer: {bert_hidden_size} -> {classifier_input_size}")
        else:
            self.projection = None
            classifier_input_size = bert_hidden_size
        
        # Quantized binary classification head using constructor
        self.classifier = create_e8_hnlq_linear(
            in_dim=classifier_input_size,
            out_dim=1,  # Binary classification: single output
            device=device,
            warmup_epochs=warmup_epochs,
            enable_diagnostics=True,
            weight_clip_value=weight_clip_value,
            q=4
        )
        
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()  # For binary classification probability
    
    def forward(self, input_ids, attention_mask=None):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        # Project to E8-compatible size if needed
        if self.projection is not None:
            pooled_output = self.projection(pooled_output)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Quantized binary classification
        logits = self.classifier(pooled_output)  # [batch_size, 1]
        logits = logits.squeeze(-1)  # [batch_size]
        
        # Apply sigmoid for probability
        probabilities = self.sigmoid(logits)
        
        return logits, probabilities
    
    def update_epoch(self, epoch):
        """Update epoch for cold start functionality."""
        self.classifier.update_epoch(epoch)
    
    def get_diagnostics(self):
        """Get diagnostics from the quantized classifier."""
        return {
            'classifier': self.classifier.get_diagnostic_summary()
        }

def create_synthetic_binary_data(num_samples=10000, max_length=128):
    """Create synthetic binary classification data (sentiment analysis)."""
    print(f"Creating synthetic binary data: {num_samples} samples, max_length={max_length}")
    
    # Sample texts for binary classification (sentiment analysis)
    positive_texts = [
        "This is an excellent product with amazing quality and great service.",
        "I absolutely love this item and would highly recommend it to everyone.",
        "Outstanding performance and exceeded all my expectations completely.",
        "Fantastic experience with this purchase, very satisfied with everything.",
        "This is the best product I have ever bought, highly recommended.",
        "Amazing quality and excellent customer service, very happy with it.",
        "Perfect product that works exactly as described, very pleased.",
        "Great value for money and excellent quality, would buy again.",
        "Wonderful product with outstanding features and great support.",
        "Excellent purchase, everything is perfect and works great."
    ]
    
    negative_texts = [
        "This is a terrible product with poor quality and bad service.",
        "I absolutely hate this item and would never recommend it to anyone.",
        "Disappointing performance and failed to meet my expectations completely.",
        "Awful experience with this purchase, very unsatisfied with everything.",
        "This is the worst product I have ever bought, avoid at all costs.",
        "Poor quality and terrible customer service, very unhappy with it.",
        "Defective product that doesn't work as described, very disappointed.",
        "Waste of money and terrible quality, would never buy again.",
        "Horrible product with no useful features and terrible support.",
        "Terrible purchase, everything is broken and doesn't work at all."
    ]
    
    texts = []
    labels = []
    
    for i in range(num_samples):
        if i % 2 == 0:
            # Positive sample
            template = positive_texts[i % len(positive_texts)]
            texts.append(template + f" Sample {i} with additional positive details.")
            labels.append(1)  # Positive = 1
        else:
            # Negative sample
            template = negative_texts[i % len(negative_texts)]
            texts.append(template + f" Sample {i} with additional negative details.")
            labels.append(0)  # Negative = 0
    
    return texts, labels

def load_data(model_name='bert-base-uncased', data_fraction=0.8, batch_size=16):
    """Load and tokenize synthetic binary text data."""
    # Create synthetic data
    texts, labels = create_synthetic_binary_data()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize texts
    print("Tokenizing texts...")
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    labels = torch.tensor(labels, dtype=torch.float)  # Float for binary classification
    
    # Create subset
    num_samples = int(len(input_ids) * data_fraction)
    indices = torch.randperm(len(input_ids))[:num_samples]
    
    input_ids = input_ids[indices]
    attention_mask = attention_mask[indices]
    labels = labels[indices]
    
    # Split into train/test
    train_size = int(0.8 * len(input_ids))
    train_input = input_ids[:train_size]
    train_attention = attention_mask[:train_size]
    train_labels = labels[:train_size]
    
    test_input = input_ids[train_size:]
    test_attention = attention_mask[train_size:]
    test_labels = labels[train_size:]
    
    # Create datasets
    train_dataset = TensorDataset(train_input, train_attention, train_labels)
    test_dataset = TensorDataset(test_input, test_attention, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    """Train the BERT binary classifier with QAT on classification head only."""
    model = model.to(device)
    
    # Only train the classification head (and projection if exists)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Training {len(trainable_params)} parameter groups")
    
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
    
    print(f"\nTraining BERT binary classifier with QAT for {epochs} epochs...")
    print("Note: Only the classification head is being trained (BERT is frozen)")
    
    for epoch in range(epochs):
        # Update epoch for cold start
        model.update_epoch(epoch)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        start_time = time.time()
        
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits, probabilities = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # Convert probabilities to predictions (threshold = 0.5)
            predictions = (probabilities > 0.5).float()
            train_correct += predictions.eq(labels).sum().item()
            train_total += labels.size(0)
            
            if batch_idx % 5 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100. * train_correct / train_total:.2f}%')
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for input_ids, attention_mask, labels in test_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                logits, probabilities = model(input_ids, attention_mask)
                test_loss += criterion(logits, labels).item()
                predictions = (probabilities > 0.5).float()
                test_correct += predictions.eq(labels).sum().item()
                test_total += labels.size(0)
        
        epoch_time = time.time() - start_time
        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total
        
        print(f'Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_acc:.2f}%')
        
        # Print QAT diagnostics
        if hasattr(model, 'get_diagnostics'):
            diagnostics = model.get_diagnostics()
            print(f'Classifier Quant Enabled: {diagnostics["classifier"]["quantization_enabled"]}')
            if diagnostics["classifier"]["quantization_enabled"]:
                print(f'Classifier Quant Error: {diagnostics["classifier"]["quantization_error"]:.4f}')
        
        print('-' * 50)

def main():
    """Main function to run the BERT binary classifier example."""
    print("Pre-trained BERT Binary Classifier with Quantized Head Example")
    print("=" * 70)
    
    # Model hyperparameters
    model_name = 'bert-base-uncased'
    warmup_epochs = 2
    weight_clip_value = 2.0
    freeze_bert = True  # Only train the classification head
    
    # Load data
    print("Loading and tokenizing synthetic binary text data...")
    train_loader, test_loader = load_data(model_name=model_name, data_fraction=0.9, batch_size=128)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\nCreating BERT binary classifier with quantized head...")
    model = QuantizedBERTBinaryClassifier(
        model_name=model_name,
        warmup_epochs=warmup_epochs,
        weight_clip_value=weight_clip_value,
        freeze_bert=freeze_bert
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Train model
    train_model(model, train_loader, test_loader, epochs=15, lr=0.001)
    
    print("\nTraining completed!")
    print("This example demonstrates:")
    print("- Pre-trained BERT with frozen parameters")
    print("- Quantized binary classification head using constructor-based API")
    print("- QAT with cold start for stable training")
    print("- E8 lattice quantization for the MLP classifier")
    print("- Binary sentiment analysis (positive/negative)")
    print("- Much more efficient than training the entire transformer!")

if __name__ == "__main__":
    main()
