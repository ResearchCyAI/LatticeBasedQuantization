#!/usr/bin/env python3
"""
Simple MLP example using Coset core APIs for quantization-aware training.
Uses the new coset/core modules instead of deprecated legacy APIs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Import coset core modules
try:
    from coset.core import HNLQLinear
    from coset.core.base import LatticeConfig
    from coset.core.e8 import E8Lattice, create_e8_hnlq_linear
    from coset.core.vq_qat_layer import HNLQLinearQAT
    print("✓ Successfully imported coset core modules")
except ImportError as e:
    print(f"✗ Failed to import coset core modules: {e}")
    print("Make sure you have installed coset with: pip install -e .")
    exit(1)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class QuantizedMLP(nn.Module):
    """Simple quantized MLP for MNIST classification using core APIs."""
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10, 
                 lattice_type="E8", q=4, M=2):
        super().__init__()
        
        # Create lattice configuration
        self.config = LatticeConfig(
            lattice_type=lattice_type,
            q=q,
            M=M,
            beta=1.0,
            alpha=1.0,
            disable_scaling=True,
            disable_overload_protection=True,
            with_tie_dither=True,
            with_dither=False,
            max_scaling_iterations=10
        )
        
        # For E8 lattice, input dimension must be divisible by 8
        # We'll pad the input if necessary
        self.input_padding = 0
        if lattice_type == "E8" and input_size % 8 != 0:
            self.input_padding = 8 - (input_size % 8)
            input_size += self.input_padding
        
        # Create quantized layers using core APIs
        if lattice_type == "E8":
            # Use E8-specific layer - pass device explicitly for GPU acceleration
            self.fc1 = create_e8_hnlq_linear(
                in_dim=input_size,
                out_dim=hidden_size,
                device=device,  # Explicitly pass device for GPU placement
                q=q,
                M=M,
                Delta0=1.5
            )
            self.fc2 = create_e8_hnlq_linear(
                in_dim=hidden_size,
                out_dim=hidden_size // 2,
                device=device,  # Explicitly pass device for GPU placement
                q=q,
                M=M,
                Delta0=1.5
            )
        else:
            # For other lattices, use generic HNLQLinear
            # Note: This would require implementing other lattice types
            raise NotImplementedError(f"Lattice type {lattice_type} not yet implemented in core APIs")
        
        # Last layer not quantized
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Activation function and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        
        # Pad input if necessary for E8
        if self.input_padding > 0:
            x = torch.cat([x, torch.zeros(x.size(0), self.input_padding, device=x.device)], dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def load_mnist_data(batch_size=64, data_fraction=0.1):
    """Load MNIST dataset with only a fraction of the data."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full datasets
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Calculate subset sizes
    train_size = int(len(train_dataset) * data_fraction)
    test_size = int(len(test_dataset) * data_fraction)
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, range(train_size))
    test_subset = torch.utils.data.Subset(test_dataset, range(test_size))
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=5, lr=0.001):
    """Train the quantized model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining quantized MLP for {epochs} epochs...")
    print(f"Model configuration: {model.config}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:  # Show progress every 10 batches
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*train_correct/train_total:.2f}%')
        
        train_time = time.time() - start_time
        
        # Testing
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total
        
        print(f'Epoch {epoch+1}/{epochs} completed in {train_time:.2f}s')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_acc:.2f}%')
        print('-' * 50)

def main():
    """Main function to run the MLP example."""
    print("Coset Core APIs Quantized MLP Example")
    print("=" * 50)
    
    # Load data (10% of the dataset)
    print("Loading MNIST dataset (10% subset)...")
    train_loader, test_loader = load_mnist_data(batch_size=64, data_fraction=0.1)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating quantized MLP model using E8 lattice...")
    model = QuantizedMLP(
        input_size=784,
        hidden_size=128,
        output_size=10,
        lattice_type="E8",
        q=4,
        M=2
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input padding for E8: {model.input_padding}")
    
    # Train model
    train_model(model, train_loader, test_loader, epochs=3, lr=0.001)
    
    print("\nTraining completed!")
    print("This example uses the new coset/core APIs:")
    print("- HNLQLinear for quantized layers")
    print("- E8Lattice for 8-dimensional lattice quantization")
    print("- LatticeConfig for configuration management")
    print("- create_e8_hnlq_linear for E8-specific layers")

if __name__ == "__main__":
    main()