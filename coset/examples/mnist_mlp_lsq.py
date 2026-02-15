#!/usr/bin/env python3
"""
MNIST Example with LSQ Scalar Quantization

This example demonstrates how to use LSQ (Learned Step Size Quantization) scalar quantization
for training a neural network on MNIST. The example shows:

1. Creating LSQ scalar quantized linear layers
2. Training with quantization-aware training (QAT)
3. Comparing quantized vs non-quantized performance
4. Analyzing quantization statistics

Usage:
    python examples/mnist_lsq_scalar.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import time
import os

# Import LSQ scalar quantization
from coset.core.scalar.layers import create_lsq_scalar_linear
from coset.core.scalar.codecs import get_lsq_scalar_bounds, get_lsq_scalar_effective_bits


class MNISTNet(nn.Module):
    """Standard MNIST network without quantization."""
    
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class MNISTNetLSQ(nn.Module):
    """MNIST network with LSQ scalar quantization."""
    
    def __init__(self, hidden_size: int = 128, q: int = 4, M: int = 2, tiling: str = 'row'):
        super().__init__()
        # Use LSQ scalar quantized linear layers
        self.fc1 = create_lsq_scalar_linear(28 * 28, hidden_size, q=q, M=M, tiling=tiling)
        self.fc2 = create_lsq_scalar_linear(hidden_size, hidden_size, q=q, M=M, tiling=tiling)
        self.fc3 = create_lsq_scalar_linear(hidden_size, 10, q=q, M=M, tiling=tiling)
        self.dropout = nn.Dropout(0.2)
        
        # Store quantization parameters for analysis
        self.q = q
        self.M = M
        self.tiling = tiling
        self.effective_bits = get_lsq_scalar_effective_bits(q, M)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def get_quantization_stats(self) -> Dict:
        """Get comprehensive quantization statistics."""
        stats = {
            'effective_bits': self.effective_bits,
            'q': self.q,
            'M': self.M,
            'tiling': self.tiling,
            'layers': {}
        }
        
        for name, module in self.named_modules():
            if hasattr(module, 'get_scaling_factors'):
                scaling_factors = module.get_scaling_factors()
                stats['layers'][name] = {
                    'scaling_factors_count': len(scaling_factors),
                    'scaling_factors_mean': scaling_factors.mean().item(),
                    'scaling_factors_std': scaling_factors.std().item(),
                    'scaling_factors_min': scaling_factors.min().item(),
                    'scaling_factors_max': scaling_factors.max().item(),
                }
                
                if hasattr(module, 'get_weight_statistics'):
                    weight_stats = module.get_weight_statistics()
                    stats['layers'][name].update(weight_stats)
        
        return stats


def load_mnist_data(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
                device: torch.device, epoch: int) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 200 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                  f'Accuracy: {100. * correct / total:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def test(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Test the model."""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy


def compare_models():
    """Compare standard and LSQ quantized models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_mnist_data(batch_size=64)
    
    # Create models
    models = {
        'Standard': MNISTNet(hidden_size=128),
        'LSQ-4bit': MNISTNetLSQ(hidden_size=128, q=4, M=2, tiling='row'),
        'LSQ-8bit': MNISTNetLSQ(hidden_size=128, q=2, M=8, tiling='row'),
        'LSQ-4bit-block': MNISTNetLSQ(hidden_size=128, q=4, M=2, tiling='block'),
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name} Model")
        print(f"{'='*60}")
        
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        start_time = time.time()
        train_accuracies = []
        test_accuracies = []
        
        for epoch in range(5):  # Reduced epochs for demo
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
            test_acc = test(model, test_loader, device)
            
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        training_time = time.time() - start_time
        
        # Get quantization stats for LSQ models
        quant_stats = None
        if hasattr(model, 'get_quantization_stats'):
            quant_stats = model.get_quantization_stats()
            print(f"\nQuantization Statistics for {model_name}:")
            print(f"Effective bits: {quant_stats['effective_bits']}")
            print(f"Q: {quant_stats['q']}, M: {quant_stats['M']}, Tiling: {quant_stats['tiling']}")
            
            for layer_name, layer_stats in quant_stats['layers'].items():
                if 'scaling_factors_count' in layer_stats:
                    print(f"  {layer_name}: {layer_stats['scaling_factors_count']} scaling factors, "
                          f"mean: {layer_stats['scaling_factors_mean']:.4f}")
        
        results[model_name] = {
            'final_test_acc': test_acc,
            'training_time': training_time,
            'quant_stats': quant_stats,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }
    
    return results


def plot_results(results: Dict):
    """Plot training results."""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training curves
    plt.subplot(1, 3, 1)
    for model_name, result in results.items():
        plt.plot(result['train_accuracies'], label=f'{model_name} (train)', linestyle='-')
        plt.plot(result['test_accuracies'], label=f'{model_name} (test)', linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Final test accuracy comparison
    plt.subplot(1, 3, 2)
    model_names = list(results.keys())
    final_accs = [results[name]['final_test_acc'] for name in model_names]
    
    bars = plt.bar(model_names, final_accs)
    plt.ylabel('Final Test Accuracy (%)')
    plt.title('Final Test Accuracy Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.grid(True, axis='y')
    
    # Plot 3: Training time comparison
    plt.subplot(1, 3, 3)
    training_times = [results[name]['training_time'] for name in model_names]
    
    bars = plt.bar(model_names, training_times)
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, time in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{time:.1f}s', ha='center', va='bottom')
    
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('mnist_lsq_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_quantization(results: Dict):
    """Analyze quantization effects."""
    print(f"\n{'='*60}")
    print("Quantization Analysis")
    print(f"{'='*60}")
    
    # Compare model sizes
    print("\nModel Size Analysis:")
    for model_name, result in results.items():
        if result['quant_stats']:
            stats = result['quant_stats']
            print(f"{model_name}:")
            print(f"  Effective bits: {stats['effective_bits']}")
            print(f"  Quantization parameters: q={stats['q']}, M={stats['M']}, tiling={stats['tiling']}")
            
            total_scaling_factors = sum(
                layer_stats['scaling_factors_count'] 
                for layer_stats in stats['layers'].values() 
                if 'scaling_factors_count' in layer_stats
            )
            print(f"  Total scaling factors: {total_scaling_factors}")
            
            # Calculate theoretical compression ratio
            # Assuming 32-bit weights vs quantized weights
            compression_ratio = 32 / stats['effective_bits']
            print(f"  Theoretical compression ratio: {compression_ratio:.1f}x")
    
    # Performance comparison
    print(f"\nPerformance Comparison:")
    standard_acc = results['Standard']['final_test_acc']
    for model_name, result in results.items():
        if model_name != 'Standard':
            acc = result['final_test_acc']
            acc_drop = standard_acc - acc
            print(f"{model_name}: {acc:.2f}% (drop: {acc_drop:.2f}%)")


def main():
    """Main function."""
    print("MNIST Example with LSQ Scalar Quantization")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run comparison
    results = compare_models()
    
    # Analyze results
    analyze_quantization(results)
    
    # Plot results
    try:
        plot_results(results)
        print(f"\nResults plot saved as 'mnist_lsq_comparison.png'")
    except ImportError:
        print(f"\nMatplotlib not available, skipping plots")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print("LSQ scalar quantization successfully demonstrated on MNIST!")
    print("\nKey findings:")
    print("- LSQ quantized models achieve competitive accuracy")
    print("- Per-row scaling provides learnable quantization parameters")
    print("- Different bit-widths (4-bit, 8-bit) show different accuracy/size tradeoffs")
    print("- Block tiling provides more granular scaling control")


if __name__ == "__main__":
    main()
