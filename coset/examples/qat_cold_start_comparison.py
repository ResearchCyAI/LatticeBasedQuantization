#!/usr/bin/env python3
"""
QAT Cold Start Comparison

Compare different cold start strategies for HNLQ quantization:
1. No cold start (QAT from epoch 0)
2. Short cold start (2 epochs warmup)
3. Long cold start (5 epochs warmup)

Includes weight diagnostics and quantization analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np

# Import coset core modules
try:
    from coset.core import HNLQLinear
    from coset.core.base import LatticeConfig
    from coset.core.e8 import E8Lattice, create_e8_hnlq_linear
    from coset.core.layers import HNLQLinearQAT
    print("✓ Successfully imported coset core modules")
except ImportError as e:
    print(f"✗ Failed to import coset core modules: {e}")
    exit(1)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class BaselineMLP(nn.Module):
    """Baseline MLP without quantization."""
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def update_epoch(self, epoch):
        """Dummy method for compatibility with QAT models."""
        pass
    
    def get_diagnostics(self):
        """Dummy method for compatibility with QAT models."""
        return {
            'fc1': {'error': 'No diagnostics for baseline model'},
            'fc2': {'error': 'No diagnostics for baseline model'}
        }

class QATMLP(nn.Module):
    """MLP with QAT cold start support."""
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10, 
                 warmup_epochs=0, enable_diagnostics=False, q=4, weight_clip_value=2.0):
        super().__init__()
        
        self.config = LatticeConfig(
            lattice_type="E8",
            q=q,
            M=2,
            beta=1.0,
            alpha=1.0,
            disable_scaling=True,
            disable_overload_protection=True,
            with_tie_dither=True,
            with_dither=False,
            max_scaling_iterations=10
        )
        
        # For E8 lattice, input dimension must be divisible by 8
        self.input_padding = 0
        if input_size % 8 != 0:
            self.input_padding = 8 - (input_size % 8)
            input_size += self.input_padding
        
        # Create QAT layers with cold start
        self.fc1 = HNLQLinearQAT(
            in_features=input_size,
            out_features=hidden_size,
            G=torch.eye(8, device=device),  # Placeholder, will be set by create_e8_hnlq_linear
            Ginv=torch.eye(8, device=device),
            quantize_fn=lambda x, q_val: x,  # Placeholder
            q=q,
            warmup_epochs=warmup_epochs,
            enable_diagnostics=enable_diagnostics,
            weight_clip_value=weight_clip_value
        )
        self.fc2 = HNLQLinearQAT(
            in_features=hidden_size,
            out_features=hidden_size // 2,
            G=torch.eye(8, device=device),
            Ginv=torch.eye(8, device=device),
            quantize_fn=lambda x, q_val: x,
            q=q,
            warmup_epochs=warmup_epochs,
            enable_diagnostics=enable_diagnostics,
            weight_clip_value=weight_clip_value
        )
        
        # Initialize with proper E8 configuration
        self._initialize_e8_layers()
        
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def _initialize_e8_layers(self):
        """Initialize layers with proper E8 configuration."""
        # Create E8 lattice for proper initialization
        lattice = E8Lattice(device=device)
        G, Ginv = lattice.get_generators()
        
        # Update the layers with proper E8 configuration
        self.fc1.G.data = G
        self.fc1.Ginv.data = Ginv
        self.fc1.quantize_fn = lambda x, q: x  # Will be set properly
        
        self.fc2.G.data = G
        self.fc2.Ginv.data = Ginv
        self.fc2.quantize_fn = lambda x, q: x
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        if self.input_padding > 0:
            x = torch.cat([x, torch.zeros(x.size(0), self.input_padding, device=x.device)], dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def update_epoch(self, epoch):
        """Update epoch for all QAT layers."""
        self.fc1.update_epoch(epoch)
        self.fc2.update_epoch(epoch)
    
    def get_diagnostics(self):
        """Get diagnostics from all QAT layers."""
        return {
            'fc1': self.fc1.get_diagnostic_summary(),
            'fc2': self.fc2.get_diagnostic_summary()
        }

def load_mnist_data(batch_size=128, data_fraction=0.8):
    """Load MNIST dataset with specified fraction."""
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
    
    train_size = int(len(train_dataset) * data_fraction)
    test_size = int(len(test_dataset) * data_fraction)
    
    train_subset = torch.utils.data.Subset(train_dataset, range(train_size))
    test_subset = torch.utils.data.Subset(test_dataset, range(test_size))
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_qat_model(model, train_loader, test_loader, epochs=15, lr=1e-3, model_name="QAT Model"):
    """Train a QAT model with cold start support."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining {model_name} for {epochs} epochs...")
    # Print warmup info (only for QAT models)
    if hasattr(model.fc1, 'warmup_epochs'):
        print(f"Warmup epochs: {model.fc1.warmup_epochs}")
    else:
        print("Warmup epochs: N/A (Baseline model)")
    
    results = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': [],
        'quantization_errors': [],
        'epoch_times': []
    }
    
    for epoch in range(epochs):
        # Update epoch for cold start
        model.update_epoch(epoch)
        
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
            
            if batch_idx % 50 == 0:
                print(f'  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*train_correct/train_total:.2f}%')
        
        epoch_time = time.time() - start_time
        results['epoch_times'].append(epoch_time)
        
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
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        results['train_loss'].append(train_loss / len(train_loader))
        results['test_loss'].append(test_loss / len(test_loader))
        
        # Get quantization diagnostics
        if hasattr(model.fc1, 'enable_diagnostics') and model.fc1.enable_diagnostics:
            diagnostics = model.get_diagnostics()
            quant_error = (diagnostics['fc1']['quantization_error'] + 
                          diagnostics['fc2']['quantization_error']) / 2
            results['quantization_errors'].append(quant_error)
            
            print(f'  Quantization Error: {quant_error:.4f}')
            print(f'  FC1 Quant Enabled: {diagnostics["fc1"]["quantization_enabled"]}')
            print(f'  FC2 Quant Enabled: {diagnostics["fc2"]["quantization_enabled"]}')
        
        print(f'  Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s')
        print(f'  Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        print('-' * 50)
    
    return results

def run_cold_start_comparison():
    """Run comprehensive cold start comparison for q=4 with extended training."""
    print("=" * 80)
    print("QAT COLD START COMPARISON - Q=4 WITH EXTENDED TRAINING")
    print("=" * 80)
    
    # Load data with larger dataset
    train_loader, test_loader = load_mnist_data(batch_size=128, data_fraction=0.8)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Test configurations: multiple warmup strategies for q=4
    configurations = []
    
    # Test multiple warmup strategies
    cold_start_strategies = [
        {"warmup_epochs": 0, "name": "No Cold Start"},
        {"warmup_epochs": 2, "name": "Short Cold Start (2 epochs)"},
        {"warmup_epochs": 5, "name": "Medium Cold Start (5 epochs)"},
        {"warmup_epochs": 10, "name": "Long Cold Start (10 epochs)"},
    ]
    
    # Focus on q=4 only
    q = 4
    for strategy in cold_start_strategies:
        config = strategy.copy()
        config["q"] = q
        config["name"] = f"{strategy['name']} (q={q})"
        configurations.append(config)
    
    # Add baseline (no quantization)
    configurations.append({
        "warmup_epochs": 0, 
        "name": "Baseline (No Quantization)", 
        "q": None,
        "is_baseline": True
    })
    
    results = {}
    
    for config in configurations:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")
        
        # Create model
        if config.get('is_baseline', False):
            model = BaselineMLP(
                input_size=784,
                hidden_size=128,
                output_size=10
            )
        else:
            model = QATMLP(
                input_size=784,
                hidden_size=128,
                output_size=10,
                warmup_epochs=config['warmup_epochs'],
                enable_diagnostics=True,
                q=config['q'],
                weight_clip_value=1.5  # Add weight clipping
            )
        
        # Train model
        model_results = train_qat_model(
            model, train_loader, test_loader, 
            epochs=15,  # Extended training to see cold start effect
            model_name=config['name']
        )
        
        results[config['name']] = model_results
        
        # Print final results
        final_test_acc = model_results['test_acc'][-1]
        final_train_acc = model_results['train_acc'][-1]
        avg_epoch_time = np.mean(model_results['epoch_times'])
        
        print(f"\n📊 {config['name']} Results:")
        print(f"  Final Test Accuracy: {final_test_acc:.2f}%")
        print(f"  Final Train Accuracy: {final_train_acc:.2f}%")
        print(f"  Average Epoch Time: {avg_epoch_time:.2f}s")
        
        if model_results['quantization_errors']:
            final_quant_error = model_results['quantization_errors'][-1]
            print(f"  Final Quantization Error: {final_quant_error:.4f}")
    
    # Print comparison summary
    print(f"\n{'='*100}")
    print("COMPARISON SUMMARY - QUANTIZATION LEVELS (q=4, 16, 32)")
    print(f"{'='*100}")
    print(f"{'Configuration':<35} {'Test Acc':<10} {'Train Acc':<10} {'Avg Time':<10} {'Quant Error':<12}")
    print(f"{'-'*100}")
    
    # Group results by q value for better analysis
    q_groups = {}
    for name, result in results.items():
        # Extract q value from name
        if 'q=4' in name:
            q_val = 4
        elif 'q=16' in name:
            q_val = 16
        elif 'q=32' in name:
            q_val = 32
        else:
            q_val = 4  # default
        
        if q_val not in q_groups:
            q_groups[q_val] = []
        q_groups[q_val].append((name, result))
    
    # Print results grouped by q value
    for q_val in sorted(q_groups.keys()):
        print(f"\n--- q={q_val} Results ---")
        for name, result in q_groups[q_val]:
            test_acc = result['test_acc'][-1]
            train_acc = result['train_acc'][-1]
            avg_time = np.mean(result['epoch_times'])
            quant_error = result['quantization_errors'][-1] if result['quantization_errors'] else 0.0
            
            print(f"{name:<35} {test_acc:<10.2f} {train_acc:<10.2f} {avg_time:<10.2f} {quant_error:<12.4f}")
    
    # Print best results for each q value
    print(f"\n{'='*100}")
    print("BEST RESULTS BY QUANTIZATION LEVEL")
    print(f"{'='*100}")
    print(f"{'Q Value':<10} {'Best Strategy':<25} {'Test Acc':<10} {'Train Acc':<10}")
    print(f"{'-'*100}")
    
    for q_val in sorted(q_groups.keys()):
        best_result = None
        best_acc = 0
        best_name = ""
        
        for name, result in q_groups[q_val]:
            test_acc = result['test_acc'][-1]
            if test_acc > best_acc:
                best_acc = test_acc
                best_result = result
                best_name = name
        
        train_acc = best_result['train_acc'][-1]
        strategy = best_name.split('(q=')[0].strip()
        print(f"{q_val:<10} {strategy:<25} {best_acc:<10.2f} {train_acc:<10.2f}")
    
    return results

if __name__ == "__main__":
    results = run_cold_start_comparison()
