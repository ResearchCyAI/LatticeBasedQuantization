# Coset: Hierarchical Nested-Lattice Quantization for PyTorch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance PyTorch library implementing **Hierarchical Nested-Lattice Quantization (HNLQ)** for quantization-aware training (QAT) with transformer models.

## Features

- **E8 Lattice Support**: High-dimensional E8 lattice quantization with optimized algorithms
- **Transformer Integration**: Pre-trained BERT with quantized classification heads
- **QAT with Cold Start**: Gradual quantization activation for stable training
- **CUDA Acceleration**: GPU-optimized quantization operations
- **Constructor-Based API**: Easy-to-use layer constructors for different lattices
- **Flexible Scale Parameters**: Learnable or fixed scale parameters for quantization
- **Comprehensive Examples**: Binary and multi-class classification examples
- **Future Support**: D4 and other lattice types will be added

## Installation

### Quick Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/coset/coset.git
cd coset

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install transformers torchvision scikit-learn matplotlib
```

### Manual Installation

```bash
# Install core package
pip install -e .

# Install example dependencies
pip install transformers torchvision scikit-learn matplotlib

# For CUDA support (if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Binary Classification with Quantized BERT

```python
import torch
from transformers import AutoTokenizer, AutoModel
from coset.core.e8.layers import create_e8_hnlq_linear

# Create a quantized BERT classifier
class QuantizedBERTClassifier(torch.nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Quantized classification head
        self.classifier = create_e8_hnlq_linear(
            in_dim=768,  # BERT hidden size
            out_dim=num_classes,
            warmup_epochs=2,  # Cold start
            enable_diagnostics=True,
            weight_clip_value=2.0,
            theta_trainable=True,  # Learnable scale parameters
            theta_init_value=0.0   # Start at midpoint of bounds
        )
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        probabilities = self.sigmoid(logits)
        return logits, probabilities

# Usage
model = QuantizedBERTClassifier(num_classes=1)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
```

### 2. Multi-Class Classification

```python
# For multi-class classification (e.g., 5 classes)
model = QuantizedBERTClassifier(num_classes=5)
# Use CrossEntropyLoss instead of BCEWithLogitsLoss
criterion = torch.nn.CrossEntropyLoss()
```

### 3. Training with QAT

```python
import torch.optim as optim

# Only train the quantized head (BERT is frozen)
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()  # For binary classification

# Training loop
model.train()
for epoch in range(15):
    # Update epoch for QAT cold start
    model.classifier.update_epoch(epoch)
    
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        logits, probabilities = model(input_ids, attention_mask)
        
        loss = criterion(logits.squeeze(), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## Examples

### Run the Examples

```bash
# Binary classification example
python examples/bert_binary_classifier_example.py

# Multi-class classification example  
python examples/bert_multiclass_classifier_example.py

# QAT comparison with different configurations
python examples/qat_cold_start_comparison.py
```

### Example Results

**Binary Classification:**
- Dataset: 10,000 synthetic samples (90% train, 10% test)
- Final Accuracy: ~84% (vs 50% random)
- Training Time: ~4.2s per epoch
- Parameters: Only 770 trainable (quantized head) vs 109M total

**Multi-Class Classification:**
- Dataset: 10,000 synthetic samples, 5 classes
- Final Accuracy: ~91% (vs 20% random)
- Efficient training of only the classification head

## API Reference

### Constructor Functions

```python
from coset.core.e8.layers import create_e8_hnlq_linear

# Create E8 quantized linear layer
layer = create_e8_hnlq_linear(
    in_dim=768,                    # Input dimension
    out_dim=10,                    # Output dimension
    q=4,                          # Quantization parameter
    M=2,                          # Hierarchical levels
    warmup_epochs=2,              # Cold start epochs
    enable_diagnostics=True,      # Enable weight diagnostics
    weight_clip_value=2.0,        # Weight clipping threshold
    theta_trainable=True,         # Learnable scale parameters (default)
    theta_init_value=0.0,         # Initial theta value (default)
    device=None                   # Auto-detect CUDA
)
```

### Scale Parameter Options

```python
# Learnable scale parameters (default behavior)
layer = create_e8_hnlq_linear(
    in_dim=768, out_dim=10,
    theta_trainable=True,         # Scale parameters are learnable
    theta_init_value=0.0          # Start at midpoint of bounds
)

# Fixed scale parameters (deterministic)
layer = create_e8_hnlq_linear(
    in_dim=768, out_dim=10,
    theta_trainable=False,        # Scale parameters are fixed
    theta_init_value=0.0          # Fixed at midpoint: (beta_min + beta_max) / 2
)

# Custom fixed scale parameters
layer = create_e8_hnlq_linear(
    in_dim=768, out_dim=10,
    theta_trainable=False,        # Scale parameters are fixed
    theta_init_value=1.0          # Fixed at: beta_min + sigmoid(1.0) * (beta_max - beta_min)
)
```

**Benefits of Fixed Scale Parameters:**
- **Deterministic**: Consistent quantization behavior across runs
- **Reduced Parameters**: Fewer trainable parameters (theta_beta becomes a buffer)
- **Stable Training**: No gradient updates for scale parameters
- **Midpoint Strategy**: `theta_init_value=0.0` gives `(beta_min + beta_max) / 2`

### QAT Methods

```python
# Cold start control
layer.update_epoch(epoch)        # Update epoch for QAT
layer.enable_quantization()       # Enable quantization
layer.disable_quantization()      # Disable quantization

# Diagnostics
diagnostics = layer.get_diagnostic_summary()
quant_error = layer.get_quantization_error()
weight_stats = layer.get_weight_statistics()
```

### Configuration

```python
from coset.core.base import LatticeConfig

config = LatticeConfig(
    lattice_type="E8",            # Currently: E8 (D4 coming soon)
    q=4,                          # Quantization parameter
    M=2,                          # Hierarchical levels
    beta=1.0,                     # Scaling parameter
    alpha=1.0,                    # Overload scaling
    decoding="full",              # Decoding method
    check_overload=False,         # Overload checking
    disable_scaling=False,        # Disable scaling
    disable_overload_protection=True,  # Disable overload protection
    with_tie_dither=False,        # Tie-breaking dither
    with_dither=False,            # Randomized dither
    max_scaling_iterations=10     # Max scaling iterations
)
```

## Performance

### Quantization Performance
- **Training Speed**: ~4.2s per epoch (batch size 128)
- **Memory Efficiency**: Only train quantized head (770 params vs 109M)
- **Accuracy**: 84-91% on synthetic datasets
- **QAT Overhead**: Minimal impact on training speed

### Model Efficiency
- **Parameter Reduction**: 99.3% reduction in trainable parameters
- **Inference Speed**: Faster due to quantized operations
- **Memory Usage**: Reduced memory footprint
- **Scalability**: Tested up to 10K samples

## Roadmap

### Current Status ✅
- [x] E8 lattice implementation
- [x] QAT with cold start
- [x] BERT integration with quantized heads
- [x] Binary and multi-class classification examples
- [x] Comprehensive QAT comparison tools
- [x] Constructor-based API
- [x] Flexible scale parameters (learnable or fixed)

### Coming Soon 🚧
- [ ] **D4 Lattice Support**: 4D checkerboard lattice implementation
- [ ] **Z2 Lattice Support**: 2D integer lattice for baseline comparison
- [ ] **Additional Lattice Types**: More lattice options for different use cases
- [ ] **Enhanced Examples**: More transformer architectures (GPT, T5, etc.)
- [ ] **Performance Benchmarks**: Comprehensive speed and accuracy comparisons
- [ ] **Documentation**: Detailed API documentation and tutorials

### Future Features 🔮
- [ ] **Distributed Training**: Gradient compression for DDP
- [ ] **Advanced QAT**: More sophisticated quantization strategies
- [ ] **Model Compression**: Full transformer quantization
- [ ] **Hardware Optimization**: Specialized kernels for different hardware

## Development

```bash
# Run tests
pytest tests/

# Run examples
python examples/bert_binary_classifier_example.py
python examples/bert_multiclass_classifier_example.py
python examples/qat_cold_start_comparison.py

# Check code quality
black coset examples
ruff check coset examples
```

## Citation

If you use this library in your research, please cite:

```bibtex
@article{kaplan2025high,
  title={High-Rate Nested-Lattice Quantized Matrix Multiplication with Small Lookup Tables},
  author={Kaplan, Haim and Ordentlich, Or},
  journal={arXiv preprint arXiv:2505.13164},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Acknowledgments

- Based on the work of Kaplan & Ordentlich (2025)
- Inspired by the PyTorch ecosystem
- Built with the scientific Python community
- Special thanks to the Hugging Face transformers library