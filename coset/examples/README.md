# LSQ Scalar Quantization Examples

This directory contains examples demonstrating LSQ (Learned Step Size Quantization) scalar quantization with the Coset library.

## Examples

### 1. Simple MNIST Example (`simple_mnist_lsq.py`)

A minimal example showing LSQ scalar quantization on MNIST classification.

**Features:**
- Compares standard vs LSQ quantized models
- 4-bit quantization (q=4, M=2)
- Per-row scaling parameters
- Quick training (2 epochs)

**Usage:**
```bash
python examples/simple_mnist_lsq.py
```

**Expected Output:**
- Training progress for both models
- Test accuracy comparison
- Quantization analysis showing learnable scaling factors

### 2. Comprehensive MNIST Example (`mnist_lsq_scalar.py`)

A detailed example with multiple quantization configurations and analysis.

**Features:**
- Multiple quantization bit-widths (4-bit, 8-bit)
- Different tiling strategies (row, block)
- Comprehensive performance analysis
- Visualization plots
- Quantization statistics

**Usage:**
```bash
python examples/mnist_lsq_scalar.py
```

**Expected Output:**
- Training curves for multiple models
- Performance comparison charts
- Detailed quantization analysis
- Model size and compression ratio analysis

## Key Concepts Demonstrated

### LSQ Scalar Quantization

LSQ scalar quantization uses learnable scaling parameters to adapt quantization to the data:

```python
# Create LSQ quantized linear layer
layer = create_lsq_scalar_linear(
    in_dim=784,      # Input features
    out_dim=128,     # Output features  
    q=4,             # Base quantization levels
    M=2,             # Hierarchical levels
    tiling='row'     # Per-row scaling
)

# Effective bits = floor(M * log2(q)) = floor(2 * log2(4)) = 4 bits
```

### Quantization Formula

The quantization follows the formula:
```
quantized = clip(round(x/s), -q_max, q_max) * s
```

Where:
- `s` is the learned scaling parameter (one per row or tile)
- `q_max = 2^(bits-1) - 1` for symmetric quantization
- `bits = floor(M * log2(q))`

### Scaling Parameters

- **Per-row scaling**: One scaling factor per output neuron
- **Per-tile scaling**: One scaling factor per tile (when using block tiling)
- **Learnable**: Scaling factors are learned during training via backpropagation

## Requirements

- PyTorch
- Torchvision
- NumPy
- Matplotlib (for comprehensive example)

## Installation

Make sure you have the Coset library installed:

```bash
pip install -e .
```

## Understanding the Results

### Accuracy Comparison
- **Standard Model**: Baseline accuracy without quantization
- **LSQ Quantized**: Accuracy with learnable scalar quantization
- **Accuracy Drop**: Difference between standard and quantized models

### Quantization Analysis
- **Scaling Factors**: Number of learnable scaling parameters
- **Effective Bits**: Actual bit-width used for quantization
- **Compression Ratio**: Theoretical compression vs 32-bit weights

### Performance Metrics
- **Training Time**: Time to train each model
- **Model Size**: Memory footprint comparison
- **Quantization Overhead**: Additional parameters for scaling

## Customization

You can modify the examples to:

1. **Change quantization bit-width**:
   ```python
   # 8-bit quantization
   layer = create_lsq_scalar_linear(784, 128, q=2, M=8)
   ```

2. **Use block tiling**:
   ```python
   # Per-tile scaling instead of per-row
   layer = create_lsq_scalar_linear(784, 128, q=4, M=2, tiling='block', block_size=8)
   ```

3. **Different network architectures**:
   ```python
   # Add more layers, different sizes, etc.
   self.fc1 = create_lsq_scalar_linear(784, 256, q=4, M=2)
   self.fc2 = create_lsq_scalar_linear(256, 128, q=4, M=2)
   self.fc3 = create_lsq_scalar_linear(128, 10, q=4, M=2)
   ```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure Coset is properly installed
2. **CUDA Error**: The examples work on both CPU and GPU
3. **Memory Error**: Reduce batch size or model size
4. **Slow Training**: LSQ quantization adds some overhead during training

### Performance Tips

1. **Use GPU**: Training is faster on GPU
2. **Batch Size**: Larger batches can improve training efficiency
3. **Learning Rate**: May need adjustment for quantized models
4. **Warmup**: Consider using warmup epochs for better quantization adaptation

## Next Steps

After running the examples, you can:

1. **Experiment with different bit-widths** (2-bit, 6-bit, etc.)
2. **Try different tiling strategies** (row vs block)
3. **Apply to other datasets** (CIFAR-10, ImageNet, etc.)
4. **Use in your own models** by replacing `nn.Linear` with `create_lsq_scalar_linear`
5. **Combine with other techniques** (pruning, knowledge distillation, etc.)
