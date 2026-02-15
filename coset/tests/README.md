# COSET Test Suite

Comprehensive test suite for the COSET library, covering both the modern core API and legacy compatibility.

## Test Structure

```
tests/
├── __init__.py
├── README.md (this file)
└── legacy/              # Legacy API tests (coset.legacy.*)
    ├── __init__.py
    ├── test_lattices.py              # Z2, D4, E8 lattice tests
    ├── test_nn.py                    # QLinear tests
    ├── test_quant.py                 # Quantization functions
    ├── test_e8_gpu.py                # E8 GPU operations
    ├── test_one_sided_vlut_pytest.py # One-sided vLUT operations
    └── test_quantized_vectors_pytest.py # Quantized vector operations
```

## Test Categories

### Legacy API Tests (`tests/legacy/`)

Tests for backward compatibility with the deprecated `coset.legacy` modules:
- **Lattice operations** - Z2, D4, E8 nearest-neighbor quantization
- **Neural network layers** - QLinear with quantization
- **Quantization functions** - encode, decode, quantize
- **E8 GPU operations** - GPU-accelerated E8 quantization
- **vLUT operations** - One-sided value lookup tables
- **Simulated quantization** - Vector simulator tests

These tests ensure the legacy API continues to work for backward compatibility.

### Core API Tests (Coming Soon)

Tests for the modern `coset.core` API will be added here:
- **HNLQLinear** - Hierarchical nested lattice quantization layers
- **STE (Straight-Through Estimator)** - Gradient flow through quantization
- **E8 quantization** - Modern E8 lattice implementation
- **Dependency injection** - Generic quantization with function injection

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Only Legacy Tests
```bash
pytest tests/legacy/ -v
```

### Run Specific Test File
```bash
pytest tests/legacy/test_lattices.py -v
pytest tests/legacy/test_nn.py -v
```

### Run Specific Test Class
```bash
pytest tests/legacy/test_lattices.py::TestE8Lattice -v
```

### Run Specific Test
```bash
pytest tests/legacy/test_lattices.py::TestE8Lattice::test_quantization -v
```

### Run Tests by Marker
```bash
# Skip slow tests
pytest tests/ -m "not slow" -v

# Run only slow (performance) tests
pytest tests/ -m "slow" -v

# Run only CUDA tests (if CUDA available)
pytest tests/ -m "cuda" -v
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=coset --cov-report=html --cov-report=term
```

### Run Tests with Different Verbosity
```bash
# Quiet mode
pytest tests/ -q

# Verbose mode
pytest tests/ -v

# Very verbose (show test names as they run)
pytest tests/ -vv
```

### Run Tests in Parallel
```bash
# Auto-detect number of CPUs
pytest tests/ -n auto

# Specific number of workers
pytest tests/ -n 4
```

## Test Configuration

Configuration is in `pytest.ini` at the project root:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    cuda: marks tests that require CUDA
    integration: marks tests as integration tests
```

## Test Markers

- `@pytest.mark.slow` - Performance benchmarks and long-running tests
- `@pytest.mark.cuda` - Tests requiring CUDA/GPU
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.unit` - Unit tests

## Legacy Test Coverage

### Lattices (`test_lattices.py`)
- ✅ Z2, D4, E8 initialization
- ✅ Nearest-neighbor quantization (Q method)
- ✅ Encode/decode coordinates
- ✅ Round-trip consistency
- ✅ Batch operations

### Neural Networks (`test_nn.py`)
- ✅ QLinear initialization
- ✅ Forward pass with/without quantization
- ✅ Gradient flow (autograd)
- ✅ Different lattice types
- ✅ Quantization scheduling

### Quantization (`test_quant.py`)
- ✅ QuantizationConfig validation
- ✅ Encode/decode operations
- ✅ Quantize wrapper
- ✅ MAC modulo-q operations
- ✅ Accumulate modulo-q
- ✅ Batch operations

### E8 GPU (`test_e8_gpu.py`)
- ✅ CPU vs GPU correctness
- ✅ Device handling
- ✅ Batch operations
- ✅ Edge cases and boundary conditions
- ✅ Performance benchmarks

### vLUT (`test_one_sided_vlut_pytest.py`)
- ✅ vLUT construction
- ✅ Dot product accuracy
- ✅ Batch processing
- ✅ Caching mechanism
- ✅ vs traditional decode-then-compute
- ✅ Performance benchmarks

### Quantized Vectors (`test_quantized_vectors_pytest.py`)
- ✅ Zero reconstruction error
- ✅ Encoding/decoding shapes
- ✅ Quantize wrapper consistency
- ✅ Various parameter combinations
- ✅ Simulator validation
- ✅ Performance benchmarks

## Continuous Integration

For CI/CD pipelines:

```bash
# Fast tests only (for PR checks)
pytest tests/ -m "not slow" --tb=short

# Full test suite (for main branch)
pytest tests/ --tb=short --cov=coset --cov-report=xml

# Legacy tests only (ensure backward compatibility)
pytest tests/legacy/ -v
```

## Debugging Failed Tests

### View full traceback
```bash
pytest tests/ --tb=long
```

### Run with pdb debugger
```bash
pytest tests/ --pdb
```

### Run last failed tests only
```bash
pytest tests/ --lf
```

### Run failed tests first
```bash
pytest tests/ --ff
```

### Show print statements
```bash
pytest tests/ -s
```

### Stop on first failure
```bash
pytest tests/ -x
```

## Requirements

### Core Requirements
- Python 3.8+
- pytest >= 7.0.0
- torch >= 2.0.0
- numpy >= 1.21.0
- coset library

### Optional Requirements
- pytest-cov >= 4.0.0 (for coverage reports)
- pytest-xdist >= 3.0.0 (for parallel execution)
- CUDA-capable GPU (for CUDA tests)

## Adding New Tests

### For Core API (Modern)
1. Create test file: `tests/test_<feature>.py`
2. Use modern API: `from coset import ...` or `from coset.core import ...`
3. Follow class-based organization: `class Test<Feature>:`
4. Use fixtures for setup: `@pytest.fixture`
5. Use parametrize for multiple inputs: `@pytest.mark.parametrize`

Example:
```python
import pytest
import torch
from coset import create_e8_hnlq_linear

class TestHNLQLinear:
    def test_forward_pass(self):
        layer = create_e8_hnlq_linear(64, 32)
        x = torch.randn(4, 64)
        output = layer(x)
        assert output.shape == (4, 32)
    
    def test_gradient_flow(self):
        layer = create_e8_hnlq_linear(64, 32)
        x = torch.randn(4, 64, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
```

### For Legacy API Tests
1. Create test file in `tests/legacy/test_<feature>.py`
2. Use legacy API: `from coset.legacy import ...`
3. Mark as legacy if needed: `@pytest.mark.legacy`

## Known Issues

- Legacy tests may show deprecation warnings - this is expected
- Some CUDA tests require specific GPU hardware
- Performance tests marked as `slow` can take significant time

## License

Same as COSET project license (MIT).
