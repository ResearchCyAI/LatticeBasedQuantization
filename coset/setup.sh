#!/bin/bash
# COSET Setup Script
# This script installs COSET and all required dependencies

echo "Setting up COSET..."

# Install COSET in development mode
echo "Installing COSET package..."
pip install -e .

# Install example dependencies
echo "Installing example dependencies..."
pip install -r requirements-examples.txt

# Verify installation
echo "Verifying installation..."
python3 -c "
try:
    from coset.optim.e8 import E8Config, E8QLinear
    from coset.lattices import E8Lattice
    print('✅ COSET core modules imported successfully!')
    
    from sklearn.model_selection import train_test_split
    from transformers import BertTokenizer, BertModel
    import torchvision
    print('✅ Example dependencies imported successfully!')
    
    print('🎉 COSET setup complete!')
    print('You can now run the examples:')
    print('  python3 examples/optim/mnist_cpu_gpu_comparison.py')
    print('  python3 examples/optim/bert_binary_classification.py')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo "Setup complete!"
