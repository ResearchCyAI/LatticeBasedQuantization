# LatticeBasedQuantization

Repository for quantization and QAT experiments.

## Structure

- **DataSet/** – CSV datasets and precomputed embeddings (`embeddings/`)
- `FP32EmbeddingsCreation.py` – Build embeddings from CSVs into `DataSet/embeddings/`
- `MLP_latticeFusedQAT` – E8 lattice QAT on MLP keeping BERT FP32
- `MLP_ScalarQAT.py` – Scalar quantization variant
- `BertMLP_latticeFused_end2endQAT.py` – BERT+MLP end-to-end QAT
- `BertMLP_latticeFused_end2endQAT_ACIQ.py` – ACIQ-based BERT+MLP end-to-end QAT

## Requirements

- Python 3.x
- PyTorch, transformers, pandas, numpy, scikit-learn
- `coset` (E8/lattice quantization)

## Data

1. Place Dataset csv file in `DataSet/`.
2. Generate embeddings: FP32EmbeddingsCreation.py for QAT on MLP keeping BERT FP32.
