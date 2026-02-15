# COSET Documentation

Documentation for the COSET (Coding-based Quantization for Efficient Transformers) library.

## 📁 Directory Structure

```
docs/
├── README.md                    # This file
└── test-summaries/              # Test documentation archive
    ├── README.md                # Test archive index
    ├── TESTING_SUMMARY.md       # Complete testing overview
    └── PYTEST_CONVERSION_SUMMARY.md  # Pytest migration details
```

## 📚 Documentation Sections

### Test Documentation

**Location:** `docs/test-summaries/`

Comprehensive documentation of all testing efforts including:
- Test results and statistics (58 tests, 100% pass rate)
- One-sided vLUT fixes and implementation details
- Pytest conversion and best practices
- Performance benchmarks and comparisons
- CI/CD integration guides

**Quick Links:**
- [Test Archive Index](test-summaries/README.md)
- [Testing Summary](test-summaries/TESTING_SUMMARY.md)
- [Pytest Conversion](test-summaries/PYTEST_CONVERSION_SUMMARY.md)

### Examples Documentation

**Location:** `examples/README.md`

Practical examples and use cases:
- One-sided vLUT vector search
- Semantic search with quantized embeddings
- k-NN retrieval
- Performance comparisons

**Link:** [Examples Documentation](../examples/README.md)

### Test Usage Documentation

**Location:** `tests/README.md`

How to run and write tests:
- Test execution commands
- Pytest markers and fixtures
- CI/CD integration
- Adding new tests

**Link:** [Test Usage Guide](../tests/README.md)

## 🚀 Quick Start

### Running Tests
```bash
cd /workspace/coset
source .venv/bin/activate
python -m pytest tests/ -v -m "not slow"
```

### Running Examples
```bash
cd /workspace/coset
PYTHONPATH=/workspace/coset:$PYTHONPATH python3 examples/one_sided_vlut_search.py
```

### Viewing Documentation
All documentation is in Markdown format and can be viewed:
- In any text editor
- On GitHub (with formatting)
- Using a Markdown viewer

## 📊 Key Statistics

### Test Coverage
- **Total Tests:** 58 (54 fast + 4 slow)
- **Pass Rate:** 100% ✅
- **Runtime:** ~55s (fast), ~90s (all)
- **Coverage Areas:** Quantized vectors, vLUT, caching, performance

### Performance Metrics
- **One-sided vLUT Caching:** 1370x+ speedup
- **Search Throughput:** 8000-9000 vectors/sec
- **Encoding:** >100K vectors/sec
- **Decoding:** >200K vectors/sec

## 🎯 Use Cases

1. **Vector Search** - Efficient search with quantized embeddings
2. **Semantic Search** - Document retrieval with vLUT
3. **k-NN Retrieval** - Fast nearest neighbor search
4. **Quantization** - Hierarchical nested-lattice quantization

## 🔧 Main Library Components

### Lattices
- `Z2Lattice` - 2D integer lattice (baseline)
- `D4Lattice` - 4D checkerboard lattice (recommended)
- `E8Lattice` - 8D optimal lattice (high precision)

### Quantization
- `encode()` - Hierarchical encoding
- `decode()` - Hierarchical decoding
- `quantize()` - Complete quantization (encode + decode)

### vLUT Operations
- `build_one_sided_vlut()` - Query-specific vLUT
- `build_two_sided_vlut()` - Dual quantization vLUT
- `vlut_mac_operation()` - MAC with vLUT

## 📖 Documentation Standards

### Adding Documentation

When adding new documentation:

1. **Choose the Right Location:**
   - Test docs → `docs/test-summaries/`
   - Examples → `examples/README.md`
   - API docs → Main `README.md`
   - Usage guides → `tests/README.md`

2. **Follow Markdown Standards:**
   - Use headers (`#`, `##`, `###`)
   - Include code blocks with syntax highlighting
   - Add links between related docs
   - Use emojis sparingly for visual organization

3. **Update Index Files:**
   - Update this `docs/README.md`
   - Update relevant section README
   - Add cross-references

4. **Include Examples:**
   - Show usage with code blocks
   - Provide expected output
   - Include edge cases

## 🔄 Document Lifecycle

### Regular Updates

- Test documentation updated after test changes
- Examples updated when new features added
- Performance metrics updated after optimization
- Version history maintained

### Archiving

Old documentation is moved to `docs/test-summaries/` or appropriate archive location with version tags.

## 📝 Version History

### v0.27 (Current)
- Created documentation archive structure
- Moved test summaries to `docs/test-summaries/`
- Added comprehensive test documentation
- Added examples documentation

## 🤝 Contributing to Documentation

1. Write clear, concise documentation
2. Include code examples
3. Add cross-references
4. Update index files
5. Test code examples
6. Commit with descriptive messages

## 📧 Support

- **Issues:** GitHub Issues
- **Tests:** See `docs/test-summaries/`
- **Examples:** See `examples/README.md`
- **API:** See main `README.md`

---

**Maintained by:** COSET Development Team  
**Last Updated:** 2025-09-30  
**Version:** v0.27
