# Wisent Guard Performance Analysis Demo

This directory contains local debugging and benchmarking scripts for Wisent Guard performance analysis.

## Files

- `benchmarking_demo.py` - Comprehensive debugging script with systematic tests
- `performance_benchmark.py` - Performance benchmarking script for measuring overhead
- `requirements.txt` - Python dependencies for local development
- `README.md` - This documentation

## Purpose

These scripts help debug and resolve the activation extraction issues encountered in the Jupyter notebook, specifically:

- **"stack expects a non-empty TensorList" error**
- **"Preparing to train logistic classifier with 0 samples" error**
- **ImportError with ActivationGuard**

## Configuration

The demo uses:
- **Model**: `meta-llama/Llama-3.2-1B` (smaller for faster local debugging)
- **Device**: `mps` (Apple Silicon) - automatically falls back to CPU if not available
- **Layer**: 8 (middle layer for 1B model)
- **Dtype**: `torch.float32` (works better with MPS)

## Usage

### 1. Install Dependencies

```bash
cd benchmarking
pip install -r requirements.txt
```

### 2. Run Debugging Demo

```bash
python benchmarking_demo.py
```

This will run 8 systematic tests:
1. **Basic Setup** - Check PyTorch, MPS, tensor operations
2. **Model Loading** - Load Llama-3.2-1B model and tokenizer
3. **Basic Generation** - Test standard model inference
4. **Activation Extraction** - Test manual activation extraction
5. **Wisent Guard Imports** - Test all required imports
6. **Wisent Model Wrapper** - Test Model wrapper functionality
7. **Training Data Preparation** - Test ContrastivePairSet creation
8. **Classifier Training** - Test classifier training with debugging

### 3. Run Performance Benchmark

```bash
python performance_benchmark.py
```

This measures:
- **Baseline latency** (without Wisent Guard)
- **Guard latency** (with activation extraction)
- **Overhead calculation** (ms and percentage)
- **Cost multiplier** for client presentations

## Expected Output

### Debugging Demo
The debugging script will show detailed progress for each test:
```
=== TEST 1: BASIC SETUP ===
[Step 1] Checking PyTorch and MPS availability
✅ MPS is available and ready
[Step 2] Testing basic tensor operations
✅ Basic tensor operations work on mps
```

### Performance Benchmark
The benchmark script will provide client-ready metrics:
```
=== FINAL SUMMARY FOR CLIENT ===
🎯 Performance Impact: 15.2% latency increase
💰 Cost Multiplier: 1.15x baseline compute cost
⏱️  Additional Latency: 45.3ms per request
✅ Low impact - suitable for production deployment
```

## Debugging Strategy

If you encounter issues:

1. **Check each test individually** - The debugging script runs tests in sequence
2. **Look for specific error messages** - Each test provides detailed error reporting
3. **Verify MPS availability** - Ensure Apple Silicon GPU is working
4. **Check model loading** - Verify Llama-3.2-1B loads correctly
5. **Test activation extraction** - This is the most common failure point

## Key Debugging Points

### Common Issues and Solutions:

1. **MPS Not Available**
   - Fallback to CPU automatically
   - Check PyTorch MPS support

2. **Model Loading Fails**
   - Check HuggingFace token if needed
   - Verify model name is correct

3. **Activation Extraction Fails**
   - Check layer index is valid
   - Verify hidden states are enabled
   - Check tensor shapes and devices

4. **Classifier Training Fails**
   - Check activation collection
   - Verify ContrastivePairSet creation
   - Look for empty tensor lists

## Next Steps

Once debugging is complete and all tests pass:
1. Use the performance benchmark to get accurate metrics
2. Apply the fixes to the main Jupyter notebook
3. Present the results to clients with confidence

## Support

If you encounter issues not covered here, the debugging script provides detailed error messages and stack traces to help identify the root cause.