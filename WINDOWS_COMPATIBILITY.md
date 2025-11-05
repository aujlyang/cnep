# Windows + Python 3.8 + PyTorch + CUDA Compatibility

This document outlines the compatibility considerations for running the tactile data training scripts on Windows with Python 3.8, PyTorch, and CUDA.

## ✅ Compatibility Status

The code has been updated to be fully compatible with:
- **OS**: Windows 10/11
- **Python**: 3.8+
- **PyTorch**: Any version (tested with 1.x and 2.x)
- **CUDA**: Any version supported by your PyTorch installation

## 🔧 Changes Made for Compatibility

### 1. **Path Handling**
- All paths use `os.path.join()` which automatically handles Windows backslash separators
- No hardcoded forward slashes in file paths
- `os.makedirs()` with `exist_ok=True` works on Windows

### 2. **PyTorch Version Compatibility**

#### `torch.set_float32_matmul_precision()`
- This feature is only available in PyTorch 2.0+
- The code now checks for its existence before calling:
  ```python
  if hasattr(torch, 'set_float32_matmul_precision'):
      torch.set_float32_matmul_precision('high')
  ```
- If not available (PyTorch 1.x), it's safely skipped without errors

#### `torch.compile()`
- This feature requires **both** Python 3.10+ AND PyTorch 2.0+
- **Important**: With Python 3.8, `torch.compile()` will NOT be used (incompatible)
- The code now checks both Python and PyTorch versions:
  ```python
  if sys.version_info >= (3, 10) and hasattr(torch, 'compile'):
      try:
          model = torch.compile(model_)
      except Exception:
          model = model_
  else:
      model = model_  # Use eager mode on Python 3.8
  ```
- Models will run in eager mode on Python 3.8, which is fully functional but may be slightly slower

### 3. **File I/O**
- CSV reading/writing uses standard Python `csv` module (cross-platform)
- Image loading uses PIL/Pillow (cross-platform)
- All file operations are Windows-compatible

### 4. **CUDA Support**
- CUDA device selection works on Windows
- GPU utilization checking is compatible with Windows CUDA drivers
- Fallback to CPU if CUDA is unavailable

## 🚀 Usage on Windows

### Installation
```bash
# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
pip install pillow numpy

# Or for CPU-only
pip install torch torchvision
pip install pillow numpy
```

### Running Scripts
```bash
# Generate example data
python data\generate_example_tactile_data.py

# Train CNEP
python training_examples\train_cnep_on_tactile.py

# Train CNMP
python training_examples\train_cnmp_on_tactile.py
```

**Note**: Windows uses backslashes `\` in commands, but Python's path functions handle this automatically.

## ⚠️ Known Limitations on Python 3.8

1. **No `torch.compile()`**: Models run in eager mode
   - This is fine for training and inference
   - Slight performance impact compared to Python 3.10+ with compilation

2. **Type hints**: Some newer type hint syntax (like `|` for unions) is not available
   - All type hints in the code use Python 3.8-compatible syntax

3. **Performance**: PyTorch 2.x with Python 3.10+ offers better performance
   - Consider upgrading to Python 3.10+ if possible for optimal performance
   - But Python 3.8 is fully functional

## 🔍 Verification

All scripts have been verified to:
- ✅ Parse correctly in Python 3.8
- ✅ Handle Windows paths properly
- ✅ Work with PyTorch 1.x and 2.x
- ✅ Support both CUDA and CPU modes
- ✅ Gracefully degrade features not available in older versions

## 💡 Recommendations

For optimal performance on Windows:
1. Use the latest CUDA version supported by your GPU
2. Ensure PyTorch is installed with CUDA support: `torch.cuda.is_available()` should return `True`
3. If possible, upgrade to Python 3.10+ for `torch.compile()` support (optional)
4. Use an SSD for data storage to improve I/O performance

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce `batch_size` in the training scripts

### Issue: Slow training
**Solutions**:
- Ensure CUDA is properly installed: `torch.cuda.is_available()`
- Reduce `num_timesteps` or use a smaller CNN model (`mobilenet_v2` instead of `resnet50`)
- Check GPU utilization in Task Manager

### Issue: Path not found
**Solution**: Use absolute paths or ensure you're running from the correct directory
```python
data_folder = r'C:\path\to\your\tactile_data'  # Use raw string for Windows paths
```

### Issue: Import errors
**Solution**: Ensure you're running from the repository root or adjust Python paths:
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
```

## 📝 Summary

The code is **fully compatible** with Windows + Python 3.8 + PyTorch + CUDA. The main difference is that `torch.compile()` will not be used on Python 3.8, but this does not affect functionality, only potentially performance optimization.
