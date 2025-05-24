# GPU Setup Guide for AI Trading Bot

## Overview

This guide documents the complete GPU setup process for optimal performance with the AI Trading Bot. The system uses YOLO object detection for pattern recognition, which benefits significantly from GPU acceleration.

## System Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with CUDA Compute Capability 6.1+
- **VRAM**: 4GB+ (8GB+ recommended)
- **System RAM**: 8GB+ (16GB+ recommended)
- **CUDA**: 11.8+ or 12.x

### Recommended Configuration (Tested)
- **GPU**: NVIDIA GeForce RTX 3070 (8GB VRAM)
- **System RAM**: 32GB
- **CPU**: 8-core i7
- **CUDA**: 12.8

## Performance Results (RTX 3070)

| Test | CPU Time | GPU Time | Speedup |
|------|----------|----------|---------|
| YOLO Inference (5 images) | 0.282s | 0.022s | **13.04x** |
| Per Image | 0.056s | 0.004s | **14x** |
| Memory Usage | N/A | 78.7MB peak | 1.0% efficient |

## Installation Guide

### Step 1: GPU Driver Installation

1. **Check GPU Detection**:
   ```bash
   lspci | grep -i nvidia
   ```

2. **Install NVIDIA Drivers**:
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install nvidia-driver-570  # Latest stable
   
   # Reboot required
   sudo reboot
   ```

3. **Verify Installation**:
   ```bash
   nvidia-smi
   ```

### Step 2: PyTorch GPU Installation

**🚨 CRITICAL: Compatibility Issues**

We encountered compatibility issues between PyTorch 2.7+ and Ultralytics YOLO models. Here's the working solution:

#### Working Configuration
```bash
# System-wide installation (RECOMMENDED)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics==8.3.144
```

#### Alternative: Virtual Environment Setup
```bash
# Create virtual environment
python3 -m venv trading_bot_env
source trading_bot_env/bin/activate

# Install GPU-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install compatible Ultralytics (if needed)
pip install ultralytics==8.0.196
```

### Step 3: Verify GPU Setup

Run our benchmark script:
```bash
python test_gpu_simple.py
```

Expected output:
```
🚀 GPU Speedup: 13.04x faster than CPU
✅ GPU setup is OPTIMAL for trading bot
💡 Recommendation: Use GPU for production
```

## Troubleshooting

### Issue 1: PyTorch 2.7+ Compatibility Error

**Error Message**:
```
WeightsUnpickler error: Unsupported global: GLOBAL ultralytics.nn.tasks.DetectionModel
```

**Solutions**:

1. **Use System Python (RECOMMENDED)**:
   ```bash
   # Install directly in system Python
   deactivate  # Exit virtual environment
   python test_gpu_simple.py
   ```

2. **Downgrade Ultralytics**:
   ```bash
   pip install ultralytics==8.0.196
   ```

3. **Environment Variable (Alternative)**:
   ```bash
   TORCH_WEIGHTS_ONLY=False python your_script.py
   ```

### Issue 2: CUDA Out of Memory

**Solution**: Reduce batch size or use gradient checkpointing:
```python
# In your code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n.pt')
model.to(device)

# Monitor memory usage
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e6:.1f} MB")
```

### Issue 3: Driver Version Mismatch

**Check compatibility**:
```bash
nvidia-smi  # Check driver version
nvcc --version  # Check CUDA version
```

**Update if needed**:
```bash
# Ubuntu
sudo apt install nvidia-driver-535  # Or latest stable
sudo reboot
```

## Configuration for Production

### 1. Model Device Selection
```python
# Automatic device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PatternDetector(device=device)
```

### 2. Memory Management
```python
import torch

# Clear GPU cache periodically
torch.cuda.empty_cache()

# Monitor memory usage
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1e6:.1f} MB")
```

### 3. Batch Processing Optimization
```python
# Process multiple charts efficiently
def process_batch(charts, batch_size=8):
    results = []
    for i in range(0, len(charts), batch_size):
        batch = charts[i:i+batch_size]
        batch_results = model(batch)
        results.extend(batch_results)
        
        # Clear cache between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results
```

## Performance Optimization

### 1. Model Precision
```python
# Use mixed precision for faster inference
model.half()  # Convert to FP16
```

### 2. Asynchronous Processing
```python
import asyncio

async def async_pattern_detection(chart_data):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, model.detect_patterns, chart_data)
    return result
```

### 3. Multi-GPU Support (Future)
```python
# For multiple GPUs
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

## Hardware Recommendations

### Development Tier ($0 - Using Existing Hardware)
- **GPU**: GTX 1060 6GB+ or equivalent
- **RAM**: 16GB system memory
- **Storage**: SSD recommended

### Training Tier ($800-1,500)
- **GPU**: RTX 3070/4060 Ti (8GB+)
- **RAM**: 32GB system memory
- **Storage**: NVMe SSD 1TB+

### Production Tier ($1,500-3,000)
- **GPU**: RTX 4070/4080 (12GB+)
- **RAM**: 64GB system memory
- **Storage**: NVMe SSD 2TB+
- **Network**: Low-latency connection

## Monitoring and Logging

### GPU Utilization Monitoring
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log GPU usage
nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv --loop=10 > gpu_usage.log
```

### Python Monitoring
```python
import psutil
import GPUtil

def log_system_status():
    # GPU info
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryPercent:.1f}%)")
    
    # CPU and memory
    print(f"CPU: {psutil.cpu_percent()}%")
    print(f"RAM: {psutil.virtual_memory().percent}%")
```

## Testing GPU Performance

Use our benchmark scripts to verify performance:

```bash
# Quick GPU test
python test_gpu_simple.py

# Comprehensive benchmark
python test_gpu_benchmark.py

# With real trading data
python test_yolo_comprehensive.py
```

## Common Commands

```bash
# Check GPU status
nvidia-smi

# Check PyTorch GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check CUDA version
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Clear Python GPU cache
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
```

## Conclusion

With proper GPU setup, the AI Trading Bot achieves:
- **13x faster pattern detection**
- **Real-time processing capability**
- **Efficient memory usage (1% GPU memory)**
- **Production-ready performance**

The RTX 3070 configuration provides excellent performance for:
- Real-time market analysis
- Multiple timeframe processing
- Batch pattern detection
- Model training and fine-tuning

For production trading systems, GPU acceleration is essential for processing multiple symbols and timeframes simultaneously while maintaining low latency for trading decisions. 