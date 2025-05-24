#!/usr/bin/env python3
"""
Simple GPU Performance Test for AI Trading Bot
Quick demonstration of CPU vs GPU speedup
"""

import time
import torch
import numpy as np
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

def print_system_info():
    """Display system information"""
    print("=" * 50)
    print("🚀 AI TRADING BOT - GPU TEST")
    print("=" * 50)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    print(f"CPU Cores: {torch.get_num_threads()}")
    print("=" * 50)

def create_dummy_candlestick_images(num_images=10):
    """Create dummy candlestick chart images for testing"""
    print(f"\n📊 Creating {num_images} dummy candlestick charts...")
    
    # Simulate realistic candlestick chart dimensions
    images = []
    for i in range(num_images):
        # Random RGB image 640x640 (YOLO input size)
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        images.append(img)
    
    print(f"  ✅ Created {len(images)} charts (640x640x3)")
    return images

def benchmark_yolo_performance():
    """Test YOLO performance on CPU and GPU"""
    print("\n🔬 YOLO Performance Benchmark")
    print("-" * 30)
    
    # Create test data
    test_images = create_dummy_candlestick_images(5)
    
    # Download YOLO model if needed
    print("\n📦 Loading YOLO model...")
    
    # Test CPU performance
    print("\n🖥️  CPU Performance Test:")
    model_cpu = YOLO('yolov8n.pt')
    model_cpu.to('cpu')
    
    # Warmup
    model_cpu(test_images[0], verbose=False)
    
    # Time CPU inference
    start_time = time.time()
    for img in test_images:
        results = model_cpu(img, verbose=False)
    cpu_time = time.time() - start_time
    
    print(f"  ⏱️  CPU Time: {cpu_time:.3f}s")
    print(f"  📊 Per image: {cpu_time/len(test_images):.3f}s")
    
    # Test GPU performance (if available)
    if torch.cuda.is_available():
        print("\n🚀 GPU Performance Test:")
        model_gpu = YOLO('yolov8n.pt')
        model_gpu.to('cuda')
        
        # GPU warmup
        model_gpu(test_images[0], verbose=False)
        torch.cuda.synchronize()  # Wait for GPU to finish
        
        # Time GPU inference
        start_time = time.time()
        for img in test_images:
            results = model_gpu(img, verbose=False)
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time
        
        print(f"  ⏱️  GPU Time: {gpu_time:.3f}s")
        print(f"  📊 Per image: {gpu_time/len(test_images):.3f}s")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"\n📈 Performance Results:")
        print(f"  🚀 GPU Speedup: {speedup:.2f}x")
        
        if speedup > 2:
            print(f"  ✅ Excellent GPU acceleration!")
        elif speedup > 1.5:
            print(f"  ✅ Good GPU performance!")
        else:
            print(f"  ⚠️  Moderate speedup (may be overhead)")
        
        return speedup
    else:
        print("\n❌ GPU not available - cannot compare performance")
        return None

def benchmark_memory_usage():
    """Test GPU memory usage"""
    if not torch.cuda.is_available():
        print("\n❌ GPU not available for memory test")
        return
    
    print("\n🧠 GPU Memory Usage Test")
    print("-" * 25)
    
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1e6
    print(f"  Initial Memory: {initial_memory:.1f} MB")
    
    # Load model
    model = YOLO('yolov8n.pt')
    model.to('cuda')
    
    model_memory = torch.cuda.memory_allocated() / 1e6
    print(f"  After Model Load: {model_memory:.1f} MB")
    
    # Run inference
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    model(dummy_img, verbose=False)
    
    inference_memory = torch.cuda.memory_allocated() / 1e6
    max_memory = torch.cuda.max_memory_allocated() / 1e6
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e6
    
    print(f"  After Inference: {inference_memory:.1f} MB")
    print(f"  Peak Memory: {max_memory:.1f} MB")
    print(f"  Total Available: {total_memory:.0f} MB")
    
    efficiency = (max_memory / total_memory) * 100
    print(f"  Memory Efficiency: {efficiency:.1f}%")

def main():
    """Run the simple GPU benchmark"""
    print_system_info()
    
    try:
        # Performance benchmark
        speedup = benchmark_yolo_performance()
        
        # Memory usage test
        benchmark_memory_usage()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 50)
        
        if speedup:
            print(f"🚀 GPU Speedup: {speedup:.2f}x faster than CPU")
            if speedup >= 2:
                print("✅ GPU setup is OPTIMAL for trading bot")
                print("💡 Recommendation: Use GPU for production")
            else:
                print("⚠️  GPU shows improvement but may have overhead")
                print("💡 Recommendation: Profile with larger batches")
        else:
            print("❌ GPU not available or not tested")
            print("💡 Recommendation: Use CPU for now")
        
        print("\n🎯 Next Steps:")
        print("  1. Test with real candlestick data")
        print("  2. Implement batch processing")
        print("  3. Optimize for real-time trading")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 