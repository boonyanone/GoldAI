# YOLOv12 Pattern Detection Training Guide

## 🎯 Overview

This guide walks you through training a YOLOv12 model to detect Cup and Handle patterns in candlestick charts, following the methodology from the research paper that achieved 76% success rate.

## 📋 Prerequisites

- ✅ Chart generation system (tested and working)
- ✅ Python environment with required dependencies
- ✅ NVIDIA GPU recommended for training (optional but faster)
- 📊 Historical market data for training

## 🔄 Training Workflow

### Phase 1: Data Collection & Chart Generation

```bash
# Generate chart images from historical data
python scripts/yolo_pattern_detector.py --mode generate_charts --symbols BTC/USDT,ETH/USDT,ADA/USDT --timeframe 1d --window_size 100 --step_size 20
```

**Expected Output:**
- 📁 `charts/` directory with candlestick chart images
- 🖼️ 800x600 pixel charts optimized for pattern recognition
- 📊 Sliding window coverage of entire dataset

### Phase 2: Manual Annotation

Use annotation tools to label Cup and Handle patterns:

1. **Install LabelImg** (recommended):
   ```bash
   pip install labelImg
   labelImg charts/
   ```

2. **Annotation Guidelines**:
   - **Class 0**: `cup_and_handle` - Complete Cup & Handle formation
   - **Class 1**: `inverse_cup_and_handle` - Inverted pattern
   - **Class 2**: `double_bottom` - Alternative bullish pattern
   - **Class 3**: `head_and_shoulders` - Reversal pattern

3. **Quality Standards**:
   - ✅ Clear pattern formation visible
   - ✅ Adequate decline (25-55%)
   - ✅ Proper recovery (80%+ retracement)
   - ✅ Volume confirmation visible
   - ❌ Skip incomplete or ambiguous patterns

### Phase 3: Training Data Preparation

```python
from scripts.yolo_pattern_detector import YOLOPatternDetector

# Initialize detector
detector = YOLOPatternDetector()

# Prepare training data from annotations
training_data = detector.prepare_training_data(
    patterns_with_labels=annotated_patterns,
    output_dir="training_data"
)
```

**File Structure:**
```
training_data/
├── dataset.yaml          # YOLO dataset configuration
├── images/               # Chart images
│   ├── pattern_0001.png
│   ├── pattern_0002.png
│   └── ...
└── labels/               # YOLO format annotations
    ├── pattern_0001.txt
    ├── pattern_0002.txt
    └── ...
```

### Phase 4: Model Training

```python
# Train YOLOv12 model
results = detector.train_model(
    dataset_path="training_data",
    epochs=100,
    batch_size=16,
    img_size=640
)
```

**Training Parameters:**
- **Epochs**: 100-200 (monitor for overfitting)
- **Batch Size**: 8-32 (based on GPU memory)
- **Image Size**: 640px (YOLO standard)
- **Device**: CUDA if available, else CPU

### Phase 5: Model Validation

```python
# Test trained model
from scripts.yolo_pattern_detector import IntegratedPatternDetector

detector = IntegratedPatternDetector("runs/detect/train/weights/best.pt")
patterns = detector.detect_and_score_patterns(test_data, "BTC/USDT")

for pattern in patterns:
    print(f"Pattern: {pattern['pattern_type']}")
    print(f"Confidence: {pattern['confidence']:.3f}")
    print(f"Combined Score: {pattern['combined_score']:.1f}")
```

## 📊 Expected Results

Based on the original research paper:

### Performance Metrics
- **Detection Accuracy**: 85-90% for well-formed patterns
- **False Positive Rate**: <15% with confidence threshold 0.5
- **Trading Success Rate**: 76% when combined with technical analysis
- **Processing Speed**: <100ms per chart on GPU

### Quality Thresholds
- **High Confidence**: >0.8 (Excellent patterns)
- **Medium Confidence**: 0.5-0.8 (Good patterns)
- **Low Confidence**: <0.5 (Review required)

## 🛠️ Training Data Requirements

### Minimum Dataset Size
- **Training Images**: 500+ annotated charts
- **Validation Images**: 100+ annotated charts
- **Pattern Classes**: Balanced distribution
- **Market Conditions**: Bull, bear, and sideways markets

### Data Quality Guidelines
- ✅ High-resolution charts (800x600+)
- ✅ Clear candlestick patterns
- ✅ Proper volume visualization
- ✅ Consistent annotation standards
- ✅ Diverse timeframes and symbols

## 🔧 Troubleshooting

### Common Issues

1. **Low Detection Accuracy**:
   - Increase training data size
   - Improve annotation quality
   - Adjust confidence thresholds
   - Add data augmentation

2. **High False Positives**:
   - Increase confidence threshold
   - Add negative examples
   - Improve pattern definition
   - Use ensemble methods

3. **Training Convergence Issues**:
   - Reduce learning rate
   - Increase batch size
   - Check data quality
   - Monitor loss curves

### Performance Optimization

```python
# GPU Memory Optimization
torch.cuda.empty_cache()

# Multi-GPU Training
model.train(device=[0, 1])  # Use multiple GPUs

# Mixed Precision Training
model.train(amp=True)  # Faster training with FP16
```

## 📈 Integration with Technical Analysis

The YOLOv12 system integrates with our existing technical analysis:

```python
# Combined detection and scoring
patterns = detector.detect_and_score_patterns(df, symbol)

for pattern in patterns:
    yolo_confidence = pattern['confidence']
    ta_score = pattern['ta_score']
    combined_score = pattern['combined_score']
    
    if combined_score >= 75:  # A-grade pattern
        print(f"🎯 High-quality pattern detected!")
        print(f"Visual: {yolo_confidence:.1%}")
        print(f"Technical: {ta_score:.1f}/100")
        print(f"Combined: {combined_score:.1f}/100")
```

## 🚀 Next Steps

1. **Collect Training Data**: Generate 1000+ chart images
2. **Annotate Patterns**: Use LabelImg or similar tools
3. **Train Model**: Start with 100 epochs
4. **Validate Performance**: Test on unseen data
5. **Deploy System**: Integrate with trading strategy
6. **Monitor Performance**: Track real-world accuracy

## 📚 Additional Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **LabelImg Tool**: https://github.com/tzutalin/labelImg
- **Pattern Recognition Research**: Original paper methodology
- **Technical Analysis Integration**: `pattern_collector_with_ta.py`

## ⚠️ Important Notes

- Start with small dataset for proof of concept
- Manually verify annotations for quality
- Use cross-validation for robust evaluation
- Monitor for overfitting during training
- Test on multiple market conditions
- Combine with risk management strategies

Remember: The goal is not just pattern detection, but **profitable pattern detection**! 🎯 