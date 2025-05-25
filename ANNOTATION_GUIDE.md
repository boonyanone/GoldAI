# YOLO Pattern Annotation Guide

## 🎯 Annotation Options for YOLOv12 Training

You have multiple options for annotating Cup and Handle patterns in your chart images.

## Option 1: LabelImg (Desktop - Recommended)

### Launch LabelImg
```bash
cd trading-bot-test
source trading_bot_env/bin/activate
labelImg test_charts/
```

### Annotation Steps
1. **Load Images**: LabelImg will open showing your chart images
2. **Select YOLO Format**: Click "PascalVOC" and change to "YOLO" format
3. **Create Bounding Box**: Press 'W' or click rectangle tool
4. **Draw Around Pattern**: Drag to create box around Cup & Handle formation
5. **Label Pattern**: Select class from predefined list
6. **Save**: Press 'Ctrl+S' to save annotation

### Pattern Classes
- **Class 0**: `cup_and_handle` - Complete Cup & Handle formation
- **Class 1**: `inverse_cup_and_handle` - Inverted pattern  
- **Class 2**: `double_bottom` - Alternative bullish pattern
- **Class 3**: `head_and_shoulders` - Reversal pattern

### LabelImg Keyboard Shortcuts
- **W**: Create bounding box
- **D**: Next image
- **A**: Previous image
- **Ctrl+S**: Save
- **Del**: Delete selected box

## Option 2: Roboflow (Web-based)

### Setup
1. **Visit**: https://roboflow.com/
2. **Create Account**: Free tier allows 1,000 images
3. **Create Project**: Select "Object Detection"
4. **Upload Images**: Drag and drop your chart files

### Advantages
- ✅ Web-based (no installation)
- ✅ Team collaboration
- ✅ Auto-export to YOLO format
- ✅ Data augmentation tools
- ✅ Model training integration

### Process
1. Upload chart images from `test_charts/`
2. Create classes: cup_and_handle, inverse_cup_and_handle, etc.
3. Draw bounding boxes around patterns
4. Export in YOLO format
5. Download trained dataset

## Option 3: Labelbox (Web-based)

### Setup
1. **Visit**: https://labelbox.com/
2. **Free Trial**: Up to 5,000 labels per month
3. **Create Project**: Object detection template
4. **Upload Images**: Batch upload chart images

### Features
- ✅ Professional annotation interface
- ✅ Quality control workflows
- ✅ Advanced labeling tools
- ✅ Export to multiple formats

## Option 4: CVAT (Computer Vision Annotation Tool)

### Setup (Self-hosted or cloud)
1. **Cloud**: https://cvat.ai/
2. **Docker**: Local installation available
3. **Features**: Advanced annotation features

### Advantages
- ✅ Free and open source
- ✅ Advanced polygon tools
- ✅ Video annotation support
- ✅ Multiple export formats

## 📊 Annotation Quality Guidelines

### Cup and Handle Pattern Requirements
- **Cup Depth**: 25-55% decline from peak
- **Cup Shape**: Rounded bottom (not V-shaped)
- **Handle**: Small consolidation after recovery
- **Volume**: Decreasing in cup, increasing at breakout
- **Duration**: Several weeks to months formation

### Bounding Box Guidelines
1. **Include Full Pattern**: Entire cup formation + handle
2. **Tight Boundaries**: Minimize empty space
3. **Volume Confirmation**: Include volume panel if visible
4. **Clear Formation**: Only annotate obvious patterns

### Quality Standards
- ✅ **Excellent**: Clear, textbook pattern formation
- ✅ **Good**: Recognizable pattern with minor deviations
- ❌ **Skip**: Incomplete, ambiguous, or poorly formed patterns

## 🚀 Quick Start Commands

### Option 1: LabelImg (Recommended)
```bash
# Launch LabelImg for your charts
cd trading-bot-test
source trading_bot_env/bin/activate
labelImg test_charts/
```

### Option 2: Generate More Charts
```bash
# Generate more training data
python scripts/test_chart_generation.py
```

### Option 3: Use Crypto Data
```bash
# Generate charts from real crypto data
python scripts/yolo_pattern_detector.py
```

## 📈 Expected Training Data

### Minimum Requirements
- **Training Images**: 500+ annotated charts
- **Validation Images**: 100+ annotated charts  
- **Pattern Distribution**: Balanced across classes
- **Market Conditions**: Bull, bear, sideways markets

### File Structure After Annotation
```
training_data/
├── dataset.yaml          # YOLO dataset config
├── images/               # Chart images
│   ├── chart_001.png
│   ├── chart_002.png
│   └── ...
└── labels/               # YOLO annotations
    ├── chart_001.txt
    ├── chart_002.txt
    └── ...
```

## 🔧 Troubleshooting

### LabelImg Issues
```bash
# If GUI doesn't launch on Linux
export DISPLAY=:0
sudo apt-get install python3-pyqt5

# Alternative installation
pip install labelImg[qt5]
```

### Web Portal Access
- **Roboflow**: Best for beginners, free tier
- **Labelbox**: Professional features, limited free
- **CVAT**: Open source, requires setup

## Next Steps

1. **Choose Tool**: LabelImg for quick start, Roboflow for web
2. **Annotate Patterns**: Start with 50-100 images
3. **Export Data**: YOLO format required
4. **Train Model**: Use annotated data for training
5. **Validate Results**: Test on unseen chart data

## 📚 Resources

- **LabelImg GitHub**: https://github.com/tzutalin/labelImg
- **Roboflow Docs**: https://docs.roboflow.com/
- **YOLO Training**: See `YOLO_TRAINING_GUIDE.md`
- **Pattern Research**: Original research paper methodology 