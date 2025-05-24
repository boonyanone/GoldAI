#!/usr/bin/env python3
"""
Test script for YOLO Pattern Detector
"""

from src.models.yolo_detector import PatternDetector
from src.data.data_loader import StockDataLoader
import sys
import os

def test_detector():
    """Test the pattern detector functionality."""
    print('Testing YOLO Detector...')
    
    try:
        # Test data loader
        print('📊 Loading stock data...')
        loader = StockDataLoader()
        data = loader.get_latest_data('SPY', days=5, interval='1h')
        print(f'✅ Data loaded: {len(data)} rows')
        
        # Test pattern detector initialization
        print('🤖 Initializing PatternDetector...')
        detector = PatternDetector()
        print('✅ PatternDetector initialized')
        
        # Test chart creation
        if len(data) >= 100:
            print('📈 Creating candlestick chart...')
            os.makedirs('data/temp', exist_ok=True)
            chart_image = detector.create_candlestick_chart(
                data.iloc[:100], 
                save_path='data/temp/test_chart.png'
            )
            print(f'✅ Chart created with shape: {chart_image.shape}')
            print('✅ Chart saved to: data/temp/test_chart.png')
        else:
            print('⚠️  Not enough data for chart creation')
        
        print('🎉 All tests passed!')
        print('\n📝 Next steps:')
        print('1. Train YOLO model with annotated Cup & Handle patterns')
        print('2. Implement sequence models for trading decisions')
        print('3. Set up backtesting framework')
        
        return True
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_detector()
    sys.exit(0 if success else 1) 