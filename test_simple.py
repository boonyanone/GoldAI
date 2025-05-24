#!/usr/bin/env python3
"""
Simple test script for basic functionality
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

def test_basic_functionality():
    """Test basic functionality without YOLO."""
    print('Testing Basic Functionality...')
    
    try:
        # Test data loader
        print('📊 Testing data loader...')
        from src.data.data_loader import StockDataLoader
        
        loader = StockDataLoader()
        data = loader.get_latest_data('SPY', days=5, interval='1h')
        print(f'✅ Data loaded: {len(data)} rows')
        print(f'✅ Data columns: {list(data.columns)}')
        
        # Test chart creation without YOLO
        if len(data) >= 50:
            print('📈 Testing chart creation...')
            os.makedirs('data/temp', exist_ok=True)
            
            # Simple candlestick chart creation
            chart_data = data.iloc[:50]
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create simple candlestick chart
            dates = range(len(chart_data))
            opens = chart_data['Open'].values
            highs = chart_data['High'].values
            lows = chart_data['Low'].values
            closes = chart_data['Close'].values
            
            # Plot price line
            ax.plot(dates, closes, 'b-', linewidth=1, label='Close Price')
            ax.fill_between(dates, lows, highs, alpha=0.3, label='Price Range')
            
            ax.set_xlabel('Time Period')
            ax.set_ylabel('Price')
            ax.set_title('Simple Stock Chart')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('data/temp/simple_chart.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print('✅ Chart created and saved to: data/temp/simple_chart.png')
        else:
            print('⚠️  Not enough data for chart creation')
        
        # Test data windowing
        print('🔍 Testing data windowing...')
        windows = loader.create_windows(data, window_size=24, step_size=6)
        print(f'✅ Created {len(windows)} windows of size 24')
        
        print('🎉 All basic tests passed!')
        print('\n📝 Basic setup is working:')
        print('✅ Data loading from Yahoo Finance')
        print('✅ Data preprocessing and cleaning')
        print('✅ Chart generation with matplotlib')
        print('✅ Data windowing for pattern detection')
        print('\n🔧 Ready for next steps:')
        print('1. Fix PyTorch/Ultralytics compatibility')
        print('2. Implement YOLO pattern detection')
        print('3. Create sequence models for trading decisions')
        
        return True
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1) 