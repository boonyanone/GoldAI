#!/usr/bin/env python3
"""
Test Chart Generation for YOLO Pattern Detection

Simple test to verify chart image generation works correctly.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add the scripts directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yolo_pattern_detector import ChartImageGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_data(days: int = 100) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducible results
    
    # Generate dates
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    
    # Generate price data with some trend and volatility
    base_price = 100.0
    prices = []
    volumes = []
    
    for i in range(days):
        # Add some trend and random walk
        trend = 0.001 * i  # Slight upward trend
        noise = np.random.normal(0, 0.02)  # Random volatility
        price_change = trend + noise
        
        if i == 0:
            price = base_price
        else:
            price = prices[i-1] * (1 + price_change)
        
        prices.append(price)
        
        # Generate volume (random around a base level)
        volume = np.random.uniform(800000, 1200000)
        volumes.append(volume)
    
    # Create OHLCV data
    data = []
    for i, (date, price, volume) in enumerate(zip(dates, prices, volumes)):
        # Simple OHLC generation around the price
        volatility = price * 0.02  # 2% daily volatility
        
        open_price = price + np.random.uniform(-volatility/2, volatility/2)
        close_price = price + np.random.uniform(-volatility/2, volatility/2)
        high_price = max(open_price, close_price) + np.random.uniform(0, volatility/2)
        low_price = min(open_price, close_price) - np.random.uniform(0, volatility/2)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def test_chart_generation():
    """Test chart image generation functionality"""
    try:
        logger.info("🔧 Testing Chart Generation System")
        
        # Generate sample data
        logger.info("Generating sample OHLCV data...")
        df = generate_sample_data(days=100)
        logger.info(f"Generated {len(df)} candles of sample data")
        
        # Initialize chart generator
        chart_gen = ChartImageGenerator(image_size=(800, 600))
        
        # Create output directory
        output_dir = "test_charts"
        os.makedirs(output_dir, exist_ok=True)
        
        # Test 1: Generate full chart
        logger.info("Test 1: Generating full chart image...")
        chart_path_1 = os.path.join(output_dir, "test_full_chart.png")
        result_1 = chart_gen.generate_chart_image(df, save_path=chart_path_1)
        
        if result_1 and os.path.exists(result_1):
            logger.info(f"✅ Full chart generated successfully: {result_1}")
        else:
            logger.error("❌ Failed to generate full chart")
        
        # Test 2: Generate sliding window charts
        logger.info("Test 2: Generating sliding window charts...")
        window_size = 50
        step_size = 10
        num_windows = (len(df) - window_size) // step_size + 1
        
        successful_windows = 0
        for i in range(0, len(df) - window_size + 1, step_size):
            window_data = df.iloc[i:i + window_size]
            window_start = window_data.index[0].strftime('%Y%m%d')
            
            chart_path = os.path.join(output_dir, f"window_{window_start}_{i:03d}.png")
            result = chart_gen.generate_chart_image(window_data, window_size=len(window_data), save_path=chart_path)
            
            if result and os.path.exists(result):
                successful_windows += 1
                logger.debug(f"Window {i//step_size + 1}/{num_windows}: {result}")
        
        logger.info(f"✅ Generated {successful_windows}/{num_windows} sliding window charts")
        
        # Test 3: Generate chart without volume
        logger.info("Test 3: Generating chart without volume...")
        chart_path_3 = os.path.join(output_dir, "test_no_volume.png")
        result_3 = chart_gen.generate_chart_image(df.tail(50), save_path=chart_path_3, show_volume=False)
        
        if result_3 and os.path.exists(result_3):
            logger.info(f"✅ No-volume chart generated successfully: {result_3}")
        else:
            logger.error("❌ Failed to generate no-volume chart")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("📊 CHART GENERATION TEST RESULTS")
        logger.info("="*60)
        logger.info(f"Sample data: {len(df)} candles")
        logger.info(f"Full chart: {'✅ Success' if result_1 else '❌ Failed'}")
        logger.info(f"Sliding windows: {successful_windows}/{num_windows} successful")
        logger.info(f"No-volume chart: {'✅ Success' if result_3 else '❌ Failed'}")
        logger.info(f"Output directory: {os.path.abspath(output_dir)}")
        
        # List generated files
        chart_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        logger.info(f"Generated {len(chart_files)} chart images:")
        for file in sorted(chart_files)[:5]:  # Show first 5
            logger.info(f"  - {file}")
        if len(chart_files) > 5:
            logger.info(f"  ... and {len(chart_files) - 5} more")
        
        return True
        
    except Exception as e:
        logger.error(f"Chart generation test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_chart_generation()
    if success:
        print("\n🎉 Chart generation test completed successfully!")
        print("You can now proceed with YOLO pattern detection training.")
    else:
        print("\n❌ Chart generation test failed.")
        print("Please check the error messages above.") 