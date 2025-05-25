#!/usr/bin/env python3
"""
Generate Training Charts for YOLO Pattern Detection

This script creates diverse chart images from multiple cryptocurrency symbols
and timeframes to build a comprehensive training dataset.
"""

import os
import sys
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.yolo_pattern_detector import ChartImageGenerator
except ImportError:
    from yolo_pattern_detector import ChartImageGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingDataGenerator:
    """Generate diverse training charts for YOLO annotation"""
    
    def __init__(self, output_dir: str = "training_charts"):
        """Initialize the generator"""
        self.output_dir = output_dir
        self.chart_generator = ChartImageGenerator(image_size=(800, 600))
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize exchange
        try:
            self.exchange = ccxt.binance()
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            self.exchange = None
    
    def get_crypto_symbols(self) -> List[str]:
        """Get list of popular crypto trading pairs"""
        symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 
            'XRP/USDT', 'SOL/USDT', 'DOGE/USDT', 'AVAX/USDT',
            'MATIC/USDT', 'DOT/USDT', 'LTC/USDT', 'LINK/USDT',
            'UNI/USDT', 'ATOM/USDT', 'FIL/USDT', 'TRX/USDT'
        ]
        return symbols
    
    def fetch_ohlcv_data(self, symbol: str, timeframe: str = '1d', limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol"""
        try:
            if not self.exchange:
                raise Exception("Exchange not initialized")
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.astype(float)
            
            logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def generate_symbol_charts(self, symbol: str, timeframes: List[str] = ['1d', '4h', '1h'], 
                             window_size: int = 100, step_size: int = 25) -> int:
        """Generate multiple charts for a single symbol across timeframes"""
        charts_generated = 0
        
        for timeframe in timeframes:
            try:
                # Fetch data
                df = self.fetch_ohlcv_data(symbol, timeframe, limit=300)
                if df.empty:
                    continue
                
                # Generate sliding window charts
                symbol_clean = symbol.replace('/', '_')
                
                for i in range(0, max(1, len(df) - window_size + 1), step_size):
                    try:
                        window_data = df.iloc[i:i + window_size]
                        if len(window_data) < 50:  # Skip small windows
                            continue
                        
                        # Create filename
                        start_date = window_data.index[0].strftime('%Y%m%d')
                        chart_filename = f"{symbol_clean}_{timeframe}_{start_date}_{i:03d}.png"
                        chart_path = os.path.join(self.output_dir, chart_filename)
                        
                        # Generate chart
                        result = self.chart_generator.generate_chart_image(
                            window_data, 
                            window_size=len(window_data), 
                            save_path=chart_path,
                            show_volume=True
                        )
                        
                        if result:
                            charts_generated += 1
                            logger.debug(f"Generated: {chart_filename}")
                    
                    except Exception as e:
                        logger.warning(f"Error generating chart for {symbol} {timeframe} window {i}: {e}")
                        continue
            
            except Exception as e:
                logger.error(f"Error processing {symbol} {timeframe}: {e}")
                continue
        
        return charts_generated
    
    def generate_all_charts(self, max_symbols: int = 10, max_charts_per_symbol: int = 20) -> Dict[str, int]:
        """Generate charts for multiple symbols"""
        symbols = self.get_crypto_symbols()[:max_symbols]
        results = {}
        total_charts = 0
        
        logger.info(f"🚀 Generating training charts for {len(symbols)} symbols")
        logger.info(f"Output directory: {os.path.abspath(self.output_dir)}")
        
        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"Processing {i}/{len(symbols)}: {symbol}")
                
                charts_count = self.generate_symbol_charts(
                    symbol, 
                    timeframes=['1d', '4h'],  # Focus on daily and 4-hour
                    window_size=100,
                    step_size=30
                )
                
                results[symbol] = charts_count
                total_charts += charts_count
                
                logger.info(f"✅ {symbol}: {charts_count} charts generated")
                
                # Limit charts per symbol
                if charts_count >= max_charts_per_symbol:
                    logger.info(f"Reached limit for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                results[symbol] = 0
                continue
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("📊 TRAINING CHART GENERATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total charts generated: {total_charts}")
        logger.info(f"Output directory: {os.path.abspath(self.output_dir)}")
        
        for symbol, count in results.items():
            if count > 0:
                logger.info(f"  {symbol}: {count} charts")
        
        logger.info(f"\n🎯 Ready for annotation using:")
        logger.info(f"  - Roboflow: https://roboflow.com/")
        logger.info(f"  - LabelImg: labelImg {self.output_dir}/")
        logger.info(f"  - Labelbox: https://labelbox.com/")
        
        return results

def main():
    """Generate training charts for YOLO annotation"""
    generator = TrainingDataGenerator()
    
    # Generate charts
    results = generator.generate_all_charts(
        max_symbols=8,  # Start with 8 symbols
        max_charts_per_symbol=15  # 15 charts per symbol
    )
    
    # Display results
    total = sum(results.values())
    print(f"\n🎉 Generated {total} training charts!")
    print(f"📁 Location: {os.path.abspath(generator.output_dir)}")
    print(f"\n🚀 Next steps:")
    print(f"1. Go to https://roboflow.com/ (web annotation)")
    print(f"2. Or run: labelImg {generator.output_dir}/ (desktop)")
    print(f"3. Annotate Cup & Handle patterns")
    print(f"4. Export in YOLO format")
    print(f"5. Train your model!")

if __name__ == "__main__":
    main() 