#!/usr/bin/env python3
"""
YOLOv12 Pattern Detection System

This module implements visual pattern detection using YOLOv12 for Cup and Handle patterns.
It generates candlestick chart images and uses computer vision for pattern recognition.

Features:
- Chart image generation from OHLCV data
- YOLOv12 model integration for pattern detection
- Sliding window analysis for comprehensive coverage
- Integration with technical analysis scoring
- Training data preparation workflow
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import torch
from ultralytics import YOLO
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DetectedPattern:
    """Container for detected pattern information"""
    symbol: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    pattern_type: str
    window_start: datetime
    window_end: datetime
    chart_path: str
    technical_score: Optional[float] = None

class ChartImageGenerator:
    """Generate candlestick chart images for YOLO training and inference"""
    
    def __init__(self, image_size: Tuple[int, int] = (800, 600)):
        """Initialize chart generator"""
        self.image_size = image_size
        self.dpi = 100
        
    def generate_chart_image(self, df: pd.DataFrame, window_size: int = 100, 
                           save_path: str = None, show_volume: bool = True) -> str:
        """Generate a candlestick chart image from OHLCV data"""
        try:
            # Use dark theme for better pattern visibility
            plt.style.use('dark_background')
            
            if show_volume:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.image_size[0]/self.dpi, self.image_size[1]/self.dpi), 
                                             dpi=self.dpi, gridspec_kw={'height_ratios': [3, 1]})
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(self.image_size[0]/self.dpi, self.image_size[1]/self.dpi), 
                                      dpi=self.dpi)
                ax2 = None
            
            # Take last window_size candles
            data = df.tail(window_size).copy()
            dates = data.index
            
            # === CANDLESTICK CHART ===
            # Calculate colors
            colors = ['#00ff88' if close >= open_price else '#ff4757' 
                     for open_price, close in zip(data['open'], data['close'])]
            
            # Draw candlesticks
            for i, (idx, row) in enumerate(data.iterrows()):
                x = i
                open_price, high, low, close = row['open'], row['high'], row['low'], row['close']
                
                # Body
                body_height = abs(close - open_price)
                body_bottom = min(open_price, close)
                body_color = colors[i]
                
                # Draw body
                ax1.add_patch(Rectangle((x-0.3, body_bottom), 0.6, body_height, 
                                      facecolor=body_color, edgecolor=body_color, alpha=0.8))
                
                # Draw wicks
                ax1.plot([x, x], [low, high], color=body_color, linewidth=1)
            
            # Chart formatting
            ax1.set_xlim(-1, len(data))
            ax1.set_ylim(data['low'].min() * 0.995, data['high'].max() * 1.005)
            ax1.set_ylabel('Price', color='white', fontsize=10)
            ax1.tick_params(colors='white', labelsize=8)
            
            # Remove axes for cleaner pattern recognition
            ax1.set_xticks([])
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['left'].set_color('gray')
            ax1.grid(True, alpha=0.2, color='gray')
            
            # === VOLUME CHART ===
            if show_volume and ax2 is not None:
                volume_colors = [colors[i] for i in range(len(data))]
                bars = ax2.bar(range(len(data)), data['volume'], color=volume_colors, alpha=0.7, width=0.8)
                
                ax2.set_xlim(-1, len(data))
                ax2.set_ylabel('Volume', color='white', fontsize=8)
                ax2.tick_params(colors='white', labelsize=6)
                ax2.set_xticks([])
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['bottom'].set_visible(False)
                ax2.spines['left'].set_color('gray')
                
            # Remove margins and padding
            plt.tight_layout()
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.1)
            
            # Save image
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"chart_{timestamp}.png"
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='black', edgecolor='none', pad_inches=0.02)
            plt.close(fig)
            
            logger.debug(f"Chart image saved: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error generating chart image: {e}")
            return ""

class YOLOPatternDetector:
    """YOLOv12-based pattern detection system"""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """Initialize YOLO pattern detector"""
        self.confidence_threshold = confidence_threshold
        self.chart_generator = ChartImageGenerator()
        
        # Load or initialize YOLO model
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading trained YOLO model from {model_path}")
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                logger.warning(f"Failed to load custom model: {e}")
                logger.info("Falling back to pre-trained YOLOv8n")
                self.model = self._load_yolo_safe()
        else:
            logger.info("Initializing new YOLOv8 model for training")
            self.model = self._load_yolo_safe()
        
        # Pattern classes
        self.pattern_classes = {
            0: 'cup_and_handle',
            1: 'inverse_cup_and_handle', 
            2: 'double_bottom',
            3: 'head_and_shoulders'
        }
        
    def _load_yolo_safe(self):
        """Safely load YOLO model with fallback options"""
        try:
            # Try to load YOLOv8n from local file if available
            if os.path.exists('yolov8n.pt'):
                logger.info("Loading local yolov8n.pt")
                return YOLO('yolov8n.pt')
            else:
                # Download and load YOLOv8n
                logger.info("Downloading YOLOv8n model...")
                return YOLO('yolov8n.pt')  # This will auto-download
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            # Create a minimal detector for testing
            logger.warning("Creating mock detector for testing purposes")
            return MockYOLOModel()

    def detect_patterns_in_image(self, image_path: str) -> List[DetectedPattern]:
        """Detect patterns in a single chart image"""
        try:
            results = self.model(image_path, conf=self.confidence_threshold)
            patterns = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract detection data
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        
                        pattern = DetectedPattern(
                            symbol="",  # Will be filled by caller
                            confidence=confidence,
                            bbox=tuple(bbox),
                            pattern_type=self.pattern_classes.get(class_id, 'unknown'),
                            window_start=datetime.now(),  # Will be filled by caller
                            window_end=datetime.now(),    # Will be filled by caller
                            chart_path=image_path
                        )
                        patterns.append(pattern)
            
            logger.info(f"Detected {len(patterns)} patterns in {image_path}")
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns in {image_path}: {e}")
            return []
    
    def sliding_window_detection(self, df: pd.DataFrame, symbol: str, 
                                window_size: int = 100, step_size: int = 20,
                                output_dir: str = "charts") -> List[DetectedPattern]:
        """Perform sliding window pattern detection across the entire dataset"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            all_patterns = []
            
            # Calculate number of windows
            num_windows = max(1, (len(df) - window_size) // step_size + 1)
            logger.info(f"Analyzing {symbol} with {num_windows} sliding windows (size={window_size}, step={step_size})")
            
            for i in range(0, len(df) - window_size + 1, step_size):
                try:
                    # Extract window data
                    window_data = df.iloc[i:i + window_size]
                    window_start = window_data.index[0]
                    window_end = window_data.index[-1]
                    
                    # Generate chart image
                    chart_filename = f"{symbol.replace('/', '_')}_{window_start.strftime('%Y%m%d')}_{i:04d}.png"
                    chart_path = os.path.join(output_dir, chart_filename)
                    
                    generated_path = self.chart_generator.generate_chart_image(
                        window_data, window_size=len(window_data), save_path=chart_path)
                    
                    if not generated_path:
                        continue
                    
                    # Detect patterns in the image
                    window_patterns = self.detect_patterns_in_image(generated_path)
                    
                    # Update pattern information
                    for pattern in window_patterns:
                        pattern.symbol = symbol
                        pattern.window_start = window_start
                        pattern.window_end = window_end
                        all_patterns.append(pattern)
                    
                    logger.debug(f"Window {i//step_size + 1}/{num_windows}: "
                               f"Found {len(window_patterns)} patterns")
                    
                except Exception as e:
                    logger.warning(f"Error processing window {i}: {e}")
                    continue
            
            logger.info(f"Total patterns detected for {symbol}: {len(all_patterns)}")
            return all_patterns
            
        except Exception as e:
            logger.error(f"Error in sliding window detection for {symbol}: {e}")
            return []
    
    def prepare_training_data(self, patterns_with_labels: List[Dict], 
                            output_dir: str = "training_data"):
        """Prepare training data in YOLO format"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
            
            # Create dataset configuration
            dataset_config = {
                'path': output_dir,
                'train': 'images',
                'val': 'images',
                'nc': len(self.pattern_classes),
                'names': list(self.pattern_classes.values())
            }
            
            with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
                import yaml
                yaml.dump(dataset_config, f)
            
            # Process each labeled pattern
            for i, pattern_data in enumerate(patterns_with_labels):
                try:
                    # Generate chart image
                    chart_path = os.path.join(output_dir, "images", f"pattern_{i:04d}.png")
                    self.chart_generator.generate_chart_image(
                        pattern_data['data'], save_path=chart_path)
                    
                    # Create YOLO label file
                    label_path = os.path.join(output_dir, "labels", f"pattern_{i:04d}.txt")
                    with open(label_path, 'w') as f:
                        for bbox in pattern_data['bboxes']:
                            # Convert to YOLO format (class x_center y_center width height)
                            class_id = bbox['class_id']
                            x_center = (bbox['x1'] + bbox['x2']) / 2
                            y_center = (bbox['y1'] + bbox['y2']) / 2
                            width = bbox['x2'] - bbox['x1']
                            height = bbox['y2'] - bbox['y1']
                            
                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                    
                except Exception as e:
                    logger.warning(f"Error processing training pattern {i}: {e}")
                    continue
            
            logger.info(f"Training data prepared in {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None
    
    def train_model(self, dataset_path: str, epochs: int = 100, 
                   batch_size: int = 16, img_size: int = 640):
        """Train the YOLO model on pattern data"""
        try:
            logger.info(f"Starting YOLO training: epochs={epochs}, batch_size={batch_size}")
            
            # Train the model
            results = self.model.train(
                data=os.path.join(dataset_path, 'dataset.yaml'),
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                project='pattern_detection',
                name='cup_and_handle_v1'
            )
            
            logger.info("Training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error training YOLO model: {e}")
            return None

class IntegratedPatternDetector:
    """Integrated pattern detection combining YOLO with technical analysis"""
    
    def __init__(self, yolo_model_path: str = None):
        """Initialize integrated detector"""
        self.yolo_detector = YOLOPatternDetector(yolo_model_path)
        
        # Import technical analysis from our existing system
        try:
            from pattern_collector_with_ta import PatternCollector
            self.ta_collector = PatternCollector()
        except ImportError:
            logger.warning("Technical analysis collector not available")
            self.ta_collector = None
    
    def detect_and_score_patterns(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Detect patterns using YOLO and enhance with technical analysis scoring"""
        try:
            # Step 1: YOLO Pattern Detection
            logger.info(f"🔍 Running YOLO pattern detection on {symbol}")
            yolo_patterns = self.yolo_detector.sliding_window_detection(df, symbol)
            
            if not yolo_patterns:
                logger.info(f"No visual patterns detected for {symbol}")
                return []
            
            # Step 2: Technical Analysis Enhancement
            enhanced_patterns = []
            for pattern in yolo_patterns:
                try:
                    # Extract relevant data window
                    pattern_start = pattern.window_start
                    pattern_end = pattern.window_end
                    
                    # Get data for this time window
                    mask = (df.index >= pattern_start) & (df.index <= pattern_end)
                    window_data = df[mask]
                    
                    if len(window_data) < 10:  # Need minimum data
                        continue
                    
                    # Calculate technical analysis if available
                    ta_score = None
                    if self.ta_collector:
                        try:
                            # Calculate technical indicators
                            ta_data = self.ta_collector.calculate_technical_indicators(window_data)
                            
                            # Simple TA scoring based on trend and momentum
                            ta_score = self._calculate_ta_score(ta_data)
                        except Exception as e:
                            logger.warning(f"TA calculation failed: {e}")
                    
                    # Combine YOLO confidence with TA score
                    combined_score = self._combine_scores(pattern.confidence, ta_score)
                    
                    enhanced_pattern = {
                        'symbol': symbol,
                        'pattern_type': pattern.pattern_type,
                        'confidence': pattern.confidence,
                        'ta_score': ta_score,
                        'combined_score': combined_score,
                        'bbox': pattern.bbox,
                        'window_start': pattern_start.strftime('%Y-%m-%d'),
                        'window_end': pattern_end.strftime('%Y-%m-%d'),
                        'chart_path': pattern.chart_path
                    }
                    
                    enhanced_patterns.append(enhanced_pattern)
                    
                except Exception as e:
                    logger.warning(f"Error enhancing pattern: {e}")
                    continue
            
            # Sort by combined score
            enhanced_patterns.sort(key=lambda x: x['combined_score'], reverse=True)
            
            logger.info(f"Found {len(enhanced_patterns)} scored patterns for {symbol}")
            return enhanced_patterns
            
        except Exception as e:
            logger.error(f"Error in integrated pattern detection: {e}")
            return []
    
    def _calculate_ta_score(self, ta_data: pd.DataFrame) -> float:
        """Calculate simple technical analysis score"""
        try:
            score = 50  # Base score
            
            # RSI momentum
            if 'rsi' in ta_data.columns:
                rsi_current = ta_data['rsi'].iloc[-1]
                if 30 <= rsi_current <= 70:  # Good momentum range
                    score += 10
            
            # MACD trend
            if 'MACD_12_26_9' in ta_data.columns and 'MACDs_12_26_9' in ta_data.columns:
                macd = ta_data['MACD_12_26_9'].iloc[-1]
                signal = ta_data['MACDs_12_26_9'].iloc[-1]
                if macd > signal:  # Bullish crossover
                    score += 15
            
            # Volume trend
            if 'volume' in ta_data.columns:
                vol_trend = ta_data['volume'].pct_change().mean()
                if vol_trend > 0:  # Increasing volume
                    score += 10
            
            return min(score, 100)
            
        except Exception as e:
            logger.warning(f"TA score calculation error: {e}")
            return 50
    
    def _combine_scores(self, yolo_confidence: float, ta_score: Optional[float]) -> float:
        """Combine YOLO confidence with TA score"""
        if ta_score is None:
            return yolo_confidence * 100
        
        # Weighted combination: 60% YOLO, 40% TA
        combined = (yolo_confidence * 0.6) + (ta_score * 0.004)  # ta_score is 0-100, normalize
        return min(combined * 100, 100)

class MockYOLOModel:
    """Mock YOLO model for testing when real model fails to load"""
    
    def __init__(self):
        logger.warning("Using mock YOLO model - no real detection will occur")
    
    def __call__(self, image_path, conf=0.5):
        """Mock detection that returns empty results"""
        return [MockResult()]
    
    def train(self, **kwargs):
        """Mock training"""
        logger.warning("Mock training - no actual training performed")
        return None

class MockResult:
    """Mock result for testing"""
    def __init__(self):
        self.boxes = None

def main():
    """Test the YOLO pattern detection system"""
    # Initialize detector
    detector = IntegratedPatternDetector()
    
    # Example: Load some test data
    import ccxt
    
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=200)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        
        # Detect patterns
        patterns = detector.detect_and_score_patterns(df, 'BTC/USDT')
        
        # Display results
        print("\n" + "="*80)
        print("🎯 YOLO PATTERN DETECTION RESULTS")
        print("="*80)
        
        for i, pattern in enumerate(patterns[:5], 1):  # Show top 5
            print(f"\nPattern {i}:")
            print(f"  Type: {pattern['pattern_type']}")
            print(f"  YOLO Confidence: {pattern['confidence']:.3f}")
            print(f"  TA Score: {pattern['ta_score']:.1f}" if pattern['ta_score'] else "  TA Score: N/A")
            print(f"  Combined Score: {pattern['combined_score']:.1f}")
            print(f"  Period: {pattern['window_start']} to {pattern['window_end']}")
            print(f"  Chart: {pattern['chart_path']}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")

if __name__ == "__main__":
    main() 