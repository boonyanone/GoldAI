"""
YOLO Pattern Detector

This module implements YOLOv12 for detecting Cup and Handle patterns in stock market charts.
Based on the research achieving 76% success rate for pattern detection.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from ultralytics import YOLO
import torch
from datetime import datetime


class PatternDetector:
    """
    YOLOv12-based detector for Cup and Handle patterns in candlestick charts.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize the pattern detector.
        
        Args:
            model_path (str, optional): Path to trained YOLO model weights
            device (str): Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
        
        # Pattern detection results cache
        self.detection_cache = {}
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize with pre-trained YOLOv8 for training
            self.model = YOLO('yolov8n.pt')
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained YOLO model.
        
        Args:
            model_path (str): Path to the model weights file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def create_candlestick_chart(
        self, 
        data: pd.DataFrame, 
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        style: str = 'default'
    ) -> np.ndarray:
        """
        Create a candlestick chart from OHLCV data.
        
        Args:
            data (pd.DataFrame): OHLCV data with datetime index
            save_path (str, optional): Path to save the chart image
            figsize (tuple): Figure size for the chart
            style (str): Chart style ('default', 'dark', 'clean')
            
        Returns:
            np.ndarray: Chart image as numpy array
        """
        # Set style
        if style == 'dark':
            plt.style.use('dark_background')
        elif style == 'clean':
            plt.style.use('seaborn-v0_8-white')
        else:
            plt.style.use('default')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for candlestick plotting
        dates = range(len(data))
        opens = data['Open'].values
        highs = data['High'].values
        lows = data['Low'].values
        closes = data['Close'].values
        
        # Color for up/down candles
        colors = ['green' if close >= open else 'red' 
                 for open, close in zip(opens, closes)]
        
        # Plot candlesticks
        for i, (date, open_price, high, low, close, color) in enumerate(
            zip(dates, opens, highs, lows, closes, colors)):
            
            # Candle body
            body_height = abs(close - open_price)
            body_bottom = min(open_price, close)
            
            # Draw body
            rect = patches.Rectangle(
                (date - 0.3, body_bottom), 0.6, body_height,
                linewidth=1, edgecolor='black', 
                facecolor=color, alpha=0.8
            )
            ax.add_patch(rect)
            
            # Draw wicks
            ax.plot([date, date], [low, high], 'k-', linewidth=1)
        
        # Customize chart
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Price')
        ax.set_title('Candlestick Chart')
        ax.grid(True, alpha=0.3)
        
        # Remove excessive whitespace
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to numpy array
        fig.canvas.draw()
        chart_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        chart_image = chart_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return chart_image
    
    def detect_patterns(
        self, 
        data: pd.DataFrame,
        window_size: int = 400,
        step_size: int = 50,
        min_confidence: float = 0.25
    ) -> List[Dict]:
        """
        Detect Cup and Handle patterns in stock data.
        
        Args:
            data (pd.DataFrame): OHLCV data
            window_size (int): Size of sliding window (default: 400)
            step_size (int): Step size for sliding window
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            List[Dict]: List of detected patterns with metadata
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a trained model first.")
        
        detections = []
        
        # Create sliding windows
        for i in range(0, len(data) - window_size + 1, step_size):
            window_data = data.iloc[i:i + window_size]
            
            if len(window_data) < window_size:
                continue
            
            # Create chart image
            chart_image = self.create_candlestick_chart(window_data)
            
            # Run detection
            results = self.model(chart_image, conf=min_confidence)
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.data.cpu().numpy()
                    
                    for box in boxes:
                        x1, y1, x2, y2, conf, cls = box
                        
                        detection = {
                            'start_index': i,
                            'end_index': i + window_size - 1,
                            'start_date': window_data.index[0],
                            'end_date': window_data.index[-1],
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class': int(cls),
                            'pattern_data': window_data.copy(),
                            'chart_image': chart_image
                        }
                        
                        detections.append(detection)
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def detect_from_image(
        self, 
        image_path: str,
        min_confidence: float = 0.25
    ) -> List[Dict]:
        """
        Detect patterns from a chart image file.
        
        Args:
            image_path (str): Path to chart image
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            List[Dict]: Detection results
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a trained model first.")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Run detection
        results = self.model(image_path, conf=min_confidence)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.data.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    
                    detection = {
                        'image_path': image_path,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf),
                        'class': int(cls)
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def visualize_detections(
        self, 
        detections: List[Dict],
        save_path: Optional[str] = None,
        max_detections: int = 5
    ) -> None:
        """
        Visualize detected patterns.
        
        Args:
            detections (List[Dict]): Detection results
            save_path (str, optional): Path to save visualization
            max_detections (int): Maximum number of detections to show
        """
        if not detections:
            print("No detections to visualize")
            return
        
        n_detections = min(len(detections), max_detections)
        fig, axes = plt.subplots(n_detections, 1, figsize=(15, 4 * n_detections))
        
        if n_detections == 1:
            axes = [axes]
        
        for i, detection in enumerate(detections[:n_detections]):
            ax = axes[i]
            
            if 'pattern_data' in detection:
                # Plot candlestick data
                data = detection['pattern_data']
                dates = range(len(data))
                
                # Simple candlestick representation
                ax.plot(dates, data['Close'], 'b-', linewidth=1, alpha=0.7)
                ax.fill_between(dates, data['Low'], data['High'], alpha=0.3)
                
                ax.set_title(f"Detection {i+1} - Confidence: {detection['confidence']:.3f}")
                ax.set_xlabel('Time Period')
                ax.set_ylabel('Price')
                ax.grid(True, alpha=0.3)
            
            else:
                # Show image with bounding box
                if 'chart_image' in detection:
                    ax.imshow(detection['chart_image'])
                    
                    # Draw bounding box
                    bbox = detection['bbox']
                    rect = patches.Rectangle(
                        (bbox[0], bbox[1]), 
                        bbox[2] - bbox[0], 
                        bbox[3] - bbox[1],
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    ax.set_title(f"Detection {i+1} - Confidence: {detection['confidence']:.3f}")
                    ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def train_model(
        self,
        data_yaml_path: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch_size: int = 16,
        save_dir: str = 'runs/train'
    ) -> str:
        """
        Train a new YOLO model for pattern detection.
        
        Args:
            data_yaml_path (str): Path to dataset YAML configuration
            epochs (int): Number of training epochs
            imgsz (int): Image size for training
            batch_size (int): Batch size
            save_dir (str): Directory to save training results
            
        Returns:
            str: Path to the trained model
        """
        if self.model is None:
            self.model = YOLO('yolov8n.pt')  # Start with pre-trained weights
        
        # Start training
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=save_dir,
            name='cup_handle_detector',
            exist_ok=True,
            verbose=True
        )
        
        # Get path to best model
        best_model_path = os.path.join(save_dir, 'cup_handle_detector', 'weights', 'best.pt')
        
        # Load the trained model
        if os.path.exists(best_model_path):
            self.load_model(best_model_path)
            print(f"Training completed. Best model saved to: {best_model_path}")
        
        return best_model_path
    
    def validate_model(
        self,
        data_yaml_path: str,
        save_dir: str = 'runs/val'
    ) -> Dict:
        """
        Validate the trained model.
        
        Args:
            data_yaml_path (str): Path to validation dataset YAML
            save_dir (str): Directory to save validation results
            
        Returns:
            Dict: Validation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a trained model first.")
        
        # Run validation
        results = self.model.val(
            data=data_yaml_path,
            project=save_dir,
            name='cup_handle_validation',
            exist_ok=True
        )
        
        return results.results_dict


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = PatternDetector()
    
    # Example: Create sample data and test chart generation
    from src.data.data_loader import StockDataLoader
    
    loader = StockDataLoader()
    data = loader.fetch_data('SPY', '2024-01-01', '2024-03-01', '1h')
    
    if not data.empty:
        print(f"Loaded {len(data)} data points")
        
        # Create a sample chart
        chart_image = detector.create_candlestick_chart(
            data.iloc[:400], 
            save_path='data/temp/sample_chart.png'
        )
        print(f"Chart created with shape: {chart_image.shape}")
        
        # Note: For actual pattern detection, you need a trained model
        # detector.load_model('data/models/cup_handle_detector.pt')
        # detections = detector.detect_patterns(data)
        # detector.visualize_detections(detections)
    else:
        print("No data available for testing") 