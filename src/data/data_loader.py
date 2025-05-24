"""
Stock Market Data Loader

This module handles fetching and preprocessing stock market data from various sources.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Union, List
import numpy as np


class StockDataLoader:
    """
    A class for loading and preprocessing stock market data.
    """
    
    def __init__(self):
        """Initialize the data loader."""
        self.data_cache = {}
    
    def fetch_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        interval: str = '15m'
    ) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            symbol (str): Stock symbol (e.g., 'SPY', 'TSLA')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            interval (str): Data interval ('1m', '5m', '15m', '1h', '1d')
            
        Returns:
            pd.DataFrame: OHLCV data with datetime index
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Clean the data
            data = self._clean_data(data)
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data by removing NaN values and ensuring data quality.
        
        Args:
            data (pd.DataFrame): Raw OHLCV data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        # Remove rows with NaN values
        data = data.dropna()
        
        # Ensure positive values for OHLCV
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                data = data[data[col] > 0]
        
        # Ensure High >= Low
        data = data[data['High'] >= data['Low']]
        
        # Ensure High >= Open, Close and Low <= Open, Close
        data = data[
            (data['High'] >= data['Open']) & 
            (data['High'] >= data['Close']) &
            (data['Low'] <= data['Open']) & 
            (data['Low'] <= data['Close'])
        ]
        
        return data
    
    def create_windows(
        self, 
        data: pd.DataFrame, 
        window_size: int = 400, 
        step_size: int = 1
    ) -> List[pd.DataFrame]:
        """
        Create sliding windows of the specified size from the data.
        
        Args:
            data (pd.DataFrame): OHLCV data
            window_size (int): Size of each window (default: 400 for Cup and Handle)
            step_size (int): Step size between windows
            
        Returns:
            List[pd.DataFrame]: List of windowed data
        """
        windows = []
        
        for i in range(0, len(data) - window_size + 1, step_size):
            window = data.iloc[i:i + window_size].copy()
            if len(window) == window_size:
                windows.append(window)
                
        return windows
    
    def get_latest_data(
        self, 
        symbol: str, 
        days: int = 30, 
        interval: str = '15m'
    ) -> pd.DataFrame:
        """
        Get the latest data for a symbol.
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days to fetch
            interval (str): Data interval
            
        Returns:
            pd.DataFrame: Latest OHLCV data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.fetch_data(
            symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            interval
        )


if __name__ == "__main__":
    # Example usage
    loader = StockDataLoader()
    
    # Fetch SPY data
    data = loader.fetch_data('SPY', '2024-01-01', '2024-12-31', '15m')
    print(f"Fetched {len(data)} rows of data for SPY")
    print(data.head())
    
    # Create windows
    windows = loader.create_windows(data, window_size=400)
    print(f"Created {len(windows)} windows of size 400") 