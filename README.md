# AI Trading Bot - Pattern Detection and Trading System

## Project Overview

This project implements an AI-powered trading bot that uses computer vision and machine learning to detect chart patterns and make trading decisions. Based on the research project "Using AI for Stock Market Pattern Detection and Trading", the system combines:

1. **YOLOv12 Object Detection** for identifying Cup and Handle patterns in candlestick charts
2. **Classification Models** for trading decision making using both full sequence data and technical indicators
3. **Automated Trading Strategy** with risk management

## Features

- **Pattern Detection**: Automated detection of Cup and Handle chart patterns using YOLOv12
- **Dual Decision Models**: 
  - Full sequence models (400 candlesticks OHLCV data)
  - Technical indicator models (10 key indicators)
- **Multi-position Trading**: Support for multiple price targets and risk levels
- **Risk Management**: Automated stop-loss and take-profit calculations
- **Real-time Data**: Integration with financial data providers
- **Backtesting**: Historical performance evaluation
- **Visualization**: Interactive charts and pattern visualization

## Project Structure

```
trading-bot-test/
├── src/
│   ├── data/
│   │   ├── data_loader.py          # Stock data fetching and preprocessing
│   │   ├── preprocessor.py         # Data cleaning and transformation
│   │   └── technical_indicators.py # Technical analysis calculations
│   ├── models/
│   │   ├── yolo_detector.py        # YOLOv12 pattern detection
│   │   ├── sequence_models.py      # Full sequence classification models
│   │   ├── indicator_models.py     # Technical indicator models
│   │   └── model_utils.py          # Model utilities and helpers
│   ├── trading/
│   │   ├── strategy.py             # Trading strategy implementation
│   │   ├── risk_management.py      # Risk management rules
│   │   └── backtester.py           # Backtesting engine
│   ├── utils/
│   │   ├── visualization.py        # Chart and pattern visualization
│   │   ├── logger.py               # Logging configuration
│   │   └── config.py               # Configuration management
│   └── api/
│       ├── main.py                 # FastAPI application
│       └── endpoints.py            # API endpoints
├── data/
│   ├── raw/                        # Raw market data
│   ├── processed/                  # Preprocessed data
│   ├── annotations/                # Pattern annotations
│   └── models/                     # Trained model weights
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pattern_annotation.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_backtesting.ipynb
│   └── 05_visualization.ipynb
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_trading.py
├── configs/
│   ├── model_config.yaml
│   ├── trading_config.yaml
│   └── data_config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd trading-bot-test
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv trading_bot_env
   source trading_bot_env/bin/activate  # On Windows: trading_bot_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install TA-Lib** (may require additional system dependencies):
   ```bash
   # On Ubuntu/Debian:
   sudo apt-get install build-essential
   # On macOS:
   brew install ta-lib
   ```

## Quick Start

1. **Data Collection**:
   ```python
   from src.data.data_loader import StockDataLoader
   
   loader = StockDataLoader()
   data = loader.fetch_data('SPY', '2020-01-01', '2025-01-01', interval='15m')
   ```

2. **Pattern Detection**:
   ```python
   from src.models.yolo_detector import PatternDetector
   
   detector = PatternDetector('data/models/yolo_cup_handle.pt')
   patterns = detector.detect_patterns(data)
   ```

3. **Trading Decision**:
   ```python
   from src.models.sequence_models import FullSequenceModel
   
   model = FullSequenceModel()
   model.load('data/models/sequence_model.pt')
   decision = model.predict(pattern_data)
   ```

4. **Run Backtesting**:
   ```python
   from src.trading.backtester import Backtester
   
   backtester = Backtester()
   results = backtester.run(start_date='2023-01-01', end_date='2024-01-01')
   ```

## Configuration

Edit the configuration files in the `configs/` directory:

- `model_config.yaml`: Model hyperparameters and architecture settings
- `trading_config.yaml`: Trading strategy parameters, risk management rules
- `data_config.yaml`: Data sources, timeframes, and preprocessing settings

## Model Performance

Based on the research results:

- **Full Sequence Models**: ~76% success rate on S&P 500 test data
- **Technical Indicator Models**: 68-71% accuracy with multi-position approach
- **Pattern Detection**: High precision YOLOv12 Cup and Handle detection

## Development

1. **Code Formatting**:
   ```bash
   black src/
   ```

2. **Linting**:
   ```bash
   flake8 src/
   ```

3. **Testing**:
   ```bash
   pytest tests/
   ```

4. **Jupyter Notebooks**:
   ```bash
   jupyter notebook
   ```

## API Usage

Start the FastAPI server:
```bash
uvicorn src.api.main:app --reload
```

Example API calls:
- `POST /detect-patterns`: Upload chart data for pattern detection
- `POST /trading-decision`: Get trading recommendation for detected pattern
- `GET /backtest`: Run backtesting with specified parameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run code quality checks
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Research Citation

This implementation is based on the research project:
"Using AI for Stock Market Pattern Detection and Trading" by Yuen Chun Ho (Student ID: 24113812G)

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results. 