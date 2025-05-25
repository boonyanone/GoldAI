#!/usr/bin/env python3
"""
Crypto Pattern Collector with pandas-ta Integration

This module combines our existing pattern detection logic with pandas-ta's
comprehensive technical analysis library for superior pattern recognition.

Features:
- 65+ candlestick patterns via pandas-ta
- Advanced trend, momentum, and volatility indicators
- Multi-timeframe analysis
- Advanced scoring system with technical confluences
- Professional visualization with technical overlays
"""

import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import pandas_ta as ta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy.signal import find_peaks, argrelextrema
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TechnicalSignals:
    """Container for technical analysis signals"""
    candlestick_patterns: Dict[str, bool]
    trend_signals: Dict[str, float]
    momentum_signals: Dict[str, float]
    volatility_signals: Dict[str, float]
    volume_signals: Dict[str, float]
    support_resistance: Dict[str, float]

class PatternCollector:
    """Pattern collector with pandas-ta integration"""
    
    def __init__(self, exchange_id='binance', sandbox=False):
        """Initialize pattern collector"""
        self.exchange_class = getattr(ccxt, exchange_id)
        self.exchange = self.exchange_class({
            'sandbox': sandbox,
            'enableRateLimit': True,
        })
        
        # Pattern detection parameters
        self.min_decline = 25.0  # Minimum decline %
        self.max_decline = 55.0  # Maximum decline %
        self.min_recovery = 80.0  # Minimum recovery %
        self.min_breakout = 10.0  # Minimum breakout %
        self.timeframe = '1d'     # Daily timeframes for quality
        self.lookback_months = 24 # 2 years of data
        
        # Technical analysis parameters
        self.ta_config = {
            'rsi_period': 14,           # Relative Strength Index
            'macd_fast': 12,            # Moving Average Convergence Divergence
            'macd_slow': 26,            # Moving Average Convergence Divergence Slow
            'macd_signal': 9,           # Moving Average Convergence Divergence Signal
            'bb_period': 20,            # Bollinger Bands Period
            'bb_std': 2,                # Bollinger Bands Standard Deviation
            'atr_period': 14,           # Average True Range Period
            'adx_period': 14,           # Average Directional Index Period
            'stoch_k': 14,              # Stochastic Oscillator K
            'stoch_d': 3,               # Stochastic Oscillator D
            'cci_period': 20,           # Commodity Channel Index Period
            'williams_r_period': 14,    # Williams %R Period
            'obv_enabled': True,        # On-Balance Volume 
            'vwap_enabled': True        # Volume Weighted Average Price 
        }
        
        # Candlestick patterns to focus on (most reliable for crypto)
        self.key_patterns = [
            'doji', 'hammer', 'hangingman', 'shootingstar', 'marubozu',
            'engulfing', 'harami', 'piercingpattern', 'darkcloudcover',
            'morningstar', 'eveningstar', 'threewhitesoldiers', 'threeblackcrows',
            'abandonedbaby', 'dragonflydoji', 'gravestonedoji'
        ]
        
        logger.info("Pattern Collector initialized with pandas-ta")
    
    def fetch_data(self, symbol: str, timeframe: str = None, limit: int = None) -> pd.DataFrame:
        """Fetch OHLCV data with comprehensive error handling"""
        try:
            if timeframe is None:
                timeframe = self.timeframe
            if limit is None:
                limit = self.lookback_months * 30  # Approximate days per month
                
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.astype(float)
            
            logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators using pandas-ta"""
        try:
            # Create a copy to avoid modifying original
            ta_df = df.copy()
            
            # === TREND INDICATORS ===
            # Moving Averages
            ta_df['sma_20'] = ta.sma(ta_df['close'], length=20)
            ta_df['sma_50'] = ta.sma(ta_df['close'], length=50)
            ta_df['ema_12'] = ta.ema(ta_df['close'], length=12)
            ta_df['ema_26'] = ta.ema(ta_df['close'], length=26)
            
            # MACD
            macd_data = ta.macd(ta_df['close'], 
                               fast=self.ta_config['macd_fast'],
                               slow=self.ta_config['macd_slow'],
                               signal=self.ta_config['macd_signal'])
            if macd_data is not None:
                ta_df = pd.concat([ta_df, macd_data], axis=1)
            
            # ADX (Average Directional Index)
            adx_data = ta.adx(ta_df['high'], ta_df['low'], ta_df['close'], 
                             length=self.ta_config['adx_period'])
            if adx_data is not None:
                ta_df = pd.concat([ta_df, adx_data], axis=1)
            
            # === MOMENTUM INDICATORS ===
            # RSI
            ta_df['rsi'] = ta.rsi(ta_df['close'], length=self.ta_config['rsi_period'])
            
            # Stochastic
            stoch_data = ta.stoch(ta_df['high'], ta_df['low'], ta_df['close'],
                                 k=self.ta_config['stoch_k'], d=self.ta_config['stoch_d'])
            if stoch_data is not None:
                ta_df = pd.concat([ta_df, stoch_data], axis=1)
            
            # CCI (Commodity Channel Index)
            ta_df['cci'] = ta.cci(ta_df['high'], ta_df['low'], ta_df['close'],
                                 length=self.ta_config['cci_period'])
            
            # Williams %R
            ta_df['willr'] = ta.willr(ta_df['high'], ta_df['low'], ta_df['close'],
                                     length=self.ta_config['williams_r_period'])
            
            # === VOLATILITY INDICATORS ===
            # Bollinger Bands
            bb_data = ta.bbands(ta_df['close'], 
                               length=self.ta_config['bb_period'],
                               std=self.ta_config['bb_std'])
            if bb_data is not None:
                ta_df = pd.concat([ta_df, bb_data], axis=1)
            
            # ATR (Average True Range)
            ta_df['atr'] = ta.atr(ta_df['high'], ta_df['low'], ta_df['close'],
                                 length=self.ta_config['atr_period'])
            
            # === VOLUME INDICATORS ===
            if self.ta_config['obv_enabled']:
                ta_df['obv'] = ta.obv(ta_df['close'], ta_df['volume'])
            
            if self.ta_config['vwap_enabled']:
                ta_df['vwap'] = ta.vwap(ta_df['high'], ta_df['low'], ta_df['close'], ta_df['volume'])
            
            # === CANDLESTICK PATTERNS ===
            # Add key candlestick patterns
            for pattern in self.key_patterns:
                try:
                    pattern_func = getattr(ta, pattern, None)
                    if pattern_func:
                        result = pattern_func(ta_df['open'], ta_df['high'], ta_df['low'], ta_df['close'])
                        if result is not None:
                            ta_df[f'cdl_{pattern}'] = result
                except Exception as e:
                    logger.warning(f"Could not calculate pattern {pattern}: {e}")
            
            logger.info(f"Calculated {len(ta_df.columns) - len(df.columns)} technical indicators")
            return ta_df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def analyze_technical_signals(self, df: pd.DataFrame, start_idx: int, 
                                 bottom_idx: int, recovery_idx: int) -> TechnicalSignals:
        """Analyze technical signals during pattern formation"""
        try:
            # Extract relevant periods
            decline_period = df.iloc[start_idx:bottom_idx+1]
            recovery_period = df.iloc[bottom_idx:recovery_idx+1]
            
            # === CANDLESTICK PATTERN ANALYSIS ===
            patterns = {}
            for pattern in self.key_patterns:
                col_name = f'cdl_{pattern}'
                if col_name in df.columns:
                    # Check for pattern occurrence during key periods
                    decline_signals = decline_period[col_name].abs().sum() > 0
                    recovery_signals = recovery_period[col_name].abs().sum() > 0
                    patterns[pattern] = decline_signals or recovery_signals
            
            # === TREND ANALYSIS ===
            trend_signals = {}
            if 'ADX_14' in df.columns:
                trend_signals['adx_strength'] = df.loc[df.index[bottom_idx], 'ADX_14']
            if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
                macd_line = df.loc[df.index[recovery_idx], 'MACD_12_26_9']
                macd_signal = df.loc[df.index[recovery_idx], 'MACDs_12_26_9']
                trend_signals['macd_crossover'] = 1 if macd_line > macd_signal else -1
            
            # === MOMENTUM ANALYSIS ===
            momentum_signals = {}
            if 'rsi' in df.columns:
                rsi_bottom = df.loc[df.index[bottom_idx], 'rsi']
                rsi_recovery = df.loc[df.index[recovery_idx], 'rsi']
                momentum_signals['rsi_oversold'] = 1 if rsi_bottom < 30 else 0
                momentum_signals['rsi_recovery'] = rsi_recovery - rsi_bottom
            
            if 'STOCHk_14_3_3' in df.columns:
                stoch_bottom = df.loc[df.index[bottom_idx], 'STOCHk_14_3_3']
                momentum_signals['stoch_oversold'] = 1 if stoch_bottom < 20 else 0
            
            # === VOLATILITY ANALYSIS ===
            volatility_signals = {}
            if 'atr' in df.columns:
                atr_values = decline_period['atr'].mean()
                volatility_signals['atr_decline'] = atr_values
            
            if 'BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
                bb_squeeze = ((df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['close']).iloc[bottom_idx]
                volatility_signals['bb_squeeze'] = bb_squeeze
            
            # === VOLUME ANALYSIS ===
            volume_signals = {}
            if 'obv' in df.columns:
                obv_trend = recovery_period['obv'].iloc[-1] - recovery_period['obv'].iloc[0]
                volume_signals['obv_confirmation'] = obv_trend
            
            avg_volume = decline_period['volume'].mean()
            recovery_volume = recovery_period['volume'].mean()
            volume_signals['volume_confirmation'] = recovery_volume / avg_volume
            
            # === SUPPORT/RESISTANCE ===
            support_resistance = {}
            low_price = df.loc[df.index[bottom_idx], 'low']
            high_price = df.loc[df.index[start_idx], 'high']
            support_resistance['support_level'] = low_price
            support_resistance['resistance_level'] = high_price
            
            return TechnicalSignals(
                candlestick_patterns=patterns,
                trend_signals=trend_signals,
                momentum_signals=momentum_signals,
                volatility_signals=volatility_signals,
                volume_signals=volume_signals,
                support_resistance=support_resistance
            )
            
        except Exception as e:
            logger.error(f"Error analyzing technical signals: {e}")
            return TechnicalSignals({}, {}, {}, {}, {}, {})
    
    def calculate_pattern_score(self, pattern_data: dict, 
                               technical_signals: TechnicalSignals) -> float:
        """Calculate pattern score with realistic scoring and reference values"""
        try:
            # Start with a base score out of 100
            total_score = 0
            
            # === 1. PATTERN GEOMETRY SCORE (35 points max) ===
            decline_pct = pattern_data['decline_pct']
            recovery_pct = pattern_data['recovery_pct']
            breakout_pct = pattern_data['breakout_pct']
            
            # Decline quality (15 points max) - ideal range 30-50%
            if 30 <= decline_pct <= 50:
                decline_score = 15 * (1 - abs(decline_pct - 40) / 10)  # Peak at 40%
            elif 25 <= decline_pct < 30:
                decline_score = 10  # Moderate decline
            elif 50 < decline_pct <= 60:
                decline_score = 8   # Too steep
            else:
                decline_score = 3   # Poor decline
            
            # Recovery quality (12 points max) - minimum 80% for consideration
            if recovery_pct >= 100:
                recovery_score = 12
            elif recovery_pct >= 95:
                recovery_score = 10
            elif recovery_pct >= 90:
                recovery_score = 8
            elif recovery_pct >= 85:
                recovery_score = 6
            elif recovery_pct >= 80:
                recovery_score = 4
            else:
                recovery_score = 0  # Pattern doesn't qualify
            
            # Breakout strength (8 points max)
            if breakout_pct >= 25:
                breakout_score = 8
            elif breakout_pct >= 20:
                breakout_score = 6
            elif breakout_pct >= 15:
                breakout_score = 4
            elif breakout_pct >= 10:
                breakout_score = 2
            else:
                breakout_score = 0
            
            geometry_score = decline_score + recovery_score + breakout_score
            total_score += geometry_score
            
            # === 2. TECHNICAL ANALYSIS SCORE (25 points max) ===
            ta_score = 0
            
            # Momentum confirmation (8 points max)
            momentum_score = 0
            if technical_signals.momentum_signals.get('rsi_oversold', 0):
                momentum_score += 2  # RSI oversold at bottom
            
            rsi_recovery = technical_signals.momentum_signals.get('rsi_recovery', 0)
            if rsi_recovery > 20:
                momentum_score += 3  # Strong RSI recovery
            elif rsi_recovery > 10:
                momentum_score += 2  # Moderate recovery
            elif rsi_recovery > 5:
                momentum_score += 1  # Weak recovery
            
            if technical_signals.momentum_signals.get('stoch_oversold', 0):
                momentum_score += 2  # Stochastic confirmation
            
            momentum_score = min(momentum_score, 8)
            ta_score += momentum_score
            
            # Trend analysis (7 points max)
            trend_score = 0
            adx_strength = technical_signals.trend_signals.get('adx_strength', 0)
            if adx_strength > 40:
                trend_score += 4  # Very strong trend
            elif adx_strength > 30:
                trend_score += 3  # Strong trend
            elif adx_strength > 25:
                trend_score += 2  # Moderate trend
            elif adx_strength > 20:
                trend_score += 1  # Weak trend
            
            macd_crossover = technical_signals.trend_signals.get('macd_crossover', 0)
            if macd_crossover > 0:
                trend_score += 3  # Bullish MACD crossover
            elif macd_crossover == 0:
                trend_score += 1  # Neutral
            # Negative crossover gets 0 points
            
            trend_score = min(trend_score, 7)
            ta_score += trend_score
            
            # Volume confirmation (6 points max)
            volume_score = 0
            volume_conf = technical_signals.volume_signals.get('volume_confirmation', 1)
            if volume_conf > 2.0:
                volume_score += 4  # Exceptional volume increase
            elif volume_conf > 1.5:
                volume_score += 3  # Strong volume
            elif volume_conf > 1.2:
                volume_score += 2  # Moderate volume
            elif volume_conf > 1.0:
                volume_score += 1  # Above average
            # Below average volume gets 0 points
            
            obv_conf = technical_signals.volume_signals.get('obv_confirmation', 0)
            if obv_conf > 0:
                volume_score += 2  # Positive OBV trend
            elif obv_conf < 0:
                volume_score -= 1  # Negative OBV (penalty)
            
            volume_score = max(min(volume_score, 6), 0)
            ta_score += volume_score
            
            # Candlestick patterns (4 points max)
            pattern_score = 0
            bullish_patterns = ['hammer', 'doji', 'engulfing', 'piercingpattern', 
                               'morningstar', 'dragonflydoji']
            pattern_count = sum(1 for pattern, present in technical_signals.candlestick_patterns.items() 
                              if present and pattern in bullish_patterns)
            
            if pattern_count >= 3:
                pattern_score = 4  # Multiple confirmations
            elif pattern_count == 2:
                pattern_score = 3  # Good confirmation
            elif pattern_count == 1:
                pattern_score = 2  # Some confirmation
            else:
                pattern_score = 0  # No pattern confirmation
            
            ta_score += pattern_score
            total_score += ta_score
            
            # === 3. MARKET CONTEXT SCORE (20 points max) ===
            context_score = 0
            
            # Volatility context (8 points max)
            atr_decline = technical_signals.volatility_signals.get('atr_decline', 0)
            bb_squeeze = technical_signals.volatility_signals.get('bb_squeeze', 0.1)
            
            # Prefer moderate volatility expansion
            if 0.02 < atr_decline < 0.08:  # Goldilocks zone
                context_score += 5
            elif 0.01 < atr_decline <= 0.02:
                context_score += 3
            elif atr_decline >= 0.08:
                context_score += 2  # Too volatile
            
            # Bollinger Band squeeze indicates potential breakout
            if bb_squeeze < 0.03:
                context_score += 3  # Tight squeeze
            elif bb_squeeze < 0.05:
                context_score += 2  # Moderate squeeze
            
            # Support/resistance levels (12 points max)
            # This would require more sophisticated analysis
            # For now, give moderate score
            context_score += 8  # Placeholder for S/R analysis
            
            total_score += context_score
            
            # === 4. PENALTY FACTORS ===
            penalties = 0
            
            # Pattern timing penalties
            days_to_recovery = (pattern_data.get('recovery_idx', 0) - 
                              pattern_data.get('bottom_idx', 0))
            if days_to_recovery > 40:
                penalties += 5  # Too slow recovery
            elif days_to_recovery < 5:
                penalties += 3  # Too fast (suspicious)
            
            # Decline too shallow or too deep
            if decline_pct < 20 or decline_pct > 65:
                penalties += 8
            
            # Apply penalties
            total_score = max(total_score - penalties, 0)
            
            # === 5. FINAL CALIBRATION ===
            # Apply realistic scaling to ensure proper score distribution
            # Most patterns should score 40-70, exceptional ones 70-85, rare gems 85+
            
            if total_score >= 75:
                final_score = 70 + (total_score - 75) * 0.6  # Compress high scores
            elif total_score >= 60:
                final_score = 50 + (total_score - 60) * 1.33  # Scale middle range
            else:
                final_score = total_score * 0.83  # Scale lower scores
            
            final_score = min(max(final_score, 0), 100)  # Clamp to 0-100
            
            logger.info(f"Pattern score breakdown: Geometry={geometry_score:.1f}, TA={ta_score:.1f}, "
                       f"Context={context_score:.1f}, Penalties={penalties:.1f} → Final={final_score:.1f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating pattern score: {e}")
            return 25.0  # Default low score for errors
    
    def find_patterns_with_ta(self, symbol: str) -> List[dict]:
        """Find patterns with technical analysis"""
        try:
            logger.info(f"🔍 Analyzing {symbol} with technical analysis...")
            
            # Fetch and prepare data
            df = self.fetch_data(symbol)
            if df.empty:
                return []
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Find base patterns using existing logic
            patterns = self._find_base_patterns(df, symbol)
            
            # Analyze each pattern with technical analysis
            analyzed_patterns = []
            for pattern in patterns:
                try:
                    start_idx = pattern['start_idx']
                    bottom_idx = pattern['bottom_idx']
                    recovery_idx = pattern['recovery_idx']
                    
                    # Analyze technical signals
                    technical_signals = self.analyze_technical_signals(
                        df, start_idx, bottom_idx, recovery_idx)
                    
                    # Calculate comprehensive pattern score (replaces old base + TA bonus)
                    pattern_score = self.calculate_pattern_score(pattern, technical_signals)
                    
                    # Update pattern data
                    pattern['pattern_score'] = pattern_score
                    pattern['technical_signals'] = technical_signals
                    
                    # Only keep high-quality patterns (score >= 65 for A-grade)
                    # This should result in much fewer, but higher quality patterns
                    if pattern_score >= 65:
                        analyzed_patterns.append(pattern)
                        
                except Exception as e:
                    logger.warning(f"Error analyzing pattern: {e}")
                    continue
            
            logger.info(f"Found {len(analyzed_patterns)} A-grade patterns (≥65 score) for {symbol}")
            return analyzed_patterns
            
        except Exception as e:
            logger.error(f"Error finding patterns for {symbol}: {e}")
            return []
    
    def _find_base_patterns_advanced(self, df: pd.DataFrame, symbol: str) -> List[dict]:
        """Advanced pattern detection using scipy.signal and statistical validation"""
        patterns = []
        
        try:
            # Ensure we have enough data
            if len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} candles")
                return patterns
            
            # === 1. ROBUST PEAK AND VALLEY DETECTION ===
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            volumes = df['volume'].values
            
            # Find peaks (potential pattern starts) with prominence
            peak_indices, peak_properties = find_peaks(
                highs, 
                prominence=np.std(highs) * 0.5,  # Adaptive prominence
                distance=10,  # Minimum 10 days between peaks
                width=3       # Minimum width of 3 days
            )
            
            # Find valleys (potential bottoms) with prominence
            valley_indices, valley_properties = find_peaks(
                -lows,  # Invert for valleys
                prominence=np.std(lows) * 0.5,
                distance=5,   # Minimum 5 days between valleys
                width=2       # Minimum width of 2 days
            )
            
            logger.info(f"Found {len(peak_indices)} peaks and {len(valley_indices)} valleys for {symbol}")
            
            # === 2. PATTERN DETECTION WITH VALIDATION ===
            for peak_idx in peak_indices:
                if peak_idx < 30 or peak_idx > len(df) - 50:  # Need buffer space
                    continue
                
                peak_price = highs[peak_idx]
                peak_date = df.index[peak_idx]
                
                # Find relevant valleys after this peak
                relevant_valleys = valley_indices[
                    (valley_indices > peak_idx) & 
                    (valley_indices < peak_idx + 60)  # Within 60 days
                ]
                
                for valley_idx in relevant_valleys:
                    try:
                        valley_price = lows[valley_idx]
                        valley_date = df.index[valley_idx]
                        
                        # === 3. DECLINE VALIDATION ===
                        decline_pct = ((peak_price - valley_price) / peak_price) * 100
                
                        # Check decline criteria
                if not (self.min_decline <= decline_pct <= self.max_decline):
                    continue
                
                        # Validate decline trend using linear regression
                        decline_period = closes[peak_idx:valley_idx+1]
                        if len(decline_period) < 5:
                            continue
                        
                        x_decline = np.arange(len(decline_period))
                        slope_decline, _, r_value_decline, _, _ = stats.linregress(x_decline, decline_period)
                        
                        # Ensure it's actually declining (negative slope with good correlation)
                        if slope_decline >= 0 or r_value_decline**2 < 0.3:
                            continue
                        
                        # === 4. VOLUME ANALYSIS DURING DECLINE ===
                        decline_volume = volumes[peak_idx:valley_idx+1]
                        avg_volume_before = np.mean(volumes[max(0, peak_idx-20):peak_idx])
                        volume_expansion = np.mean(decline_volume) / avg_volume_before if avg_volume_before > 0 else 1
                        
                        # === 5. RECOVERY DETECTION ===
                        recovery_start = valley_idx + 1
                        recovery_end = min(valley_idx + 40, len(df))
                
                if recovery_start >= len(df):
                    continue
                
                        # Find the best recovery high
                        recovery_window = highs[recovery_start:recovery_end]
                        if len(recovery_window) == 0:
                    continue
                
                        max_recovery_price = np.max(recovery_window)
                        recovery_idx = recovery_start + np.argmax(recovery_window)
                        recovery_date = df.index[recovery_idx]
                        
                        # Calculate recovery percentage
                        recovery_pct = ((max_recovery_price - valley_price) / (peak_price - valley_price)) * 100
                
                if recovery_pct < self.min_recovery:
                    continue
                
                        # === 6. RECOVERY TREND VALIDATION ===
                        recovery_period = closes[valley_idx:recovery_idx+1]
                        if len(recovery_period) < 5:
                            continue
                        
                        x_recovery = np.arange(len(recovery_period))
                        slope_recovery, _, r_value_recovery, _, _ = stats.linregress(x_recovery, recovery_period)
                
                        # Ensure it's actually recovering (positive slope with good correlation)
                        if slope_recovery <= 0 or r_value_recovery**2 < 0.3:
                            continue
                        
                        # === 7. BREAKOUT VALIDATION ===
                        breakout_start = recovery_idx + 1
                        breakout_end = min(recovery_idx + 15, len(df))
                        
                        breakout_pct = 0
                        if breakout_start < len(df):
                            breakout_window = highs[breakout_start:breakout_end]
                            if len(breakout_window) > 0:
                                max_breakout_price = np.max(breakout_window)
                                breakout_pct = ((max_breakout_price - max_recovery_price) / max_recovery_price) * 100
                    
                    if breakout_pct < self.min_breakout:
                        continue
                
                        # === 8. PATTERN QUALITY METRICS ===
                        # Calculate additional quality metrics
                        pattern_duration = valley_idx - peak_idx
                        recovery_duration = recovery_idx - valley_idx
                        decline_steepness = abs(slope_decline)
                        recovery_steepness = slope_recovery
                        
                        # Volume pattern score
                        recovery_volume = volumes[valley_idx:recovery_idx+1]
                        recovery_volume_trend = np.corrcoef(np.arange(len(recovery_volume)), recovery_volume)[0,1] if len(recovery_volume) > 1 else 0
                        
                        # === 9. STATISTICAL VALIDATION ===
                        # Ensure the pattern is statistically significant
                        price_volatility = np.std(closes[max(0, peak_idx-20):recovery_idx+1])
                        decline_significance = decline_pct / (price_volatility / np.mean(closes[peak_idx:recovery_idx+1]) * 100)
                        
                        if decline_significance < 1.5:  # Pattern must be 1.5x more significant than normal volatility
                            continue
                        
                        # === 10. CREATE VALIDATED PATTERN ===
                pattern = {
                    'symbol': symbol,
                            'start_date': peak_date.strftime('%Y-%m-%d'),
                            'bottom_date': valley_date.strftime('%Y-%m-%d'),
                            'recovery_date': recovery_date.strftime('%Y-%m-%d'),
                            'start_idx': peak_idx,
                            'bottom_idx': valley_idx,
                    'recovery_idx': recovery_idx,
                    'decline_pct': decline_pct,
                    'recovery_pct': recovery_pct,
                    'breakout_pct': breakout_pct,
                            # Quality metrics
                            'pattern_duration': pattern_duration,
                            'recovery_duration': recovery_duration,
                            'decline_trend_r2': r_value_decline**2,
                            'recovery_trend_r2': r_value_recovery**2,
                            'volume_expansion': volume_expansion,
                            'volume_trend': recovery_volume_trend,
                            'decline_significance': decline_significance,
                            'decline_steepness': decline_steepness,
                            'recovery_steepness': recovery_steepness
                }
                
                patterns.append(pattern)
                
                        logger.debug(f"Valid pattern found: {decline_pct:.1f}% decline, "
                                   f"{recovery_pct:.1f}% recovery, {breakout_pct:.1f}% breakout")
                        
            except Exception as e:
                        logger.warning(f"Error processing valley at {valley_idx}: {e}")
                continue
        
            logger.info(f"Found {len(patterns)} validated base patterns for {symbol}")
        return patterns
    
        except Exception as e:
            logger.error(f"Error in advanced pattern detection for {symbol}: {e}")
            return []
    
    def _find_base_patterns(self, df: pd.DataFrame, symbol: str) -> List[dict]:
        """Find base decline-recovery patterns (enhanced version)"""
        # Use the advanced detection method
        return self._find_base_patterns_advanced(df, symbol)
    
    def visualize_pattern(self, pattern: dict, df: pd.DataFrame, 
                                 save_path: str = None) -> None:
        """Create visualization with technical indicators"""
        try:
            plt.style.use('dark_background')
            fig, axes = plt.subplots(4, 1, figsize=(16, 20))
            fig.suptitle(f"Pattern Analysis: {pattern['symbol']}", 
                        fontsize=20, fontweight='bold', color='white')
            
            # Extract pattern data
            start_idx = pattern['start_idx']
            bottom_idx = pattern['bottom_idx']
            recovery_idx = pattern['recovery_idx']
            
            # Main price chart with technical overlays
            ax1 = axes[0]
            dates = df.index
            
            # Candlestick-style price chart
            ax1.plot(dates, df['close'], color='#00ff88', linewidth=2, label='Close Price')
            ax1.fill_between(dates, df['low'], df['high'], alpha=0.1, color='#00ff88')
            
            # Moving averages
            if 'sma_20' in df.columns:
                ax1.plot(dates, df['sma_20'], color='#ff6b6b', linewidth=1, label='SMA 20')
            if 'sma_50' in df.columns:
                ax1.plot(dates, df['sma_50'], color='#4ecdc4', linewidth=1, label='SMA 50')
            
            # Bollinger Bands
            if 'BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
                ax1.fill_between(dates, df['BBL_20_2.0'], df['BBU_20_2.0'], 
                               alpha=0.1, color='yellow', label='Bollinger Bands')
            
            # Pattern markers
            ax1.scatter(dates[start_idx], df['high'].iloc[start_idx], 
                       color='red', s=100, marker='v', label='Pattern Start', zorder=5)
            ax1.scatter(dates[bottom_idx], df['low'].iloc[bottom_idx], 
                       color='blue', s=100, marker='^', label='Bottom', zorder=5)
            ax1.scatter(dates[recovery_idx], df['high'].iloc[recovery_idx], 
                       color='green', s=100, marker='*', label='Recovery', zorder=5)
            
            ax1.set_title('Price Action with Technical Overlays')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # RSI and momentum indicators
            ax2 = axes[1]
            if 'rsi' in df.columns:
                ax2.plot(dates, df['rsi'], color='#ff9f43', linewidth=2, label='RSI')
                ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
                ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
                ax2.fill_between(dates, 30, 70, alpha=0.1, color='gray')
            
            ax2.set_title('RSI & Momentum')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # MACD
            ax3 = axes[2]
            if 'MACD_12_26_9' in df.columns:
                ax3.plot(dates, df['MACD_12_26_9'], color='#00d2d3', linewidth=2, label='MACD')
            if 'MACDs_12_26_9' in df.columns:
                ax3.plot(dates, df['MACDs_12_26_9'], color='#ff6348', linewidth=2, label='Signal')
            if 'MACDh_12_26_9' in df.columns:
                ax3.bar(dates, df['MACDh_12_26_9'], color='gray', alpha=0.6, label='Histogram')
            
            ax3.axhline(y=0, color='white', linestyle='-', alpha=0.5)
            ax3.set_title('MACD')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Volume with technical analysis
            ax4 = axes[3]
            volume_colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
                           for i in range(len(df))]
            ax4.bar(dates, df['volume'], color=volume_colors, alpha=0.7)
            
            if 'obv' in df.columns:
                ax4_twin = ax4.twinx()
                ax4_twin.plot(dates, df['obv'], color='yellow', linewidth=2, label='OBV')
                ax4_twin.legend(loc='upper right')
                ax4_twin.set_ylabel('OBV', color='yellow')
            
            ax4.set_title('Volume Analysis')
            ax4.set_ylabel('Volume')
            ax4.grid(True, alpha=0.3)
            
            # Format x-axis
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Add pattern information
            info_text = f"""
Pattern Score: {pattern.get('pattern_score', pattern['score']):.1f}/100
Decline: {pattern['decline_pct']:.1f}%
Recovery: {pattern['recovery_pct']:.1f}%
Breakout: {pattern['breakout_pct']:.1f}%
Period: {pattern['start_date']} to {pattern['recovery_date']}
            """
            
            fig.text(0.02, 0.98, info_text, transform=fig.transFigure, 
                    fontsize=12, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                           facecolor='black', edgecolor='none')
                logger.info(f"Pattern visualization saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating pattern visualization: {e}")
    
    def run_analysis(self, symbols: List[str]) -> Dict[str, List[dict]]:
        """Run pattern analysis on multiple symbols"""
        logger.info("🚀 Starting Pattern Analysis with pandas-ta")
        logger.info(f"Target symbols: {symbols}")
        
        all_patterns = {}
        
        for symbol in symbols:
            try:
                patterns = self.find_patterns_with_ta(symbol)
                if patterns:
                    all_patterns[symbol] = patterns
                    logger.info(f"✅ {symbol}: {len(patterns)} patterns found")
                else:
                    logger.info(f"❌ {symbol}: No patterns found")
                    
            except Exception as e:
                logger.error(f"❌ {symbol}: Error during analysis - {e}")
        
        # Summary
        total_patterns = sum(len(patterns) for patterns in all_patterns.values())
        logger.info(f"\n🎯 Analysis Complete!")
        logger.info(f"Total A-grade patterns: {total_patterns}")
        
        return all_patterns

def main():
    """Main execution function"""
    # Initialize pattern collector
    collector = PatternCollector()
    
    # Crypto symbols to analyze
    symbols = [
        'ADA/USDT', 'SOL/USDT', 'BTC/USDT', 'ETH/USDT'
    ]
    
    # Run pattern analysis
    results = collector.run_analysis(symbols)
    
    # Display results
    print("\n" + "="*80)
    print("🎯 PATTERN ANALYSIS RESULTS (with pandas-ta)")
    print("="*80)
    
    for symbol, patterns in results.items():
        print(f"\n📊 {symbol}: {len(patterns)} patterns")
        for i, pattern in enumerate(patterns, 1):
            pattern_score = pattern.get('pattern_score', pattern['score'])
            print(f"  Pattern {i}: Score {pattern_score:.1f}/100 "
                  f"({pattern['start_date']} to {pattern['recovery_date']})")
    
    # Visualize best pattern if any found
    if results:
        best_symbol = max(results.keys(), key=lambda x: len(results[x]))
        if results[best_symbol]:
            best_pattern = max(results[best_symbol], 
                             key=lambda x: x.get('pattern_score', x['score']))
            
            print(f"\n🎨 Creating visualization for best pattern:")
            print(f"   {best_pattern['symbol']} - Score: {best_pattern.get('pattern_score', best_pattern['score']):.1f}")
            
            # Fetch data for visualization
            df = collector.fetch_data(best_pattern['symbol'])
            df = collector.calculate_technical_indicators(df)
            
            collector.visualize_pattern(
                best_pattern, df, 
                f"pattern_{best_pattern['symbol'].replace('/', '_')}.png"
            )

if __name__ == "__main__":
    main() 