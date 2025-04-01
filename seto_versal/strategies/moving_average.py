#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Moving Average strategy module for SETO-Versal
Identifies moving average crossovers and trend confirmations
"""

import logging
import numpy as np
from datetime import datetime

from seto_versal.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class MovingAverageStrategy(BaseStrategy):
    """
    Moving Average strategy for detecting crossovers and trend changes
    
    Uses various combinations of simple and exponential moving averages
    to generate trading signals based on crossovers and trend confirmations
    """
    
    def __init__(self, name="ma_crossover", **kwargs):
        """
        Initialize the moving average strategy
        
        Args:
            name (str): Strategy name
            **kwargs: Strategy parameters
        """
        super().__init__(name, category="trend", **kwargs)
        
        # Initialize parameters with defaults
        self.parameters = {
            'fast_period': kwargs.get('fast_period', 10),
            'slow_period': kwargs.get('slow_period', 30),
            'signal_period': kwargs.get('signal_period', 9),
            'ma_type': kwargs.get('ma_type', 'ema'),  # 'sma', 'ema', 'wma'
            'confirmation_periods': kwargs.get('confirmation_periods', 2),
            'min_trend_strength': kwargs.get('min_trend_strength', 0.01),
            'use_macd': kwargs.get('use_macd', False)
        }
        
        logger.debug(f"Moving Average strategy '{self.name}' initialized with params: {self.parameters}")
    
    def generate_signals(self, market_data, **kwargs):
        """
        Generate moving average crossover signals
        
        Args:
            market_data (dict): Market data including price history
            **kwargs: Additional parameters
            
        Returns:
            list: List of moving average signals
        """
        signals = []
        
        # Validate inputs
        if not market_data or 'history' not in market_data:
            logger.warning(f"Strategy '{self.name}': Invalid market data format")
            return signals
        
        # Extract parameters
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        signal_period = self.parameters['signal_period']
        ma_type = self.parameters['ma_type']
        confirmation_periods = self.parameters['confirmation_periods']
        
        # Process each symbol in market data
        for symbol, data in market_data.get('symbols', {}).items():
            try:
                # Get price data
                if 'ohlcv' not in data:
                    continue
                
                ohlcv = data['ohlcv']
                
                # Need enough data for calculations
                min_required = max(fast_period, slow_period, signal_period) + confirmation_periods + 5
                if len(ohlcv) < min_required:
                    logger.debug(f"Not enough data for {symbol} to calculate moving averages")
                    continue
                
                # Extract close prices
                close_prices = [bar['close'] for bar in ohlcv]
                
                # Calculate moving averages
                if ma_type == 'sma':
                    fast_ma = self._calculate_sma(close_prices, fast_period)
                    slow_ma = self._calculate_sma(close_prices, slow_period)
                elif ma_type == 'ema':
                    fast_ma = self._calculate_ema(close_prices, fast_period)
                    slow_ma = self._calculate_ema(close_prices, slow_period)
                else:  # Default to EMA
                    fast_ma = self._calculate_ema(close_prices, fast_period)
                    slow_ma = self._calculate_ema(close_prices, slow_period)
                
                # Check for crossovers
                if self.parameters['use_macd']:
                    macd, signal, histogram = self._calculate_macd(
                        close_prices, 
                        fast_period, 
                        slow_period, 
                        signal_period
                    )
                    
                    # Check for MACD signal
                    signal_type = self._check_macd_signal(macd, signal, histogram)
                    
                    if signal_type:
                        current_price = close_prices[-1]
                        confidence = self._calculate_macd_confidence(macd, signal, histogram)
                        
                        signal = {
                            'symbol': symbol,
                            'type': signal_type,
                            'timestamp': datetime.now(),
                            'price': current_price,
                            'confidence': confidence,
                            'parameters': {k: v for k, v in self.parameters.items()},
                            'reason': f"MACD {signal_type} signal with {confidence:.2f} confidence"
                        }
                        
                        signals.append(signal)
                        logger.debug(f"MACD signal generated for {symbol}: {signal_type}")
                else:
                    # Check for standard moving average crossover
                    crossover_type = self._check_crossover(fast_ma, slow_ma, confirmation_periods)
                    
                    if crossover_type:
                        current_price = close_prices[-1]
                        confidence = self._calculate_ma_confidence(fast_ma, slow_ma, close_prices)
                        
                        signal = {
                            'symbol': symbol,
                            'type': crossover_type,
                            'timestamp': datetime.now(),
                            'price': current_price,
                            'confidence': confidence,
                            'parameters': {k: v for k, v in self.parameters.items()},
                            'reason': f"Moving average {crossover_type} with {confidence:.2f} confidence"
                        }
                        
                        signals.append(signal)
                        logger.debug(f"MA signal generated for {symbol}: {crossover_type}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol} for moving average strategy: {e}")
        
        # Update strategy metrics
        self.signals_generated += len(signals)
        
        return signals
    
    def _calculate_sma(self, prices, period):
        """
        Calculate Simple Moving Average
        
        Args:
            prices (list): List of prices
            period (int): Moving average period
            
        Returns:
            list: SMA values
        """
        if len(prices) < period:
            return []
        
        sma = []
        for i in range(len(prices) - period + 1):
            window = prices[i:i+period]
            sma.append(sum(window) / period)
        
        return sma
    
    def _calculate_ema(self, prices, period):
        """
        Calculate Exponential Moving Average
        
        Args:
            prices (list): List of prices
            period (int): Moving average period
            
        Returns:
            list: EMA values
        """
        if len(prices) < period:
            return []
        
        # Calculate the multiplier
        multiplier = 2 / (period + 1)
        
        # Calculate the first EMA as SMA
        ema = [sum(prices[:period]) / period]
        
        # Calculate the rest of the EMAs
        for price in prices[period:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])
        
        return ema
    
    def _calculate_macd(self, prices, fast_period, slow_period, signal_period):
        """
        Calculate MACD, Signal Line, and Histogram
        
        Args:
            prices (list): List of prices
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal EMA period
            
        Returns:
            tuple: (MACD, Signal Line, Histogram)
        """
        # Calculate fast and slow EMAs
        fast_ema = self._calculate_ema(prices, fast_period)
        slow_ema = self._calculate_ema(prices, slow_period)
        
        # Adjust lengths
        min_len = min(len(fast_ema), len(slow_ema))
        fast_ema = fast_ema[-min_len:]
        slow_ema = slow_ema[-min_len:]
        
        # Calculate MACD
        macd = [fast - slow for fast, slow in zip(fast_ema, slow_ema)]
        
        # Calculate signal line
        signal = self._calculate_ema(macd, signal_period)
        
        # Adjust lengths again
        min_len = min(len(macd), len(signal))
        macd = macd[-min_len:]
        signal = signal[-min_len:]
        
        # Calculate histogram
        histogram = [m - s for m, s in zip(macd, signal)]
        
        return macd, signal, histogram
    
    def _check_crossover(self, fast_ma, slow_ma, confirmation_periods=1):
        """
        Check for moving average crossovers
        
        Args:
            fast_ma (list): Fast moving average values
            slow_ma (list): Slow moving average values
            confirmation_periods (int): Number of periods to confirm crossover
            
        Returns:
            str: 'buy', 'sell', or None
        """
        if len(fast_ma) < confirmation_periods + 2 or len(slow_ma) < confirmation_periods + 2:
            return None
        
        # Get the relevant segments
        fast_current = fast_ma[-confirmation_periods-1:]
        slow_current = slow_ma[-confirmation_periods-1:]
        
        # Check for golden cross (fast crosses above slow)
        if all(fast_current[i] <= slow_current[i] for i in range(1)) and \
           all(fast_current[i] > slow_current[i] for i in range(1, confirmation_periods+1)):
            return 'buy'
        
        # Check for death cross (fast crosses below slow)
        if all(fast_current[i] >= slow_current[i] for i in range(1)) and \
           all(fast_current[i] < slow_current[i] for i in range(1, confirmation_periods+1)):
            return 'sell'
        
        return None
    
    def _check_macd_signal(self, macd, signal, histogram, confirmation_periods=1):
        """
        Check for MACD signals
        
        Args:
            macd (list): MACD values
            signal (list): Signal line values
            histogram (list): Histogram values
            confirmation_periods (int): Number of periods to confirm signal
            
        Returns:
            str: 'buy', 'sell', or None
        """
        if len(macd) < confirmation_periods + 2 or len(signal) < confirmation_periods + 2:
            return None
        
        # Check for bullish crossover (MACD crosses above signal)
        if macd[-confirmation_periods-1] < signal[-confirmation_periods-1] and \
           all(macd[-i] > signal[-i] for i in range(1, confirmation_periods+1)):
            return 'buy'
        
        # Check for bearish crossover (MACD crosses below signal)
        if macd[-confirmation_periods-1] > signal[-confirmation_periods-1] and \
           all(macd[-i] < signal[-i] for i in range(1, confirmation_periods+1)):
            return 'sell'
        
        # Check for divergence (more complex signals - basic implementation)
        if len(histogram) > 5:
            # Bullish divergence: histogram getting less negative
            if histogram[-3] < histogram[-2] < histogram[-1] < 0:
                return 'buy'
            
            # Bearish divergence: histogram getting less positive
            if histogram[-3] > histogram[-2] > histogram[-1] > 0:
                return 'sell'
        
        return None
    
    def _calculate_ma_confidence(self, fast_ma, slow_ma, prices):
        """
        Calculate confidence level for moving average signal
        
        Args:
            fast_ma (list): Fast moving average values
            slow_ma (list): Slow moving average values
            prices (list): Price data
            
        Returns:
            float: Confidence level (0.0 to 1.0)
        """
        # Get the most recent values
        fast_recent = fast_ma[-3:]
        slow_recent = slow_ma[-3:]
        
        # Calculate the spread between fast and slow MAs
        spread = abs(fast_recent[-1] - slow_recent[-1])
        avg_price = sum(prices[-5:]) / 5
        
        # Normalize the spread as a percentage of price
        normalized_spread = spread / avg_price
        
        # Calculate the slope of the fast MA
        fast_slope = (fast_recent[-1] - fast_recent[0]) / (len(fast_recent) - 1)
        
        # Normalize the slope
        normalized_slope = abs(fast_slope) / avg_price
        
        # Combine factors for confidence score
        confidence = min(1.0, (normalized_spread * 50) + (normalized_slope * 50))
        
        return confidence
    
    def _calculate_macd_confidence(self, macd, signal, histogram):
        """
        Calculate confidence level for MACD signal
        
        Args:
            macd (list): MACD values
            signal (list): Signal line values
            histogram (list): Histogram values
            
        Returns:
            float: Confidence level (0.0 to 1.0)
        """
        # Get the most recent values
        recent_macd = macd[-3:]
        recent_signal = signal[-3:]
        recent_histogram = histogram[-3:]
        
        # Calculate factors for confidence
        
        # Factor 1: Spread between MACD and signal
        spread = abs(recent_macd[-1] - recent_signal[-1])
        max_macd = max(abs(m) for m in recent_macd)
        normalized_spread = min(1.0, spread / (max_macd * 0.5) if max_macd > 0 else 0)
        
        # Factor 2: Histogram momentum
        if len(recent_histogram) >= 3:
            momentum = (recent_histogram[-1] - recent_histogram[-3]) / 2
            max_hist = max(abs(h) for h in recent_histogram)
            normalized_momentum = min(1.0, abs(momentum) / (max_hist * 0.3) if max_hist > 0 else 0)
        else:
            normalized_momentum = 0.0
        
        # Combine factors
        confidence = 0.4 * normalized_spread + 0.6 * normalized_momentum
        
        return min(1.0, confidence)
    
    def optimize_parameters(self, historical_data, target_metric='win_rate'):
        """
        Optimize strategy parameters based on historical data
        
        Args:
            historical_data (dict): Historical market data
            target_metric (str): Metric to optimize for
            
        Returns:
            dict: Optimized parameters
        """
        # This is a placeholder for a parameter optimization method
        # In a real implementation, this would test different parameter combinations
        logger.info(f"Parameter optimization for '{self.name}' not implemented yet")
        
        return self.parameters.copy() 