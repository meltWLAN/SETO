#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Breakout strategy module for SETO-Versal
Identifies price and volume breakouts
"""

import logging
import numpy as np
from datetime import datetime

from seto_versal.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy for detecting price and volume breakouts
    
    Identifies trading opportunities when price breaks above resistance
    with confirmation from increased volume
    """
    
    def __init__(self, name="breakout", **kwargs):
        """
        Initialize the breakout strategy
        
        Args:
            name (str): Strategy name
            **kwargs: Strategy parameters
        """
        super().__init__(name, category="breakout", **kwargs)
        
        # Initialize parameters with defaults
        self.parameters = {
            'lookback_period': kwargs.get('lookback_period', 20),
            'volume_threshold': kwargs.get('volume_threshold', 2.0),
            'price_threshold': kwargs.get('price_threshold', 0.03),
            'use_atr': kwargs.get('use_atr', True),
            'atr_period': kwargs.get('atr_period', 14),
            'atr_multiplier': kwargs.get('atr_multiplier', 1.0),
            'confirmation_periods': kwargs.get('confirmation_periods', 1)
        }
        
        logger.debug(f"Breakout strategy '{self.name}' initialized")
    
    def generate_signals(self, market_data, **kwargs):
        """
        Generate breakout signals based on price and volume patterns
        
        Args:
            market_data (dict): Market data including price and volume history
            **kwargs: Additional parameters
            
        Returns:
            list: List of breakout signals
        """
        signals = []
        
        # Validate inputs
        if not market_data or 'history' not in market_data:
            logger.warning(f"Strategy '{self.name}': Invalid market data format")
            return signals
        
        history = market_data['history']
        if len(history) < self.parameters['lookback_period']:
            logger.debug(f"Strategy '{self.name}': Not enough historical data")
            return signals
        
        # Extract parameters
        lookback_period = self.parameters['lookback_period']
        volume_threshold = self.parameters['volume_threshold']
        price_threshold = self.parameters['price_threshold']
        
        # Process each symbol in market data
        for symbol, data in market_data.get('symbols', {}).items():
            try:
                # Get price and volume data
                if 'ohlcv' not in data:
                    continue
                
                ohlcv = data['ohlcv']
                
                # Calculate price resistance level
                close_prices = [bar['close'] for bar in ohlcv[-lookback_period:]]
                high_prices = [bar['high'] for bar in ohlcv[-lookback_period:]]
                volumes = [bar['volume'] for bar in ohlcv[-lookback_period:]]
                
                if not close_prices or not high_prices or not volumes:
                    continue
                
                # Calculate resistance level (recent highest high)
                resistance = max(high_prices[:-1])  # Exclude current bar
                
                # Get current values
                current_close = close_prices[-1]
                current_high = high_prices[-1]
                current_volume = volumes[-1]
                
                # Calculate average volume
                avg_volume = np.mean(volumes[:-1])  # Exclude current volume
                
                # Check for price breakout
                price_breakout = current_high > resistance
                
                # Calculate price change percentage
                price_change = (current_close - close_prices[-2]) / close_prices[-2]
                significant_move = price_change > price_threshold
                
                # Check for volume surge
                volume_surge = current_volume > (avg_volume * volume_threshold)
                
                # Check breakout conditions
                if price_breakout and volume_surge:
                    # Calculate signal confidence
                    price_confidence = min(1.0, price_change / (price_threshold * 2))
                    volume_confidence = min(1.0, (current_volume / avg_volume) / (volume_threshold * 2))
                    
                    # Overall confidence is average of price and volume confidence
                    confidence = (price_confidence + volume_confidence) / 2
                    
                    # Create signal
                    signal = {
                        'symbol': symbol,
                        'type': 'buy',
                        'timestamp': datetime.now(),
                        'price': current_close,
                        'confidence': confidence,
                        'parameters': {k: v for k, v in self.parameters.items()},
                        'reason': f"Price breakout above {resistance:.2f} with {current_volume / avg_volume:.1f}x volume"
                    }
                    
                    signals.append(signal)
                    
                    logger.debug(f"Breakout signal generated for {symbol}: confidence={confidence:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol} for breakout: {e}")
        
        # Update strategy metrics
        self.signals_generated += len(signals)
        
        return signals
    
    def calculate_atr(self, high_prices, low_prices, close_prices, period=14):
        """
        Calculate Average True Range (ATR)
        
        Args:
            high_prices (list): List of high prices
            low_prices (list): List of low prices
            close_prices (list): List of close prices
            period (int): ATR period
            
        Returns:
            float: ATR value
        """
        if len(high_prices) < period + 1:
            return None
        
        # Calculate true ranges
        true_ranges = []
        
        for i in range(1, len(high_prices)):
            high = high_prices[i]
            low = low_prices[i]
            prev_close = close_prices[i-1]
            
            # True range is the greatest of:
            # - Current high - current low
            # - Absolute value of current high - previous close
            # - Absolute value of current low - previous close
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        # ATR is average of true ranges
        return np.mean(true_ranges[-period:])
    
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