#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MACD Divergence strategy module for SETO-Versal
Identifies MACD divergence patterns for reversal signals
"""

import logging
import numpy as np
from datetime import datetime

from seto_versal.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class MacdDivergenceStrategy(BaseStrategy):
    """
    MACD Divergence strategy for detecting price-indicator divergences
    
    This strategy identifies bullish and bearish divergences between price and MACD,
    which can signal potential trend reversals.
    """
    
    def __init__(self, name="macd_divergence", **kwargs):
        """
        Initialize the MACD Divergence strategy
        
        Args:
            name (str): Strategy name
            **kwargs: Strategy parameters
        """
        kwargs['name'] = name
        kwargs['category'] = 'reversal'
        super().__init__(**kwargs)
        
        # Initialize parameters with defaults
        self.parameters = {
            'fast_period': kwargs.get('fast_period', 12),  # Fast EMA period
            'slow_period': kwargs.get('slow_period', 26),  # Slow EMA period
            'signal_period': kwargs.get('signal_period', 9),  # Signal line period
            'lookback_period': kwargs.get('lookback_period', 60),  # History to analyze
            'min_price': kwargs.get('min_price', 5.0),  # Minimum price to consider
            'divergence_window': kwargs.get('divergence_window', 20),  # Window to look for divergence
            'confirmation_bars': kwargs.get('confirmation_bars', 2),  # Bars to confirm divergence
            'filter_consolidation': kwargs.get('filter_consolidation', True)  # Filter out consolidation patterns
        }
        
        logger.info(f"Initialized MacdDivergenceStrategy: fast={self.parameters['fast_period']}, slow={self.parameters['slow_period']}")
    
    def generate_signals(self, market_data, **kwargs):
        """
        Generate signals based on MACD divergence patterns
        
        Args:
            market_data (dict): Market data including price and volume history
            **kwargs: Additional parameters
            
        Returns:
            list: List of MACD divergence signals
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
        lookback_period = self.parameters['lookback_period']
        min_price = self.parameters['min_price']
        divergence_window = self.parameters['divergence_window']
        confirmation_bars = self.parameters['confirmation_bars']
        filter_consolidation = self.parameters['filter_consolidation']
        
        # Need enough data for calculations
        min_data_points = max(lookback_period, slow_period * 3)
        
        # Process each symbol in market data
        for symbol, data in market_data.get('symbols', {}).items():
            try:
                # Get price and volume data
                if 'ohlcv' not in data:
                    continue
                
                ohlcv = data['ohlcv']
                if len(ohlcv) < min_data_points:
                    continue
                
                # Extract price data
                close_prices = np.array([bar['close'] for bar in ohlcv[-lookback_period:]])
                low_prices = np.array([bar['low'] for bar in ohlcv[-lookback_period:]])
                high_prices = np.array([bar['high'] for bar in ohlcv[-lookback_period:]])
                
                # Check minimum price
                current_price = close_prices[-1]
                if current_price < min_price:
                    continue  # Price too low, skip
                
                # Calculate MACD and signal line
                macd_line, signal_line, histogram = self._calculate_macd(
                    close_prices, fast_period, slow_period, signal_period)
                
                if len(macd_line) < divergence_window:
                    continue  # Not enough MACD data
                
                # Look for bullish divergence (price lower low, MACD higher low)
                bullish_divergence = self._check_bullish_divergence(
                    close_prices, low_prices, macd_line, divergence_window, confirmation_bars)
                
                # Look for bearish divergence (price higher high, MACD lower high)
                bearish_divergence = self._check_bearish_divergence(
                    close_prices, high_prices, macd_line, divergence_window, confirmation_bars)
                
                # Filter out consolidation patterns if required
                if filter_consolidation:
                    # Check if price is in a tight range (consolidation)
                    recent_prices = close_prices[-divergence_window:]
                    price_range = (np.max(recent_prices) - np.min(recent_prices)) / np.mean(recent_prices)
                    
                    is_consolidation = price_range < 0.05  # Less than 5% range
                    
                    if is_consolidation:
                        bullish_divergence = False
                        bearish_divergence = False
                
                # Check if MACD crossed signal line (confirmation)
                macd_cross_up = (macd_line[-2] < signal_line[-2]) and (macd_line[-1] > signal_line[-1])
                macd_cross_down = (macd_line[-2] > signal_line[-2]) and (macd_line[-1] < signal_line[-1])
                
                # Strengthen bullish divergence if MACD crosses up
                if bullish_divergence and macd_cross_up:
                    bullish_divergence = 2  # Stronger signal
                
                # Strengthen bearish divergence if MACD crosses down
                if bearish_divergence and macd_cross_down:
                    bearish_divergence = 2  # Stronger signal
                
                # Generate signals based on divergences
                if bullish_divergence:
                    # Calculate confidence
                    confidence_base = 0.7 if bullish_divergence == 2 else 0.5
                    
                    # Adjust based on histogram direction
                    histogram_rising = histogram[-1] > histogram[-2]
                    confidence = confidence_base + (0.1 if histogram_rising else -0.1)
                    
                    # Adjust based on how oversold we are
                    recent_lows = np.min(low_prices[-divergence_window:])
                    current_to_low_ratio = (current_price - recent_lows) / recent_lows
                    if current_to_low_ratio < 0.02:  # Very close to recent lows
                        confidence += 0.1
                    
                    confidence = min(0.95, max(0.2, confidence))  # Keep in range
                    
                    # Calculate target and stop
                    recent_swing_high = np.max(high_prices[-30:])
                    target_price = max(current_price * 1.05, (current_price + recent_swing_high) / 2)
                    stop_loss = min(recent_lows * 0.99, current_price * 0.97)
                    
                    # Create signal
                    signal = {
                        'symbol': symbol,
                        'type': "buy",
                        'timestamp': datetime.now(),
                        'price': current_price,
                        'confidence': confidence,
                        'parameters': {k: v for k, v in self.parameters.items()},
                        'reason': f"Bullish MACD divergence with {'MACD crossover' if bullish_divergence == 2 else 'positive divergence'}",
                        'target_price': target_price,
                        'stop_loss': stop_loss,
                        'expected_holding_period': '1-3 weeks'
                    }
                    
                    signals.append(signal)
                    logger.debug(f"Bullish MACD Divergence signal for {symbol}: confidence={confidence:.2f}")
                
                elif bearish_divergence:
                    # Calculate confidence
                    confidence_base = 0.7 if bearish_divergence == 2 else 0.5
                    
                    # Adjust based on histogram direction
                    histogram_falling = histogram[-1] < histogram[-2]
                    confidence = confidence_base + (0.1 if histogram_falling else -0.1)
                    
                    # Adjust based on how overbought we are
                    recent_highs = np.max(high_prices[-divergence_window:])
                    high_to_current_ratio = (recent_highs - current_price) / current_price
                    if high_to_current_ratio < 0.02:  # Very close to recent highs
                        confidence += 0.1
                    
                    confidence = min(0.95, max(0.2, confidence))  # Keep in range
                    
                    # Calculate target and stop
                    recent_swing_low = np.min(low_prices[-30:])
                    target_price = min(current_price * 0.95, (current_price + recent_swing_low) / 2)
                    stop_loss = max(recent_highs * 1.01, current_price * 1.03)
                    
                    # Create signal
                    signal = {
                        'symbol': symbol,
                        'type': "sell",
                        'timestamp': datetime.now(),
                        'price': current_price,
                        'confidence': confidence,
                        'parameters': {k: v for k, v in self.parameters.items()},
                        'reason': f"Bearish MACD divergence with {'MACD crossover' if bearish_divergence == 2 else 'negative divergence'}",
                        'target_price': target_price,
                        'stop_loss': stop_loss,
                        'expected_holding_period': '1-3 weeks'
                    }
                    
                    signals.append(signal)
                    logger.debug(f"Bearish MACD Divergence signal for {symbol}: confidence={confidence:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol} for MACD divergence: {e}")
        
        # Update strategy metrics
        self.signals_generated += len(signals)
        
        return signals
    
    def _calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate MACD, signal line, and histogram
        
        Args:
            prices (np.array): Array of prices
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
            
        Returns:
            tuple: (MACD line, signal line, histogram)
        """
        if len(prices) < slow_period * 2:
            return np.array([]), np.array([]), np.array([])
            
        # Calculate EMAs
        fast_ema = self._calculate_ema(prices, fast_period)
        slow_ema = self._calculate_ema(prices, slow_period)
        
        # Make sure arrays are the same length
        min_length = min(len(fast_ema), len(slow_ema))
        fast_ema = fast_ema[-min_length:]
        slow_ema = slow_ema[-min_length:]
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD line)
        signal_line = self._calculate_ema(macd_line, signal_period)
        
        # Make sure arrays are the same length again
        min_length = min(len(macd_line), len(signal_line))
        macd_line = macd_line[-min_length:]
        signal_line = signal_line[-min_length:]
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_ema(self, prices, period):
        """
        Calculate Exponential Moving Average
        
        Args:
            prices (np.array): Array of prices
            period (int): EMA period
            
        Returns:
            np.array: EMA values
        """
        if len(prices) < period * 2:
            return np.array([])
            
        # Calculate initial SMA for seeding the EMA
        sma = np.mean(prices[:period])
        
        # Calculate EMA
        multiplier = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[period-1] = sma
        
        # Calculate EMA for the rest of the data
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
            
        # Return only the valid part of the EMA
        return ema[period-1:]
    
    def _check_bullish_divergence(self, prices, lows, macd, window, confirmation=2):
        """
        Check for bullish divergence (price lower low, MACD higher low)
        
        Args:
            prices (np.array): Close prices
            lows (np.array): Low prices
            macd (np.array): MACD line values
            window (int): Window to look for divergence
            confirmation (int): Bars to confirm the pattern
            
        Returns:
            bool: True if bullish divergence detected
        """
        if len(prices) < window or len(macd) < window:
            return False
            
        # Look at most recent window of data
        recent_prices = lows[-window:]
        recent_macd = macd[-window:]
        
        # Find the two most recent lows in price
        # First find all local minima
        price_minima_idx = []
        for i in range(1, len(recent_prices) - 1):
            if recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                price_minima_idx.append(i)
        
        # Need at least two lows to compare
        if len(price_minima_idx) < 2:
            return False
            
        # Get two most recent lows
        recent_low_idx = price_minima_idx[-1]
        previous_low_idx = price_minima_idx[-2]
        
        # Check if price made a lower low
        if recent_prices[recent_low_idx] >= recent_prices[previous_low_idx]:
            return False  # No lower low in price
            
        # Find MACD values at these price low points
        macd_at_recent_low = recent_macd[recent_low_idx]
        macd_at_previous_low = recent_macd[previous_low_idx]
        
        # Check for bullish divergence (MACD higher low)
        if macd_at_recent_low <= macd_at_previous_low:
            return False  # No higher low in MACD
            
        # Check for confirmation (price moving up after the divergence)
        if len(recent_prices) - 1 - recent_low_idx < confirmation:
            return False  # Not enough bars after the recent low
            
        # Check if price has started moving up after the low
        for i in range(1, confirmation + 1):
            if recent_prices[recent_low_idx + i] < recent_prices[recent_low_idx]:
                return False  # Price not confirming upward movement
        
        # Check current MACD slope (should be positive or flat)
        if macd[-1] < macd[-2]:
            return False  # MACD currently falling
        
        return True  # Bullish divergence confirmed
    
    def _check_bearish_divergence(self, prices, highs, macd, window, confirmation=2):
        """
        Check for bearish divergence (price higher high, MACD lower high)
        
        Args:
            prices (np.array): Close prices
            highs (np.array): High prices
            macd (np.array): MACD line values
            window (int): Window to look for divergence
            confirmation (int): Bars to confirm the pattern
            
        Returns:
            bool: True if bearish divergence detected
        """
        if len(prices) < window or len(macd) < window:
            return False
            
        # Look at most recent window of data
        recent_prices = highs[-window:]
        recent_macd = macd[-window:]
        
        # Find the two most recent highs in price
        # First find all local maxima
        price_maxima_idx = []
        for i in range(1, len(recent_prices) - 1):
            if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                price_maxima_idx.append(i)
        
        # Need at least two highs to compare
        if len(price_maxima_idx) < 2:
            return False
            
        # Get two most recent highs
        recent_high_idx = price_maxima_idx[-1]
        previous_high_idx = price_maxima_idx[-2]
        
        # Check if price made a higher high
        if recent_prices[recent_high_idx] <= recent_prices[previous_high_idx]:
            return False  # No higher high in price
            
        # Find MACD values at these price high points
        macd_at_recent_high = recent_macd[recent_high_idx]
        macd_at_previous_high = recent_macd[previous_high_idx]
        
        # Check for bearish divergence (MACD lower high)
        if macd_at_recent_high >= macd_at_previous_high:
            return False  # No lower high in MACD
            
        # Check for confirmation (price moving down after the divergence)
        if len(recent_prices) - 1 - recent_high_idx < confirmation:
            return False  # Not enough bars after the recent high
            
        # Check if price has started moving down after the high
        for i in range(1, confirmation + 1):
            if recent_prices[recent_high_idx + i] > recent_prices[recent_high_idx]:
                return False  # Price not confirming downward movement
        
        # Check current MACD slope (should be negative or flat)
        if macd[-1] > macd[-2]:
            return False  # MACD currently rising
        
        return True  # Bearish divergence confirmed 