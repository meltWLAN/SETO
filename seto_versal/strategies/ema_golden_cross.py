#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EMA Golden Cross strategy module for SETO-Versal
Identifies exponential moving average crossovers (golden cross)
"""

import logging
import numpy as np
from datetime import datetime

from seto_versal.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class EmaGoldenCrossStrategy(BaseStrategy):
    """
    EMA Golden Cross strategy for detecting exponential moving average crossovers
    
    This strategy identifies when a shorter-term EMA crosses above a longer-term EMA,
    indicating a potential uptrend (golden cross).
    """
    
    def __init__(self, name="ema_golden_cross", **kwargs):
        """
        Initialize the EMA Golden Cross strategy
        
        Args:
            name (str): Strategy name
            **kwargs: Strategy parameters
        """
        kwargs['name'] = name
        kwargs['category'] = 'trend'
        super().__init__(**kwargs)
        
        # Initialize parameters with defaults
        self.parameters = {
            'fast_period': kwargs.get('fast_period', 8),
            'slow_period': kwargs.get('slow_period', 21),
            'signal_period': kwargs.get('signal_period', 9),
            'confirmation_bars': kwargs.get('confirmation_bars', 2),
            'min_volume_increase': kwargs.get('min_volume_increase', 1.2),
            'min_price': kwargs.get('min_price', 5.0),
            'filter_sideways_market': kwargs.get('filter_sideways_market', True)
        }
        
        logger.info(f"Initialized EmaGoldenCrossStrategy: fast={self.parameters['fast_period']}, slow={self.parameters['slow_period']}")
    
    def generate_signals(self, market_data, **kwargs):
        """
        Generate golden cross signals based on EMA crossovers
        
        Args:
            market_data (dict): Market data including price and volume history
            **kwargs: Additional parameters
            
        Returns:
            list: List of golden cross signals
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
        min_uptrend_days = self.parameters['confirmation_bars']
        volume_confirm = self.parameters['filter_sideways_market']
        min_price = self.parameters['min_price']
        min_slope = self.parameters['min_slope']
        
        # Need enough data for slow EMA calculation
        min_data_points = max(slow_period * 3, 50)  # At least 3x slow period or 50 days
        
        # Process each symbol in market data
        for symbol, data in market_data.get('symbols', {}).items():
            try:
                # Get price and volume data
                if 'ohlcv' not in data:
                    continue
                
                ohlcv = data['ohlcv']
                if len(ohlcv) < min_data_points:
                    continue
                
                # Extract price and volume data
                close_prices = np.array([bar['close'] for bar in ohlcv])
                volumes = np.array([bar['volume'] for bar in ohlcv])
                
                # Check minimum price threshold
                current_price = close_prices[-1]
                if current_price < min_price:
                    continue  # Price too low, skip
                
                # Calculate EMAs
                fast_ema = self._calculate_ema(close_prices, fast_period)
                slow_ema = self._calculate_ema(close_prices, slow_period)
                
                if len(fast_ema) < 5 or len(slow_ema) < 5:
                    continue  # Not enough EMA data
                
                # Check for crossover (golden cross)
                # A golden cross occurs when fast EMA crosses above slow EMA
                current_fast = fast_ema[-1]
                current_slow = slow_ema[-1]
                prev_fast = fast_ema[-2]
                prev_slow = slow_ema[-2]
                
                # Check if crossover happened recently (last bar)
                is_golden_cross = prev_fast <= prev_slow and current_fast > current_slow
                
                if not is_golden_cross:
                    continue  # No golden cross, skip
                
                # Calculate fast EMA slope for trend strength
                fast_ema_slope = (current_fast - fast_ema[-min_uptrend_days-1]) / min_uptrend_days
                fast_ema_slope_pct = fast_ema_slope / current_fast  # As percentage of price
                
                # Check minimum slope requirement
                if fast_ema_slope_pct < min_slope:
                    continue  # Trend not strong enough
                
                # Check uptrend confirmation (fast EMA trending up for minimum days)
                is_confirmed_uptrend = True
                for i in range(1, min_uptrend_days + 1):
                    if fast_ema[-i] < fast_ema[-(i+1)]:
                        is_confirmed_uptrend = False
                        break
                        
                if not is_confirmed_uptrend:
                    continue  # Uptrend not confirmed
                
                # Check volume confirmation if required
                if volume_confirm:
                    recent_volume_avg = np.mean(volumes[-min_uptrend_days:])
                    prev_volume_avg = np.mean(volumes[-(min_uptrend_days*2):-min_uptrend_days])
                    volume_increase = recent_volume_avg / prev_volume_avg if prev_volume_avg > 0 else 1.0
                    
                    if volume_increase < 1.2:  # Require 20% volume increase
                        continue  # Volume not confirming
                
                # Calculate MACD for additional confirmation
                macd_line = fast_ema - slow_ema
                signal_line = self._calculate_ema(macd_line, signal_period)
                
                # Check if MACD is also showing bullish divergence
                macd_bullish = macd_line[-1] > signal_line[-1] and macd_line[-1] > macd_line[-2]
                
                # All conditions met, generate signal
                
                # Calculate confidence based on multiple factors
                # 1. Strength of crossover (difference between EMAs)
                crossover_strength = (current_fast - current_slow) / current_slow
                crossover_confidence = min(1.0, crossover_strength * 100)  # Normalized to 0-1
                
                # 2. Slope of fast EMA
                slope_confidence = min(1.0, fast_ema_slope_pct / (min_slope * 5))
                
                # 3. MACD confirmation
                macd_confidence = 0.7 if macd_bullish else 0.4
                
                # Combined confidence with weights
                confidence = (crossover_confidence * 0.4) + (slope_confidence * 0.3) + (macd_confidence * 0.3)
                confidence = min(0.95, confidence)  # Cap at 0.95
                
                # Calculate reasonable target based on recent swing highs
                recent_swing_high = np.max(close_prices[-30:])
                target_price = max(current_price * 1.05, recent_swing_high * 1.02)
                
                # Calculate stop loss based on slow EMA
                stop_loss = min(slow_ema[-1] * 0.98, close_prices[-1] * 0.97)
                
                # Create signal
                signal = {
                    'symbol': symbol,
                    'type': 'buy',
                    'timestamp': datetime.now(),
                    'price': current_price,
                    'confidence': confidence,
                    'parameters': {k: v for k, v in self.parameters.items()},
                    'reason': f"EMA Golden Cross: {fast_period}-day EMA crossed above {slow_period}-day EMA",
                    'target_price': target_price,
                    'stop_loss': stop_loss,
                    'expected_holding_period': '1-4 weeks'  # Medium-term trend following
                }
                
                signals.append(signal)
                logger.debug(f"EMA Golden Cross signal for {symbol}: confidence={confidence:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol} for EMA golden cross: {e}")
        
        # Update strategy metrics
        self.signals_generated += len(signals)
        
        return signals
    
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