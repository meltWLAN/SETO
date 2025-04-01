#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Breakout Volume strategy module for SETO-Versal
Identifies volume-confirmed price breakouts
"""

import logging
import numpy as np
from datetime import datetime

from seto_versal.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class BreakoutVolumeStrategy(BaseStrategy):
    """
    Breakout Volume strategy for detecting volume-confirmed price breakouts
    
    This strategy specifically focuses on significant volume surges alongside
    price breakouts, suggesting strong market participation confirming the move.
    """
    
    def __init__(self, name="breakout_volume", **kwargs):
        """
        Initialize the breakout volume strategy
        
        Args:
            name (str): Strategy name
            **kwargs: Strategy parameters
        """
        kwargs['name'] = name
        kwargs['category'] = 'breakout'
        super().__init__(**kwargs)
        
        # Initialize parameters with defaults
        self.parameters = {
            'lookback_period': kwargs.get('lookback_period', 20),
            'volume_surge_threshold': kwargs.get('volume_surge_threshold', 2.5),  # Minimum volume ratio to average
            'price_breakout_threshold': kwargs.get('price_breakout_threshold', 0.02),  # Min price change %
            'consolidation_days': kwargs.get('consolidation_days', 5),  # Days of consolidation before breakout
            'min_consolidation_tightness': kwargs.get('min_consolidation_tightness', 0.03),  # Max price range % during consolidation
            'confirmation_bars': kwargs.get('confirmation_bars', 1)  # Number of bars to confirm the breakout
        }
        
        logger.info(f"Initialized BreakoutVolumeStrategy: volume_surge={self.parameters['volume_surge_threshold']}")
    
    def generate_signals(self, market_data, **kwargs):
        """
        Generate breakout volume signals based on price and volume patterns
        
        Args:
            market_data (dict): Market data including price and volume history
            **kwargs: Additional parameters
            
        Returns:
            list: List of breakout volume signals
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
        volume_surge_threshold = self.parameters['volume_surge_threshold']
        price_breakout_threshold = self.parameters['price_breakout_threshold']
        consolidation_days = self.parameters['consolidation_days']
        min_consolidation_tightness = self.parameters['min_consolidation_tightness']
        
        # Process each symbol in market data
        for symbol, data in market_data.get('symbols', {}).items():
            try:
                # Get price and volume data
                if 'ohlcv' not in data:
                    continue
                
                ohlcv = data['ohlcv']
                if len(ohlcv) < lookback_period:
                    continue
                
                # Extract price and volume data
                close_prices = [bar['close'] for bar in ohlcv[-lookback_period:]]
                high_prices = [bar['high'] for bar in ohlcv[-lookback_period:]]
                low_prices = [bar['low'] for bar in ohlcv[-lookback_period:]]
                volumes = [bar['volume'] for bar in ohlcv[-lookback_period:]]
                
                # Calculate average volume (excluding current day)
                avg_volume = np.mean(volumes[:-1])
                current_volume = volumes[-1]
                
                # Volume surge check
                volume_ratio = current_volume / avg_volume
                has_volume_surge = volume_ratio > volume_surge_threshold
                
                if not has_volume_surge:
                    continue  # No volume surge, skip further analysis
                
                # Check for price breakout
                current_price = close_prices[-1]
                prev_price = close_prices[-2]
                
                # Calculate recent resistance level (highest high in lookback excluding current day)
                recent_resistance = max(high_prices[-(consolidation_days+1):-1])
                
                # Check for breakout above resistance
                is_price_breakout = current_price > recent_resistance and (current_price - prev_price) / prev_price > price_breakout_threshold
                
                if not is_price_breakout:
                    continue  # No price breakout, skip further analysis
                
                # Check for prior consolidation (tight trading range)
                consolidation_range = ohlcv[-(consolidation_days+1):-1]  # Exclude current breakout day
                consolidation_highs = [bar['high'] for bar in consolidation_range]
                consolidation_lows = [bar['low'] for bar in consolidation_range]
                
                highest_high = max(consolidation_highs)
                lowest_low = min(consolidation_lows)
                
                # Calculate consolidation tightness as a percentage of price
                avg_price = np.mean([bar['close'] for bar in consolidation_range])
                tightness = (highest_high - lowest_low) / avg_price
                
                is_tight_consolidation = tightness < min_consolidation_tightness
                
                if not is_tight_consolidation:
                    continue  # Not a tight consolidation, skip
                
                # All conditions met, generate signal
                
                # Calculate confidence based on volume surge and price breakout strength
                volume_confidence = min(1.0, (volume_ratio - volume_surge_threshold) / 3.0 + 0.6)
                price_change_pct = (current_price - prev_price) / prev_price
                price_confidence = min(1.0, (price_change_pct / price_breakout_threshold) * 0.7)
                
                # Combined confidence
                confidence = (volume_confidence * 0.6) + (price_confidence * 0.4)
                confidence = min(0.95, confidence)  # Cap at 0.95
                
                # Create signal
                signal = {
                    'symbol': symbol,
                    'type': 'buy',
                    'timestamp': datetime.now(),
                    'price': current_price,
                    'confidence': confidence,
                    'parameters': {k: v for k, v in self.parameters.items()},
                    'reason': f"Volume surge ({volume_ratio:.1f}x) with price breakout above {recent_resistance:.2f}",
                    'target_price': current_price * 1.05,  # 5% target
                    'stop_loss': low_prices[-1] * 0.98  # 2% below current low
                }
                
                signals.append(signal)
                logger.debug(f"Volume Breakout signal for {symbol}: confidence={confidence:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol} for volume breakout: {e}")
        
        # Update strategy metrics
        self.signals_generated += len(signals)
        
        return signals
    
    def calculate_confidence(self, signal_data):
        """
        Calculate confidence level for a volume breakout signal
        
        Args:
            signal_data (dict): Signal data including price and volume metrics
            
        Returns:
            float: Confidence level (0.0 to 1.0)
        """
        base_confidence = signal_data.get('confidence', 0.5)
        
        # Additional confidence factors could be included here
        return base_confidence 