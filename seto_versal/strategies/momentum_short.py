#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Momentum Short strategy module for SETO-Versal
Identifies short-term momentum opportunities
"""

import logging
import numpy as np
from datetime import datetime

from seto_versal.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class MomentumShortStrategy(BaseStrategy):
    """
    Momentum Short strategy for detecting short-term momentum opportunities
    
    This strategy focuses on identifying stocks with strong short-term momentum,
    suitable for quick profit opportunities with relatively short holding periods.
    """
    
    def __init__(self, name="momentum_short", **kwargs):
        """
        Initialize the momentum short strategy
        
        Args:
            name (str): Strategy name
            **kwargs: Strategy parameters
        """
        kwargs['name'] = name
        kwargs['category'] = 'momentum'
        super().__init__(**kwargs)
        
        # Initialize parameters with defaults
        self.parameters = {
            'lookback_period': kwargs.get('lookback_period', 10),
            'momentum_period': kwargs.get('momentum_period', 5),  # Days to calculate momentum
            'min_momentum_threshold': kwargs.get('min_momentum_threshold', 0.03),  # 3% minimum momentum
            'volume_increase_threshold': kwargs.get('volume_increase_threshold', 1.5),  # 1.5x volume increase
            'max_volatility': kwargs.get('max_volatility', 0.025),  # Maximum allowed volatility
            'rsi_lower_bound': kwargs.get('rsi_lower_bound', 50),  # Minimum RSI value
            'rsi_upper_bound': kwargs.get('rsi_upper_bound', 75)   # Maximum RSI value
        }
        
        logger.info(f"Initialized MomentumShortStrategy: period={self.parameters['momentum_period']}, threshold={self.parameters['min_momentum_threshold']}")
    
    def generate_signals(self, market_data, **kwargs):
        """
        Generate short-term momentum signals
        
        Args:
            market_data (dict): Market data including price and volume history
            **kwargs: Additional parameters
            
        Returns:
            list: List of momentum signals
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
        momentum_period = self.parameters['momentum_period']
        min_momentum_threshold = self.parameters['min_momentum_threshold']
        volume_increase_threshold = self.parameters['volume_increase_threshold']
        max_volatility = self.parameters['max_volatility']
        rsi_lower_bound = self.parameters['rsi_lower_bound']
        rsi_upper_bound = self.parameters['rsi_upper_bound']
        
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
                close_prices = np.array([bar['close'] for bar in ohlcv[-lookback_period:]])
                volumes = np.array([bar['volume'] for bar in ohlcv[-lookback_period:]])
                
                # Calculate short-term momentum (current price vs n days ago)
                if len(close_prices) <= momentum_period:
                    continue
                    
                current_price = close_prices[-1]
                momentum_start_price = close_prices[-(momentum_period+1)]
                momentum = (current_price - momentum_start_price) / momentum_start_price
                
                # Check momentum threshold
                if momentum < min_momentum_threshold:
                    continue  # Not enough momentum
                
                # Calculate volume increase
                recent_volume_avg = np.mean(volumes[-momentum_period:])
                prev_volume_avg = np.mean(volumes[-(lookback_period):-momentum_period])
                volume_increase = recent_volume_avg / prev_volume_avg if prev_volume_avg > 0 else 1.0
                
                # Check volume increase
                if volume_increase < volume_increase_threshold:
                    continue  # Not enough volume increase
                
                # Calculate volatility (standard deviation of returns)
                returns = np.diff(close_prices) / close_prices[:-1]
                volatility = np.std(returns)
                
                # Check volatility
                if volatility > max_volatility:
                    continue  # Too volatile
                
                # Calculate RSI
                rsi = self._calculate_rsi(close_prices, 14)
                
                # Check RSI bounds - we want intermediate RSI (not overbought or oversold)
                if rsi is None or rsi < rsi_lower_bound or rsi > rsi_upper_bound:
                    continue  # RSI out of desired range
                
                # All conditions met, generate signal
                
                # Calculate confidence based on momentum strength and volume increase
                momentum_confidence = min(1.0, momentum / (min_momentum_threshold * 3))
                volume_confidence = min(1.0, (volume_increase - volume_increase_threshold) / 3.0 + 0.6)
                
                # Add RSI factor - prefer middle of our range
                rsi_confidence = 1.0 - abs((rsi - ((rsi_lower_bound + rsi_upper_bound) / 2)) / 
                                          ((rsi_upper_bound - rsi_lower_bound) / 2))
                
                # Combined confidence with weights
                confidence = (momentum_confidence * 0.5) + (volume_confidence * 0.3) + (rsi_confidence * 0.2)
                confidence = min(0.95, confidence)  # Cap at 0.95
                
                # Calculate target price (quick profit target based on momentum)
                target_price = current_price * (1 + (momentum * 0.5))  # 50% of the momentum as additional gain
                
                # Create signal
                signal = {
                    'symbol': symbol,
                    'type': 'buy',
                    'timestamp': datetime.now(),
                    'price': current_price,
                    'confidence': confidence,
                    'parameters': {k: v for k, v in self.parameters.items()},
                    'reason': f"Short-term momentum of {momentum:.1%} with {volume_increase:.1f}x volume increase",
                    'target_price': target_price,
                    'stop_loss': current_price * 0.97,  # 3% stop loss
                    'expected_holding_period': '1-5 days'  # Short-term trading
                }
                
                signals.append(signal)
                logger.debug(f"Momentum Short signal for {symbol}: confidence={confidence:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol} for momentum short: {e}")
        
        # Update strategy metrics
        self.signals_generated += len(signals)
        
        return signals
    
    def _calculate_rsi(self, prices, period=14):
        """
        Calculate the Relative Strength Index (RSI)
        
        Args:
            prices (np.array): Array of prices
            period (int): RSI calculation period
            
        Returns:
            float: RSI value (0-100)
        """
        if len(prices) <= period:
            return None
            
        # Calculate price changes
        delta = np.diff(prices)
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi 