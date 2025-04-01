#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Momentum strategy module for SETO-Versal
Identifies stocks with strong price momentum
"""

import logging
import numpy as np
from datetime import datetime

from seto_versal.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy that identifies stocks with strong price momentum
    
    Generates signals based on price momentum across different timeframes,
    with optional RSI filtering for better entry timing
    """
    
    def __init__(self, name="momentum", **kwargs):
        """
        Initialize the momentum strategy
        
        Args:
            name (str): Strategy name
            **kwargs: Strategy parameters
        """
        super().__init__(name, category="momentum", **kwargs)
        
        # Initialize parameters with defaults
        self.parameters = {
            'timeframe': kwargs.get('timeframe', 'medium'),
            'momentum_period': kwargs.get('momentum_period', 10),
            'threshold': kwargs.get('threshold', 0.05),
            'rsi_filter': kwargs.get('rsi_filter', True),
            'rsi_lower': kwargs.get('rsi_lower', 30),
            'rsi_upper': kwargs.get('rsi_upper', 70),
            'volume_confirmation': kwargs.get('volume_confirmation', True),
            'use_multi_timeframe': kwargs.get('use_multi_timeframe', False)
        }
        
        # Adjust period based on timeframe
        if self.parameters['timeframe'] == 'short':
            self.parameters['momentum_period'] = min(self.parameters['momentum_period'], 5)
        elif self.parameters['timeframe'] == 'long':
            self.parameters['momentum_period'] = max(self.parameters['momentum_period'], 20)
        
        logger.debug(f"Momentum strategy '{self.name}' initialized with params: {self.parameters}")
    
    def generate_signals(self, market_data, **kwargs):
        """
        Generate momentum signals based on price data
        
        Args:
            market_data (dict): Market data including price history
            **kwargs: Additional parameters
            
        Returns:
            list: List of momentum signals
        """
        signals = []
        
        # Validate inputs
        if not market_data or 'history' not in market_data:
            logger.warning(f"Strategy '{self.name}': Invalid market data format")
            return signals
        
        # Extract parameters
        momentum_period = self.parameters['momentum_period']
        threshold = self.parameters['threshold']
        rsi_filter = self.parameters['rsi_filter']
        rsi_upper = self.parameters['rsi_upper']
        rsi_lower = self.parameters['rsi_lower']
        volume_confirmation = self.parameters['volume_confirmation']
        
        # Process each symbol in market data
        for symbol, data in market_data.get('symbols', {}).items():
            try:
                # Get price and volume data
                if 'ohlcv' not in data:
                    continue
                
                ohlcv = data['ohlcv']
                
                # Need enough data for calculations
                if len(ohlcv) < momentum_period + 1:
                    logger.debug(f"Not enough data for {symbol} to calculate momentum")
                    continue
                
                # Extract needed data
                close_prices = [bar['close'] for bar in ohlcv]
                volumes = [bar.get('volume', 0) for bar in ohlcv]
                
                # Calculate momentum
                current_price = close_prices[-1]
                reference_price = close_prices[-(momentum_period+1)]
                momentum = current_price / reference_price - 1
                
                # Get RSI if available
                rsi_value = None
                if 'indicators' in data and 'rsi' in data['indicators']:
                    rsi_value = data['indicators']['rsi'][-1]
                
                # Check momentum threshold
                if abs(momentum) > threshold:
                    # Determine direction
                    signal_type = 'buy' if momentum > 0 else 'sell'
                    
                    # Apply RSI filter if enabled and available
                    rsi_filter_passed = True
                    if rsi_filter and rsi_value is not None:
                        if signal_type == 'buy' and rsi_value > rsi_upper:
                            rsi_filter_passed = False  # Overbought, skip buy
                        elif signal_type == 'sell' and rsi_value < rsi_lower:
                            rsi_filter_passed = False  # Oversold, skip sell
                    
                    # Apply volume confirmation if enabled
                    volume_confirmed = True
                    if volume_confirmation and len(volumes) > 5:
                        avg_volume = sum(volumes[-6:-1]) / 5  # Last 5 periods excluding current
                        current_volume = volumes[-1]
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                        
                        # For strong momentum, we want to see increasing volume
                        if signal_type == 'buy' and volume_ratio < 1.0:
                            volume_confirmed = False
                        elif signal_type == 'sell' and volume_ratio < 1.0:
                            volume_confirmed = False
                    
                    # Generate signal if all filters pass
                    if rsi_filter_passed and volume_confirmed:
                        # Calculate confidence
                        confidence = self._calculate_momentum_confidence(momentum, threshold, rsi_value)
                        
                        # Create signal
                        signal = {
                            'symbol': symbol,
                            'type': signal_type,
                            'timestamp': datetime.now(),
                            'price': current_price,
                            'confidence': confidence,
                            'parameters': {k: v for k, v in self.parameters.items()},
                            'reason': self._generate_signal_reason(
                                momentum, 
                                momentum_period, 
                                rsi_value, 
                                volume_ratio if volume_confirmation else None
                            )
                        }
                        
                        signals.append(signal)
                        logger.debug(f"Momentum signal generated for {symbol}: {signal_type}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol} for momentum: {e}")
        
        # Update strategy metrics
        self.signals_generated += len(signals)
        
        return signals
    
    def _calculate_momentum_confidence(self, momentum, threshold, rsi=None):
        """
        Calculate confidence level for a momentum signal
        
        Args:
            momentum (float): Calculated momentum value
            threshold (float): Momentum threshold
            rsi (float, optional): RSI value if available
            
        Returns:
            float: Confidence level (0.0 to 1.0)
        """
        # Base confidence on strength relative to threshold
        confidence = min(1.0, abs(momentum) / (threshold * 2))
        
        # Adjust confidence based on RSI if available
        if rsi is not None:
            if momentum > 0:  # Buy signal
                # Ideal RSI for buy is between lower bound and 50
                if rsi < self.parameters['rsi_lower']:
                    # Oversold, good entry (boost confidence)
                    rsi_factor = 1.2
                elif rsi < 50:
                    # Moderate RSI, good for trend (slight boost)
                    rsi_factor = 1.1
                elif rsi < self.parameters['rsi_upper']:
                    # Above 50 but not overbought (neutral)
                    rsi_factor = 1.0
                else:
                    # Overbought, less ideal (reduce confidence)
                    rsi_factor = 0.8
            else:  # Sell signal
                # Ideal RSI for sell is between 50 and upper bound
                if rsi > self.parameters['rsi_upper']:
                    # Overbought, good for selling (boost confidence)
                    rsi_factor = 1.2
                elif rsi > 50:
                    # Moderate high RSI, good for sell (slight boost)
                    rsi_factor = 1.1
                elif rsi > self.parameters['rsi_lower']:
                    # Below 50 but not oversold (neutral)
                    rsi_factor = 1.0
                else:
                    # Oversold, less ideal for selling (reduce confidence)
                    rsi_factor = 0.8
            
            # Apply RSI adjustment
            confidence = min(1.0, confidence * rsi_factor)
        
        return confidence
    
    def _generate_signal_reason(self, momentum, period, rsi=None, volume_ratio=None):
        """
        Generate descriptive reason for the momentum signal
        
        Args:
            momentum (float): Calculated momentum value
            period (int): Momentum calculation period
            rsi (float, optional): RSI value if available
            volume_ratio (float, optional): Volume ratio if available
            
        Returns:
            str: Descriptive reason for the signal
        """
        # Base reason on momentum
        momentum_desc = "Strong"
        if abs(momentum) > self.parameters['threshold'] * 3:
            momentum_desc = "Very strong"
        elif abs(momentum) < self.parameters['threshold'] * 1.5:
            momentum_desc = "Moderate"
        
        direction = "bullish" if momentum > 0 else "bearish"
        timeframe = self.parameters['timeframe']
        
        reason = f"{momentum_desc} {direction} momentum ({momentum:.1%} over {period} periods)"
        
        # Add RSI context if available
        if rsi is not None:
            if momentum > 0:  # Bullish
                if rsi < self.parameters['rsi_lower']:
                    reason += f" with oversold RSI ({rsi:.0f})"
                elif rsi > self.parameters['rsi_upper']:
                    reason += f" with overbought RSI ({rsi:.0f})"
                else:
                    reason += f" with neutral RSI ({rsi:.0f})"
            else:  # Bearish
                if rsi > self.parameters['rsi_upper']:
                    reason += f" with overbought RSI ({rsi:.0f})"
                elif rsi < self.parameters['rsi_lower']:
                    reason += f" with oversold RSI ({rsi:.0f})"
                else:
                    reason += f" with neutral RSI ({rsi:.0f})"
        
        # Add volume context if available
        if volume_ratio is not None:
            if volume_ratio > 1.5:
                reason += f" on strong volume ({volume_ratio:.1f}x average)"
            elif volume_ratio > 1.0:
                reason += f" on above average volume ({volume_ratio:.1f}x)"
            else:
                reason += f" on below average volume ({volume_ratio:.1f}x)"
        
        return reason
    
    def calculate_multi_timeframe_momentum(self, prices):
        """
        Calculate momentum across multiple timeframes
        
        Args:
            prices (list): List of closing prices
            
        Returns:
            dict: Momentum values for different timeframes
        """
        if len(prices) < 30:
            return {}
        
        try:
            # Calculate short, medium and long-term momentum
            short_momentum = prices[-1] / prices[-6] - 1  # 5 periods
            medium_momentum = prices[-1] / prices[-11] - 1  # 10 periods
            long_momentum = prices[-1] / prices[-21] - 1  # 20 periods
            
            # Calculate weighted score
            score = 0.5 * short_momentum + 0.3 * medium_momentum + 0.2 * long_momentum
            
            return {
                'short': short_momentum,
                'medium': medium_momentum,
                'long': long_momentum,
                'weighted_score': score
            }
            
        except Exception as e:
            logger.warning(f"Error calculating multi-timeframe momentum: {e}")
            return {}