#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example Moving Average Crossover strategy for SETO-Versal.

This module demonstrates how to implement a simple trading strategy
using the Strategy class from the strategy manager.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import base Strategy class
from seto_versal.strategy.manager import Strategy


class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover Strategy.
    
    This strategy generates buy signals when a fast moving average crosses above
    a slow moving average, and sell signals when the fast MA crosses below the slow MA.
    
    Parameters:
        fast_period: Period for the fast moving average
        slow_period: Period for the slow moving average
        use_ema: Boolean to use exponential MA instead of simple MA
        position_size: Position size as a percentage of portfolio value
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
    """
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        """
        Initialize the strategy with parameters.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
        """
        # Initialize base class
        super().__init__(name, parameters)
        
        # Set default parameters if not provided
        if not parameters:
            self.parameters = {
                'fast_period': 10,
                'slow_period': 30,
                'use_ema': False,
                'position_size': 0.1,  # 10% of portfolio
                'stop_loss_pct': 0.02,  # 2% stop loss
                'take_profit_pct': 0.04  # 4% take profit
            }
        
        # Price history
        self.price_history = []
        
        # Position tracking
        self.active_position = None
    
    def initialize(self, context: Dict[str, Any]) -> None:
        """
        Initialize the strategy with trading context.
        
        Args:
            context: Dictionary containing account, broker, etc.
        """
        self.logger.info(f"Initializing {self.name} with parameters: {self.parameters}")
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate strategy parameters."""
        # Ensure fast_period < slow_period
        if self.parameters['fast_period'] >= self.parameters['slow_period']:
            self.logger.warning(f"Invalid MA periods: fast ({self.parameters['fast_period']}) >= slow ({self.parameters['slow_period']})")
            # Adjust parameters to valid values
            self.parameters['fast_period'] = min(self.parameters['fast_period'], self.parameters['slow_period'] - 1)
    
    def calculate_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate moving averages from price data.
        
        Args:
            data: Dictionary with OHLCV data
            
        Returns:
            Dictionary with calculated indicators
        """
        # Extract prices
        close_price = data.get('close', 0)
        
        # Update price history
        self.price_history.append(close_price)
        
        # Keep only necessary history based on slow period
        max_period = max(self.parameters['slow_period'] * 3, 100)  # Keep extra history for stability
        if len(self.price_history) > max_period:
            self.price_history = self.price_history[-max_period:]
        
        # Check if we have enough data
        if len(self.price_history) < self.parameters['slow_period']:
            return {
                'fast_ma': None,
                'slow_ma': None,
                'cross_above': False,
                'cross_below': False
            }
        
        # Calculate moving averages
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        use_ema = self.parameters['use_ema']
        
        if use_ema:
            # Calculate exponential moving averages
            fast_ma = self._calculate_ema(self.price_history, fast_period)
            slow_ma = self._calculate_ema(self.price_history, slow_period)
        else:
            # Calculate simple moving averages
            fast_ma = self._calculate_sma(self.price_history, fast_period)
            slow_ma = self._calculate_sma(self.price_history, slow_period)
        
        # Detect crossovers
        cross_above = False
        cross_below = False
        
        if len(fast_ma) >= 2 and len(slow_ma) >= 2:
            # Current values
            current_fast = fast_ma[-1]
            current_slow = slow_ma[-1]
            
            # Previous values
            prev_fast = fast_ma[-2]
            prev_slow = slow_ma[-2]
            
            # Detect crossovers
            cross_above = prev_fast <= prev_slow and current_fast > current_slow
            cross_below = prev_fast >= prev_slow and current_fast < current_slow
        
        return {
            'fast_ma': fast_ma[-1] if len(fast_ma) > 0 else None,
            'slow_ma': slow_ma[-1] if len(slow_ma) > 0 else None,
            'cross_above': cross_above,
            'cross_below': cross_below,
            'current_price': close_price
        }
    
    def _calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: List of price values
            period: SMA period
            
        Returns:
            List of SMA values
        """
        if len(prices) < period:
            return []
        
        sma = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            sma.append(avg)
        
        return sma
    
    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: List of price values
            period: EMA period
            
        Returns:
            List of EMA values
        """
        if len(prices) < period:
            return []
        
        # Start with SMA for first value
        ema = [sum(prices[:period]) / period]
        
        # Calculate multiplier
        multiplier = 2 / (period + 1)
        
        # Calculate EMA for remaining prices
        for i in range(period, len(prices)):
            ema.append((prices[i] - ema[-1]) * multiplier + ema[-1])
        
        return ema
    
    def calculate_signals(self, data: Dict[str, Any], indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on indicators.
        
        Args:
            data: Dictionary with OHLCV data
            indicators: Dictionary with calculated indicators
            
        Returns:
            Dictionary with trading signals
        """
        if not indicators.get('fast_ma') or not indicators.get('slow_ma'):
            return {'signal': 'neutral'}
        
        # Check for crossovers
        if indicators.get('cross_above'):
            return {'signal': 'buy'}
        elif indicators.get('cross_below'):
            return {'signal': 'sell'}
        
        # No signal
        return {'signal': 'neutral'}
    
    def determine_action(self, signals: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Determine trading action based on signals and context.
        
        Args:
            signals: Dictionary with trading signals
            context: Trading context with portfolio, market data, etc.
            
        Returns:
            Trade action or None if no action
        """
        signal = signals.get('signal', 'neutral')
        current_price = self.indicators.get('current_price')
        
        if not current_price:
            return None
        
        # Get portfolio information
        portfolio = context.get('portfolio')
        if not portfolio:
            self.logger.warning("No portfolio information available")
            return None
        
        # Get current position if any
        symbol = context.get('symbol', 'UNKNOWN')
        current_position = self.active_position or portfolio.get_position(symbol)
        
        # Calculate position size
        portfolio_value = portfolio.get_total_value()
        position_value = portfolio_value * self.parameters['position_size']
        
        # Buy signal
        if signal == 'buy' and not current_position:
            # Calculate quantity
            quantity = position_value / current_price
            
            # Calculate stop loss and take profit prices
            stop_loss_price = current_price * (1 - self.parameters['stop_loss_pct'])
            take_profit_price = current_price * (1 + self.parameters['take_profit_pct'])
            
            # Create trade action
            action = {
                'action': 'buy',
                'symbol': symbol,
                'quantity': quantity,
                'price': current_price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'timestamp': datetime.now().isoformat(),
                'reason': 'Moving average crossover (bullish)'
            }
            
            # Update position tracking
            self.active_position = {
                'symbol': symbol,
                'entry_price': current_price,
                'quantity': quantity,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price
            }
            
            return action
        
        # Sell signal
        elif signal == 'sell' and current_position:
            # Create trade action
            action = {
                'action': 'sell',
                'symbol': symbol,
                'quantity': current_position.get('quantity', 0),
                'price': current_price,
                'timestamp': datetime.now().isoformat(),
                'reason': 'Moving average crossover (bearish)'
            }
            
            # Reset position tracking
            self.active_position = None
            
            return action
        
        # Check stop loss and take profit for existing position
        elif current_position:
            entry_price = current_position.get('entry_price')
            stop_loss = current_position.get('stop_loss')
            take_profit = current_position.get('take_profit')
            
            if entry_price and stop_loss and current_price <= stop_loss:
                # Stop loss triggered
                action = {
                    'action': 'sell',
                    'symbol': symbol,
                    'quantity': current_position.get('quantity', 0),
                    'price': current_price,
                    'timestamp': datetime.now().isoformat(),
                    'reason': 'Stop loss triggered'
                }
                
                # Reset position tracking
                self.active_position = None
                
                return action
            
            elif entry_price and take_profit and current_price >= take_profit:
                # Take profit triggered
                action = {
                    'action': 'sell',
                    'symbol': symbol,
                    'quantity': current_position.get('quantity', 0),
                    'price': current_price,
                    'timestamp': datetime.now().isoformat(),
                    'reason': 'Take profit triggered'
                }
                
                # Reset position tracking
                self.active_position = None
                
                return action
        
        # No action
        return None 