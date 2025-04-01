#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test cases for the example strategy
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from datetime import datetime, timedelta

from seto_versal.strategy.example_strategy import MovingAverageCrossover


class TestMovingAverageCrossover(unittest.TestCase):
    """Test cases for the MovingAverageCrossover strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.strategy = MovingAverageCrossover(name="test_ma_crossover")
        
        # Mock portfolio for testing
        self.mock_portfolio = MagicMock()
        self.mock_portfolio.get_total_value.return_value = 10000.0
        self.mock_portfolio.get_position.return_value = None
    
    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.strategy.name, "test_ma_crossover")
        self.assertIsNotNone(self.strategy.parameters)
        self.assertEqual(self.strategy.parameters['fast_period'], 10)
        self.assertEqual(self.strategy.parameters['slow_period'], 30)
        self.assertEqual(self.strategy.parameters['position_size'], 0.1)
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters"""
        custom_params = {
            'fast_period': 5,
            'slow_period': 15,
            'use_ema': True,
            'position_size': 0.2
        }
        
        strategy = MovingAverageCrossover(name="custom_ma", parameters=custom_params)
        
        self.assertEqual(strategy.parameters['fast_period'], 5)
        self.assertEqual(strategy.parameters['slow_period'], 15)
        self.assertTrue(strategy.parameters['use_ema'])
        self.assertEqual(strategy.parameters['position_size'], 0.2)
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Create strategy with invalid parameters (fast > slow)
        invalid_params = {
            'fast_period': 30,
            'slow_period': 20
        }
        
        strategy = MovingAverageCrossover(name="invalid_ma", parameters=invalid_params)
        
        # Initialize should validate and correct parameters
        strategy.initialize({})
        
        # Fast period should be adjusted to be less than slow period
        self.assertLess(strategy.parameters['fast_period'], strategy.parameters['slow_period'])
    
    def test_calculate_sma(self):
        """Test SMA calculation"""
        prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        
        # Calculate SMA with period 3
        sma = self.strategy._calculate_sma(prices, 3)
        
        # Expected values: [11, 12, 13, 14, 15, 16, 17, 18, 19]
        expected = [11, 12, 13, 14, 15, 16, 17, 18, 19]
        
        self.assertEqual(len(sma), len(expected))
        for i in range(len(expected)):
            self.assertAlmostEqual(sma[i], expected[i])
    
    def test_calculate_ema(self):
        """Test EMA calculation"""
        prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        
        # Calculate EMA with period 3
        ema = self.strategy._calculate_ema(prices, 3)
        
        # First EMA should be SMA of first 3 prices (11)
        self.assertAlmostEqual(ema[0], 11)
        
        # EMA should be more responsive to recent prices than SMA
        self.assertGreater(ema[-1], 19)  # Last EMA should be higher than SMA
    
    def test_calculate_indicators_insufficient_data(self):
        """Test indicator calculation with insufficient data"""
        # Test with insufficient price history
        data = {'close': 100.0}
        
        indicators = self.strategy.calculate_indicators(data)
        
        self.assertIsNone(indicators['fast_ma'])
        self.assertIsNone(indicators['slow_ma'])
        self.assertFalse(indicators['cross_above'])
        self.assertFalse(indicators['cross_below'])
    
    def test_calculate_indicators_sufficient_data(self):
        """Test indicator calculation with sufficient data"""
        # Add enough data to meet slow period requirement
        for i in range(50):
            self.strategy.price_history.append(100.0 + i)
        
        data = {'close': 150.0}
        
        indicators = self.strategy.calculate_indicators(data)
        
        self.assertIsNotNone(indicators['fast_ma'])
        self.assertIsNotNone(indicators['slow_ma'])
        self.assertNotEqual(indicators['fast_ma'], indicators['slow_ma'])
    
    def test_crossover_detection(self):
        """Test crossover detection"""
        # Create a price pattern that will generate a crossover
        # First, prices trending down (fast MA below slow MA)
        for i in range(40):
            self.strategy.price_history.append(120.0 - i * 0.5)
        
        # Then, prices rapidly turning up (will cause fast MA to cross above slow MA)
        for i in range(10):
            self.strategy.price_history.append(100.0 + i * 2)
        
        data = {'close': 120.0}
        
        indicators = self.strategy.calculate_indicators(data)
        
        # Should detect a crossover
        self.assertTrue(indicators['cross_above'])
        self.assertFalse(indicators['cross_below'])
    
    def test_crossunder_detection(self):
        """Test crossunder detection"""
        # Create a price pattern that will generate a crossunder
        # First, prices trending up (fast MA above slow MA)
        for i in range(40):
            self.strategy.price_history.append(100.0 + i * 0.5)
        
        # Then, prices rapidly turning down (will cause fast MA to cross below slow MA)
        for i in range(10):
            self.strategy.price_history.append(120.0 - i * 2)
        
        data = {'close': 100.0}
        
        indicators = self.strategy.calculate_indicators(data)
        
        # Should detect a crossunder
        self.assertFalse(indicators['cross_above'])
        self.assertTrue(indicators['cross_below'])
    
    def test_calculate_signals_buy(self):
        """Test signal generation for buy signal"""
        indicators = {
            'fast_ma': 105.0,
            'slow_ma': 100.0,
            'cross_above': True,
            'cross_below': False
        }
        
        data = {'close': 110.0}
        
        signals = self.strategy.calculate_signals(data, indicators)
        
        self.assertEqual(signals['signal'], 'buy')
    
    def test_calculate_signals_sell(self):
        """Test signal generation for sell signal"""
        indicators = {
            'fast_ma': 95.0,
            'slow_ma': 100.0,
            'cross_above': False,
            'cross_below': True
        }
        
        data = {'close': 90.0}
        
        signals = self.strategy.calculate_signals(data, indicators)
        
        self.assertEqual(signals['signal'], 'sell')
    
    def test_determine_action_buy(self):
        """Test action determination for buy signal"""
        self.strategy.indicators = {'current_price': 100.0}
        signals = {'signal': 'buy'}
        
        context = {
            'portfolio': self.mock_portfolio,
            'symbol': 'AAPL'
        }
        
        action = self.strategy.determine_action(signals, context)
        
        self.assertIsNotNone(action)
        self.assertEqual(action['action'], 'buy')
        self.assertEqual(action['symbol'], 'AAPL')
        self.assertEqual(action['quantity'], 1000.0)  # $10,000 * 0.1 / $100
        self.assertLess(action['stop_loss'], 100.0)  # Stop loss below current price
        self.assertGreater(action['take_profit'], 100.0)  # Take profit above current price
    
    def test_determine_action_sell(self):
        """Test action determination for sell signal"""
        self.strategy.indicators = {'current_price': 100.0}
        signals = {'signal': 'sell'}
        
        # Create a mock position
        mock_position = {
            'quantity': 100,
            'entry_price': 90.0
        }
        
        # Set active position
        self.strategy.active_position = mock_position
        
        context = {
            'portfolio': self.mock_portfolio,
            'symbol': 'AAPL'
        }
        
        action = self.strategy.determine_action(signals, context)
        
        self.assertIsNotNone(action)
        self.assertEqual(action['action'], 'sell')
        self.assertEqual(action['symbol'], 'AAPL')
        self.assertEqual(action['quantity'], 100)
    
    def test_determine_action_stop_loss(self):
        """Test action determination for stop loss trigger"""
        self.strategy.indicators = {'current_price': 85.0}
        signals = {'signal': 'neutral'}
        
        # Create a mock position with stop loss
        mock_position = {
            'quantity': 100,
            'entry_price': 90.0,
            'stop_loss': 88.0,
            'take_profit': 100.0
        }
        
        # Set active position
        self.strategy.active_position = mock_position
        
        context = {
            'portfolio': self.mock_portfolio,
            'symbol': 'AAPL'
        }
        
        action = self.strategy.determine_action(signals, context)
        
        self.assertIsNotNone(action)
        self.assertEqual(action['action'], 'sell')
        self.assertEqual(action['symbol'], 'AAPL')
        self.assertEqual(action['quantity'], 100)
        self.assertEqual(action['reason'], 'Stop loss triggered')
    
    def test_determine_action_take_profit(self):
        """Test action determination for take profit trigger"""
        self.strategy.indicators = {'current_price': 105.0}
        signals = {'signal': 'neutral'}
        
        # Create a mock position with take profit
        mock_position = {
            'quantity': 100,
            'entry_price': 90.0,
            'stop_loss': 85.0,
            'take_profit': 102.0
        }
        
        # Set active position
        self.strategy.active_position = mock_position
        
        context = {
            'portfolio': self.mock_portfolio,
            'symbol': 'AAPL'
        }
        
        action = self.strategy.determine_action(signals, context)
        
        self.assertIsNotNone(action)
        self.assertEqual(action['action'], 'sell')
        self.assertEqual(action['symbol'], 'AAPL')
        self.assertEqual(action['quantity'], 100)
        self.assertEqual(action['reason'], 'Take profit triggered')
    
    def test_on_bar_integration(self):
        """Test complete on_bar workflow"""
        # Initialize with enough price history
        for i in range(40):
            self.strategy.price_history.append(100.0 - i * 0.5)  # Downtrend
        
        # Now we'll simulate a reversal that should trigger a buy
        data = {'close': 85.0}  # Reversal price point
        
        context = {
            'portfolio': self.mock_portfolio,
            'symbol': 'AAPL'
        }
        
        # Process first bar - should generate indicators but no action yet
        action = self.strategy.on_bar(data, context)
        self.assertIsNone(action)  # No action on first reversal bar
        
        # Process another bar with higher price - should trigger crossover
        data = {'close': 90.0}
        action = self.strategy.on_bar(data, context)
        
        # Should now have a buy action
        self.assertIsNotNone(action)
        self.assertEqual(action['action'], 'buy')
        
        # Process another bar with lower price near stop loss
        self.strategy.indicators = {'current_price': 89.0}
        data = {'close': 89.0}
        action = self.strategy.on_bar(data, context)
        
        # Should be no action (price still above stop loss)
        self.assertIsNone(action)
        
        # Process another bar with price below stop loss
        self.strategy.indicators = {'current_price': 87.0}
        data = {'close': 87.0}
        action = self.strategy.on_bar(data, context)
        
        # Should be a sell action due to stop loss
        self.assertIsNotNone(action)
        self.assertEqual(action['action'], 'sell')
        self.assertEqual(action['reason'], 'Stop loss triggered')


if __name__ == '__main__':
    unittest.main() 