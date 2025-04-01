#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Strategy

A simple strategy for testing the strategy loading mechanism.
"""

import logging
import random
from datetime import datetime
from typing import Dict, List, Any

from seto_versal.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class TestStrategy(BaseStrategy):
    """
    Test Strategy - Generates random signals for testing
    
    This strategy:
    1. Generates random buy/sell signals with configurable probabilities
    2. Is intended for testing the strategy loading mechanism
    3. Should not be used for actual trading
    """
    
    def __init__(
        self,
        signal_probability: float = 0.3,
        buy_bias: float = 0.6,
        min_confidence: float = 0.3,
        max_confidence: float = 0.8,
        **kwargs
    ):
        """
        Initialize Test Strategy
        
        Args:
            signal_probability: Probability of generating a signal (0-1)
            buy_bias: Probability of generating a buy signal vs. sell (0-1)
            min_confidence: Minimum confidence level for signals
            max_confidence: Maximum confidence level for signals
        """
        super().__init__(**kwargs)
        self.signal_probability = signal_probability
        self.buy_bias = buy_bias
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        
        logger.info(
            f"Initialized TestStrategy: prob={signal_probability}, "
            f"buy_bias={buy_bias}"
        )
    
    def generate_signals(
        self, 
        market_data: Dict[str, Dict[datetime, Dict[str, float]]], 
        positions: Dict[str, Dict[str, Any]] = None,
        market_state: Dict[str, Any] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate random trading signals for testing
        
        Args:
            market_data: Dictionary of market data by symbol
            positions: Current positions
            market_state: Current market state information
            
        Returns:
            List of trading signals as dictionaries
        """
        signals = []
        
        try:
            # Early exit if no market data
            if not market_data:
                return signals
                
            # Process each symbol
            for symbol, data in market_data.items():
                # Skip with probability (1 - signal_probability)
                if random.random() > self.signal_probability:
                    continue
                
                # Skip if not enough data
                if not data:
                    continue
                    
                # Get current price
                dates = sorted(data.keys())
                if not dates:
                    continue
                    
                current_date = dates[-1]
                current_price = data[current_date].get('close', 0)
                
                if current_price <= 0:
                    continue
                
                # Determine signal type (buy or sell)
                is_buy = random.random() < self.buy_bias
                
                # Generate random confidence
                confidence = random.uniform(self.min_confidence, self.max_confidence)
                
                # Calculate target price and stop loss
                price_volatility = current_price * 0.05  # Assume 5% volatility
                if is_buy:
                    target_price = current_price * (1 + random.uniform(0.05, 0.2))
                    stop_loss_price = current_price * (1 - random.uniform(0.03, 0.1))
                    
                    signals.append({
                        'symbol': symbol,
                        'signal_type': 'buy',
                        'price': current_price,
                        'quantity': random.randint(1, 10),
                        'order_type': 'market',
                        'confidence': confidence,
                        'target_price': target_price,
                        'stop_loss_price': stop_loss_price,
                        'reason': "Test buy signal",
                        'metadata': {
                            'strategy': 'test_strategy',
                            'random_value': random.random()
                        }
                    })
                else:
                    target_price = current_price * (1 - random.uniform(0.05, 0.2))
                    stop_loss_price = current_price * (1 + random.uniform(0.03, 0.1))
                    
                    signals.append({
                        'symbol': symbol,
                        'signal_type': 'sell',
                        'price': current_price,
                        'quantity': random.randint(1, 10),
                        'order_type': 'market',
                        'confidence': confidence,
                        'target_price': target_price,
                        'stop_loss_price': stop_loss_price,
                        'reason': "Test sell signal",
                        'metadata': {
                            'strategy': 'test_strategy',
                            'random_value': random.random()
                        }
                    })
            
            logger.info(f"TestStrategy generated {len(signals)} signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error in TestStrategy: {str(e)}", exc_info=True)
            return [] 