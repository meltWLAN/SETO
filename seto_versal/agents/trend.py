#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trend following agent module
"""

import logging
import numpy as np
from .base import Agent

logger = logging.getLogger(__name__)

class TrendAgent(Agent):
    """Trend following agent class"""
    
    def __init__(self, name, confidence_threshold=0.7, max_positions=5, weight=1.0, strategies=None,
                 lookback_period=20, trend_threshold=0.05):
        """
        Initialize trend following agent
        
        Args:
            name (str): Agent name
            confidence_threshold (float): Confidence threshold for trading
            max_positions (int): Maximum number of positions
            weight (float): Agent weight in ensemble
            strategies (list): List of strategies
            lookback_period (int): Period for trend calculation
            trend_threshold (float): Minimum trend strength threshold
        """
        super().__init__(name, confidence_threshold, max_positions, weight, strategies)
        
        self.lookback_period = lookback_period
        self.trend_threshold = trend_threshold
        
        logger.info(f"Initialized trend agent {name} with lookback={lookback_period}")
        
    def calculate_trend(self, prices):
        """
        Calculate trend strength using linear regression
        
        Args:
            prices (np.array): Historical price data
            
        Returns:
            float: Trend strength indicator (-1 to 1)
        """
        if len(prices) < self.lookback_period:
            return 0
            
        x = np.arange(len(prices))
        y = prices
        
        # Calculate linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope to -1 to 1 range
        trend = np.clip(slope / self.trend_threshold, -1, 1)
        
        return trend
        
    def analyze(self, market_state, symbols=None):
        """
        Analyze market state and generate trading decisions
        
        Args:
            market_state: Current market state
            symbols: List of symbols to analyze
            
        Returns:
            list: List of trading decisions
        """
        decisions = []
        
        # Get symbols to analyze
        if symbols is None:
            symbols = market_state.get_tradable_symbols()
            
        for symbol in symbols:
            # Get historical prices
            prices = market_state.get_price_history(symbol, self.lookback_period)
            if prices is None or len(prices) < self.lookback_period:
                continue
                
            # Calculate trend
            trend = self.calculate_trend(prices)
            
            # Generate trading decision
            if abs(trend) > self.trend_threshold:
                confidence = abs(trend)
                if confidence >= self.confidence_threshold:
                    decision = {
                        'symbol': symbol,
                        'action': 'buy' if trend > 0 else 'sell',
                        'confidence': confidence,
                        'weight': self.weight
                    }
                    decisions.append(decision)
                    
        return decisions 