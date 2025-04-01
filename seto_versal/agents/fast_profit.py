#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fast profit agent module for SETO-Versal
Specializes in T+1 breakout and explosive trading strategies
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid

from seto_versal.agents.base import Agent

logger = logging.getLogger(__name__)

class FastProfitAgent(Agent):
    """
    Fast Profit Agent (刀锋)
    
    Specializes in T+1 breakout trading with the following characteristics:
    - Quick entry on volume and price breakouts
    - Short holding periods (typically 1-3 days)
    - Higher risk tolerance for greater profit potential
    - Focuses on volatile stocks with momentum
    """
    
    def __init__(self, name, config):
        """
        Initialize the fast profit agent
        
        Args:
            name (str): Agent name
            config (dict): Agent configuration
        """
        super().__init__(name, config)
        
        self.type = "fast_profit"
        self.description = "Fast profit agent focusing on T+1 breakout trading"
        
        # Specific settings for fast profit agent
        self.volume_threshold = config.get('volume_threshold', 2.0)  # Min volume surge ratio
        self.price_threshold = config.get('price_threshold', 0.03)  # Min price movement (3%)
        self.profit_target = config.get('profit_target', 0.05)  # Target profit (5%)
        self.stop_loss = config.get('stop_loss', 0.03)  # Stop loss (3%)
        self.max_holding_days = config.get('max_holding_days', 3)  # Max holding period
        
        # Set higher risk tolerance by default
        self.risk_tolerance = config.get('risk_tolerance', 0.7)  # Higher risk tolerance (0.0-1.0)
        
        # Watchlist of potential breakout candidates
        self.watchlist = {}
        
        logger.info(f"Fast profit agent '{self.name}' initialized")
    
    def generate_intentions(self, market_state):
        """
        Generate trading intentions based on breakout patterns
        
        Args:
            market_state (MarketState): Current market state
            
        Returns:
            list: List of trading intentions
        """
        intentions = []
        
        # Skip if agent is paused
        if not self.is_active:
            return intentions
        
        # Select appropriate strategy based on market conditions
        strategy = self.select_strategy(market_state)
        if not strategy:
            logger.warning(f"Agent '{self.name}' has no active strategy")
            return intentions
        
        # Update watchlist with new candidates
        self._update_watchlist(market_state)
        
        # Process each symbol in our watchlist
        for symbol, data in list(self.watchlist.items()):
            # Get latest price data
            current_data = market_state.get_ohlcv(symbol)
            if not current_data:
                continue
            
            # Check for breakout conditions
            breakout_detected, reason, confidence = self._check_breakout(symbol, current_data, market_state)
            
            if breakout_detected:
                # Calculate appropriate position size
                signal_strength = 0.8  # Strong signal for breakouts
                position_size = self.calculate_position_size(symbol, signal_strength, confidence)
                
                # Calculate target price and stop loss
                current_price = current_data['close']
                target_price = current_price * (1 + self.profit_target)
                stop_loss_price = current_price * (1 - self.stop_loss)
                
                # Create trade intention
                intention = self._create_intention(
                    symbol=symbol,
                    direction='buy',
                    size=position_size,
                    reason=reason,
                    confidence=confidence,
                    target_price=target_price,
                    stop_loss=stop_loss_price
                )
                
                intentions.append(intention)
                logger.info(f"FastProfit agent generated BUY intention for {symbol}: {reason}")
        
        return intentions
    
    def _update_watchlist(self, market_state):
        """
        Update the watchlist with potential breakout candidates
        
        Args:
            market_state (MarketState): Current market state
        """
        # Scan all available symbols for potential candidates
        for symbol in market_state.symbols:
            # Skip if already in watchlist
            if symbol in self.watchlist:
                continue
            
            # Get current and historical data
            current_data = market_state.get_ohlcv(symbol)
            if not current_data:
                continue
            
            # Get indicators
            indicators = market_state.indicators.get(symbol, {})
            
            # Check if the stock meets our screening criteria
            if self._meets_screening_criteria(symbol, current_data, indicators, market_state):
                # Add to watchlist
                self.watchlist[symbol] = {
                    'added_time': datetime.now(),
                    'screening_price': current_data['close'],
                    'screening_volume': current_data['volume'],
                    'notes': 'Potential breakout candidate'
                }
                logger.debug(f"Added {symbol} to FastProfit agent watchlist")
        
        # Remove symbols that have been in watchlist too long without triggering
        current_time = datetime.now()
        for symbol in list(self.watchlist.keys()):
            if (current_time - self.watchlist[symbol]['added_time']).days > 5:
                del self.watchlist[symbol]
                logger.debug(f"Removed {symbol} from FastProfit agent watchlist (too old)")
    
    def _meets_screening_criteria(self, symbol, current_data, indicators, market_state):
        """
        Check if a symbol meets the initial screening criteria
        
        Args:
            symbol (str): Symbol to check
            current_data (dict): Current OHLCV data
            indicators (dict): Technical indicators
            market_state (MarketState): Current market state
            
        Returns:
            bool: True if the symbol meets criteria
        """
        # This is a simplified implementation
        # In a real system, we would have more sophisticated screening
        
        # Require minimum volume
        if current_data['volume'] < 500000:
            return False
        
        # Check for recent price movement
        price_range = (current_data['high'] - current_data['low']) / current_data['low']
        if price_range < 0.02:  # Require at least 2% intraday range
            return False
        
        # Check market regime - prefers bull or sideways markets
        if market_state.get_market_regime() == 'bear':
            # Be more selective in bear markets
            if price_range < 0.03:  # Require larger range in bear markets
                return False
        
        # Check RSI if available
        rsi = indicators.get('rsi', 50)
        if rsi < 40 or rsi > 80:
            return False  # Avoid extreme RSI values for breakout plays
        
        # Passed all criteria
        return True
    
    def _check_breakout(self, symbol, current_data, market_state):
        """
        Check if a symbol is showing breakout conditions
        
        Args:
            symbol (str): Symbol to check
            current_data (dict): Current OHLCV data
            market_state (MarketState): Current market state
            
        Returns:
            tuple: (breakout_detected, reason, confidence)
        """
        # Get watchlist data
        watch_data = self.watchlist.get(symbol, {})
        if not watch_data:
            return False, "", 0.0
        
        # Get indicators
        indicators = market_state.indicators.get(symbol, {})
        
        # Check volume surge
        screening_volume = watch_data.get('screening_volume', current_data['volume'])
        volume_ratio = current_data['volume'] / max(1, screening_volume)
        
        # Check price breakout
        screening_price = watch_data.get('screening_price', current_data['close'])
        price_change = (current_data['close'] - screening_price) / screening_price
        
        # Calculate confidence based on multiple factors
        confidence_factors = []
        
        # Volume factor
        volume_confidence = min(1.0, (volume_ratio - 1.0) / (self.volume_threshold - 1.0))
        if volume_confidence > 0:
            confidence_factors.append((volume_confidence, 0.4))  # 40% weight
        
        # Price movement factor
        price_confidence = min(1.0, price_change / self.price_threshold)
        if price_confidence > 0:
            confidence_factors.append((price_confidence, 0.4))  # 40% weight
        
        # Market regime factor
        regime = market_state.get_market_regime()
        regime_scores = {'bull': 0.95, 'sideways': 0.7, 'bear': 0.3}
        regime_confidence = regime_scores.get(regime, 0.5)
        confidence_factors.append((regime_confidence, 0.2))  # 20% weight
        
        # Calculate overall confidence
        if not confidence_factors:
            return False, "", 0.0
            
        total_weight = sum(weight for _, weight in confidence_factors)
        weighted_confidence = sum(conf * weight for conf, weight in confidence_factors) / total_weight
        
        # Breakout detection rules
        breakout_detected = False
        reason = ""
        
        # Volume breakout rule
        if volume_ratio >= self.volume_threshold:
            if price_change > 0:
                breakout_detected = True
                reason = f"Volume surge ({volume_ratio:.1f}x) with positive price movement"
        
        # Price breakout rule
        if price_change >= self.price_threshold:
            breakout_detected = True
            reason = f"Price breakout of {price_change:.1%} on {volume_ratio:.1f}x volume"
        
        # Combined pattern
        if volume_ratio >= 1.5 and price_change >= 0.02:
            breakout_detected = True
            reason = f"Combined pattern: {price_change:.1%} move on {volume_ratio:.1f}x volume"
        
        # Final check - require minimum confidence
        if breakout_detected and weighted_confidence < 0.5:
            breakout_detected = False
        
        return breakout_detected, reason, weighted_confidence
    
    def reset(self):
        """
        Reset agent state, including watchlist
        
        Returns:
            None
        """
        super().reset()
        self.watchlist = {}
        logger.info(f"Fast profit agent '{self.name}' reset (watchlist cleared)")
        
    def get_watchlist(self):
        """
        Get the current watchlist
        
        Returns:
            dict: Watchlist dictionary
        """
        return self.watchlist

    def select_strategy(self, market_state):
        """
        Select the most appropriate strategy for current market conditions
        
        Args:
            market_state (MarketState): Current market state
            
        Returns:
            Strategy: Selected strategy or None
        """
        if not self.strategies:
            return None
        
        # Simple implementation: return first enabled strategy
        for strategy in self.strategies:
            if strategy.is_enabled():
                return strategy
            
        return None

    def calculate_position_size(self, symbol, signal_strength, confidence):
        """
        Calculate appropriate position size based on signal strength and confidence
        
        Args:
            symbol (str): The trading symbol
            signal_strength (float): Strength of the signal (0.0-1.0)
            confidence (ConfidenceLevel): Confidence level of the signal
            
        Returns:
            float: Position size as a percentage of available capital (0.0-1.0)
        """
        # Base position size based on max position size
        base_size = self.max_position_size
        
        # Adjust for signal strength
        size = base_size * signal_strength
        
        # Adjust for confidence level
        conf_multiplier = {
            'VERY_LOW': 0.2,
            'LOW': 0.4,
            'MEDIUM': 0.6,
            'HIGH': 0.8,
            'VERY_HIGH': 1.0
        }
        
        # Get confidence name if it's an enum
        conf_name = confidence.name if hasattr(confidence, 'name') else str(confidence)
        multiplier = conf_multiplier.get(conf_name, 0.5)
        
        # Apply confidence multiplier
        size *= multiplier
        
        # Apply risk tolerance
        size *= self.risk_tolerance
        
        # Ensure size is within limits
        size = min(self.max_position_size, max(0.01, size))
        
        logger.debug(f"Calculated position size for {symbol}: {size:.2%}")
        return size

    def _create_intention(self, symbol, direction, size, reason, confidence, target_price=None, stop_loss=None):
        """
        Create a trading intention
        
        Args:
            symbol (str): Symbol to trade
            direction (str): 'buy' or 'sell'
            size (float): Position size (0.0-1.0)
            reason (str): Reason for the intention
            confidence (ConfidenceLevel): Confidence level
            target_price (float, optional): Target price
            stop_loss (float, optional): Stop loss price
            
        Returns:
            dict: Trading intention
        """
        # Create a unique ID for this intention
        intention_id = str(uuid.uuid4())
        
        intention = {
            'id': intention_id,
            'agent_id': self.id,
            'agent_name': self.name,
            'symbol': symbol,
            'direction': direction,
            'size': size,
            'reason': reason,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'target_price': target_price,
            'stop_loss': stop_loss
        }
        
        return intention 