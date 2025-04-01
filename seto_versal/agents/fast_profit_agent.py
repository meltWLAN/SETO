#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fast profit agent module for SETO-Versal
Specializes in T+1 breakout trading strategies for quick profits
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from seto_versal.agents.base_agent import BaseAgent, AgentType

logger = logging.getLogger(__name__)

class FastProfitAgent(BaseAgent):
    """
    FastProfitAgent specializes in quick profit T+1 trading strategies
    
    This agent looks for breakout opportunities and rapid momentum
    shifts that can be capitalized on in T+1 trading environments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the fast profit agent
        
        Args:
            config (dict): Agent configuration
        """
        super().__init__(config)
        self.type = AgentType.FAST_PROFIT
        
        # FastProfit specific configuration
        self.breakout_threshold = config.get('breakout_threshold', 0.03)  # 3% breakout threshold
        self.volume_factor = config.get('volume_factor', 2.0)  # Volume must be 2x average
        self.profit_target = config.get('profit_target', 0.05)  # 5% profit target
        self.stop_loss = config.get('stop_loss', 0.03)  # 3% stop loss
        self.max_hold_days = config.get('max_hold_days', 3)  # Maximum days to hold
        
        # Special tracking for fast profit agent
        self.potential_breakouts = {}  # Stocks showing potential breakout patterns
        self.watched_stocks = {}  # Stocks being monitored but not traded yet
        
        logger.info(f"FastProfitAgent {self.name} initialized with breakout threshold: {self.breakout_threshold*100}%, "
                   f"volume factor: {self.volume_factor}x")
    
    def process_market_state(self, market_state):
        """
        Process market state with special focus on breakout opportunities
        
        Args:
            market_state: Current market state object
            
        Returns:
            bool: Whether the market state was processed successfully
        """
        success = super().process_market_state(market_state)
        if not success:
            return False
        
        try:
            # Update watched stocks and potential breakouts
            self._scan_for_breakouts(market_state)
            return True
        except Exception as e:
            logger.error(f"Error in FastProfitAgent processing market state: {str(e)}")
            return False
    
    def _scan_for_breakouts(self, market_state):
        """
        Scan market data for potential breakout opportunities
        
        Args:
            market_state: Current market state
        """
        # Skip if market state doesn't have needed data
        if not hasattr(market_state, 'stocks_data') or not market_state.stocks_data:
            return
        
        # Clear old potential breakouts
        self.potential_breakouts = {}
        
        # Get market regime and overall sentiment
        market_regime = getattr(market_state, 'market_regime', 'unknown')
        market_sentiment = getattr(market_state, 'market_sentiment', 0.0)
        
        # Adjust breakout search parameters based on market regime
        volume_factor = self.volume_factor
        breakout_threshold = self.breakout_threshold
        
        # In bull markets, we can be less strict with breakout criteria
        if market_regime == 'bull' and market_sentiment > 0.5:
            volume_factor *= 0.8  # Less volume needed in bull markets
            breakout_threshold *= 0.8  # Lower threshold in bull markets
        
        # In bear markets, be more conservative
        elif market_regime == 'bear' and market_sentiment < -0.3:
            volume_factor *= 1.5  # Require more volume confirmation
            breakout_threshold *= 1.2  # Require stronger breakouts
        
        # Check each stock for breakout patterns
        for ticker, data in market_state.stocks_data.items():
            # Skip stocks that don't meet basic criteria
            if not self._meets_basic_criteria(ticker, data):
                continue
            
            # Check for breakout pattern
            breakout_score = self._calculate_breakout_score(data, volume_factor, breakout_threshold)
            
            # If breakout score is high enough, add to potential breakouts
            if breakout_score >= self.confidence_threshold:
                self.potential_breakouts[ticker] = {
                    'score': breakout_score,
                    'timestamp': datetime.now().isoformat(),
                    'breakout_level': data.get('resistance', data.get('close', 0)),
                    'volume_factor': data.get('volume_factor', 1.0),
                    'price': data.get('close', 0),
                    'target_price': data.get('close', 0) * (1 + self.profit_target),
                    'stop_loss_price': data.get('close', 0) * (1 - self.stop_loss)
                }
    
    def _meets_basic_criteria(self, ticker, data):
        """
        Check if stock meets basic criteria for consideration
        
        Args:
            ticker (str): Stock ticker
            data (dict): Stock market data
            
        Returns:
            bool: Whether stock meets basic criteria
        """
        # Skip if missing essential data
        if not data.get('close') or not data.get('volume'):
            return False
        
        # Skip penny stocks below threshold (adjust as needed)
        if data.get('close', 0) < 5.0:
            return False
        
        # Skip stocks with too low volume
        if data.get('volume', 0) < 100000:
            return False
        
        # Skip stocks we're already holding (T+1 restriction)
        if ticker in self.current_positions:
            return False
        
        return True
    
    def _calculate_breakout_score(self, data, volume_factor, breakout_threshold):
        """
        Calculate a breakout score for a stock
        
        Args:
            data (dict): Stock market data
            volume_factor (float): Required volume factor
            breakout_threshold (float): Required price breakout percentage
            
        Returns:
            float: Breakout score between 0.0 and 1.0
        """
        score = 0.0
        
        # Check if price is breaking out
        close = data.get('close', 0)
        prev_close = data.get('prev_close', close)
        resistance = data.get('resistance', close * 1.1)  # Fallback resistance level
        
        # Price breakout components
        if close > resistance:
            # Breaking resistance is a strong signal
            score += 0.4
        
        price_change = (close - prev_close) / prev_close if prev_close > 0 else 0
        if price_change > breakout_threshold:
            # Significant daily move
            score += min(0.3, price_change * 5)  # Cap at 0.3
        
        # Volume components
        volume = data.get('volume', 0)
        avg_volume = data.get('avg_volume', volume)
        
        if avg_volume > 0 and volume > avg_volume * volume_factor:
            # Strong volume confirmation
            vol_ratio = volume / avg_volume
            score += min(0.3, (vol_ratio - 1) * 0.1)  # Cap at 0.3
        
        # Technical indicator components
        if data.get('rsi', 50) > 70:
            # Strong momentum but potentially overbought
            score += 0.1
        
        # Market breadth components
        if data.get('sector_strength', 0.5) > 0.7:
            # Sector is strong
            score += 0.1
        
        return min(1.0, score)  # Cap at 1.0
    
    def generate_intentions(self, market_state=None):
        """
        Generate trading intentions focused on breakout opportunities
        
        Args:
            market_state: Current market state (optional)
            
        Returns:
            dict: Trading intentions for various tickers
        """
        # Use provided market state or fall back to last known state
        market_state = market_state or self.last_market_state
        if market_state is None:
            logger.warning(f"{self.name}: Cannot generate intentions without market state")
            return {}
        
        # Clear previous intentions
        self.current_intentions = {}
        
        # Check if agent is active
        if not self.is_active:
            return {}
        
        # Process potential breakouts into intentions
        for ticker, breakout_data in self.potential_breakouts.items():
            confidence = breakout_data['score']
            
            # Skip if confidence is below threshold
            if confidence < self.confidence_threshold:
                continue
            
            # Create intention
            intention = {
                'ticker': ticker,
                'action': 'BUY',
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'agent_id': self.id,
                'agent_type': self.type.value,
                'reason': 'breakout',
                'target_price': breakout_data['target_price'],
                'stop_loss_price': breakout_data['stop_loss_price'],
                'breakout_level': breakout_data['breakout_level'],
                'expected_hold_time': self.max_hold_days,
                'strategy_type': 'T+1_BREAKOUT'
            }
            
            # Add to current intentions
            self.current_intentions[ticker] = intention
        
        # Sort intentions by confidence and limit to top 5
        sorted_intentions = {k: v for k, v in sorted(
            self.current_intentions.items(), 
            key=lambda item: item[1]['confidence'], 
            reverse=True
        )[:5]}
        
        self.current_intentions = sorted_intentions
        
        if sorted_intentions:
            logger.info(f"{self.name} generated {len(sorted_intentions)} trading intentions")
        
        return self.current_intentions
    
    def evaluate_confidence(self, intention, market_state=None):
        """
        Evaluate confidence for a specific intention
        
        Args:
            intention (dict): Trading intention to evaluate
            market_state: Current market state (optional)
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        # If this is our intention, return the stored confidence
        ticker = intention.get('ticker')
        if ticker in self.current_intentions:
            return self.current_intentions[ticker].get('confidence', 0.5)
        
        # For intentions from other agents
        if intention.get('strategy_type') in ['BREAKOUT', 'MOMENTUM', 'GAP_UP', 'T+1_BREAKOUT']:
            # These strategies align with our specialty
            return 0.7
        
        # Default confidence for other types of intentions
        return 0.3
    
    def suggest_position_size(self, ticker, intention=None, market_state=None):
        """
        Suggest position size based on breakout strength and confidence
        
        Args:
            ticker (str): Ticker symbol
            intention (dict, optional): Trading intention
            market_state (object, optional): Current market state
            
        Returns:
            float: Suggested position size as proportion of total capital
        """
        # Default position size
        base_size = min(self.max_position_size, 0.1)
        
        # If we have an intention for this ticker, adjust based on confidence
        if ticker in self.current_intentions:
            intention = self.current_intentions[ticker]
            confidence = intention.get('confidence', 0.5)
            
            # Scale position size based on confidence
            confidence_factor = (confidence - self.confidence_threshold) / (1.0 - self.confidence_threshold)
            size_multiplier = 0.5 + (confidence_factor * 0.5)  # 0.5x to 1.0x multiplier
            
            # Apply risk tolerance factor
            risk_factor = self.risk_tolerance
            
            # Calculate adjusted position size
            adjusted_size = base_size * size_multiplier * risk_factor
            
            # Ensure within limits
            return min(self.max_position_size, max(0.05, adjusted_size))
        
        return base_size
    
    def adapt(self, adaptation_changes, reason=None):
        """
        Adapt agent parameters based on feedback or market conditions
        
        Args:
            adaptation_changes (dict): Changes to apply
            reason (str, optional): Reason for adaptation
            
        Returns:
            dict: Adaptation results
        """
        # Apply base class adaptations
        results = super().adapt(adaptation_changes, reason)
        
        if not results['success']:
            return results
        
        try:
            # FastProfitAgent specific adaptations
            if 'breakout_threshold' in adaptation_changes:
                self.breakout_threshold = adaptation_changes['breakout_threshold']
            
            if 'volume_factor' in adaptation_changes:
                self.volume_factor = adaptation_changes['volume_factor']
            
            if 'profit_target' in adaptation_changes:
                self.profit_target = adaptation_changes['profit_target']
            
            if 'stop_loss' in adaptation_changes:
                self.stop_loss = adaptation_changes['stop_loss']
            
            # Update results with additional parameters changed
            results['current_state'].update({
                'breakout_threshold': self.breakout_threshold,
                'volume_factor': self.volume_factor,
                'profit_target': self.profit_target,
                'stop_loss': self.stop_loss
            })
            
            return results
        
        except Exception as e:
            logger.error(f"Error in FastProfitAgent adaptation: {str(e)}")
            return {
                'success': False,
                'agent_id': self.id,
                'error': str(e)
            } 