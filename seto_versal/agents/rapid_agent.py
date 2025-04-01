#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rapid Agent module for SETO-Versal
Implements the "Blade" agent for T+1 rapid breakout trading
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from seto_versal.agents.base import Agent, AgentType, AgentDecision, ConfidenceLevel

logger = logging.getLogger(__name__)

class RapidAgent(Agent):
    """
    Rapid profit agent (a.k.a "Blade")
    Specializes in identifying short-term breakout opportunities for T+1 trading
    Focuses on explosive momentum patterns with tight risk management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Rapid Agent with configuration
        
        Args:
            config (dict): Agent configuration
        """
        super().__init__(config, agent_type=AgentType.RAPID)
        
        # Specialized parameters for rapid trading
        self.parameters.update({
            # Breakout detection
            'volume_threshold': config.get('volume_threshold', 2.0),  # Volume increase threshold
            'price_threshold': config.get('price_threshold', 0.03),   # Price breakout threshold (%)
            'consolidation_days': config.get('consolidation_days', 5),  # Days of consolidation before breakout
            'breakout_window': config.get('breakout_window', 3),  # Lookback window for breakout pattern
            
            # Risk management
            'stop_loss_pct': config.get('stop_loss_pct', 0.03),  # Stop loss as % from entry
            'target_pct': config.get('target_pct', 0.05),      # Target profit as % from entry
            'max_holding_days': config.get('max_holding_days', 2),  # Maximum days to hold position
            
            # Filtering
            'min_volume': config.get('min_volume', 500000),    # Minimum volume requirement
            'min_price': config.get('min_price', 5.0),         # Minimum price requirement
            'max_gap_up': config.get('max_gap_up', 0.05),      # Maximum gap up to allow entry
            
            # Timing
            'preferred_entry_time': config.get('preferred_entry_time', '10:30-14:00'),  # Preferred entry timing
            'avoid_news_days': config.get('avoid_news_days', True),  # Avoid trading on major news days
        })
        
        # Historical pattern performance
        self.pattern_performance = {}
        
        logger.info(f"Rapid Agent '{self.name}' initialized with breakout thresholds: "
                   f"vol={self.parameters['volume_threshold']}, price={self.parameters['price_threshold']}")
    
    def analyze(self, market_state, symbols=None, **kwargs) -> List[AgentDecision]:
        """
        Analyze market data and generate trading decisions based on breakout patterns
        
        Args:
            market_state: Current market state
            symbols (list, optional): Specific symbols to analyze
            **kwargs: Additional analysis parameters
            
        Returns:
            list: List of AgentDecision objects
        """
        decisions = []
        
        # Skip analysis if agent is not active
        if not self.is_active:
            logger.debug(f"Agent {self.name} is not active, skipping analysis")
            return decisions
        
        # Check market regime suitability
        regime = market_state.get_market_regime()
        regime_suitability = self._check_regime_suitability(regime)
        if regime_suitability < 0.5:
            logger.info(f"Market regime {regime} is not optimal for rapid trading (suitability: {regime_suitability:.2f})")
            # Still proceed but with caution
        
        # Get symbols to analyze
        if symbols is None:
            # In a real implementation, this would get a filtered list of stocks
            # For now, use a sample set
            symbols = ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOG']
            
        # Get data source from market state
        data_source = getattr(market_state, 'data_source', None)
        if data_source is None:
            logger.error("No data source available in market state")
            return decisions
        
        # Loop through symbols and look for breakout patterns
        for symbol in symbols:
            try:
                # Get historical data for analysis
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                
                # Get price data
                df = data_source.get_price_data(symbol, start_date, end_date, 'daily')
                
                # Skip if insufficient data
                if df is None or len(df) < 10:
                    continue
                
                # Detect breakout patterns
                is_breakout, breakout_data = self._detect_breakout(df, symbol)
                
                if is_breakout:
                    # Calculate expected entry, target and stop loss
                    current_price = df['close'].iloc[-1]
                    stop_loss = current_price * (1 - self.parameters['stop_loss_pct'])
                    target_price = current_price * (1 + self.parameters['target_pct'])
                    
                    # Calculate confidence level
                    confidence = self._calculate_breakout_confidence(breakout_data, df, market_state)
                    
                    # Create decision only if confidence is sufficient
                    if confidence.value >= ConfidenceLevel.MEDIUM.value:
                        decision = AgentDecision(
                            agent_id=self.id,
                            symbol=symbol,
                            decision_type="buy",
                            confidence=confidence,
                            target_price=target_price,
                            stop_loss=stop_loss,
                            timeframe=f"{self.parameters['max_holding_days']} days",
                            reasoning=f"Breakout detected: {breakout_data['reason']}",
                            metrics=breakout_data,
                            entry_window=("09:30", "14:30")  # Trading window for next day
                        )
                        
                        decisions.append(decision)
                        self.record_decision(decision)
                        
                        logger.info(f"Breakout opportunity identified: {symbol} - Confidence: {confidence.name}")
            
            except Exception as e:
                logger.warning(f"Error analyzing {symbol}: {str(e)}")
        
        return decisions
    
    def _detect_breakout(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect breakout patterns in price data
        
        Args:
            df (pd.DataFrame): Price data for the symbol
            symbol (str): Symbol being analyzed
            
        Returns:
            tuple: (is_breakout, breakout_data)
                - is_breakout (bool): Whether a breakout is detected
                - breakout_data (dict): Data related to the breakout
        """
        # Initialize result
        breakout_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'breakout_type': None,
            'reason': None,
            'strength': 0.0,
            'support_level': None,
            'volume_increase': None,
            'consolidation_quality': None
        }
        
        # Check if we have enough data
        if len(df) < self.parameters['consolidation_days'] + 3:
            return False, breakout_data
        
        # Get recent data
        recent_data = df.iloc[-self.parameters['breakout_window']:]
        consolidation_data = df.iloc[-(self.parameters['consolidation_days'] + self.parameters['breakout_window']):-self.parameters['breakout_window']]
        
        # Price checks
        last_close = recent_data['close'].iloc[-1]
        
        # Minimum price filter
        if last_close < self.parameters['min_price']:
            return False, breakout_data
        
        # Find consolidation range
        consolidation_high = consolidation_data['high'].max()
        consolidation_low = consolidation_data['low'].min()
        consolidation_range = (consolidation_high - consolidation_low) / consolidation_low
        
        # Check if consolidation is tight enough
        tight_consolidation = consolidation_range < 0.08  # 8% range or less
        
        # Calculate volume increase
        recent_volume = recent_data['volume'].iloc[-1]
        avg_volume = consolidation_data['volume'].mean()
        volume_increase = recent_volume / avg_volume if avg_volume > 0 else 0
        
        # Check if volume threshold is met
        volume_breakout = volume_increase > self.parameters['volume_threshold']
        
        # Check minimum volume
        if recent_volume < self.parameters['min_volume']:
            return False, breakout_data
        
        # Check price breakout - closing above consolidation high
        price_breakout = last_close > consolidation_high * (1 + self.parameters['price_threshold'])
        
        # Check for gap up - avoid chasing too far
        prev_close = df['close'].iloc[-2] if len(df) > 1 else last_close
        gap_up_pct = (df['open'].iloc[-1] - prev_close) / prev_close
        if gap_up_pct > self.parameters['max_gap_up']:
            return False, breakout_data
        
        # Combine conditions for breakout detection
        is_breakout = price_breakout and (volume_breakout or tight_consolidation)
        
        if is_breakout:
            # Calculate breakout strength
            strength = 0.0
            reasons = []
            
            if price_breakout:
                price_strength = (last_close - consolidation_high) / consolidation_high
                strength += min(price_strength * 5, 0.5)  # Cap at 0.5
                reasons.append(f"Price breakout: +{price_strength:.1%}")
                
            if volume_breakout:
                vol_strength = min((volume_increase - 1) / 5, 0.3)  # Cap at 0.3
                strength += vol_strength
                reasons.append(f"Volume surge: {volume_increase:.1f}x")
                
            if tight_consolidation:
                cons_strength = min((0.1 - consolidation_range) / 0.1, 0.2)  # Cap at 0.2
                strength += max(cons_strength, 0)
                reasons.append(f"Tight consolidation: {consolidation_range:.1%}")
            
            # Update breakout data
            breakout_data.update({
                'breakout_type': 'volume_price' if volume_breakout else 'consolidation_breakout',
                'reason': "; ".join(reasons),
                'strength': strength,
                'support_level': consolidation_high,
                'volume_increase': volume_increase,
                'consolidation_quality': 1.0 - consolidation_range,
                'consolidation_days': self.parameters['consolidation_days'],
                'price_data': {
                    'last_close': last_close,
                    'consolidation_high': consolidation_high,
                    'consolidation_low': consolidation_low,
                    'breakout_percent': (last_close - consolidation_high) / consolidation_high
                }
            })
        
        return is_breakout, breakout_data
    
    def _calculate_breakout_confidence(self, breakout_data: Dict[str, Any], 
                                     df: pd.DataFrame, market_state) -> ConfidenceLevel:
        """
        Calculate confidence level for a breakout signal
        
        Args:
            breakout_data (dict): Breakout detection results
            df (pd.DataFrame): Price data for the symbol
            market_state: Current market state
            
        Returns:
            ConfidenceLevel: Confidence level for the breakout
        """
        # Extract key metrics
        signal_strength = breakout_data['strength']
        
        # Market trend alignment
        regime = market_state.get_market_regime()
        market_alignment = {
            'bull': 0.9,
            'recovery': 0.8,
            'range': 0.7,
            'topping': 0.5,
            'volatile': 0.4,
            'bear': 0.3,
            'unknown': 0.5
        }.get(regime, 0.5)
        
        # Volume confirmation
        volume_confirmation = min(breakout_data['volume_increase'] / 3, 1.0) if breakout_data['volume_increase'] else 0.5
        
        # Historical accuracy - would be based on actual agent performance
        # For now use a placeholder
        historical_accuracy = 0.7
        
        # Calculate confidence
        return self.calculate_confidence(
            signal_strength=signal_strength,
            trend_alignment=market_alignment,
            historical_accuracy=historical_accuracy,
            volume_confirmation=volume_confirmation
        )
    
    def _check_regime_suitability(self, regime: str) -> float:
        """
        Check how suitable the current market regime is for this agent
        
        Args:
            regime (str): Current market regime
            
        Returns:
            float: Suitability score (0.0-1.0)
        """
        # Define suitability for each regime
        suitability_map = {
            'bull': 0.9,        # Very suitable in bull markets
            'recovery': 0.8,    # Strong in recovery phases
            'range': 0.7,       # Works well in range-bound markets
            'volatile': 0.6,    # Can work in volatile markets but needs careful selection
            'topping': 0.5,     # Need caution in topping markets
            'bear': 0.3,        # Not ideal in bear markets
            'unknown': 0.5      # Neutral when regime is unknown
        }
        
        return suitability_map.get(regime, 0.5)
    
    def adapt(self, changes: Dict[str, Any], reason: str = None) -> Dict[str, Any]:
        """
        Adapt agent behavior based on feedback
        
        Args:
            changes (dict): Parameter changes to apply
            reason (str, optional): Reason for adaptation
            
        Returns:
            dict: Results of adaptation
        """
        # Apply common adaptation logic from parent class
        result = super().adapt(changes, reason)
        
        # Check for RapidAgent specific adaptations
        if 'volatility_adjustment' in changes:
            vol_adj = changes['volatility_adjustment']
            
            # Adjust thresholds based on volatility
            if vol_adj > 0:  # Volatility increased
                # In higher volatility, require stronger breakouts
                self.parameters['volume_threshold'] *= (1 + vol_adj * 0.2)
                self.parameters['price_threshold'] *= (1 + vol_adj * 0.3)
            else:  # Volatility decreased
                # In lower volatility, can be less stringent
                self.parameters['volume_threshold'] /= (1 + abs(vol_adj) * 0.1)
                self.parameters['price_threshold'] /= (1 + abs(vol_adj) * 0.15)
            
            logger.info(f"Adjusted breakout thresholds due to volatility change: "
                       f"vol={self.parameters['volume_threshold']:.2f}, price={self.parameters['price_threshold']:.3f}")
        
        # Adjust risk parameters if specified
        if 'stop_loss_multiplier' in changes:
            self.parameters['stop_loss_pct'] *= changes['stop_loss_multiplier']
            logger.info(f"Adjusted stop loss to {self.parameters['stop_loss_pct']:.1%}")
        
        if 'target_multiplier' in changes:
            self.parameters['target_pct'] *= changes['target_multiplier']
            logger.info(f"Adjusted profit target to {self.parameters['target_pct']:.1%}")
        
        # Record the adaptation in the result
        result['adjusted_parameters'] = {
            'volume_threshold': self.parameters['volume_threshold'],
            'price_threshold': self.parameters['price_threshold'],
            'stop_loss_pct': self.parameters['stop_loss_pct'],
            'target_pct': self.parameters['target_pct']
        }
        
        return result 