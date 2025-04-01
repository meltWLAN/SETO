#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
防御性代理
专注于风险控制和资金保护
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid

from seto_versal.agents.base import Agent

logger = logging.getLogger(__name__)

class DefensiveAgent(Agent):
    """
    Defensive Agent (防御)
    
    Specializes in risk management with the following characteristics:
    - Focuses on capital preservation and risk control
    - Increases activity during high volatility and market downturns
    - Reduces exposure when market uncertainty rises
    - Prefers stable, low-volatility investments with downside protection
    """
    
    def __init__(self, name, config):
        """
        Initialize the defensive agent
        
        Args:
            name (str): Agent name
            config (dict): Agent configuration
        """
        super().__init__(name, config)
        
        self.type = "defensive"
        self.description = "Defensive agent focusing on capital preservation"
        
        # Specific settings for defensive agent
        self.volatility_threshold = config.get('volatility_threshold', 0.02)  # 2% daily volatility threshold
        self.drawdown_threshold = config.get('drawdown_threshold', 0.1)  # 10% market drawdown threshold
        self.position_reduction_ratio = config.get('position_reduction_ratio', 0.5)  # Position size reduction in risky markets
        self.max_allocation = config.get('max_allocation', 0.6)  # Maximum capital allocation
        self.market_hedge_threshold = config.get('market_hedge_threshold', 0.15)  # Market drawdown for hedge activation
        
        # Lower risk tolerance by default
        self.risk_tolerance = config.get('risk_tolerance', 0.3)  # Low risk tolerance (0.0-1.0)
        self.max_position_size = config.get('max_position_size', 0.1)  # Smaller position sizes
        
        # Additional state tracking
        self.current_market_assessment = 'normal'  # normal, cautious, defensive, hedging
        self.defensive_mode_active = False
        self.hedge_positions = {}  # Hedge positions (typically shorts on index)
        self.safe_asset_positions = {}  # Safe assets (typically bonds, low-vol stocks)
        self.current_positions = {}  # Track current positions
        
        logger.info(f"Defensive agent '{self.name}' initialized")
    
    def generate_intentions(self, market_state):
        """
        Generate trading intentions based on defensive strategy
        
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
        
        # Update market assessment
        self._update_market_assessment(market_state)
        
        # Generate position reduction intentions if needed
        if self.defensive_mode_active:
            reduce_intentions = self._generate_position_reduction_intentions(market_state)
            intentions.extend(reduce_intentions)
        
        # Generate hedge intentions if needed
        if self.current_market_assessment == 'hedging':
            hedge_intentions = self._generate_hedge_intentions(market_state)
            intentions.extend(hedge_intentions)
        
        # Generate safe asset intentions
        if self.current_market_assessment in ['cautious', 'defensive', 'hedging']:
            safe_intentions = self._generate_safe_asset_intentions(market_state)
            intentions.extend(safe_intentions)
        
        # Only look for normal opportunities if not in hedging mode
        if self.current_market_assessment != 'hedging':
            # Generate normal trading intentions with reduced size
            normal_intentions = self._generate_normal_intentions(market_state)
            intentions.extend(normal_intentions)
        
        logger.debug(f"Defensive agent generated {len(intentions)} intentions in {self.current_market_assessment} mode")
        return intentions
    
    def _update_market_assessment(self, market_state):
        """
        Update assessment of market conditions
        
        Args:
            market_state (MarketState): Current market state
        """
        # Get market metrics
        market_volatility = self._calculate_market_volatility(market_state)
        market_drawdown = self._calculate_market_drawdown(market_state)
        market_trend = market_state.get_market_regime()
        
        # Store previous assessment for logging
        previous_assessment = self.current_market_assessment
        previous_defensive_mode = self.defensive_mode_active
        
        # Update market assessment based on conditions
        if market_drawdown >= self.market_hedge_threshold:
            # Significant drawdown - implement hedging
            self.current_market_assessment = 'hedging'
            self.defensive_mode_active = True
        elif market_drawdown >= self.drawdown_threshold or market_volatility >= self.volatility_threshold * 1.5:
            # High drawdown or very high volatility - defensive mode
            self.current_market_assessment = 'defensive'
            self.defensive_mode_active = True
        elif market_volatility >= self.volatility_threshold or market_trend == 'bear':
            # Elevated volatility or bear market - cautious mode
            self.current_market_assessment = 'cautious'
            self.defensive_mode_active = True
        else:
            # Normal market conditions
            self.current_market_assessment = 'normal'
            self.defensive_mode_active = False
        
        # Log changes in assessment
        if previous_assessment != self.current_market_assessment:
            logger.info(f"Market assessment changed from {previous_assessment} to {self.current_market_assessment}")
            
        # Log defensive mode changes
        if previous_defensive_mode != self.defensive_mode_active:
            if self.defensive_mode_active:
                logger.info(f"Defensive mode activated due to {self.current_market_assessment} conditions")
            else:
                logger.info(f"Defensive mode deactivated, returning to normal operations")
    
    def _calculate_market_volatility(self, market_state):
        """
        Calculate current market volatility
        
        Args:
            market_state (MarketState): Current market state
            
        Returns:
            float: Market volatility estimate
        """
        # Use index as market proxy
        index_symbol = market_state.config.get('market_index', '000001.SH')
        
        # Get historical prices (20 days for volatility)
        history = market_state.get_history(index_symbol, 20)
        
        if not history or len(history) < 10:
            return 0.01  # Default low volatility if not enough data
        
        # Calculate daily returns
        closes = [bar['close'] for bar in history]
        returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
        
        # Calculate volatility as standard deviation of returns
        if returns:
            volatility = float(np.std(returns))
            return volatility
            
        return 0.01  # Default
    
    def _calculate_market_drawdown(self, market_state):
        """
        Calculate current market drawdown from recent high
        
        Args:
            market_state (MarketState): Current market state
            
        Returns:
            float: Market drawdown from recent high
        """
        # Use index as market proxy
        index_symbol = market_state.config.get('market_index', '000001.SH')
        
        # Get historical prices (60 days for drawdown)
        history = market_state.get_history(index_symbol, 60)
        
        if not history or len(history) < 5:
            return 0.0  # Default no drawdown if not enough data
        
        # Find recent high
        closes = [bar['close'] for bar in history]
        recent_high = max(closes[:-1])  # Exclude current day
        current_price = closes[-1]
        
        # Calculate drawdown
        if recent_high > 0:
            drawdown = 1 - (current_price / recent_high)
            return max(0, drawdown)  # Ensure non-negative
            
        return 0.0  # Default
    
    def _generate_position_reduction_intentions(self, market_state):
        """
        Generate intentions to reduce existing positions in defensive mode
        
        Args:
            market_state (MarketState): Current market state
            
        Returns:
            list: Position reduction intentions
        """
        intentions = []
        
        # Calculate reduction factor based on market conditions
        if self.current_market_assessment == 'hedging':
            reduction_factor = 0.7  # Reduce positions by 70% in hedging mode
        elif self.current_market_assessment == 'defensive':
            reduction_factor = 0.5  # Reduce positions by 50% in defensive mode
        elif self.current_market_assessment == 'cautious':
            reduction_factor = 0.3  # Reduce positions by 30% in cautious mode
        else:
            reduction_factor = 0.0  # No reduction in normal mode
        
        # Check existing positions for reduction
        for symbol, position in list(self.current_positions.items()):
            # Skip positions recently adjusted
            last_adjusted = position.get('last_adjusted', datetime.now() - timedelta(days=30))
            if (datetime.now() - last_adjusted).days < 1:
                continue
            
            # Calculate amount to reduce
            current_size = position.get('position_size', 0)
            reduction_size = current_size * reduction_factor
            
            # Minimum threshold to avoid tiny trades
            if reduction_size < 0.01:
                continue
            
            # Create reduction intention
            reason = f"Defensive position reduction ({self.current_market_assessment} mode)"
            
            intention = self._create_intention(
                symbol=symbol,
                direction='sell',
                size=reduction_size,
                reason=reason,
                confidence=0.8
            )
            
            intentions.append(intention)
            
            # Update position record
            self.current_positions[symbol]['position_size'] -= reduction_size
            self.current_positions[symbol]['last_adjusted'] = datetime.now()
            
            logger.info(f"Defensive agent reducing {symbol} position by {reduction_size:.2f} due to {self.current_market_assessment} conditions")
        
        return intentions
    
    def _generate_hedge_intentions(self, market_state):
        """
        Generate intentions for market hedging in severe downturns
        
        Args:
            market_state (MarketState): Current market state
            
        Returns:
            list: Hedge intentions
        """
        intentions = []
        
        # Skip if hedging is disabled
        if not market_state.config.get('enable_hedging', True):
            return intentions
        
        # Calculate total exposure as sum of all positions
        total_exposure = sum(pos.get('position_size', 0) for pos in self.current_positions.values())
        
        # Calculate desired hedge ratio based on market conditions
        market_drawdown = self._calculate_market_drawdown(market_state)
        target_hedge_ratio = min(0.5, market_drawdown * 2)  # Max 50% hedge
        
        # Calculate target hedge amount
        target_hedge_amount = total_exposure * target_hedge_ratio
        
        # Get current hedge amount
        current_hedge_amount = sum(pos.get('position_size', 0) for pos in self.hedge_positions.values())
        
        # Determine if we need to add hedges
        hedge_deficit = target_hedge_amount - current_hedge_amount
        
        if hedge_deficit > 0.05:  # Minimum threshold for new hedge
            # Select hedge instrument (typically a market index ETF or futures)
            hedge_symbol = market_state.config.get('hedge_instrument', '510050.SH')  # Default to SSE 50 ETF
            
            # Get current price
            current_data = market_state.get_ohlcv(hedge_symbol)
            if not current_data:
                return intentions
            
            current_price = current_data['close']
            
            # Create hedge intention
            reason = f"Market hedge in {self.current_market_assessment} conditions (drawdown: {market_drawdown:.1%})"
            
            intention = self._create_intention(
                symbol=hedge_symbol,
                direction='sell',  # Short sell for hedging
                size=hedge_deficit,
                reason=reason,
                confidence=0.9
            )
            
            intentions.append(intention)
            
            # Update hedge positions record
            if hedge_symbol in self.hedge_positions:
                self.hedge_positions[hedge_symbol]['position_size'] += hedge_deficit
            else:
                self.hedge_positions[hedge_symbol] = {
                    'position_size': hedge_deficit,
                    'entry_price': current_price,
                    'entry_time': datetime.now()
                }
            
            logger.info(f"Defensive agent adding hedge position in {hedge_symbol} of size {hedge_deficit:.2f}")
        
        # Check if we need to reduce hedges (market improving)
        elif hedge_deficit < -0.05 and current_hedge_amount > 0:
            # Reduce some hedges as conditions improve
            reduction_amount = min(abs(hedge_deficit), current_hedge_amount)
            
            for hedge_symbol, position in list(self.hedge_positions.items()):
                # Skip if this position is empty
                if position['position_size'] <= 0:
                    continue
                
                # Calculate reduction for this position (proportional)
                position_ratio = position['position_size'] / current_hedge_amount
                position_reduction = reduction_amount * position_ratio
                
                # Create hedge reduction intention
                reason = f"Reducing hedge as market conditions improve"
                
                intention = self._create_intention(
                    symbol=hedge_symbol,
                    direction='buy',  # Buy to cover short
                    size=position_reduction,
                    reason=reason,
                    confidence=0.7
                )
                
                intentions.append(intention)
                
                # Update hedge position record
                self.hedge_positions[hedge_symbol]['position_size'] -= position_reduction
                
                logger.info(f"Defensive agent reducing hedge in {hedge_symbol} by {position_reduction:.2f}")
        
        return intentions
    
    def _generate_safe_asset_intentions(self, market_state):
        """
        Generate intentions to add safe assets in defensive conditions
        
        Args:
            market_state (MarketState): Current market state
            
        Returns:
            list: Safe asset intentions
        """
        intentions = []
        
        # Define safe assets based on market conditions
        if self.current_market_assessment == 'hedging':
            # In severe downturns, focus on cash and bonds
            safe_assets = market_state.config.get('safe_assets', {}).get('hedging', ['511010.SH', '511030.SH'])
        elif self.current_market_assessment == 'defensive':
            # In defensive mode, focus on bonds and defensive stocks
            safe_assets = market_state.config.get('safe_assets', {}).get('defensive', ['511010.SH', '600900.SH'])
        else:
            # In cautious mode, include some utilities and consumer staples
            safe_assets = market_state.config.get('safe_assets', {}).get('cautious', ['600900.SH', '600519.SH'])
        
        # Calculate total current allocation
        total_allocation = sum(pos.get('position_size', 0) for pos in 
                              list(self.current_positions.values()) + 
                              list(self.safe_asset_positions.values()))
        
        # Calculate available allocation
        available_allocation = max(0, self.max_allocation - total_allocation)
        
        # Skip if no room for allocation
        if available_allocation < 0.05:
            return intentions
        
        # Allocate to safe assets
        allocation_per_asset = available_allocation / len(safe_assets) if safe_assets else 0
        
        for symbol in safe_assets:
            # Skip if already at max allocation
            if symbol in self.safe_asset_positions and self.safe_asset_positions[symbol].get('position_size', 0) >= allocation_per_asset:
                continue
            
            # Get current data
            current_data = market_state.get_ohlcv(symbol)
            if not current_data:
                continue
            
            current_price = current_data['close']
            
            # Calculate how much to add
            current_size = self.safe_asset_positions.get(symbol, {}).get('position_size', 0)
            add_size = allocation_per_asset - current_size
            
            if add_size < 0.02:  # Minimum threshold
                continue
            
            # Create safe asset intention
            reason = f"Adding safe asset in {self.current_market_assessment} conditions"
            
            intention = self._create_intention(
                symbol=symbol,
                direction='buy',
                size=add_size,
                reason=reason,
                confidence=0.8
            )
            
            intentions.append(intention)
            
            # Update safe asset positions record
            if symbol in self.safe_asset_positions:
                self.safe_asset_positions[symbol]['position_size'] += add_size
            else:
                self.safe_asset_positions[symbol] = {
                    'position_size': add_size,
                    'entry_price': current_price,
                    'entry_time': datetime.now()
                }
            
            logger.info(f"Defensive agent adding safe asset {symbol} with size {add_size:.2f}")
        
        return intentions
    
    def _generate_normal_intentions(self, market_state):
        """
        Generate normal trading intentions with reduced size
        
        Args:
            market_state (MarketState): Current market state
            
        Returns:
            list: Normal trading intentions
        """
        intentions = []
        
        # Skip in hedging mode
        if self.current_market_assessment == 'hedging':
            return intentions
        
        # Apply size reduction in defensive modes
        size_reduction = {
            'normal': 1.0,     # No reduction
            'cautious': 0.7,   # 30% reduction
            'defensive': 0.5   # 50% reduction
        }.get(self.current_market_assessment, 0.5)
        
        # Select stocks that meet defensive criteria
        candidates = self._select_defensive_candidates(market_state)
        
        # Generate intentions for top candidates
        for symbol, metrics in candidates[:3]:  # Top 3 candidates
            # Skip if already have a position
            if symbol in self.current_positions:
                continue
            
            # Get current data
            current_data = market_state.get_ohlcv(symbol)
            if not current_data:
                continue
            
            # Calculate position size with reduction
            signal_strength = metrics.get('signal_strength', 0.6)
            confidence = metrics.get('confidence', 0.6)
            
            position_size = self.calculate_position_size(symbol, signal_strength, confidence) * size_reduction
            
            # Create intention
            reason = metrics.get('reason', 'Defensive stock selection')
            
            intention = self._create_intention(
                symbol=symbol,
                direction='buy',
                size=position_size,
                reason=reason,
                confidence=confidence
            )
            
            intentions.append(intention)
            
            # Add to current positions
            self.current_positions[symbol] = {
                'position_size': position_size,
                'entry_price': current_data['close'],
                'entry_time': datetime.now()
            }
            
            logger.info(f"Defensive agent selected {symbol} with {position_size:.2f} position: {reason}")
        
        return intentions
    
    def _select_defensive_candidates(self, market_state):
        """
        Select candidate stocks meeting defensive criteria
        
        Args:
            market_state (MarketState): Current market state
            
        Returns:
            list: List of (symbol, metrics) tuples
        """
        candidates = []
        
        # Analyze symbols based on defensive criteria
        for symbol in market_state.symbols:
            # Skip if already in positions
            if symbol in self.current_positions:
                continue
            
            # Get data and indicators
            current_data = market_state.get_ohlcv(symbol)
            if not current_data:
                continue
            
            indicators = market_state.indicators.get(symbol, {})
            
            # Calculate defensive metrics
            metrics = self._calculate_defensive_metrics(symbol, current_data, indicators, market_state)
            
            if metrics and metrics.get('score', 0) > 0.5:
                candidates.append((symbol, metrics))
        
        # Sort by defensive score
        candidates.sort(key=lambda x: x[1].get('score', 0), reverse=True)
        
        return candidates
    
    def _calculate_defensive_metrics(self, symbol, current_data, indicators, market_state):
        """
        Calculate defensive metrics for stock selection
        
        Args:
            symbol (str): Symbol to analyze
            current_data (dict): Current OHLCV data
            indicators (dict): Technical indicators
            market_state (MarketState): Current market state
            
        Returns:
            dict: Defensive metrics
        """
        metrics = {}
        reasons = []
        
        try:
            # Get historical data
            history = market_state.get_history(symbol, 60)
            if not history or len(history) < 30:
                return None
            
            # Calculate volatility
            closes = [bar['close'] for bar in history]
            returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
            
            volatility = float(np.std(returns)) if returns else 0.02
            
            # Calculate beta (market relativity)
            market_returns = market_state.indicators.get('market_returns', [])
            if len(market_returns) > 20 and len(returns) > 20:
                # Use simple correlation as proxy for beta
                min_len = min(len(market_returns), len(returns))
                correlation = np.corrcoef(market_returns[-min_len:], returns[-min_len:])[0, 1]
                beta = correlation  # Simplified beta calculation
            else:
                beta = 1.0  # Default
            
            # Get other indicators
            rsi = indicators.get('rsi', 50)
            
            # Calculate defensive score components
            volatility_score = max(0, 1 - volatility * 20)  # Lower volatility is better
            beta_score = max(0, 1 - beta)  # Lower beta is better
            rsi_score = 0
            
            # RSI score - prefer moderate RSI in defensive conditions
            if 40 <= rsi <= 60:
                rsi_score = 1.0  # Ideal range
            elif 30 <= rsi < 40 or 60 < rsi <= 70:
                rsi_score = 0.7  # Acceptable range
            else:
                rsi_score = 0.3  # Less ideal
            
            # Fundamental metrics if available
            dividend_yield = 0
            pe_ratio = 20
            
            if hasattr(market_state, 'fundamentals') and symbol in market_state.fundamentals:
                dividend_yield = market_state.fundamentals.get(symbol, {}).get('dividend_yield', 0)
                pe_ratio = market_state.fundamentals.get(symbol, {}).get('pe_ratio', 20)
            
            # Score fundamental factors
            dividend_score = min(1.0, dividend_yield / 0.03)  # Higher yield is better
            pe_score = max(0, 1 - pe_ratio / 40)  # Lower PE is better
            
            # Combine scores with weights
            weights = {
                'volatility': 0.3,
                'beta': 0.25,
                'rsi': 0.15,
                'dividend': 0.2,
                'pe': 0.1
            }
            
            total_score = (
                volatility_score * weights['volatility'] +
                beta_score * weights['beta'] +
                rsi_score * weights['rsi'] +
                dividend_score * weights['dividend'] +
                pe_score * weights['pe']
            )
            
            # Build reason string
            if volatility_score > 0.7:
                reasons.append(f"low volatility ({volatility:.1%})")
            if beta_score > 0.6:
                reasons.append(f"low beta ({beta:.1f})")
            if dividend_score > 0.5:
                reasons.append(f"good yield ({dividend_yield:.1%})")
            if pe_score > 0.6:
                reasons.append(f"reasonable valuation (PE: {pe_ratio:.1f})")
            
            reason = f"Defensive selection: " + ", ".join(reasons)
            
            # Store metrics
            metrics = {
                'score': total_score,
                'volatility': volatility,
                'beta': beta,
                'rsi': rsi,
                'dividend_yield': dividend_yield,
                'pe_ratio': pe_ratio,
                'reason': reason,
                'signal_strength': 0.6,  # Moderate signal strength
                'confidence': min(0.9, 0.5 + total_score * 0.4)  # Scale confidence with score
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating defensive metrics for {symbol}: {e}")
            return None
    
    def get_market_assessment(self):
        """
        Get current market assessment
        
        Returns:
            dict: Market assessment information
        """
        return {
            'state': self.current_market_assessment,
            'defensive_mode': self.defensive_mode_active,
            'hedge_positions': self.hedge_positions,
            'safe_assets': self.safe_asset_positions
        }
    
    def reset(self):
        """
        Reset agent state
        
        Returns:
            None
        """
        super().reset()
        self.current_market_assessment = 'normal'
        self.defensive_mode_active = False
        self.hedge_positions = {}
        self.safe_asset_positions = {}
        self.current_positions = {}
        logger.info(f"Defensive agent '{self.name}' reset")
    
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
        
        # Special adjustment for defensive agent - further reduce size based on market assessment
        if self.current_market_assessment == 'hedging':
            size *= 0.3  # 70% reduction in hedging mode
        elif self.current_market_assessment == 'defensive':
            size *= 0.5  # 50% reduction in defensive mode
        elif self.current_market_assessment == 'cautious':
            size *= 0.7  # 30% reduction in cautious mode
        
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