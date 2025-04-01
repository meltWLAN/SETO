#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
反转交易代理
专注于市场反转机会
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid

from seto_versal.agents.base import Agent

logger = logging.getLogger(__name__)

class ReversalAgent(Agent):
    """
    Reversal Agent (翻转)
    
    Specializes in contrarian trading with the following characteristics:
    - Enters positions when markets are overbought or oversold
    - Identifies potential reversal points using technical indicators
    - Focuses on mean reversion opportunities
    - Higher win rate but smaller profit per trade
    """
    
    def __init__(self, name, config):
        """
        Initialize the reversal agent
        
        Args:
            name (str): Agent name
            config (dict): Agent configuration
        """
        super().__init__(name, config)
        
        self.type = "reversal"
        self.description = "Reversal agent focusing on overbought/oversold conditions"
        
        # Specific settings for reversal agent
        self.rsi_upper = config.get('rsi_upper', 70)  # RSI overbought threshold
        self.rsi_lower = config.get('rsi_lower', 30)  # RSI oversold threshold
        self.bb_threshold = config.get('bb_threshold', 0.9)  # % from price to bollinger band
        self.profit_target = config.get('profit_target', 0.04)  # Target profit (4%)
        self.stop_loss = config.get('stop_loss', 0.03)  # Stop loss (3%)
        
        # Since reversal trades are more frequent but smaller gains
        self.max_position_size = config.get('max_position_size', 0.15)  # Smaller position size
        self.risk_tolerance = config.get('risk_tolerance', 0.4)  # Lower risk tolerance
        
        # Potential opportunities tracking
        self.reversal_candidates = {}
        self.current_positions = {}  # Track current positions
        
        logger.info(f"Reversal agent '{self.name}' initialized")
    
    def generate_intentions(self, market_state):
        """
        Generate trading intentions based on reversal patterns
        
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
        
        # Update reversal candidates
        self._update_reversal_candidates(market_state)
        
        # Generate buy signals (from oversold conditions)
        buy_intentions = self._generate_buy_intentions(market_state)
        intentions.extend(buy_intentions)
        
        # Generate sell signals (from overbought conditions and existing positions)
        sell_intentions = self._generate_sell_intentions(market_state)
        intentions.extend(sell_intentions)
        
        logger.debug(f"Reversal agent generated {len(intentions)} intentions")
        return intentions
    
    def _update_reversal_candidates(self, market_state):
        """
        Update the list of potential reversal candidates
        
        Args:
            market_state (MarketState): Current market state
        """
        # Scan all available symbols
        for symbol in market_state.symbols:
            # Get current data and indicators
            current_data = market_state.get_ohlcv(symbol)
            if not current_data:
                continue
            
            indicators = market_state.indicators.get(symbol, {})
            
            # Check if the symbol shows reversal potential
            reversal_score, direction, reason = self._check_reversal_potential(
                symbol, current_data, indicators, market_state
            )
            
            # If score is above threshold, add or update in candidates list
            if reversal_score > 0.5:
                self.reversal_candidates[symbol] = {
                    'updated_time': datetime.now(),
                    'score': reversal_score,
                    'direction': direction,
                    'reason': reason,
                    'price': current_data['close']
                }
                logger.debug(f"Added/updated {symbol} in reversal candidates: {direction} with score {reversal_score:.2f}")
        
        # Clean up old candidates
        current_time = datetime.now()
        for symbol in list(self.reversal_candidates.keys()):
            if (current_time - self.reversal_candidates[symbol]['updated_time']) > timedelta(days=3):
                del self.reversal_candidates[symbol]
    
    def _check_reversal_potential(self, symbol, current_data, indicators, market_state):
        """
        Check if a symbol has reversal potential
        
        Args:
            symbol (str): Symbol to check
            current_data (dict): Current OHLCV data
            indicators (dict): Technical indicators
            market_state (MarketState): Current market state
            
        Returns:
            tuple: (reversal_score, direction, reason)
        """
        score = 0.0
        signals = []
        reason = []
        direction = None
        
        # Check RSI for oversold/overbought
        rsi = indicators.get('rsi')
        if rsi is not None:
            if rsi < self.rsi_lower:
                # Oversold - bullish reversal potential
                rsi_score = (self.rsi_lower - rsi) / self.rsi_lower
                signals.append((rsi_score, 'buy'))
                reason.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > self.rsi_upper:
                # Overbought - bearish reversal potential
                rsi_score = (rsi - self.rsi_upper) / (100 - self.rsi_upper)
                signals.append((rsi_score, 'sell'))
                reason.append(f"RSI overbought ({rsi:.1f})")
        
        # Check Bollinger Bands
        bb = indicators.get('bollinger_bands')
        if bb and 'upper' in bb and 'lower' in bb:
            price = current_data['close']
            upper = bb['upper']
            lower = bb['lower']
            middle = bb.get('middle', (upper + lower) / 2)
            
            # Calculate distance to bands as percentage
            bb_range = upper - lower
            if bb_range > 0:
                upper_dist = (upper - price) / bb_range
                lower_dist = (price - lower) / bb_range
                
                if lower_dist < 0:  # Price below lower band
                    bb_score = min(1.0, abs(lower_dist) * 2)
                    signals.append((bb_score, 'buy'))
                    reason.append(f"Price below lower BB")
                elif upper_dist < 0:  # Price above upper band
                    bb_score = min(1.0, abs(upper_dist) * 2)
                    signals.append((bb_score, 'sell'))
                    reason.append(f"Price above upper BB")
        
        # Check for oversold/overbought stochastic
        stoch = indicators.get('stochastic')
        if stoch and 'k' in stoch and 'd' in stoch:
            k = stoch['k']
            d = stoch['d']
            
            if k < 20 and d < 20:
                # Oversold
                stoch_score = (20 - min(k, d)) / 20
                signals.append((stoch_score, 'buy'))
                reason.append(f"Stochastic oversold ({k:.1f}/{d:.1f})")
            elif k > 80 and d > 80:
                # Overbought
                stoch_score = (max(k, d) - 80) / 20
                signals.append((stoch_score, 'sell'))
                reason.append(f"Stochastic overbought ({k:.1f}/{d:.1f})")
        
        # Check for doji or hammer/shooting star candlestick patterns
        if 'open' in current_data and 'high' in current_data and 'low' in current_data:
            open_price = current_data['open']
            high = current_data['high']
            low = current_data['low']
            close = current_data['close']
            
            body_size = abs(close - open_price)
            total_range = high - low
            
            if total_range > 0:
                body_ratio = body_size / total_range
                
                # Doji pattern (small body)
                if body_ratio < 0.2:
                    doji_score = 0.5
                    if close > open_price:
                        signals.append((doji_score, 'buy'))
                    else:
                        signals.append((doji_score, 'sell'))
                    reason.append(f"Doji pattern")
                
                # Hammer pattern (bullish reversal)
                lower_wick = min(open_price, close) - low
                if lower_wick / total_range > 0.6 and body_ratio < 0.3:
                    hammer_score = 0.7
                    signals.append((hammer_score, 'buy'))
                    reason.append(f"Hammer pattern")
                
                # Shooting star pattern (bearish reversal)
                upper_wick = high - max(open_price, close)
                if upper_wick / total_range > 0.6 and body_ratio < 0.3:
                    star_score = 0.7
                    signals.append((star_score, 'sell'))
                    reason.append(f"Shooting star pattern")
        
        # Determine overall direction and score
        if signals:
            buy_signals = [score for score, dir in signals if dir == 'buy']
            sell_signals = [score for score, dir in signals if dir == 'sell']
            
            if buy_signals and sum(buy_signals) > sum(sell_signals):
                direction = 'buy'
                score = sum(buy_signals) / len(buy_signals)
            elif sell_signals:
                direction = 'sell'
                score = sum(sell_signals) / len(sell_signals)
        
        return score, direction, ", ".join(reason)
    
    def _generate_buy_intentions(self, market_state):
        """
        Generate buy intentions from oversold conditions
        
        Args:
            market_state (MarketState): Current market state
            
        Returns:
            list: List of buy intentions
        """
        intentions = []
        
        # Process each reversal candidate
        for symbol, data in list(self.reversal_candidates.items()):
            # Only consider buy signals
            if data['direction'] != 'buy':
                continue
            
            # Skip if already in our positions
            if symbol in self.current_positions:
                continue
            
            # Get current data
            current_data = market_state.get_ohlcv(symbol)
            if not current_data:
                continue
            
            # Check for confirmation of reversal
            confirmed, confirmation_reason, confidence = self._check_reversal_confirmation(
                symbol, data, current_data, market_state, 'buy'
            )
            
            if confirmed:
                # Calculate position size
                signal_strength = 0.6  # More conservative for reversals
                position_size = self.calculate_position_size(symbol, signal_strength, confidence)
                
                # Calculate target price and stop loss
                current_price = current_data['close']
                target_price = current_price * (1 + self.profit_target)
                stop_loss_price = current_price * (1 - self.stop_loss)
                
                # Generate full reason
                full_reason = f"Reversal BUY: {data['reason']} - {confirmation_reason}"
                
                # Create intention
                intention = self._create_intention(
                    symbol=symbol,
                    direction='buy',
                    size=position_size,
                    reason=full_reason,
                    confidence=confidence,
                    target_price=target_price,
                    stop_loss=stop_loss_price
                )
                
                intentions.append(intention)
                
                # Add to current positions
                self.current_positions[symbol] = {
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'position_size': position_size,
                    'stop_loss': stop_loss_price,
                    'target_price': target_price
                }
                
                logger.info(f"Reversal agent generated BUY intention for {symbol}: {confidence:.2f} confidence")
        
        return intentions
    
    def _generate_sell_intentions(self, market_state):
        """
        Generate sell intentions from overbought conditions and existing positions
        
        Args:
            market_state (MarketState): Current market state
            
        Returns:
            list: List of sell intentions
        """
        intentions = []
        
        # Process reversal candidates for new short positions
        for symbol, data in list(self.reversal_candidates.items()):
            # Only consider sell signals
            if data['direction'] != 'sell':
                continue
                
            # Get current data
            current_data = market_state.get_ohlcv(symbol)
            if not current_data:
                continue
            
            # Check for confirmation of reversal
            confirmed, confirmation_reason, confidence = self._check_reversal_confirmation(
                symbol, data, current_data, market_state, 'sell'
            )
            
            if confirmed:
                # For reversal agent, we focus more on exit signals than shorting
                # In a real implementation, this would be a short position
                
                # For demonstration, we'll just log a detection of shorting opportunity
                logger.debug(f"Reversal agent detected shorting opportunity for {symbol}: {data['reason']} - {confirmation_reason}")
        
        # Check existing positions for exit signals
        for symbol, position in list(self.current_positions.items()):
            # Get current data
            current_data = market_state.get_ohlcv(symbol)
            if not current_data:
                continue
            
            current_price = current_data['close']
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            target_price = position['target_price']
            position_size = position['position_size']
            
            # Check exit criteria
            exit_signal, exit_reason, exit_confidence = self._check_exit_signal(
                symbol, current_data, position, market_state
            )
            
            if exit_signal:
                # Create exit intention
                intention = self._create_intention(
                    symbol=symbol,
                    direction='sell',
                    size=position_size,  # Sell entire position
                    reason=exit_reason,
                    confidence=exit_confidence
                )
                
                intentions.append(intention)
                
                # Remove from current positions
                del self.current_positions[symbol]
                
                logger.info(f"Reversal agent generated SELL intention for {symbol}: {exit_reason}")
        
        return intentions
    
    def _check_reversal_confirmation(self, symbol, reversal_data, current_data, market_state, direction):
        """
        Check for confirmation of a reversal signal
        
        Args:
            symbol (str): Symbol to check
            reversal_data (dict): Reversal candidate data
            current_data (dict): Current OHLCV data
            market_state (MarketState): Current market state
            direction (str): 'buy' or 'sell'
            
        Returns:
            tuple: (confirmed, reason, confidence)
        """
        indicators = market_state.indicators.get(symbol, {})
        price = current_data['close']
        original_price = reversal_data['price']
        price_change = (price - original_price) / original_price
        
        # Default values
        confirmed = False
        confidence = 0.0
        reasons = []
        
        # Time-based decay of confidence
        time_elapsed = (datetime.now() - reversal_data['updated_time']).total_seconds() / 3600  # hours
        time_factor = max(0, 1 - (time_elapsed / 48))  # Decay over 48 hours
        
        if direction == 'buy':
            # Bullish confirmation: Positive momentum after oversold condition
            
            # Check for price confirmation (higher low or positive momentum)
            if price_change > 0.01:
                confirmed = True
                confidence += 0.3
                reasons.append(f"Price up {price_change:.1%} from signal")
            
            # Check for RSI confirmation (rising from oversold)
            rsi = indicators.get('rsi')
            prev_rsi = indicators.get('prev_rsi', 0)
            if rsi is not None and prev_rsi and rsi > prev_rsi and rsi < 50:
                confirmed = True
                confidence += 0.2
                reasons.append(f"RSI rising from oversold ({rsi:.1f})")
            
            # Check for volume confirmation
            if 'volume' in current_data:
                # 安全获取平均成交量
                avg_vol = current_data['volume']  # 默认使用当前成交量
                
                # 尝试从history中获取平均成交量（如果存在）
                if hasattr(market_state, 'history') and symbol in market_state.history:
                    avg_vol = market_state.history[symbol].get('avg_volume', current_data['volume'])
                
                if current_data['volume'] > avg_vol * 1.2:
                    confidence += 0.2
                    reasons.append(f"Above average volume")
        
        elif direction == 'sell':
            # Bearish confirmation: Negative momentum after overbought condition
            
            # Check for price confirmation (lower high or negative momentum)
            if price_change < -0.01:
                confirmed = True
                confidence += 0.3
                reasons.append(f"Price down {abs(price_change):.1%} from signal")
            
            # Check for RSI confirmation (falling from overbought)
            rsi = indicators.get('rsi')
            prev_rsi = indicators.get('prev_rsi', 100)
            if rsi is not None and prev_rsi and rsi < prev_rsi and rsi > 50:
                confirmed = True
                confidence += 0.2
                reasons.append(f"RSI falling from overbought ({rsi:.1f})")
            
            # Check for volume confirmation
            if 'volume' in current_data:
                # 安全获取平均成交量
                avg_vol = current_data['volume']  # 默认使用当前成交量
                
                # 尝试从history中获取平均成交量（如果存在）
                if hasattr(market_state, 'history') and symbol in market_state.history:
                    avg_vol = market_state.history[symbol].get('avg_volume', current_data['volume'])
                
                if current_data['volume'] > avg_vol * 1.2:
                    confidence += 0.2
                    reasons.append(f"Above average volume")
        
        # Final confirmation requires at least one confirmation signal
        if confirmed:
            # Apply original signal confidence as a factor
            base_confidence = reversal_data['score']
            confidence = (confidence + base_confidence) / 2
            
            # Apply time decay
            confidence *= time_factor
            
            # Apply market regime adjustment
            regime = market_state.get_market_regime()
            if regime == 'bear' and direction == 'buy':
                confidence *= 0.8  # Lower confidence for buy in bear market
            elif regime == 'bull' and direction == 'sell':
                confidence *= 0.8  # Lower confidence for sell in bull market
        
        # Ensure confidence is in valid range
        confidence = min(0.95, max(0, confidence))
        
        return confirmed, ", ".join(reasons), confidence
    
    def _check_exit_signal(self, symbol, current_data, position, market_state):
        """
        Check if a position should be exited
        
        Args:
            symbol (str): Symbol to check
            current_data (dict): Current OHLCV data
            position (dict): Position data
            market_state (MarketState): Current market state
            
        Returns:
            tuple: (exit_signal, exit_reason, exit_confidence)
        """
        current_price = current_data['close']
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        target_price = position['target_price']
        
        # Check stop loss
        if current_price <= stop_loss:
            return True, f"Stop loss triggered at {current_price:.2f}", 0.9
        
        # Check profit target
        if current_price >= target_price:
            return True, f"Profit target reached at {current_price:.2f}", 0.8
        
        # Check reversal of the reversal (return to mean complete)
        indicators = market_state.indicators.get(symbol, {})
        rsi = indicators.get('rsi')
        
        if rsi is not None:
            # If RSI has moved from oversold to overbought range, exit the position
            if rsi > 60:  # Moderately overbought - suitable exit point
                return True, f"RSI indicates potential reversal completion ({rsi:.1f})", 0.7
        
        # Check time-based exit
        days_held = (datetime.now() - position['entry_time']).days
        if days_held > 10:  # Max holding period for reversal trades
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct > 0:
                return True, f"Time-based exit after {days_held} days with {profit_pct:.1%} profit", 0.7
        
        # No exit signal
        return False, "", 0.0
    
    def reset(self):
        """
        Reset agent state, including reversal candidates and positions
        
        Returns:
            None
        """
        super().reset()
        self.reversal_candidates = {}
        self.current_positions = {}
        logger.info(f"Reversal agent '{self.name}' reset")
    
    def get_reversal_candidates(self):
        """
        Get the current reversal candidates
        
        Returns:
            dict: Reversal candidates dictionary
        """
        return self.reversal_candidates
    
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