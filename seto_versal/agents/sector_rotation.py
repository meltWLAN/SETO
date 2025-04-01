#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sector rotation agent module for SETO-Versal
Specializes in finding sector rotation opportunities across market cycles
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any, Optional
import uuid

from seto_versal.agents.base import Agent

logger = logging.getLogger(__name__)

class SectorRotationAgent(Agent):
    """
    Sector Rotation Agent (轮动)
    
    Specializes in sector rotation trading with the following characteristics:
    - Identifies leading and lagging sectors in different market phases
    - Rotates capital between sectors based on economic cycle and relative strength
    - Medium-term holding periods (typically 1-6 months)
    - Focuses on broader market patterns rather than individual stock selection
    """
    
    def __init__(self, name, config):
        """
        Initialize the sector rotation agent
        
        Args:
            name (str): Agent name
            config (dict): Agent configuration
        """
        super().__init__(name, config)
        
        self.type = "sector_rotation"
        self.description = "Sector rotation agent focusing on economic cycles"
        
        # Specific settings for sector rotation agent
        self.relative_strength_period = config.get('relative_strength_period', 60)  # 60 trading days
        self.rotation_threshold = config.get('rotation_threshold', 0.05)  # Min relative strength diff
        self.sector_holding_limit = config.get('sector_holding_limit', 3)  # Max number of sectors to hold
        self.min_holding_period = config.get('min_holding_period', 20)  # Min holding days
        self.max_holding_period = config.get('max_holding_period', 90)  # Max holding days
        
        # Set medium risk tolerance by default
        self.risk_tolerance = config.get('risk_tolerance', 0.5)  # Medium risk tolerance (0.0-1.0)
        
        # Economic cycle classification (early, mid, late, recession)
        self.current_cycle = config.get('default_cycle', 'mid')
        
        # Current sector allocations and history
        self.sector_positions = {}  # Current sector positions
        self.sector_history = {}    # Historical sector performance
        
        # Sector mappings
        self.sectors = {
            'tech': ['000001.SZ', '000063.SZ', '002415.SZ'],        # Technology stocks
            'finance': ['600000.SH', '600036.SH', '601398.SH'],     # Financial stocks
            'consumer': ['600519.SH', '000858.SZ', '603288.SH'],    # Consumer stocks
            'healthcare': ['600276.SH', '300015.SZ', '000538.SZ'],  # Healthcare stocks
            'industry': ['601088.SH', '601766.SH', '601857.SH'],    # Industrial stocks
            'materials': ['601899.SH', '600019.SH', '601600.SH'],   # Materials stocks
            'utility': ['600900.SH', '600886.SH', '601985.SH'],     # Utility stocks
            'realestate': ['600048.SH', '001979.SZ', '600606.SH'],  # Real estate stocks
        }
        
        # Define sector preferences for different economic cycles
        self.cycle_preferences = {
            'early': ['materials', 'industry', 'finance'],  # Early cycle preferences
            'mid': ['tech', 'consumer', 'healthcare'],      # Mid cycle preferences
            'late': ['utility', 'finance', 'healthcare'],   # Late cycle preferences
            'recession': ['utility', 'consumer', 'healthcare']  # Recession preferences
        }
        
        logger.info(f"Sector rotation agent '{self.name}' initialized")
        
    def generate_intentions(self, market_state):
        """
        Generate trading intentions based on sector rotation analysis
        
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
        
        # Update economic cycle classification
        self._update_economic_cycle(market_state)
        
        # Update sector relative strength
        sector_strength = self._calculate_sector_strength(market_state)
        
        # Generate buy intentions for strong sectors
        buy_intentions = self._generate_sector_buy_intentions(market_state, sector_strength)
        intentions.extend(buy_intentions)
        
        # Generate sell intentions for weak sectors
        sell_intentions = self._generate_sector_sell_intentions(market_state, sector_strength)
        intentions.extend(sell_intentions)
        
        logger.debug(f"Sector rotation agent generated {len(intentions)} intentions")
        return intentions
    
    def _update_economic_cycle(self, market_state):
        """
        Update the current economic cycle classification
        
        Args:
            market_state (MarketState): Current market state
        """
        # In a real implementation, this would use macroeconomic indicators
        # For now, we'll use a simplified approach based on market trend and interest rates
        
        # Get market trend
        market_trend = market_state.get_market_regime()
        
        # Get interest rate trend (placeholder)
        rate_trend = market_state.indicators.get('interest_rate_trend', 'stable')
        
        # Simple classification rules
        if market_trend == 'bull' and rate_trend == 'rising':
            new_cycle = 'mid'
        elif market_trend == 'bull' and rate_trend == 'stable':
            new_cycle = 'early'
        elif market_trend == 'bull' and rate_trend == 'falling':
            new_cycle = 'late'
        elif market_trend == 'bear':
            new_cycle = 'recession'
        else:
            # Default to mid cycle if unsure
            new_cycle = 'mid'
        
        # Only log if cycle changes
        if new_cycle != self.current_cycle:
            logger.info(f"Economic cycle changed from {self.current_cycle} to {new_cycle}")
            self.current_cycle = new_cycle
    
    def _calculate_sector_strength(self, market_state):
        """
        Calculate relative strength for each sector
        
        Args:
            market_state (MarketState): Current market state
            
        Returns:
            dict: Sector strength metrics
        """
        sector_metrics = {}
        
        # Calculate market benchmark return
        benchmark_return = self._calculate_benchmark_return(market_state)
        
        # Calculate each sector's return
        for sector_name, symbols in self.sectors.items():
            # Get returns for this sector
            returns = []
            momentum_scores = []
            volume_ratios = []
            
            for symbol in symbols:
                symbol_metrics = self._calculate_symbol_metrics(symbol, market_state)
                if symbol_metrics:
                    returns.append(symbol_metrics.get('return', 0))
                    momentum_scores.append(symbol_metrics.get('momentum', 0))
                    volume_ratios.append(symbol_metrics.get('volume_ratio', 1))
            
            if not returns:
                continue
            
            # Calculate average metrics
            avg_return = np.mean(returns)
            avg_momentum = np.mean(momentum_scores)
            avg_volume = np.mean(volume_ratios)
            
            # Calculate relative strength vs market
            relative_return = avg_return - benchmark_return
            
            # Store metrics
            sector_metrics[sector_name] = {
                'return': avg_return,
                'relative_return': relative_return,
                'momentum': avg_momentum,
                'volume_ratio': avg_volume,
                'composite_score': relative_return * 0.6 + avg_momentum * 0.3 + (avg_volume - 1) * 0.1
            }
            
            logger.debug(f"Sector {sector_name}: relative return {relative_return:.2%}, score {sector_metrics[sector_name]['composite_score']:.2f}")
        
        return sector_metrics
    
    def _calculate_benchmark_return(self, market_state):
        """
        Calculate market benchmark return
        
        Args:
            market_state (MarketState): Current market state
            
        Returns:
            float: Benchmark return
        """
        # Use index as benchmark
        benchmark_symbol = market_state.config.get('market_index', '000001.SH')
        
        # Get historical prices
        history = market_state.get_history(benchmark_symbol, self.relative_strength_period)
        
        if not history or len(history) < 2:
            return 0
        
        # Calculate return
        start_price = history[0]['close']
        end_price = history[-1]['close']
        
        if start_price <= 0:
            return 0
            
        return (end_price / start_price) - 1
    
    def _calculate_symbol_metrics(self, symbol, market_state):
        """
        Calculate metrics for a single symbol
        
        Args:
            symbol (str): Symbol to calculate metrics for
            market_state (MarketState): Current market state
            
        Returns:
            dict: Symbol metrics
        """
        # Get historical prices
        history = market_state.get_history(symbol, self.relative_strength_period)
        
        if not history or len(history) < 10:
            return None
        
        try:
            # Calculate return
            start_price = history[0]['close']
            end_price = history[-1]['close']
            
            if start_price <= 0:
                return None
                
            period_return = (end_price / start_price) - 1
            
            # Calculate momentum (10-day rate of change)
            recent_return = (end_price / history[-11]['close']) - 1
            
            # Calculate volume trend
            recent_volumes = [bar['volume'] for bar in history[-10:]]
            earlier_volumes = [bar['volume'] for bar in history[-20:-10]]
            
            avg_recent_volume = np.mean(recent_volumes) if recent_volumes else 0
            avg_earlier_volume = np.mean(earlier_volumes) if earlier_volumes else 0
            
            volume_ratio = avg_recent_volume / avg_earlier_volume if avg_earlier_volume > 0 else 1.0
            
            return {
                'return': period_return,
                'momentum': recent_return,
                'volume_ratio': volume_ratio
            }
            
        except (IndexError, KeyError, ZeroDivisionError):
            logger.warning(f"Error calculating metrics for {symbol}")
            return None
    
    def _generate_sector_buy_intentions(self, market_state, sector_strength):
        """
        Generate buy intentions for strong sectors
        
        Args:
            market_state (MarketState): Current market state
            sector_strength (dict): Sector strength metrics
            
        Returns:
            list: List of buy intentions
        """
        intentions = []
        
        # Get preferred sectors for current cycle
        preferred_sectors = self.cycle_preferences.get(self.current_cycle, [])
        
        # Rank sectors by strength
        ranked_sectors = sorted(
            [(name, metrics['composite_score']) for name, metrics in sector_strength.items()],
            key=lambda x: x[1],
            reverse=True  # Highest score first
        )
        
        # Determine how many new positions we can take
        available_slots = self.sector_holding_limit - len(self.sector_positions)
        if available_slots <= 0:
            return intentions
        
        # Consider top sectors for buy intentions
        considered_sectors = 0
        for sector_name, score in ranked_sectors:
            # Skip if already holding this sector
            if sector_name in self.sector_positions:
                continue
                
            # Skip if score isn't strong enough
            if score < self.rotation_threshold:
                continue
                
            # Extra points for sectors that match the current cycle
            cycle_match_bonus = 0.2 if sector_name in preferred_sectors else 0.0
            
            # Final score with cycle preference
            adjusted_score = score + cycle_match_bonus
            
            # Calculate confidence based on adjusted score
            confidence = min(0.9, 0.5 + adjusted_score)
            
            # Choose a representative stock from the sector
            best_symbol = self._select_best_sector_stock(sector_name, market_state)
            if not best_symbol:
                continue
            
            # Calculate position size
            signal_strength = 0.7  # Medium-strong signal for sector rotation
            position_size = self.calculate_position_size(best_symbol, signal_strength, confidence)
            
            # Get current price
            current_data = market_state.get_ohlcv(best_symbol)
            if not current_data:
                continue
                
            current_price = current_data['close']
            
            # Create intention - no specific target price since this is longer term
            intention = self._create_intention(
                symbol=best_symbol,
                direction='buy',
                size=position_size,
                reason=f"Sector rotation into {sector_name} ({adjusted_score:.2f} score, {sector_strength[sector_name]['relative_return']:.1%} rel return)",
                confidence=confidence
            )
            
            intentions.append(intention)
            
            # Add to sector positions
            self.sector_positions[sector_name] = {
                'symbol': best_symbol,
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'position_size': position_size,
                'cycle_at_entry': self.current_cycle
            }
            
            logger.info(f"Sector rotation agent adding {sector_name} position via {best_symbol}")
            
            # Count this sector
            considered_sectors += 1
            if considered_sectors >= available_slots:
                break
        
        return intentions
    
    def _generate_sector_sell_intentions(self, market_state, sector_strength):
        """
        Generate sell intentions for weak sectors or sectors that have been held long enough
        
        Args:
            market_state (MarketState): Current market state
            sector_strength (dict): Sector strength metrics
            
        Returns:
            list: List of sell intentions
        """
        intentions = []
        
        # Process each current sector position
        for sector_name, position in list(self.sector_positions.items()):
            symbol = position['symbol']
            entry_price = position['entry_price']
            entry_time = position['entry_time']
            position_size = position['position_size']
            
            # Get current data
            current_data = market_state.get_ohlcv(symbol)
            if not current_data:
                continue
                
            current_price = current_data['close']
            
            # Calculate metrics
            days_held = (datetime.now() - entry_time).days
            price_change = (current_price - entry_price) / entry_price
            
            # Get current sector strength
            current_strength = sector_strength.get(sector_name, {'composite_score': 0, 'relative_return': 0})
            
            # Determine if we should exit this position
            exit_signal, exit_reason, exit_confidence = self._check_sector_exit_signal(
                sector_name, days_held, price_change, current_strength, market_state
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
                
                # Remove from sector positions
                del self.sector_positions[sector_name]
                
                # Record in history
                self.sector_history[f"{sector_name}_{entry_time.strftime('%Y%m%d')}"] = {
                    'sector': sector_name,
                    'symbol': symbol,
                    'entry_time': entry_time,
                    'exit_time': datetime.now(),
                    'price_change': price_change,
                    'days_held': days_held
                }
                
                logger.info(f"Sector rotation agent exiting {sector_name} position via {symbol}: {exit_reason}")
        
        return intentions
    
    def _select_best_sector_stock(self, sector_name, market_state):
        """
        Select the best performing stock in a sector
        
        Args:
            sector_name (str): Sector name
            market_state (MarketState): Current market state
            
        Returns:
            str: Symbol of the best stock
        """
        sector_symbols = self.sectors.get(sector_name, [])
        if not sector_symbols:
            return None
        
        # Compare recent performance
        best_symbol = None
        best_score = -float('inf')
        
        for symbol in sector_symbols:
            metrics = self._calculate_symbol_metrics(symbol, market_state)
            if not metrics:
                continue
                
            # Simple scoring: return + momentum + volume boost
            score = metrics['return'] + metrics['momentum'] * 0.5 + (metrics['volume_ratio'] - 1) * 0.2
            
            if score > best_score:
                best_score = score
                best_symbol = symbol
        
        return best_symbol
    
    def _check_sector_exit_signal(self, sector_name, days_held, price_change, current_strength, market_state):
        """
        Check if a sector position should be exited
        
        Args:
            sector_name (str): Sector name
            days_held (int): Days the position has been held
            price_change (float): Price change since entry
            current_strength (dict): Current sector strength metrics
            market_state (MarketState): Current market state
            
        Returns:
            tuple: (exit_signal, exit_reason, exit_confidence)
        """
        # Check minimum holding period
        if days_held < self.min_holding_period:
            return False, "", 0.0
        
        # Check reasons to exit:
        
        # 1. Held too long
        if days_held > self.max_holding_period:
            reason = f"Maximum holding period reached ({days_held} days)"
            confidence = 0.8
            return True, reason, confidence
        
        # 2. Sector has significantly weakened
        composite_score = current_strength.get('composite_score', 0)
        if composite_score < -self.rotation_threshold:
            reason = f"Sector significantly weakened (score: {composite_score:.2f})"
            confidence = 0.7 + min(0.2, abs(composite_score) * 0.5)
            return True, reason, confidence
        
        # 3. Economic cycle changed and sector doesn't match new cycle
        cycle_at_entry = self.sector_positions[sector_name].get('cycle_at_entry')
        if self.current_cycle != cycle_at_entry and sector_name not in self.cycle_preferences.get(self.current_cycle, []):
            reason = f"Economic cycle changed from {cycle_at_entry} to {self.current_cycle}"
            confidence = 0.6
            return True, reason, confidence
        
        # 4. Good profit already achieved
        if price_change > 0.2:  # 20% profit
            reason = f"Profit target reached ({price_change:.1%})"
            confidence = 0.7
            return True, reason, confidence
        
        # 5. Cut loss if significant
        if price_change < -0.1:  # 10% loss
            reason = f"Stop loss triggered ({price_change:.1%})"
            confidence = 0.8
            return True, reason, confidence
        
        # No exit signal
        return False, "", 0.0
    
    def get_current_cycle(self):
        """
        Get the current economic cycle classification
        
        Returns:
            str: Current economic cycle
        """
        return self.current_cycle
    
    def get_sector_positions(self):
        """
        Get current sector positions
        
        Returns:
            dict: Sector positions
        """
        return self.sector_positions
    
    def get_sector_history(self):
        """
        Get historical sector trades
        
        Returns:
            dict: Sector trade history
        """
        return self.sector_history
    
    def reset(self):
        """
        Reset agent state
        
        Returns:
            None
        """
        super().reset()
        self.sector_positions = {}
        # We keep the history for analysis
        logger.info(f"Sector rotation agent '{self.name}' reset")
    
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