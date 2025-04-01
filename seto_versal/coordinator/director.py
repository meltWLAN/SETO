#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Director module for SETO-Versal
Manages the coordination of multiple trading agents
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class Director:
    """
    Director class that coordinates multiple trading agents
    
    Responsible for:
    - Collecting trading intentions from all agents
    - Weighting agent influence based on performance and priority
    - Resolving conflicts between agent intentions
    - Generating final decisions for the executor
    """
    
    def __init__(self, config):
        """
        Initialize the director
        
        Args:
            config (dict): Director configuration
        """
        self.config = config
        self.name = config.get('name', 'main_director')
        
        # Weight configuration
        self.performance_weight = config.get('performance_weight', 0.5)
        self.priority_weight = config.get('priority_weight', 0.3)
        self.confidence_weight = config.get('confidence_weight', 0.2)
        
        # Minimum consensus threshold
        self.consensus_threshold = config.get('consensus_threshold', 0.6)
        
        # Decision history
        self.decisions_history = []
        
        # Agent weights
        self.agent_weights = {}
        
        # Agent performance cache
        self.agent_performance = {}
        
        logger.info(f"Director '{self.name}' initialized")
    
    def coordinate(self, agents, market_state):
        """
        Coordinate trading intentions from multiple agents into final decisions
        
        Args:
            agents (list): List of active agents
            market_state (MarketState): Current market state
            
        Returns:
            list: List of trading decisions
        """
        # Collect all intentions from agents
        intentions = self._collect_intentions(agents)
        
        # Update agent weights
        self._update_agent_weights(agents)
        
        # Resolve conflicts and generate decisions
        decisions = self._resolve_conflicts(intentions, market_state)
        
        # Record decisions for historical analysis
        self._record_decisions(decisions)
        
        logger.debug(f"Director created {len(decisions)} decisions from {len(intentions)} intentions")
        
        return decisions
    
    def _collect_intentions(self, agents):
        """
        Collect trading intentions from all active agents
        
        Args:
            agents (list): List of active agents
            
        Returns:
            list: List of all trading intentions
        """
        all_intentions = []
        
        for agent in agents:
            # Skip paused agents
            if hasattr(agent, 'state') and agent.state == 'paused':
                continue
                
            # Get current intentions
            if hasattr(agent, 'current_intentions'):
                all_intentions.extend(agent.current_intentions)
        
        return all_intentions
    
    def _update_agent_weights(self, agents):
        """
        Update influence weights for each agent based on performance and priority
        
        Args:
            agents (list): List of active agents
        """
        # Reset weights
        self.agent_weights = {}
        
        # Get performance metrics for all agents
        for agent in agents:
            if not hasattr(agent, 'name'):
                continue
                
            performance = agent.get_performance() if hasattr(agent, 'get_performance') else {}
            
            # Store latest performance
            self.agent_performance[agent.name] = performance
            
            # Extract key metrics (with defaults if not available)
            win_rate = performance.get('win_rate', 0.5)
            profit_factor = performance.get('profit_factor', 1.0)
            priority = getattr(agent, 'priority', 1)
            
            # Calculate performance score (0.0-1.0)
            performance_score = (win_rate * 0.4) + (min(profit_factor, 3.0) / 3.0 * 0.6)
            
            # Normalize priority (1-10)
            norm_priority = max(1, min(10, priority)) / 10.0
            
            # Calculate overall weight
            weight = (
                performance_score * self.performance_weight +
                norm_priority * self.priority_weight
            )
            
            # Store weight
            self.agent_weights[agent.name] = weight
            
            logger.debug(f"Agent '{agent.name}' assigned weight: {weight:.2f}")
    
    def _resolve_conflicts(self, intentions, market_state):
        """
        Resolve conflicts between agent intentions and generate final decisions
        
        Args:
            intentions (list): List of all trading intentions
            market_state (MarketState): Current market state
            
        Returns:
            list: List of final trading decisions
        """
        decisions = []
        
        # Group intentions by symbol and direction
        grouped_intentions = self._group_intentions(intentions)
        
        # Process each symbol
        for symbol, directions in grouped_intentions.items():
            # Check if there are conflicting directions
            if 'buy' in directions and 'sell' in directions:
                # Resolve the conflict
                decision = self._resolve_direction_conflict(
                    symbol, directions['buy'], directions['sell'], market_state
                )
                if decision:
                    decisions.append(decision)
            else:
                # No conflict in direction, process each direction
                for direction, direction_intentions in directions.items():
                    decision = self._create_decision_from_intentions(
                        symbol, direction, direction_intentions, market_state
                    )
                    if decision:
                        decisions.append(decision)
        
        return decisions
    
    def _group_intentions(self, intentions):
        """
        Group intentions by symbol and direction
        
        Args:
            intentions (list): List of trading intentions
            
        Returns:
            dict: Grouped intentions by symbol and direction
        """
        grouped = {}
        
        for intention in intentions:
            symbol = intention.get('symbol')
            direction = intention.get('direction')
            
            if not symbol or not direction:
                continue
            
            if symbol not in grouped:
                grouped[symbol] = {}
                
            if direction not in grouped[symbol]:
                grouped[symbol][direction] = []
                
            grouped[symbol][direction].append(intention)
        
        return grouped
    
    def _resolve_direction_conflict(self, symbol, buy_intentions, sell_intentions, market_state):
        """
        Resolve conflict between buy and sell intentions for the same symbol
        
        Args:
            symbol (str): Symbol with conflicting intentions
            buy_intentions (list): List of buy intentions
            sell_intentions (list): List of sell intentions
            market_state (MarketState): Current market state
            
        Returns:
            dict: Resolved decision or None
        """
        # Calculate total weighted confidence for each direction
        buy_score = self._calculate_weighted_score(buy_intentions)
        sell_score = self._calculate_weighted_score(sell_intentions)
        
        # Calculate magnitude of decision (proportional to confidence difference)
        confidence_diff = abs(buy_score - sell_score)
        
        # Check if the difference is significant
        if confidence_diff < self.consensus_threshold:
            logger.debug(f"No clear consensus for {symbol}: buy={buy_score:.2f}, sell={sell_score:.2f}")
            return None
            
        # Determine winning direction
        if buy_score > sell_score:
            winning_direction = 'buy'
            winning_intentions = buy_intentions
            score = buy_score
        else:
            winning_direction = 'sell'
            winning_intentions = sell_intentions
            score = sell_score
        
        # Calculate the size based on confidence difference
        size_factor = min(1.0, confidence_diff / (1.0 + confidence_diff))
        
        # Create decision from winning intentions
        return self._create_decision_from_intentions(
            symbol, winning_direction, winning_intentions, market_state, 
            confidence_boost=confidence_diff, size_factor=size_factor
        )
    
    def _calculate_weighted_score(self, intentions):
        """
        Calculate weighted score for a set of intentions
        
        Args:
            intentions (list): List of intentions
            
        Returns:
            float: Weighted score
        """
        if not intentions:
            return 0.0
            
        total_score = 0.0
        total_weight = 0.0
        
        for intention in intentions:
            agent_name = intention.get('agent_name')
            if not agent_name or agent_name not in self.agent_weights:
                continue
                
            agent_weight = self.agent_weights[agent_name]
            confidence = intention.get('confidence', 0.5)
            
            # Combined weight for this intention
            intention_weight = agent_weight + (confidence * self.confidence_weight)
            
            total_score += confidence * intention_weight
            total_weight += intention_weight
        
        # Calculate weighted average
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _create_decision_from_intentions(self, symbol, direction, intentions, market_state, 
                                        confidence_boost=0.0, size_factor=1.0):
        """
        Create a final decision from a set of intentions in the same direction
        
        Args:
            symbol (str): Trading symbol
            direction (str): Trading direction
            intentions (list): List of intentions in the same direction
            market_state (MarketState): Current market state
            confidence_boost (float): Additional confidence from conflict resolution
            size_factor (float): Size adjustment factor
            
        Returns:
            dict: Final trading decision
        """
        if not intentions:
            return None
            
        # Calculate weighted size, confidence and other parameters
        total_weighted_size = 0.0
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        reasons = []
        stops = []
        targets = []
        
        for intention in intentions:
            agent_name = intention.get('agent_name')
            if not agent_name or agent_name not in self.agent_weights:
                continue
                
            agent_weight = self.agent_weights[agent_name]
            confidence = intention.get('confidence', 0.5)
            size = intention.get('size', 0.0)
            
            # Skip intentions with zero size
            if size <= 0:
                continue
                
            # Combined weight for this intention
            intention_weight = agent_weight + (confidence * self.confidence_weight)
            
            # Accumulate weighted values
            total_weighted_size += size * intention_weight
            total_weighted_confidence += confidence * intention_weight
            total_weight += intention_weight
            
            # Collect reasons, stop losses and targets
            if 'reason' in intention and intention['reason'] not in reasons:
                reasons.append(intention['reason'])
                
            if 'stop_loss' in intention and intention['stop_loss'] is not None:
                stops.append(intention['stop_loss'])
                
            if 'target_price' in intention and intention['target_price'] is not None:
                targets.append(intention['target_price'])
        
        # If no valid intentions, return None
        if total_weight <= 0:
            return None
            
        # Calculate final values
        final_size = (total_weighted_size / total_weight) * size_factor
        final_confidence = (total_weighted_confidence / total_weight) + confidence_boost
        
        # Set stop loss and target as average of provided values
        stop_loss = np.mean(stops) if stops else None
        target_price = np.mean(targets) if targets else None
        
        # Get current price for reference
        current_data = market_state.get_ohlcv(symbol)
        current_price = current_data['close'] if current_data else None
        
        # Create the decision
        decision = {
            'symbol': symbol,
            'direction': direction,
            'size': final_size,
            'confidence': min(0.95, final_confidence),  # Cap at 95%
            'timestamp': datetime.now(),
            'price': current_price,
            'stop_loss': stop_loss,
            'target_price': target_price,
            'reasons': reasons[:3],  # Top 3 reasons only
            'contributing_agents': [intention.get('agent_name') for intention in intentions],
            'intention_count': len(intentions)
        }
        
        return decision
    
    def _record_decisions(self, decisions):
        """
        Record decisions for historical analysis
        
        Args:
            decisions (list): List of final trading decisions
        """
        # Add timestamp to decisions
        timestamped_decisions = {
            'timestamp': datetime.now(),
            'decisions': decisions
        }
        
        # Add to history, limiting size
        self.decisions_history.append(timestamped_decisions)
        
        # Limit history size
        max_history = self.config.get('max_history', 100)
        if len(self.decisions_history) > max_history:
            self.decisions_history = self.decisions_history[-max_history:]
    
    def get_decisions_history(self, count=None):
        """
        Get history of recent decisions
        
        Args:
            count (int, optional): Number of recent decisions to return
            
        Returns:
            list: Recent decisions
        """
        if count is None:
            return self.decisions_history
            
        return self.decisions_history[-count:]
    
    def get_agent_weights(self):
        """
        Get current agent weights
        
        Returns:
            dict: Agent weights
        """
        return self.agent_weights
    
    def get_agent_performance(self):
        """
        Get latest agent performance metrics
        
        Returns:
            dict: Agent performance metrics
        """
        return self.agent_performance 