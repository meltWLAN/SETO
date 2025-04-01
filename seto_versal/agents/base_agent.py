#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base agent module for SETO-Versal
Provides the foundation for all specialized trading agents
"""

import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Enumeration of agent types"""
    FAST_PROFIT = "fast_profit"
    TREND = "trend"
    REVERSAL = "reversal"
    SECTOR_ROTATION = "sector_rotation"
    DEFENSIVE = "defensive"
    GENERIC = "generic"

class BaseAgent:
    """
    Base class for all trading agents in SETO-Versal
    
    Defines the common interface and functionality that all agents must implement
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base agent
        
        Args:
            config (dict): Agent configuration
        """
        # Basic identification
        self.id = config.get('id', str(uuid.uuid4())[:8])
        self.name = config.get('name', f"Agent-{self.id}")
        self.type = AgentType.GENERIC
        self.description = config.get('description', "Generic trading agent")
        
        # Operational state
        self.is_active = config.get('is_active', True)
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.generation = config.get('generation', 0)
        self.ancestry = config.get('ancestry', [])
        
        # Performance tracking
        self.trades_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.performance_metrics = {}
        
        # Agent-specific settings
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.max_position_size = config.get('max_position_size', 0.25)  # As proportion of total capital
        self.risk_tolerance = config.get('risk_tolerance', 0.5)  # 0.0-1.0 scale
        
        # Strategy references
        self.strategies = []
        self.preferred_strategies = config.get('preferred_strategies', [])
        
        # State variables
        self.current_intentions = {}  # Ticker -> intention details
        self.current_positions = {}   # Ticker -> position details
        self.last_market_state = None  # Last observed market state
        
        logger.info(f"Initialized {self.name} of type {self.type.value}")
    
    def process_market_state(self, market_state):
        """
        Process current market state and update agent's internal state
        
        Args:
            market_state: Current market state object
            
        Returns:
            bool: Whether the market state was processed successfully
        """
        try:
            self.last_market_state = market_state
            self.updated_at = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Error processing market state: {str(e)}")
            return False
    
    def generate_intentions(self, market_state=None):
        """
        Generate trading intentions based on the agent's strategies and market state
        
        Args:
            market_state: Current market state (optional, uses last state if None)
            
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
        
        # This is a placeholder that should be overridden by subclasses
        logger.info(f"{self.name}: Base implementation of generate_intentions called")
        return self.current_intentions
    
    def evaluate_confidence(self, intention, market_state=None):
        """
        Evaluate the confidence level for a specific intention
        
        Args:
            intention (dict): Trading intention to evaluate
            market_state: Current market state (optional)
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        # This is a placeholder that should be overridden by subclasses
        # Default implementation assigns a neutral confidence
        return 0.5
    
    def update_positions(self, positions):
        """
        Update the agent's knowledge of current positions
        
        Args:
            positions (dict): Current positions mapping
            
        Returns:
            bool: Whether positions were updated successfully
        """
        try:
            self.current_positions = positions
            return True
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
            return False
    
    def suggest_position_size(self, ticker, intention=None, market_state=None):
        """
        Suggest position size for a potential trade
        
        Args:
            ticker (str): Ticker symbol
            intention (dict, optional): Trading intention
            market_state (object, optional): Current market state
            
        Returns:
            float: Suggested position size as proportion of total capital
        """
        # This is a placeholder that should be overridden by subclasses
        # Default implementation uses a fixed position size
        return min(self.max_position_size, 0.1)
    
    def update_performance(self, trade_result):
        """
        Update agent performance based on a completed trade
        
        Args:
            trade_result (dict): Result of a trade
            
        Returns:
            dict: Updated performance metrics
        """
        try:
            # Increment counters
            self.trades_count += 1
            if trade_result.get('profit', 0) > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
            
            # Calculate metrics
            win_rate = self.win_count / self.trades_count if self.trades_count > 0 else 0
            
            # Update metrics
            self.performance_metrics.update({
                'trades_count': self.trades_count,
                'win_count': self.win_count,
                'loss_count': self.loss_count,
                'win_rate': win_rate,
                'last_updated': datetime.now().isoformat()
            })
            
            return self.performance_metrics
        
        except Exception as e:
            logger.error(f"Error updating performance: {str(e)}")
            return self.performance_metrics
    
    def adapt(self, adaptation_changes, reason=None):
        """
        Adapt the agent based on external changes or feedback
        
        Args:
            adaptation_changes (dict): Changes to apply
            reason (str, optional): Reason for adaptation
            
        Returns:
            dict: Adaptation results
        """
        try:
            # Apply changes to agent
            self.updated_at = datetime.now()
            
            # Handle generic adaptation values
            if 'active' in adaptation_changes:
                self.is_active = adaptation_changes['active']
            
            if 'risk_multiplier' in adaptation_changes:
                risk_multiplier = adaptation_changes['risk_multiplier']
                self.risk_tolerance = min(1.0, max(0.1, self.risk_tolerance * risk_multiplier))
            
            if 'confidence_threshold' in adaptation_changes:
                conf_multiplier = adaptation_changes.get('confidence_threshold', 1.0)
                self.confidence_threshold = min(0.95, max(0.5, self.confidence_threshold * conf_multiplier))
            
            logger.info(f"{self.name} adapted: {reason or 'unspecified reason'}")
            
            return {
                'success': True,
                'agent_id': self.id,
                'agent_type': self.type.value,
                'changes_applied': adaptation_changes,
                'current_state': {
                    'is_active': self.is_active,
                    'risk_tolerance': self.risk_tolerance,
                    'confidence_threshold': self.confidence_threshold
                }
            }
        
        except Exception as e:
            logger.error(f"Error adapting agent: {str(e)}")
            return {
                'success': False,
                'agent_id': self.id,
                'error': str(e)
            }
    
    def reset(self):
        """
        Reset the agent to its initial state
        
        Returns:
            bool: Whether reset was successful
        """
        try:
            self.current_intentions = {}
            self.current_positions = {}
            self.last_market_state = None
            self.updated_at = datetime.now()
            
            logger.info(f"{self.name} reset to initial state")
            return True
        
        except Exception as e:
            logger.error(f"Error resetting agent: {str(e)}")
            return False
    
    def get_status(self):
        """
        Get the current status of the agent
        
        Returns:
            dict: Agent status information
        """
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'generation': self.generation,
            'performance': self.performance_metrics,
            'current_intentions_count': len(self.current_intentions),
            'current_positions_count': len(self.current_positions)
        }
    
    def __str__(self):
        """String representation of the agent"""
        return f"{self.name} ({self.type.value})"
    
    def __repr__(self):
        """Detailed representation of the agent"""
        return f"<Agent: {self.name}, Type: {self.type.value}, Active: {self.is_active}>" 