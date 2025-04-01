#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base strategy module for SETO-Versal
Defines base strategy class that all trading strategies inherit from
"""

import logging
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class BaseStrategy:
    """
    Base Strategy class that all trading strategies inherit from
    
    This class provides the interface and common functionality for 
    trading strategies in the SETO-Versal system.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize base strategy
        
        Args:
            **kwargs: Strategy parameters
        """
        self.name = kwargs.get('name', self.__class__.__name__)
        self.config = kwargs
        self.category = kwargs.get('category', 'general')
        self.enabled = kwargs.get('enabled', True)
        self.weight = kwargs.get('weight', 1.0)
        
        logger.debug(f"BaseStrategy initialized: {self.name}, category={self.category}")
    
    def generate_signals(
        self, 
        market_data: Dict[str, Dict[datetime, Dict[str, float]]], 
        positions: Dict[str, Dict[str, Any]] = None,
        market_state: Dict[str, Any] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on market data
        
        This is the main method that subclasses should implement.
        
        Args:
            market_data: Dictionary of market data by symbol
            positions: Current positions
            market_state: Current market state information
            **kwargs: Additional parameters
            
        Returns:
            List of trading signals
        """
        raise NotImplementedError("Subclasses must implement generate_signals()")
    
    def get_name(self) -> str:
        """Get strategy name"""
        return self.name
    
    def get_category(self) -> str:
        """Get strategy category"""
        return self.category
    
    def is_enabled(self) -> bool:
        """Check if strategy is enabled"""
        return self.enabled
    
    def enable(self):
        """Enable the strategy"""
        self.enabled = True
        
    def disable(self):
        """Disable the strategy"""
        self.enabled = False
    
    def get_weight(self) -> float:
        """Get strategy weight"""
        return self.weight
    
    def set_weight(self, weight: float):
        """Set strategy weight"""
        self.weight = weight 