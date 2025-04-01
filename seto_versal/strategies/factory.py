#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strategy Factory module for SETO-Versal
Handles dynamic creation and management of trading strategies
"""

import logging
import importlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

class StrategyFactory:
    """
    Factory class for creating and managing trading strategies
    
    Responsible for:
    - Creating strategies based on configuration
    - Managing strategy lifecycle
    - Strategy parameter optimization
    - Strategy performance tracking
    """
    
    def __init__(self, config=None):
        """
        Initialize the strategy factory
        
        Args:
            config (dict, optional): Configuration dictionary with strategy definitions
        """
        self.strategies = {}  # name -> strategy object
        self.strategies_by_category = {}  # category -> list of strategy objects
        
        # Register known strategy types
        self.strategy_types = {
            'breakout': 'seto_versal.strategies.breakout.BreakoutStrategy',
            'moving_average': 'seto_versal.strategies.moving_average.MovingAverageStrategy',
            'macd': 'seto_versal.strategies.moving_average.MovingAverageStrategy',
            'momentum': 'seto_versal.strategies.momentum.MomentumStrategy',
        }
        
        # Initialize from config if provided
        if config:
            self.create_strategies(config)
        
        logger.info(f"Strategy factory initialized with {len(self.strategy_types)} strategy types")
    
    def create_strategies(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create all strategies defined in the configuration
        
        Args:
            config (dict): Configuration dictionary with strategy definitions
            
        Returns:
            dict: Dictionary of created strategies (name -> strategy object)
        """
        created_strategies = {}
        
        if not config or 'strategies' not in config:
            logger.warning("No strategies defined in configuration")
            return created_strategies
        
        for strategy_config in config['strategies']:
            try:
                # Check for required fields
                if 'name' not in strategy_config:
                    logger.warning("Strategy missing name, skipping")
                    continue
                
                if 'type' not in strategy_config:
                    logger.warning(f"Strategy '{strategy_config['name']}' missing type, skipping")
                    continue
                
                # Skip disabled strategies
                if not strategy_config.get('enabled', True):
                    logger.info(f"Strategy '{strategy_config['name']}' is disabled, skipping")
                    continue
                
                # Create the strategy
                strategy = self.create_strategy(
                    strategy_type=strategy_config['type'],
                    name=strategy_config['name'],
                    parameters=strategy_config.get('parameters', {})
                )
                
                if strategy:
                    created_strategies[strategy_config['name']] = strategy
            
            except Exception as e:
                logger.error(f"Error creating strategy: {e}")
        
        # Update the strategies dictionary
        self.strategies.update(created_strategies)
        
        # Update categories
        for name, strategy in created_strategies.items():
            category = getattr(strategy, 'category', 'uncategorized')
            if category not in self.strategies_by_category:
                self.strategies_by_category[category] = []
            self.strategies_by_category[category].append(strategy)
        
        logger.info(f"Created {len(created_strategies)} strategies")
        return created_strategies
    
    def create_strategy(self, strategy_type: str, name: str, parameters: Dict[str, Any] = None) -> Optional[Any]:
        """
        Create a single strategy of the specified type
        
        Args:
            strategy_type (str): Type of strategy to create
            name (str): Name for the strategy
            parameters (dict, optional): Strategy parameters
            
        Returns:
            object: Strategy instance or None if creation failed
        """
        if not parameters:
            parameters = {}
        
        try:
            # Check if the strategy type is supported
            if strategy_type not in self.strategy_types:
                logger.warning(f"Unsupported strategy type: {strategy_type}")
                return None
            
            # Get the class path
            class_path = self.strategy_types[strategy_type]
            
            # Split the module path and class name
            module_path, class_name = class_path.rsplit('.', 1)
            
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the class
            strategy_class = getattr(module, class_name)
            
            # Create the strategy instance
            strategy = strategy_class(name=name, **parameters)
            
            logger.debug(f"Created strategy '{name}' of type '{strategy_type}'")
            
            return strategy
        
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import strategy class: {e}")
        except Exception as e:
            logger.error(f"Error creating strategy '{name}' of type '{strategy_type}': {e}")
        
        return None
    
    def get_strategy(self, name: str) -> Optional[Any]:
        """
        Get a strategy by name
        
        Args:
            name (str): Strategy name
            
        Returns:
            object: Strategy instance or None if not found
        """
        return self.strategies.get(name)
    
    def get_strategies_by_category(self, category: str) -> List[Any]:
        """
        Get strategies by category
        
        Args:
            category (str): Strategy category
            
        Returns:
            list: List of strategy instances
        """
        return self.strategies_by_category.get(category, [])
    
    def get_all_strategies(self) -> Dict[str, Any]:
        """
        Get all registered strategies
        
        Returns:
            dict: Dictionary of strategy instances (name -> strategy object)
        """
        return self.strategies
    
    def optimize_strategy(self, strategy_name: str, historical_data: Dict, target_metric: str = 'win_rate') -> Dict:
        """
        Optimize parameters for a specific strategy
        
        Args:
            strategy_name (str): Name of the strategy to optimize
            historical_data (dict): Historical market data for optimization
            target_metric (str): Metric to optimize for
            
        Returns:
            dict: Optimized parameters
        """
        strategy = self.get_strategy(strategy_name)
        
        if not strategy:
            logger.warning(f"Strategy '{strategy_name}' not found")
            return {}
        
        if not hasattr(strategy, 'optimize_parameters'):
            logger.warning(f"Strategy '{strategy_name}' does not support parameter optimization")
            return {}
        
        try:
            optimized_params = strategy.optimize_parameters(historical_data, target_metric)
            
            logger.info(f"Optimized parameters for strategy '{strategy_name}': {optimized_params}")
            
            # Update strategy with optimized parameters
            for param, value in optimized_params.items():
                if hasattr(strategy, 'parameters'):
                    strategy.parameters[param] = value
            
            return optimized_params
        
        except Exception as e:
            logger.error(f"Error optimizing strategy '{strategy_name}': {e}")
            return {}
    
    def reset_strategies(self):
        """
        Reset all strategy performance metrics
        """
        for name, strategy in self.strategies.items():
            if hasattr(strategy, 'reset_performance'):
                strategy.reset_performance()
            elif hasattr(strategy, 'signals_generated'):
                strategy.signals_generated = 0
                strategy.successful_signals = 0
        
        logger.info(f"Reset performance metrics for {len(self.strategies)} strategies")
    
    def export_strategies(self, file_path: str = None) -> Dict:
        """
        Export strategy configurations to a dictionary or file
        
        Args:
            file_path (str, optional): File path to save the configurations
            
        Returns:
            dict: Strategy configurations
        """
        config = {'strategies': []}
        
        for name, strategy in self.strategies.items():
            strategy_config = {
                'name': name,
                'type': getattr(strategy, 'name', name).split('_')[0],  # Extract type from name
                'enabled': True,
                'parameters': {},
            }
            
            # Get parameters if available
            if hasattr(strategy, 'parameters'):
                strategy_config['parameters'] = strategy.parameters
            elif hasattr(strategy, 'get_parameters'):
                strategy_config['parameters'] = strategy.get_parameters()
            
            # Get performance metrics if available
            if hasattr(strategy, 'get_performance'):
                strategy_config['performance'] = strategy.get_performance()
            
            config['strategies'].append(strategy_config)
        
        # Save to file if requested
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                logger.info(f"Exported {len(self.strategies)} strategies to {file_path}")
            except Exception as e:
                logger.error(f"Error exporting strategies to {file_path}: {e}")
        
        return config
    
    def import_strategies(self, config: Union[Dict, str]) -> int:
        """
        Import strategies from a configuration dictionary or file
        
        Args:
            config (dict or str): Strategy configurations or file path
            
        Returns:
            int: Number of strategies imported
        """
        # If config is a string, assume it's a file path
        if isinstance(config, str):
            try:
                with open(config, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Error importing strategies from {config}: {e}")
                return 0
        
        # Create strategies
        created = self.create_strategies(config)
        
        return len(created) 