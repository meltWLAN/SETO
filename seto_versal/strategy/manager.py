#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strategy manager for SETO-Versal trading system.

This module implements strategy definition, management, and execution.
It serves as an integration point between the risk controller and
strategy evolution components.
"""

import os
import enum
import logging
import json
import importlib
import inspect
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Type
from datetime import datetime, timedelta

# Import risk and evolution components
from seto_versal.risk.controller import RiskController, RiskLevel


class Strategy:
    """
    Base class for all trading strategies.
    
    Strategies must implement the following methods:
    - initialize: Set up strategy with parameters and indicators
    - calculate_signals: Generate trading signals from market data
    - on_bar: Process a new price bar and decide on action
    """
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name identifier
            parameters: Strategy parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.indicators = {}
        self.position = None
        self.last_signal = None
        self.logger = logging.getLogger(f"strategy.{name}")
    
    def initialize(self, context: Dict[str, Any]) -> None:
        """
        Initialize the strategy with context data.
        
        Args:
            context: Dictionary containing additional data like
                     universe, account, broker, etc.
        """
        self.logger.info(f"Initializing strategy: {self.name}")
        # To be implemented by subclasses
    
    def calculate_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate technical indicators based on price data.
        
        Args:
            data: Dictionary containing OHLCV data
            
        Returns:
            Dictionary of calculated indicator values
        """
        # To be implemented by subclasses
        return {}
    
    def calculate_signals(self, data: Dict[str, Any], indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on data and indicators.
        
        Args:
            data: Dictionary containing OHLCV data
            indicators: Dictionary of technical indicators
            
        Returns:
            Dictionary containing trading signals
        """
        # To be implemented by subclasses
        return {}
    
    def on_bar(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a new price bar and decide on action.
        
        Args:
            data: Dictionary containing OHLCV data
            context: Current trading context
            
        Returns:
            Trade action dictionary or None if no action
        """
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        self.indicators = indicators
        
        # Generate signals
        signals = self.calculate_signals(data, indicators)
        self.last_signal = signals
        
        # Determine action based on signals
        action = self.determine_action(signals, context)
        
        return action
    
    def determine_action(self, signals: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Determine trading action based on signals and context.
        
        Args:
            signals: Dictionary of trading signals
            context: Current trading context
            
        Returns:
            Trade action dictionary or None if no action
        """
        # To be implemented by subclasses
        return None
    
    def validate(self) -> bool:
        """
        Validate that the strategy has all required components.
        
        Returns:
            Boolean indicating if strategy is valid
        """
        # Check if required methods are implemented
        required_methods = ['initialize', 'calculate_signals', 'on_bar']
        
        for method in required_methods:
            if not hasattr(self, method) or not callable(getattr(self, method)):
                self.logger.error(f"Strategy {self.name} missing required method: {method}")
                return False
        
        return True
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the strategy to a dictionary.
        
        Returns:
            Dictionary representation of strategy
        """
        return {
            'name': self.name,
            'parameters': self.parameters,
            'type': self.__class__.__name__
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'Strategy':
        """
        Create a strategy instance from serialized data.
        
        Args:
            data: Dictionary containing serialized strategy
            
        Returns:
            Strategy instance
        """
        name = data.get('name', 'unnamed')
        parameters = data.get('parameters', {})
        
        return cls(name=name, parameters=parameters)


class StrategyManager:
    """
    Manager for trading strategies.
    
    The strategy manager:
    - Loads and registers strategies
    - Integrates with risk controller
    - Provides strategy execution
    - Tracks performance metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the strategy manager.
        
        Args:
            config: Dictionary with configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing strategy manager")
        
        # Configuration
        self.name = config.get('name', 'strategy_manager')
        self.strategy_dir = config.get('strategy_dir', 'strategies')
        
        # Integration with risk controller
        risk_config = config.get('risk', {})
        self.risk_controller = RiskController(risk_config) if risk_config else None
        
        # Register known strategy types
        self.strategy_types = {}
        self.load_strategy_types()
        
        # Active strategies
        self.strategies = {}
        
        # Strategy performance tracking
        self.performance = {}
        
        # Strategy state history
        self.state_history = []
        
        self.logger.info(f"Strategy manager initialized with {len(self.strategy_types)} strategy types")
    
    def load_strategy_types(self) -> None:
        """
        Load available strategy types from the strategy directory.
        """
        try:
            # Check if strategy directory exists
            if not os.path.exists(self.strategy_dir):
                self.logger.warning(f"Strategy directory {self.strategy_dir} does not exist.")
                return
            
            # Get all Python files in the directory
            strategy_files = [f for f in os.listdir(self.strategy_dir) 
                             if f.endswith('.py') and not f.startswith('_')]
            
            for file in strategy_files:
                module_name = file[:-3]  # Remove .py extension
                
                try:
                    # Import the module
                    module_path = f"{self.strategy_dir}.{module_name}"
                    module = importlib.import_module(module_path)
                    
                    # Find strategy classes in the module
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and issubclass(obj, Strategy) 
                            and obj != Strategy):
                            self.strategy_types[name] = obj
                            self.logger.info(f"Registered strategy type: {name}")
                
                except Exception as e:
                    self.logger.error(f"Error loading strategy from {file}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error loading strategy types: {str(e)}")
    
    def register_strategy(self, strategy_instance: Strategy) -> bool:
        """
        Register a strategy with the manager.
        
        Args:
            strategy_instance: Strategy instance to register
            
        Returns:
            Boolean indicating success
        """
        if not isinstance(strategy_instance, Strategy):
            self.logger.error(f"Cannot register non-Strategy object: {strategy_instance}")
            return False
        
        if not strategy_instance.validate():
            self.logger.error(f"Strategy validation failed: {strategy_instance.name}")
            return False
        
        # Add to active strategies
        self.strategies[strategy_instance.name] = strategy_instance
        
        # Initialize performance tracking
        self.performance[strategy_instance.name] = {
            'trades': [],
            'metrics': {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'expectancy': 0.0
            },
            'last_update': datetime.now().isoformat()
        }
        
        self.logger.info(f"Registered strategy: {strategy_instance.name}")
        return True
    
    def create_strategy(self, 
                      strategy_type: str, 
                      name: str, 
                      parameters: Dict[str, Any]) -> Optional[Strategy]:
        """
        Create a new strategy instance.
        
        Args:
            strategy_type: Type of strategy to create
            name: Name for the new strategy
            parameters: Strategy parameters
            
        Returns:
            Strategy instance or None if creation fails
        """
        if strategy_type not in self.strategy_types:
            self.logger.error(f"Unknown strategy type: {strategy_type}")
            return None
        
        try:
            # Create the strategy instance
            strategy_class = self.strategy_types[strategy_type]
            strategy = strategy_class(name=name, parameters=parameters)
            
            # Validate the strategy
            if not strategy.validate():
                self.logger.error(f"Created strategy failed validation: {name}")
                return None
            
            return strategy
        
        except Exception as e:
            self.logger.error(f"Error creating strategy {name}: {str(e)}")
            return None
    
    def initialize_strategies(self, context: Dict[str, Any]) -> None:
        """
        Initialize all registered strategies with context.
        
        Args:
            context: Common context dictionary for all strategies
        """
        self.logger.info(f"Initializing {len(self.strategies)} strategies")
        
        for name, strategy in self.strategies.items():
            try:
                strategy.initialize(context)
                self.logger.info(f"Initialized strategy: {name}")
            except Exception as e:
                self.logger.error(f"Error initializing strategy {name}: {str(e)}")
    
    def on_bar(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Process a new price bar with all strategies.
        
        Args:
            data: Dictionary containing OHLCV data
            context: Current trading context
            
        Returns:
            Dictionary mapping strategy names to their actions
        """
        actions = {}
        
        for name, strategy in self.strategies.items():
            try:
                action = strategy.on_bar(data, context)
                
                # Validate action with risk controller if available
                if action and self.risk_controller:
                    is_valid, reason = self.risk_controller.validate_trade(
                        action, context.get('portfolio'), context.get('market'))
                    
                    if not is_valid:
                        self.logger.warning(f"Trade rejected by risk controller for {name}: {reason}")
                        action = None
                
                actions[name] = action
            
            except Exception as e:
                self.logger.error(f"Error running strategy {name} on_bar: {str(e)}")
                actions[name] = None
        
        return actions
    
    def record_trade_result(self, 
                          strategy_name: str, 
                          trade_result: Dict[str, Any]) -> None:
        """
        Record the result of a trade for performance tracking.
        
        Args:
            strategy_name: Name of the strategy
            trade_result: Trade result dictionary
        """
        if strategy_name not in self.strategies:
            self.logger.error(f"Cannot record trade for unknown strategy: {strategy_name}")
            return
        
        # Record the trade
        self.performance[strategy_name]['trades'].append(trade_result)
        
        # Update performance metrics
        self._update_performance_metrics(strategy_name)
        
        # Update last update timestamp
        self.performance[strategy_name]['last_update'] = datetime.now().isoformat()
        
        # Update risk controller if available
        if self.risk_controller:
            self.risk_controller.record_trade_result(trade_result)
    
    def _update_performance_metrics(self, strategy_name: str) -> None:
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy_name: Name of the strategy
        """
        if strategy_name not in self.performance:
            return
        
        trades = self.performance[strategy_name]['trades']
        
        if not trades:
            return
        
        # Calculate basic metrics
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        losses = sum(1 for t in trades if t.get('pnl', 0) < 0)
        
        win_rate = wins / len(trades) if len(trades) > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate expectancy
        avg_win = gross_profit / wins if wins > 0 else 0
        avg_loss = gross_loss / losses if losses > 0 else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Simplified Sharpe ratio (just using PnL series standard deviation)
        pnl_series = [t.get('pnl', 0) for t in trades]
        avg_pnl = sum(pnl_series) / len(pnl_series) if pnl_series else 0
        
        # Calculate variance and standard deviation
        variance = sum((p - avg_pnl) ** 2 for p in pnl_series) / len(pnl_series) if pnl_series else 0
        std_dev = variance ** 0.5 if variance > 0 else 1  # Avoid division by zero
        
        sharpe_ratio = avg_pnl / std_dev if std_dev > 0 else 0
        
        # Calculate max drawdown
        # This is a simplified calculation that doesn't consider timing
        cumulative_pnl = [sum(pnl_series[:i+1]) for i in range(len(pnl_series))]
        max_drawdown = 0
        peak = cumulative_pnl[0]
        
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = (peak - pnl) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Update metrics
        self.performance[strategy_name]['metrics'] = {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'expectancy': expectancy,
            'trade_count': len(trades),
            'win_count': wins,
            'loss_count': losses
        }
    
    def get_performance(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for a strategy or all strategies.
        
        Args:
            strategy_name: Name of the strategy or None for all
            
        Returns:
            Dictionary with performance metrics
        """
        if strategy_name:
            if strategy_name not in self.performance:
                self.logger.error(f"Unknown strategy: {strategy_name}")
                return {}
            return self.performance[strategy_name]
        
        # Return all strategy performance
        return self.performance
    
    def get_strategy(self, strategy_name: str) -> Optional[Strategy]:
        """
        Get a strategy by name.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy instance or None if not found
        """
        return self.strategies.get(strategy_name)
    
    def save_strategies(self, filepath: str = None) -> bool:
        """
        Save all strategies to a file.
        
        Args:
            filepath: Path to save strategies file
            
        Returns:
            Boolean indicating success
        """
        if filepath is None:
            filepath = f"data/strategies/{self.name}_strategies.json"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Serialize all strategies
        serialized = {
            'name': self.name,
            'strategies': {
                name: strategy.serialize()
                for name, strategy in self.strategies.items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(serialized, f, indent=2)
            
            self.logger.info(f"Saved {len(self.strategies)} strategies to {filepath}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving strategies: {str(e)}")
            return False
    
    def load_strategies(self, filepath: str = None) -> bool:
        """
        Load strategies from a file.
        
        Args:
            filepath: Path to strategies file
            
        Returns:
            Boolean indicating success
        """
        if filepath is None:
            filepath = f"data/strategies/{self.name}_strategies.json"
        
        if not os.path.exists(filepath):
            self.logger.warning(f"Strategies file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Clear existing strategies
            self.strategies = {}
            
            # Load each strategy
            strategies_data = data.get('strategies', {})
            
            for name, strategy_data in strategies_data.items():
                strategy_type = strategy_data.get('type')
                
                if strategy_type not in self.strategy_types:
                    self.logger.warning(f"Unknown strategy type: {strategy_type}")
                    continue
                
                try:
                    strategy_class = self.strategy_types[strategy_type]
                    strategy = strategy_class.deserialize(strategy_data)
                    
                    # Register the strategy
                    self.register_strategy(strategy)
                
                except Exception as e:
                    self.logger.error(f"Error deserializing strategy {name}: {str(e)}")
            
            self.logger.info(f"Loaded {len(self.strategies)} strategies from {filepath}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error loading strategies: {str(e)}")
            return False
    
    def generate_strategy_report(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a report for a strategy or all strategies.
        
        Args:
            strategy_name: Name of the strategy or None for all
            
        Returns:
            Dictionary with strategy information
        """
        if strategy_name:
            if strategy_name not in self.strategies:
                self.logger.error(f"Unknown strategy: {strategy_name}")
                return {}
            
            strategy = self.strategies[strategy_name]
            performance = self.performance.get(strategy_name, {})
            
            # Generate report for a single strategy
            report = {
                'name': strategy_name,
                'type': strategy.__class__.__name__,
                'parameters': strategy.parameters,
                'performance': performance.get('metrics', {}),
                'trade_count': len(performance.get('trades', [])),
                'last_update': performance.get('last_update')
            }
            
            return report
        
        # Generate report for all strategies
        all_reports = {
            'timestamp': datetime.now().isoformat(),
            'strategy_count': len(self.strategies),
            'strategies': {}
        }
        
        for name in self.strategies:
            all_reports['strategies'][name] = self.generate_strategy_report(name)
        
        return all_reports
    
    def integrate_with_evolution(self, evolution_manager: Any) -> None:
        """
        Integrate with an evolution manager for strategy improvement.
        
        Args:
            evolution_manager: EvolutionManager instance
        """
        if not evolution_manager:
            self.logger.error("No evolution manager provided for integration")
            return
        
        self.logger.info("Integrating strategy manager with evolution manager")
        
        # Define the evaluation function for strategies
        def evaluate_strategy(individual: Dict[str, Any]) -> Dict[str, float]:
            """
            Evaluate a strategy individual from the evolution manager.
            
            Args:
                individual: Strategy individual dictionary
                
            Returns:
                Dictionary of performance metrics
            """
            # Extract strategy information
            strategy_type = individual.get('type')
            parameters = individual.get('parameters', {})
            name = f"evolved_{individual.get('id')}"
            
            # Create and register the strategy
            strategy = self.create_strategy(strategy_type, name, parameters)
            
            if not strategy:
                # Return poor performance if creation fails
                return {
                    'sharpe_ratio': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'expectancy': 0.0,
                    'max_drawdown': 1.0  # Maximum possible drawdown
                }
            
            # Register the strategy temporarily
            self.register_strategy(strategy)
            
            # Get performance (assumes strategy has been backtested)
            # In a real system, this would likely run a backtest here
            performance = self.performance.get(name, {}).get('metrics', {})
            
            # Clean up
            if name in self.strategies:
                del self.strategies[name]
            
            if name in self.performance:
                del self.performance[name]
            
            # Return performance metrics
            return {
                'sharpe_ratio': performance.get('sharpe_ratio', 0.0),
                'win_rate': performance.get('win_rate', 0.0),
                'profit_factor': performance.get('profit_factor', 0.0),
                'return_to_drawdown': (
                    performance.get('expectancy', 0.0) / 
                    max(performance.get('max_drawdown', 0.01), 0.01)
                ),
                'expectancy': performance.get('expectancy', 0.0),
                'max_drawdown': performance.get('max_drawdown', 1.0)
            }
        
        # Set the evaluation function in the evolution manager
        setattr(evolution_manager, "strategy_evaluator", evaluate_strategy)
        
        # Store a reference to the evolution manager
        self.evolution_manager = evolution_manager
    
    def evolve_strategies(self, 
                        strategy_template: Dict[str, Any], 
                        param_ranges: Dict[str, Any],
                        generations: int = 5) -> Dict[str, Any]:
        """
        Evolve strategies using the evolution manager.
        
        Args:
            strategy_template: Template strategy
            param_ranges: Parameter ranges for evolution
            generations: Number of generations to evolve
            
        Returns:
            Dictionary with evolution results
        """
        if not hasattr(self, 'evolution_manager'):
            self.logger.error("No evolution manager available for evolving strategies")
            return {'error': 'No evolution manager available'}
        
        # Initialize the population
        self.evolution_manager.initialize_population(strategy_template, param_ranges)
        
        # Run evolution for specified generations
        for i in range(generations):
            continue_evolution = self.evolution_manager.evolve(
                self.evolution_manager.strategy_evaluator)
            
            if not continue_evolution:
                self.logger.info(f"Evolution stopped after {i+1} generations")
                break
        
        # Get the best strategy
        best_strategy = self.evolution_manager.get_best_strategy()
        
        if best_strategy:
            # Create and register the best strategy
            strategy_type = best_strategy.get('type')
            parameters = best_strategy.get('parameters', {})
            name = f"evolved_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            strategy = self.create_strategy(strategy_type, name, parameters)
            
            if strategy:
                self.register_strategy(strategy)
                self.logger.info(f"Registered best evolved strategy as {name}")
        
        # Generate and return evolution report
        return self.evolution_manager.generate_evolution_report() 