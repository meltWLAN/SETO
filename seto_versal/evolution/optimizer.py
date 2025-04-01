#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evolution optimizer module for SETO-Versal
Evolves trading strategies based on feedback and performance
"""

import logging
import random
import numpy as np
import json
import copy
from datetime import datetime
from enum import Enum
import os

logger = logging.getLogger(__name__)

class OptimizationGoal(Enum):
    """Enum for optimization goals"""
    MAX_RETURN = "max_return"
    MAX_SHARPE = "max_sharpe"
    MIN_DRAWDOWN = "min_drawdown"
    MIN_VOLATILITY = "min_volatility"
    BALANCED = "balanced"

class EvolutionOptimizer:
    """
    Evolution optimizer for SETO-Versal
    
    Evolves trading strategies based on feedback and performance data
    """
    
    def __init__(self, config, feedback_analyzer=None):
        """
        Initialize the evolution optimizer
        
        Args:
            config (dict): Configuration dictionary
            feedback_analyzer (FeedbackAnalyzer, optional): Feedback analyzer instance
        """
        self.config = config
        self.feedback_analyzer = feedback_analyzer
        
        # Evolution settings
        self.evolution_interval = config.get('evolution_interval', 30)  # days
        self.population_size = config.get('population_size', 10)
        self.mutation_rate = config.get('mutation_rate', 0.2)
        self.crossover_rate = config.get('crossover_rate', 0.7)
        self.selection_pressure = config.get('selection_pressure', 0.3)
        self.fitness_metrics = config.get('fitness_metrics', ['profit_factor', 'win_rate', 'sharpe'])
        self.metric_weights = config.get('metric_weights', {'profit_factor': 0.4, 'win_rate': 0.3, 'sharpe': 0.3})
        
        # Strategy population tracking
        self.strategy_population = {}
        self.strategy_performance = {}
        self.evolution_history = []
        
        # Generation counter
        self.generation = 0
        self.last_evolution_time = None
        
        # Optimization settings
        self.optimization_goal = OptimizationGoal(config.get('optimization_goal', 'balanced'))
        
        logger.info("Evolution optimizer initialized")
    
    def register_strategy(self, strategy_name, strategy_params, agent_name=None):
        """
        Register a strategy for evolution
        
        Args:
            strategy_name (str): Name of the strategy
            strategy_params (dict): Strategy parameters
            agent_name (str, optional): Name of the agent using this strategy
            
        Returns:
            str: Unique ID for this strategy instance
        """
        strategy_id = f"{strategy_name}_{len(self.strategy_population) + 1}"
        
        # Clone the parameters to avoid reference issues
        params_copy = copy.deepcopy(strategy_params)
        
        self.strategy_population[strategy_id] = {
            'name': strategy_name,
            'agent_name': agent_name,
            'params': params_copy,
            'created_at': datetime.now(),
            'generation': self.generation,
            'parent_ids': [],
            'active': True
        }
        
        logger.info(f"Registered strategy {strategy_id} for evolution")
        return strategy_id
    
    def should_evolve(self):
        """
        Check if it's time to evolve strategies
        
        Returns:
            bool: True if evolution should be performed, False otherwise
        """
        if not self.last_evolution_time:
            return False
        
        days_since_last_evolution = (datetime.now() - self.last_evolution_time).days
        return days_since_last_evolution >= self.evolution_interval
    
    def update_performance_data(self):
        """
        Update performance data for all strategies
        
        Args:
            None
            
        Returns:
            bool: True if performance data updated successfully, False otherwise
        """
        if not self.feedback_analyzer:
            logger.warning("Feedback analyzer not available, cannot update performance data")
            return False
        
        # Get performance data from feedback analyzer
        performance_data = self.feedback_analyzer.analyze_strategy_performance()
        
        # Update our tracked performance data
        for strategy_name, performance in performance_data.items():
            # Find all strategy instances with this name
            for strategy_id, strategy_info in self.strategy_population.items():
                if strategy_info['name'] == strategy_name and strategy_info['active']:
                    self.strategy_performance[strategy_id] = performance
        
        logger.info(f"Updated performance data for {len(self.strategy_performance)} strategies")
        return True
    
    def calculate_fitness(self, strategy_id):
        """
        Calculate fitness score for a strategy
        
        Args:
            strategy_id (str): Strategy ID
            
        Returns:
            float: Fitness score (higher is better)
        """
        if strategy_id not in self.strategy_performance:
            return 0.0
        
        performance = self.strategy_performance[strategy_id]
        fitness = 0.0
        
        # Calculate weighted average of metrics
        for metric, weight in self.metric_weights.items():
            if metric in performance:
                # Apply normalization based on metric type
                if metric == 'profit_factor':
                    # Cap extremely high profit factors to prevent over-optimization
                    value = min(performance[metric], 10.0)
                    normalized = value / 10.0  # Normalize to 0-1 range
                elif metric == 'win_rate':
                    normalized = performance[metric]  # Already in 0-1 range
                elif metric == 'sharpe':
                    # Normalize Sharpe ratio (typical range 0-4)
                    normalized = min(performance[metric], 4.0) / 4.0
                else:
                    # Default normalization for other metrics
                    normalized = performance[metric] / 100.0
                
                fitness += normalized * weight
        
        return fitness
    
    def select_parents(self):
        """
        Select parent strategies for breeding using tournament selection
        
        Returns:
            list: List of selected parent strategy IDs
        """
        if not self.strategy_performance:
            logger.warning("No strategy performance data available for selection")
            return []
        
        # Calculate fitness for all strategies
        fitness_scores = {}
        for strategy_id in self.strategy_population:
            if self.strategy_population[strategy_id]['active']:
                fitness_scores[strategy_id] = self.calculate_fitness(strategy_id)
        
        if not fitness_scores:
            logger.warning("No active strategies with fitness scores")
            return []
        
        # Tournament selection
        tournament_size = max(2, int(len(fitness_scores) * self.selection_pressure))
        parents = []
        
        # Select top strategies based on fitness
        sorted_strategies = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        num_parents = min(self.population_size // 2, len(sorted_strategies))
        
        for _ in range(num_parents):
            # Select random contestants for tournament
            contestants = random.sample(list(fitness_scores.keys()), 
                                       min(tournament_size, len(fitness_scores)))
            # Find contestant with highest fitness
            winner = max(contestants, key=lambda s: fitness_scores[s])
            parents.append(winner)
        
        return parents
    
    def mutate_params(self, params):
        """
        Apply mutation to strategy parameters
        
        Args:
            params (dict): Strategy parameters
            
        Returns:
            dict: Mutated parameters
        """
        mutated = copy.deepcopy(params)
        
        for param_name, param_value in mutated.items():
            # Only mutate numeric parameters
            if isinstance(param_value, (int, float)):
                # Decide if this parameter should be mutated
                if random.random() < self.mutation_rate:
                    # Apply mutation based on parameter type
                    if isinstance(param_value, int):
                        # For integers, add/subtract a small integer
                        mutated[param_name] = max(1, param_value + random.randint(-2, 2))
                    else:
                        # For floats, adjust by a percentage
                        change = param_value * random.uniform(-0.3, 0.3)
                        mutated[param_name] = max(0.001, param_value + change)
            
            # Handle other parameter types if needed
            elif isinstance(param_value, bool) and random.random() < self.mutation_rate:
                mutated[param_name] = not param_value
            
            # Handle lists (e.g., for feature lists)
            elif isinstance(param_value, list) and random.random() < self.mutation_rate:
                if param_value and all(isinstance(item, str) for item in param_value):
                    # For string lists (e.g., features), randomly include/exclude
                    if len(param_value) > 1 and random.random() < 0.5:
                        # Remove a random element
                        item_to_remove = random.choice(param_value)
                        mutated[param_name].remove(item_to_remove)
                    # We could also add new features here if we had a master list
        
        return mutated
    
    def crossover(self, params1, params2):
        """
        Perform crossover between two parameter sets
        
        Args:
            params1 (dict): First parameter set
            params2 (dict): Second parameter set
            
        Returns:
            dict: New parameter set created from crossover
        """
        offspring = {}
        
        # Simple uniform crossover
        for param_name in params1:
            if param_name in params2:
                # Randomly select parameter from either parent
                if random.random() < 0.5:
                    offspring[param_name] = copy.deepcopy(params1[param_name])
                else:
                    offspring[param_name] = copy.deepcopy(params2[param_name])
            else:
                # If parameter only exists in one parent, use that one
                offspring[param_name] = copy.deepcopy(params1[param_name])
        
        # Add any parameters unique to the second parent
        for param_name in params2:
            if param_name not in params1:
                offspring[param_name] = copy.deepcopy(params2[param_name])
        
        return offspring
    
    def evolve_strategies(self):
        """
        Perform one generation of strategy evolution
        
        Returns:
            list: List of new strategy IDs created
        """
        # Update performance data first
        if not self.update_performance_data():
            logger.warning("Failed to update performance data, evolution aborted")
            return []
        
        # Select parents
        parents = self.select_parents()
        if not parents:
            logger.warning("No parents selected for evolution")
            return []
        
        # Create new generation
        new_strategies = []
        
        # Generate offspring through crossover and mutation
        while len(new_strategies) < self.population_size:
            # Select two parents
            if len(parents) >= 2:
                parent1, parent2 = random.sample(parents, 2)
            else:
                logger.warning("Not enough parents for crossover")
                break
            
            parent1_info = self.strategy_population[parent1]
            parent2_info = self.strategy_population[parent2]
            
            # Create new parameter set through crossover
            if random.random() < self.crossover_rate and len(parents) >= 2:
                offspring_params = self.crossover(parent1_info['params'], parent2_info['params'])
            else:
                # No crossover, just clone one parent
                offspring_params = copy.deepcopy(parent1_info['params'])
            
            # Apply mutation
            offspring_params = self.mutate_params(offspring_params)
            
            # Register new strategy
            strategy_id = self.register_strategy(
                parent1_info['name'],
                offspring_params,
                parent1_info['agent_name']
            )
            
            # Record parents
            self.strategy_population[strategy_id]['parent_ids'] = [parent1, parent2]
            self.strategy_population[strategy_id]['generation'] = self.generation + 1
            
            new_strategies.append(strategy_id)
        
        # Increment generation counter
        self.generation += 1
        self.last_evolution_time = datetime.now()
        
        # Record evolution event
        self.evolution_history.append({
            'timestamp': self.last_evolution_time,
            'generation': self.generation,
            'parents': parents,
            'offspring': new_strategies
        })
        
        logger.info(f"Evolved generation {self.generation}, created {len(new_strategies)} new strategies")
        return new_strategies
    
    def get_best_strategies(self, strategy_type=None, limit=5):
        """
        Get the best performing strategies of a specific type
        
        Args:
            strategy_type (str, optional): Filter by strategy type
            limit (int, optional): Maximum number of strategies to return
            
        Returns:
            list: List of strategy IDs and their fitness scores
        """
        fitness_scores = {}
        
        for strategy_id, strategy_info in self.strategy_population.items():
            # Filter by type if specified
            if strategy_type and strategy_info['name'] != strategy_type:
                continue
            
            if strategy_info['active']:
                fitness_scores[strategy_id] = self.calculate_fitness(strategy_id)
        
        # Sort by fitness (descending)
        sorted_strategies = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N strategies
        return sorted_strategies[:limit]
    
    def get_evolution_stats(self):
        """
        Get statistics about the evolution process
        
        Returns:
            dict: Evolution statistics
        """
        return {
            'generation': self.generation,
            'last_evolution_time': self.last_evolution_time,
            'population_size': len([s for s in self.strategy_population.values() if s['active']]),
            'total_strategies_created': len(self.strategy_population),
            'evolution_count': len(self.evolution_history)
        }
    
    def export_population(self, file_path):
        """
        Export the current strategy population to a file
        
        Args:
            file_path (str): Path to output file
            
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            data = {
                'generation': self.generation,
                'last_evolution_time': self.last_evolution_time,
                'evolution_history': self.evolution_history,
                'strategy_population': self.strategy_population,
                'strategy_performance': self.strategy_performance
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Exported strategy population to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting strategy population: {e}")
            return False
    
    def import_population(self, file_path):
        """
        Import a strategy population from a file
        
        Args:
            file_path (str): Path to input file
            
        Returns:
            bool: True if import successful, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.generation = data.get('generation', 0)
            self.last_evolution_time = data.get('last_evolution_time')
            self.evolution_history = data.get('evolution_history', [])
            self.strategy_population = data.get('strategy_population', {})
            self.strategy_performance = data.get('strategy_performance', {})
            
            logger.info(f"Imported strategy population from {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error importing strategy population: {e}")
            return False 