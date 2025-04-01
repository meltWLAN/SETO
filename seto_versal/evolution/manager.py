#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evolution manager for SETO-Versal trading system.

This module implements strategy evolution, genetic algorithms, and
continuous improvement mechanisms to adapt trading strategies
based on performance feedback.
"""

import os
import enum
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import random


class EvolutionMetric(enum.Enum):
    """
    Metrics used for evaluating and evolving strategies.
    """
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    RETURN_TO_DRAWDOWN = "return_to_drawdown"
    AVERAGE_PROFIT = "average_profit"
    EXPECTANCY = "expectancy"
    
    def __str__(self):
        return self.value


class EvolutionManager:
    """
    Manager for strategy evolution, adaptation, and improvement.
    
    The evolution manager:
    - Evaluates strategy performance
    - Applies genetic algorithm techniques to evolve strategies
    - Implements continuous improvement mechanisms
    - Tracks strategy lineage and performance history
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evolution manager.
        
        Args:
            config: Dictionary with configuration parameters.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing evolution manager")
        
        # Configuration
        self.name = config.get('name', 'evolution_manager')
        self.mutation_rate = config.get('mutation_rate', 0.2)
        self.crossover_rate = config.get('crossover_rate', 0.7)
        self.population_size = config.get('population_size', 20)
        self.tournament_size = config.get('tournament_size', 3)
        self.generation_limit = config.get('generation_limit', 50)
        self.convergence_threshold = config.get('convergence_threshold', 0.001)
        
        # Set fitness function and weights
        self.fitness_metrics = config.get('fitness_metrics', {
            EvolutionMetric.SHARPE_RATIO.value: 0.4,
            EvolutionMetric.WIN_RATE.value: 0.2,
            EvolutionMetric.PROFIT_FACTOR.value: 0.3,
            EvolutionMetric.RETURN_TO_DRAWDOWN.value: 0.1
        })
        
        # Population data structures
        self.current_population = []
        self.best_individual = None
        self.current_generation = 0
        self.population_history = []
        
        # Strategy lineage tracking
        self.strategy_lineage = {}
        
        # Mutation parameters (range modifiers)
        self.mutation_params = config.get('mutation_params', {
            'numeric_param_scale': 0.2,  # Scale for numeric parameter mutations
            'categorical_mutation_prob': 0.3,  # Probability to change categorical params
            'boolean_flip_prob': 0.2,    # Probability to flip boolean params
        })
        
        # Continuous improvement
        self.improvement_threshold = config.get('improvement_threshold', 0.05)
        self.no_improvement_generations = 0
        self.max_no_improvement = config.get('max_no_improvement', 10)
        
        self.logger.info(f"Evolution manager initialized with population size: {self.population_size}")
    
    def initialize_population(self, strategy_template: Dict[str, Any], param_ranges: Dict[str, Any]):
        """
        Initialize a population of strategies based on a template and parameter ranges.
        
        Args:
            strategy_template: Base strategy template
            param_ranges: Dictionary of parameter ranges and constraints
        """
        self.current_population = []
        
        # Store parameter ranges for future mutations
        self.param_ranges = param_ranges
        
        for i in range(self.population_size):
            # Create a new individual based on the template
            individual = self._create_individual(strategy_template, param_ranges)
            individual['id'] = f"gen0_ind{i}"
            individual['generation'] = 0
            individual['lineage'] = [individual['id']]
            individual['fitness'] = 0.0
            individual['performance'] = {}
            
            self.current_population.append(individual)
        
        self.current_generation = 0
        self.logger.info(f"Initialized population with {len(self.current_population)} individuals")
    
    def _create_individual(self, strategy_template: Dict[str, Any], param_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new individual by randomizing parameters within ranges.
        
        Args:
            strategy_template: Base strategy template
            param_ranges: Parameter ranges and constraints
            
        Returns:
            New strategy individual
        """
        # Create a deep copy of the template
        individual = {key: value for key, value in strategy_template.items()}
        
        # Randomize parameters within ranges
        params = individual.get('parameters', {})
        
        for param_name, param_range in param_ranges.items():
            param_type = param_range.get('type', 'numeric')
            
            if param_type == 'numeric':
                min_val = param_range.get('min', 0)
                max_val = param_range.get('max', 1)
                is_int = param_range.get('is_integer', False)
                
                if is_int:
                    value = random.randint(min_val, max_val)
                else:
                    value = random.uniform(min_val, max_val)
                    # Round to specified decimal places if provided
                    decimals = param_range.get('decimals')
                    if decimals is not None:
                        value = round(value, decimals)
                
                # Set the parameter value
                params[param_name] = value
                
            elif param_type == 'categorical':
                options = param_range.get('options', [])
                if options:
                    params[param_name] = random.choice(options)
                    
            elif param_type == 'boolean':
                params[param_name] = random.choice([True, False])
        
        # Update the individual with new parameters
        individual['parameters'] = params
        
        return individual
    
    def evaluate_population(self, evaluation_func: Callable[[Dict[str, Any]], Dict[str, float]]):
        """
        Evaluate the fitness of each individual in the population.
        
        Args:
            evaluation_func: Function that evaluates a strategy and returns performance metrics
        """
        previous_best_fitness = 0
        if self.best_individual:
            previous_best_fitness = self.best_individual.get('fitness', 0)
        
        # Evaluate each individual
        for individual in self.current_population:
            # Call the provided evaluation function
            performance = evaluation_func(individual)
            
            # Store the performance metrics
            individual['performance'] = performance
            
            # Calculate fitness
            fitness = self._calculate_fitness(performance)
            individual['fitness'] = fitness
        
        # Sort population by fitness
        self.current_population.sort(key=lambda x: x.get('fitness', 0), reverse=True)
        
        # Update best individual
        if self.current_population:
            current_best = self.current_population[0]
            
            if not self.best_individual or current_best.get('fitness', 0) > self.best_individual.get('fitness', 0):
                self.best_individual = current_best.copy()
                self.no_improvement_generations = 0
            else:
                # Check if we have sufficient improvement
                improvement = (current_best.get('fitness', 0) - previous_best_fitness) / previous_best_fitness if previous_best_fitness else 0
                if improvement < self.improvement_threshold:
                    self.no_improvement_generations += 1
                else:
                    self.no_improvement_generations = 0
        
        self.logger.info(f"Evaluated generation {self.current_generation}. Best fitness: {self.best_individual.get('fitness', 0):.4f}")
        
        # Store this generation in history
        self._record_generation()
    
    def _calculate_fitness(self, performance: Dict[str, float]) -> float:
        """
        Calculate fitness score from performance metrics using weighted sum.
        
        Args:
            performance: Dictionary of performance metrics
            
        Returns:
            Fitness score
        """
        fitness = 0.0
        
        for metric, weight in self.fitness_metrics.items():
            value = performance.get(metric, 0)
            
            # Special handling for metrics where lower is better
            if metric in [EvolutionMetric.MAX_DRAWDOWN.value]:
                # Convert to a positive score (lower drawdown is better)
                if value > 0:
                    value = 1 / value
            
            fitness += value * weight
        
        return fitness
    
    def _record_generation(self):
        """
        Record the current generation in history.
        """
        generation_record = {
            'generation': self.current_generation,
            'timestamp': datetime.now().isoformat(),
            'population': [
                {
                    'id': ind.get('id'),
                    'fitness': ind.get('fitness', 0),
                    'key_metrics': {
                        k: ind.get('performance', {}).get(k, 0)
                        for k in self.fitness_metrics.keys()
                    }
                }
                for ind in self.current_population
            ],
            'best_individual': self.best_individual.get('id') if self.best_individual else None,
            'best_fitness': self.best_individual.get('fitness', 0) if self.best_individual else 0
        }
        
        self.population_history.append(generation_record)
    
    def evolve(self, evaluation_func: Callable[[Dict[str, Any]], Dict[str, float]]) -> bool:
        """
        Evolve the population to the next generation.
        
        Args:
            evaluation_func: Function to evaluate individual performance
            
        Returns:
            Boolean indicating if evolution should continue
        """
        # First, evaluate the current population
        self.evaluate_population(evaluation_func)
        
        # Check termination conditions
        if self._check_termination():
            self.logger.info(f"Evolution terminated after {self.current_generation} generations")
            return False
        
        # Generate the next generation
        next_population = []
        
        # Elitism: keep the best individual(s)
        elites = self.current_population[:2]  # Keep top 2
        for elite in elites:
            next_individual = elite.copy()
            next_individual['id'] = f"gen{self.current_generation+1}_elite{len(next_population)}"
            next_individual['generation'] = self.current_generation + 1
            next_individual['lineage'] = elite.get('lineage', [])[:]  # Copy lineage
            next_individual['lineage'].append(next_individual['id'])
            
            next_population.append(next_individual)
        
        # Fill the rest with offspring
        while len(next_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                offspring = self._crossover(parent1, parent2)
            else:
                # If no crossover, just copy one parent
                offspring = parent1.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                offspring = self._mutate(offspring)
            
            # Update metadata
            offspring['id'] = f"gen{self.current_generation+1}_ind{len(next_population)}"
            offspring['generation'] = self.current_generation + 1
            offspring['lineage'] = []
            offspring['lineage'].extend(parent1.get('lineage', [])[:])
            if parent1 != parent2 and 'lineage' in parent2:
                offspring['lineage'].extend(parent2.get('lineage', [])[:])
            offspring['lineage'].append(offspring['id'])
            
            # Record lineage
            self._update_lineage(offspring, parent1, parent2)
            
            # Reset fitness and performance
            offspring['fitness'] = 0.0
            offspring['performance'] = {}
            
            next_population.append(offspring)
        
        # Update the population
        self.current_population = next_population
        self.current_generation += 1
        
        self.logger.info(f"Generated generation {self.current_generation} with {len(self.current_population)} individuals")
        
        return True  # Continue evolution
    
    def _check_termination(self) -> bool:
        """
        Check if evolution should terminate.
        
        Returns:
            Boolean indicating if termination condition is met
        """
        # Check generation limit
        if self.current_generation >= self.generation_limit:
            return True
        
        # Check for convergence or lack of improvement
        if self.no_improvement_generations >= self.max_no_improvement:
            self.logger.info(f"No significant improvement for {self.no_improvement_generations} generations")
            return True
        
        return False
    
    def _tournament_selection(self) -> Dict[str, Any]:
        """
        Select an individual using tournament selection.
        
        Returns:
            Selected individual
        """
        # Randomly select tournament participants
        tournament = random.sample(self.current_population, min(self.tournament_size, len(self.current_population)))
        
        # Select the best individual from the tournament
        tournament.sort(key=lambda x: x.get('fitness', 0), reverse=True)
        
        return tournament[0]
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Offspring individual
        """
        # Create a new individual
        offspring = {key: value for key, value in parent1.items() if key not in ['parameters', 'id', 'generation', 'lineage', 'fitness', 'performance']}
        
        # Get parent parameters
        params1 = parent1.get('parameters', {})
        params2 = parent2.get('parameters', {})
        
        # Create new parameters with crossover
        new_params = {}
        
        for param_name in set(params1.keys()).union(params2.keys()):
            # If parameter exists in both parents, choose randomly
            if param_name in params1 and param_name in params2:
                # For some parameters we might want to do averaging instead of selection
                param_range = self.param_ranges.get(param_name, {})
                param_type = param_range.get('type', 'numeric')
                
                if param_type == 'numeric' and random.random() < 0.5:
                    # Average numeric values
                    new_params[param_name] = (params1[param_name] + params2[param_name]) / 2
                    
                    # Round to integer if needed
                    if param_range.get('is_integer', False):
                        new_params[param_name] = int(round(new_params[param_name]))
                    
                    # Round to specified decimal places if provided
                    decimals = param_range.get('decimals')
                    if decimals is not None:
                        new_params[param_name] = round(new_params[param_name], decimals)
                else:
                    # Otherwise random selection
                    new_params[param_name] = params1[param_name] if random.random() < 0.5 else params2[param_name]
            elif param_name in params1:
                new_params[param_name] = params1[param_name]
            else:
                new_params[param_name] = params2[param_name]
        
        offspring['parameters'] = new_params
        
        return offspring
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate an individual.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        # Create a mutated copy
        mutated = individual.copy()
        params = mutated.get('parameters', {}).copy()
        
        # Get mutation parameters
        numeric_scale = self.mutation_params.get('numeric_param_scale', 0.2)
        categorical_prob = self.mutation_params.get('categorical_mutation_prob', 0.3)
        boolean_prob = self.mutation_params.get('boolean_flip_prob', 0.2)
        
        # Select a random subset of parameters to mutate
        param_keys = list(params.keys())
        num_to_mutate = max(1, int(len(param_keys) * self.mutation_rate))
        params_to_mutate = random.sample(param_keys, min(num_to_mutate, len(param_keys)))
        
        for param_name in params_to_mutate:
            param_range = self.param_ranges.get(param_name, {})
            param_type = param_range.get('type', 'numeric')
            
            if param_type == 'numeric':
                current_value = params[param_name]
                min_val = param_range.get('min', 0)
                max_val = param_range.get('max', 1)
                is_int = param_range.get('is_integer', False)
                
                # Calculate mutation range
                range_size = max_val - min_val
                mutation_amount = random.uniform(-range_size * numeric_scale, range_size * numeric_scale)
                
                # Apply mutation
                new_value = current_value + mutation_amount
                
                # Ensure within bounds
                new_value = max(min_val, min(max_val, new_value))
                
                # Convert to integer if needed
                if is_int:
                    new_value = int(round(new_value))
                else:
                    # Round to specified decimal places if provided
                    decimals = param_range.get('decimals')
                    if decimals is not None:
                        new_value = round(new_value, decimals)
                
                params[param_name] = new_value
                
            elif param_type == 'categorical':
                if random.random() < categorical_prob:
                    options = param_range.get('options', [])
                    if options:
                        current = params[param_name]
                        # Choose a different option
                        other_options = [opt for opt in options if opt != current]
                        if other_options:
                            params[param_name] = random.choice(other_options)
                            
            elif param_type == 'boolean':
                if random.random() < boolean_prob:
                    # Flip the boolean value
                    params[param_name] = not params[param_name]
        
        mutated['parameters'] = params
        
        return mutated
    
    def _update_lineage(self, offspring: Dict[str, Any], parent1: Dict[str, Any], parent2: Dict[str, Any]):
        """
        Update the strategy lineage tracking.
        
        Args:
            offspring: The new offspring individual
            parent1: First parent
            parent2: Second parent
        """
        offspring_id = offspring.get('id')
        if offspring_id:
            self.strategy_lineage[offspring_id] = {
                'parents': [
                    parent1.get('id'),
                    parent2.get('id') if parent1 != parent2 else None
                ],
                'generation': self.current_generation + 1,
                'created': datetime.now().isoformat()
            }
    
    def get_best_strategy(self) -> Dict[str, Any]:
        """
        Get the best strategy found during evolution.
        
        Returns:
            Best strategy individual
        """
        return self.best_individual.copy() if self.best_individual else None
    
    def save_state(self, filepath: str = None):
        """
        Save the evolution state to a file.
        
        Args:
            filepath: Path to save the state file, default is based on manager name
        """
        if filepath is None:
            filepath = f"data/evolution/{self.name}_state.json"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare state data
        state = {
            'name': self.name,
            'current_generation': self.current_generation,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'population_size': self.population_size,
            'fitness_metrics': self.fitness_metrics,
            'best_individual': self.best_individual,
            'population_history': self.population_history[-10:],  # Last 10 generations
            'strategy_lineage': self.strategy_lineage,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            self.logger.info(f"Evolution state saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving evolution state: {str(e)}")
            return False
    
    def load_state(self, filepath: str = None):
        """
        Load evolution state from a file.
        
        Args:
            filepath: Path to the state file, default is based on manager name
            
        Returns:
            Boolean indicating if state was loaded successfully
        """
        if filepath is None:
            filepath = f"data/evolution/{self.name}_state.json"
        
        if not os.path.exists(filepath):
            self.logger.warning(f"Evolution state file {filepath} not found")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore state
            self.name = state.get('name', self.name)
            self.current_generation = state.get('current_generation', 0)
            self.mutation_rate = state.get('mutation_rate', self.mutation_rate)
            self.crossover_rate = state.get('crossover_rate', self.crossover_rate)
            self.population_size = state.get('population_size', self.population_size)
            self.fitness_metrics = state.get('fitness_metrics', self.fitness_metrics)
            self.best_individual = state.get('best_individual')
            self.population_history = state.get('population_history', [])
            self.strategy_lineage = state.get('strategy_lineage', {})
            
            self.logger.info(f"Evolution state loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading evolution state: {str(e)}")
            return False
    
    def generate_evolution_report(self) -> Dict[str, Any]:
        """
        Generate a report on the evolution process.
        
        Returns:
            Dictionary with evolution statistics and metrics
        """
        if not self.population_history:
            return {
                'status': 'Not started',
                'generations': 0,
                'best_fitness': 0
            }
        
        # Get generation fitness progression
        generations = [g.get('generation') for g in self.population_history]
        best_fitness = [g.get('best_fitness') for g in self.population_history]
        avg_fitness = [
            sum(ind.get('fitness', 0) for ind in g.get('population', [])) / len(g.get('population', []))
            if g.get('population') else 0
            for g in self.population_history
        ]
        
        # Calculate improvement metrics
        first_gen_fitness = best_fitness[0] if best_fitness else 0
        last_gen_fitness = best_fitness[-1] if best_fitness else 0
        improvement = (last_gen_fitness - first_gen_fitness) / first_gen_fitness if first_gen_fitness else 0
        
        # Get the best strategy metrics
        best_metrics = {}
        if self.best_individual and 'performance' in self.best_individual:
            best_metrics = self.best_individual.get('performance', {})
        
        # Generate report
        report = {
            'status': 'Completed' if self._check_termination() else 'In progress',
            'generations': self.current_generation,
            'population_size': self.population_size,
            'fitness_metrics': self.fitness_metrics,
            'best_fitness': last_gen_fitness,
            'improvement': improvement * 100,  # As percentage
            'best_strategy_id': self.best_individual.get('id') if self.best_individual else None,
            'best_strategy_metrics': best_metrics,
            'generations_without_improvement': self.no_improvement_generations,
            'fitness_progression': {
                'generations': generations,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness
            },
            'termination_reason': self._get_termination_reason()
        }
        
        return report
    
    def _get_termination_reason(self) -> str:
        """
        Get the reason for termination if evolution has stopped.
        
        Returns:
            String explaining why evolution terminated
        """
        if self.current_generation >= self.generation_limit:
            return f"Reached generation limit ({self.generation_limit})"
        
        if self.no_improvement_generations >= self.max_no_improvement:
            return f"No improvement for {self.no_improvement_generations} generations"
        
        return "Evolution still in progress"
    
    def export_strategies(self, filepath: str = None, n_best: int = 5):
        """
        Export the best strategies to a file.
        
        Args:
            filepath: Path to save the strategies file
            n_best: Number of best strategies to export
        """
        if filepath is None:
            filepath = f"data/evolution/{self.name}_best_strategies.json"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Sort current population by fitness
        sorted_population = sorted(self.current_population, key=lambda x: x.get('fitness', 0), reverse=True)
        
        # Get top N strategies
        best_strategies = sorted_population[:n_best]
        
        # Extract key information
        export_data = {
            'name': self.name,
            'generation': self.current_generation,
            'timestamp': datetime.now().isoformat(),
            'strategies': []
        }
        
        for i, strategy in enumerate(best_strategies):
            strategy_data = {
                'rank': i + 1,
                'id': strategy.get('id'),
                'fitness': strategy.get('fitness', 0),
                'performance': strategy.get('performance', {}),
                'parameters': strategy.get('parameters', {}),
                'generation': strategy.get('generation', 0),
                'lineage_length': len(strategy.get('lineage', []))
            }
            export_data['strategies'].append(strategy_data)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.logger.info(f"Exported {len(best_strategies)} best strategies to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting strategies: {str(e)}")
            return False 