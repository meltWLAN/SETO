#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evolution engine module for SETO-Versal
Manages the evolution of trading strategies based on feedback
"""

import logging
import os
import json
from datetime import datetime, timedelta
import random

from seto_versal.evolution.optimizer import EvolutionOptimizer
from seto_versal.evolution.strategy_templates import StrategyTemplateFactory

logger = logging.getLogger(__name__)

class EvolutionEngine:
    """
    Evolution engine for SETO-Versal
    
    Manages the evolution of trading strategies based on feedback analysis
    """
    
    def __init__(self, config):
        """
        Initialize the evolution engine
        
        Args:
            config (dict): Evolution configuration
        """
        self.config = config
        self.mode = config.get('mode', 'backtest')
        self.mode_config = config.get('mode_configs', {}).get(self.mode, {})
        
        # Get mode-specific parameters
        self.mutation_rate = self.mode_config.get('mutation_rate', config.get('parameter_mutation_rate', 0.2))
        self.breeding_threshold = self.mode_config.get('breeding_threshold', config.get('agent_breeding_threshold', 0.5))
        self.optimization_frequency = self.mode_config.get('optimization_frequency', config.get('optimization_frequency', 'weekly'))
        self.performance_metric = config.get('performance_metric', 'sharpe')
        self.sliding_window = config.get('sliding_window', 90)
        
        # Initialize evolution state
        self.generation = 0
        self.best_agents = []
        self.agent_history = []
        self.performance_history = []
        
        self.template_factory = StrategyTemplateFactory()
        self.optimizer = EvolutionOptimizer(config)
        
        # Track when last evolution occurred
        self.last_evolution_time = None
        
        # Track strategy mappings (agent -> strategies)
        self.agent_strategies = {}
        
        # Directory for storing evolution data
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'evolution')
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info("Evolution engine initialized")
    
    def evolve(self, agents, market_state):
        """
        Evolve the agent population
        
        Args:
            agents (list): List of agents to evolve
            market_state (MarketState): Current market state
            
        Returns:
            list: Evolved agent population
        """
        try:
            # Evaluate current population
            performance_scores = self._evaluate_agents(agents, market_state)
            
            # Select best agents
            self.best_agents = self._select_best_agents(agents, performance_scores)
            
            # Create new generation
            new_agents = self._create_new_generation(agents, performance_scores)
            
            # Update history
            self._update_history(agents, performance_scores)
            
            # Increment generation
            self.generation += 1
            
            return new_agents
            
        except Exception as e:
            logger.error(f"Error in evolution: {e}")
            return agents
            
    def _evaluate_agents(self, agents, market_state):
        """Evaluate agent performance"""
        scores = {}
        for agent in agents:
            try:
                # Get agent performance metrics
                metrics = agent.get_performance_metrics()
                
                # Calculate score based on performance metric
                if self.performance_metric == 'sharpe':
                    score = metrics.get('sharpe_ratio', 0)
                elif self.performance_metric == 'sortino':
                    score = metrics.get('sortino_ratio', 0)
                elif self.performance_metric == 'calmar':
                    score = metrics.get('calmar_ratio', 0)
                else:
                    score = metrics.get('total_return', 0)
                    
                scores[agent] = score
                
            except Exception as e:
                logger.error(f"Error evaluating agent {agent.name}: {e}")
                scores[agent] = 0
                
        return scores
        
    def _select_best_agents(self, agents, scores):
        """Select best performing agents"""
        # Sort agents by score
        sorted_agents = sorted(agents, key=lambda x: scores.get(x, 0), reverse=True)
        
        # Select top performers based on breeding threshold
        num_best = int(len(agents) * self.breeding_threshold)
        return sorted_agents[:num_best]
        
    def _create_new_generation(self, agents, scores):
        """Create new generation of agents"""
        new_agents = []
        
        # Keep best agents
        new_agents.extend(self.best_agents)
        
        # Create new agents through breeding and mutation
        while len(new_agents) < len(agents):
            # Select parents
            parent1 = random.choice(self.best_agents)
            parent2 = random.choice(self.best_agents)
            
            # Create child through crossover
            child = self._crossover(parent1, parent2)
            
            # Apply mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
                
            new_agents.append(child)
            
        return new_agents
        
    def _crossover(self, parent1, parent2):
        """Perform crossover between two agents"""
        # Create new agent with combined strategies
        child = parent1.__class__(f"{parent1.name}_{parent2.name}_child")
        
        # Combine strategies from both parents
        strategies = []
        strategies.extend(parent1.get_strategies())
        strategies.extend(parent2.get_strategies())
        
        # Remove duplicates
        strategies = list(set(strategies))
        
        # Add strategies to child
        for strategy in strategies:
            child.add_strategy(strategy)
            
        return child
        
    def _mutate(self, agent):
        """Apply mutation to an agent"""
        # Randomly modify agent parameters
        if random.random() < 0.5:
            agent.confidence_threshold *= random.uniform(0.9, 1.1)
            
        if random.random() < 0.5:
            agent.max_positions = max(1, int(agent.max_positions * random.uniform(0.8, 1.2)))
            
        return agent
        
    def _update_history(self, agents, scores):
        """Update evolution history"""
        # Record agent performance
        self.agent_history.append({
            'generation': self.generation,
            'agents': [agent.name for agent in agents],
            'scores': {agent.name: score for agent, score in scores.items()}
        })
        
        # Record best performance
        best_score = max(scores.values())
        self.performance_history.append({
            'generation': self.generation,
            'best_score': best_score,
            'best_agent': max(scores.items(), key=lambda x: x[1])[0].name
        })
        
    def get_history(self):
        """Get evolution history"""
        return {
            'agent_history': self.agent_history,
            'performance_history': self.performance_history
        }
        
    def get_best_agents(self):
        """Get best performing agents"""
        return self.best_agents
    
    def _should_evolve(self):
        """
        Check if it's time to evolve strategies
        
        Returns:
            bool: True if evolution should be performed
        """
        if self.last_evolution_time is None:
            # First run, check if we have enough data
            return self.optimizer.strategy_population and len(self.optimizer.strategy_performance) >= 5
        
        # Check based on frequency
        if self.optimization_frequency == 'daily':
            return (datetime.now() - self.last_evolution_time) > timedelta(days=1)
        elif self.optimization_frequency == 'weekly':
            return (datetime.now() - self.last_evolution_time) > timedelta(days=7)
        elif self.optimization_frequency == 'monthly':
            return (datetime.now() - self.last_evolution_time) > timedelta(days=30)
        else:
            # Default to weekly
            return (datetime.now() - self.last_evolution_time) > timedelta(days=7)
    
    def _register_agent_strategies(self, agents):
        """
        Register all current strategies from agents
        
        Args:
            agents (list): List of agent instances
        """
        for agent in agents:
            agent_name = agent.name
            
            if not hasattr(agent, 'strategies'):
                logger.warning(f"Agent {agent_name} has no strategies attribute")
                continue
            
            self.agent_strategies[agent_name] = []
            
            for strategy in agent.strategies:
                strategy_name = strategy.name
                strategy_params = strategy.get_parameters()
                
                # Register with optimizer
                strategy_id = self.optimizer.register_strategy(
                    strategy_name=strategy_name,
                    strategy_params=strategy_params,
                    agent_name=agent_name
                )
                
                self.agent_strategies[agent_name].append(strategy_id)
        
        logger.debug(f"Registered strategies for {len(self.agent_strategies)} agents")
    
    def _apply_evolved_strategies(self, agents, new_strategy_ids):
        """
        Apply evolved strategies to the appropriate agents
        
        Args:
            agents (list): List of agent instances
            new_strategy_ids (list): List of new strategy IDs
        """
        # Get new strategies by agent
        strategies_by_agent = {}
        
        for strategy_id in new_strategy_ids:
            strategy_info = self.optimizer.strategy_population[strategy_id]
            agent_name = strategy_info['agent_name']
            
            if agent_name not in strategies_by_agent:
                strategies_by_agent[agent_name] = []
            
            strategies_by_agent[agent_name].append((strategy_id, strategy_info))
        
        # Apply to agents
        for agent in agents:
            if agent.name not in strategies_by_agent:
                continue
            
            new_strategies = strategies_by_agent[agent.name]
            logger.info(f"Applying {len(new_strategies)} evolved strategies to agent {agent.name}")
            
            for strategy_id, strategy_info in new_strategies:
                # Here we would apply the new strategy
                # This would need to be implemented specific to each agent type
                self._apply_strategy_to_agent(agent, strategy_info)
    
    def _apply_strategy_to_agent(self, agent, strategy_info):
        """
        Apply a new strategy to an agent
        
        Args:
            agent: Agent instance
            strategy_info (dict): Strategy information
        """
        # This is a placeholder implementation
        # In a real system, we would instantiate the appropriate strategy
        # with the evolved parameters and add it to the agent
        
        strategy_name = strategy_info['name']
        strategy_params = strategy_info['params']
        
        logger.info(f"Would apply strategy {strategy_name} with evolved parameters to {agent.name}")
        logger.debug(f"Strategy parameters: {strategy_params}")
        
        # In a real implementation, we would do something like:
        # if hasattr(agent, 'add_strategy'):
        #     new_strategy = create_strategy(strategy_name, strategy_params)
        #     agent.add_strategy(new_strategy)
    
    def get_best_strategies(self):
        """
        Get the best performing strategies by type
        
        Returns:
            dict: Dictionary of best strategies by type
        """
        best_strategies = {}
        
        # Get all templates
        templates = self.template_factory.get_all_templates()
        
        # For each template, get the best strategies
        for template_name in templates.keys():
            best_of_type = self.optimizer.get_best_strategies(template_name, limit=3)
            if best_of_type:
                best_strategies[template_name] = best_of_type
        
        return best_strategies
    
    def create_strategy_from_template(self, template_name, custom_params=None):
        """
        Create a new strategy from a template
        
        Args:
            template_name (str): Name of the template
            custom_params (dict, optional): Custom parameters
            
        Returns:
            tuple: (parameters, is_valid, error_message)
        """
        return self.template_factory.create_strategy_parameters(
            template_name, custom_params
        )
    
    def export_evolution_state(self, file_path=None):
        """
        Export the current evolution state
        
        Args:
            file_path (str, optional): Path to output file
            
        Returns:
            str: Path to the export file
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.data_dir, f"evolution_state_{timestamp}.json")
        
        try:
            state = {
                'last_evolution_time': self.last_evolution_time,
                'agent_strategies': self.agent_strategies,
                'evolution_stats': self.optimizer.get_evolution_stats()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=str)
            
            # Export optimizer population as well
            population_path = os.path.splitext(file_path)[0] + "_population.json"
            self.optimizer.export_population(population_path)
            
            logger.info(f"Exported evolution state to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error exporting evolution state: {e}")
            return None
    
    def import_evolution_state(self, file_path):
        """
        Import evolution state from file
        
        Args:
            file_path (str): Path to input file
            
        Returns:
            bool: True if import was successful
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.last_evolution_time = state.get('last_evolution_time')
            self.agent_strategies = state.get('agent_strategies', {})
            
            # Import optimizer population as well
            population_path = os.path.splitext(file_path)[0] + "_population.json"
            if os.path.exists(population_path):
                self.optimizer.import_population(population_path)
            
            logger.info(f"Imported evolution state from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing evolution state: {e}")
            return False 