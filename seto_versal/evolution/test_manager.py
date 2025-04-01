#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test cases for EvolutionManager
"""

import unittest
import os
import tempfile
import json
from unittest.mock import MagicMock, patch

from seto_versal.evolution.manager import EvolutionManager, EvolutionMetric


class TestEvolutionManager(unittest.TestCase):
    """Test cases for the EvolutionManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.config = {
            'name': 'test_manager',
            'mutation_rate': 0.2,
            'crossover_rate': 0.7,
            'population_size': 10,
            'tournament_size': 3,
            'generation_limit': 5,
            'fitness_metrics': {
                EvolutionMetric.SHARPE_RATIO.value: 0.5,
                EvolutionMetric.WIN_RATE.value: 0.5
            }
        }
        
        # Create manager
        self.manager = EvolutionManager(self.config)
        
        # Create strategy template
        self.strategy_template = {
            'name': 'TestStrategy',
            'type': 'moving_average_crossover',
            'description': 'MA Crossover strategy for testing',
            'parameters': {}
        }
        
        # Create parameter ranges
        self.param_ranges = {
            'fast_period': {
                'type': 'numeric',
                'min': 5,
                'max': 50,
                'is_integer': True
            },
            'slow_period': {
                'type': 'numeric',
                'min': 20,
                'max': 200,
                'is_integer': True
            },
            'signal_type': {
                'type': 'categorical',
                'options': ['crossover', 'level', 'divergence']
            },
            'use_adaptive_sizing': {
                'type': 'boolean'
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        os.rmdir(self.temp_dir)
    
    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.manager.name, 'test_manager')
        self.assertEqual(self.manager.population_size, 10)
        self.assertEqual(self.manager.current_generation, 0)
        self.assertEqual(len(self.manager.current_population), 0)
        self.assertIsNone(self.manager.best_individual)
    
    def test_initialize_population(self):
        """Test population initialization"""
        self.manager.initialize_population(self.strategy_template, self.param_ranges)
        
        # Check population size
        self.assertEqual(len(self.manager.current_population), 10)
        
        # Check individual structure
        individual = self.manager.current_population[0]
        self.assertIn('id', individual)
        self.assertIn('generation', individual)
        self.assertIn('lineage', individual)
        self.assertIn('parameters', individual)
        
        # Check parameters were created
        params = individual['parameters']
        self.assertIn('fast_period', params)
        self.assertIn('slow_period', params)
        self.assertIn('signal_type', params)
        self.assertIn('use_adaptive_sizing', params)
        
        # Check parameter bounds
        self.assertGreaterEqual(params['fast_period'], 5)
        self.assertLessEqual(params['fast_period'], 50)
        self.assertGreaterEqual(params['slow_period'], 20)
        self.assertLessEqual(params['slow_period'], 200)
        self.assertIn(params['signal_type'], ['crossover', 'level', 'divergence'])
        self.assertIsInstance(params['use_adaptive_sizing'], bool)
    
    def test_create_individual(self):
        """Test individual creation"""
        individual = self.manager._create_individual(self.strategy_template, self.param_ranges)
        
        # Check individual has the right structure
        self.assertEqual(individual['name'], 'TestStrategy')
        self.assertEqual(individual['type'], 'moving_average_crossover')
        
        # Check parameters were created correctly
        params = individual['parameters']
        self.assertIsInstance(params['fast_period'], int)
        self.assertIsInstance(params['slow_period'], int)
        self.assertIn(params['signal_type'], ['crossover', 'level', 'divergence'])
        self.assertIsInstance(params['use_adaptive_sizing'], bool)
    
    def test_evaluate_population(self):
        """Test population evaluation"""
        # Initialize population
        self.manager.initialize_population(self.strategy_template, self.param_ranges)
        
        # Create mock evaluation function
        def mock_evaluator(individual):
            # Return higher fitness for larger fast_period
            fast = individual['parameters']['fast_period']
            return {
                'sharpe_ratio': fast / 50.0,  # Normalize to 0-1
                'win_rate': fast / 100.0      # Normalize to 0-0.5
            }
        
        # Evaluate the population
        self.manager.evaluate_population(mock_evaluator)
        
        # Check fitness was calculated
        for individual in self.manager.current_population:
            self.assertIn('fitness', individual)
            self.assertIn('performance', individual)
            self.assertIn('sharpe_ratio', individual['performance'])
            self.assertIn('win_rate', individual['performance'])
        
        # Check population is sorted by fitness
        for i in range(len(self.manager.current_population) - 1):
            self.assertGreaterEqual(
                self.manager.current_population[i]['fitness'],
                self.manager.current_population[i+1]['fitness']
            )
        
        # Check best individual is set
        self.assertIsNotNone(self.manager.best_individual)
        self.assertEqual(
            self.manager.best_individual['id'],
            self.manager.current_population[0]['id']
        )
    
    def test_calculate_fitness(self):
        """Test fitness calculation"""
        # Test with normal metrics
        performance = {
            'sharpe_ratio': 1.5,
            'win_rate': 0.6
        }
        
        fitness = self.manager._calculate_fitness(performance)
        expected = 1.5 * 0.5 + 0.6 * 0.5
        self.assertAlmostEqual(fitness, expected)
        
        # Test with inverted metrics (max_drawdown)
        performance = {
            'sharpe_ratio': 1.5,
            'win_rate': 0.6,
            'max_drawdown': 0.2
        }
        
        # Add max_drawdown to fitness metrics with weight 0.2
        original_metrics = self.manager.fitness_metrics.copy()
        self.manager.fitness_metrics[EvolutionMetric.MAX_DRAWDOWN.value] = 0.2
        
        # Need to adjust other weights
        self.manager.fitness_metrics[EvolutionMetric.SHARPE_RATIO.value] = 0.4
        self.manager.fitness_metrics[EvolutionMetric.WIN_RATE.value] = 0.4
        
        fitness = self.manager._calculate_fitness(performance)
        
        # For max_drawdown, lower is better, so it's inverted (1/0.2 = 5)
        expected = 1.5 * 0.4 + 0.6 * 0.4 + 5 * 0.2
        self.assertAlmostEqual(fitness, expected)
        
        # Restore original metrics
        self.manager.fitness_metrics = original_metrics
    
    def test_tournament_selection(self):
        """Test tournament selection"""
        # Initialize population with known fitness values
        self.manager.initialize_population(self.strategy_template, self.param_ranges)
        
        # Assign fitness values (individuals at even indices have higher fitness)
        for i, individual in enumerate(self.manager.current_population):
            individual['fitness'] = 10.0 if i % 2 == 0 else 5.0
        
        # Mock the random.sample function to always return the first tournament_size individuals
        with patch('random.sample', return_value=self.manager.current_population[:3]):
            selected = self.manager._tournament_selection()
            
            # Should select individual with highest fitness in tournament
            self.assertEqual(selected['fitness'], 10.0)
    
    def test_crossover(self):
        """Test crossover between individuals"""
        # Create two parent individuals with different parameter values
        parent1 = {
            'name': 'Parent1',
            'parameters': {
                'fast_period': 10,
                'slow_period': 50,
                'signal_type': 'crossover',
                'use_adaptive_sizing': True
            }
        }
        
        parent2 = {
            'name': 'Parent2',
            'parameters': {
                'fast_period': 20,
                'slow_period': 100,
                'signal_type': 'level',
                'use_adaptive_sizing': False
            }
        }
        
        # Set param ranges for crossover
        self.manager.param_ranges = self.param_ranges
        
        # Perform crossover
        with patch('random.random', return_value=0.4):  # Will choose averaging for numeric params
            offspring = self.manager._crossover(parent1, parent2)
            
            # Check the offspring has parameters from both parents
            params = offspring['parameters']
            
            # Numeric parameters should be averaged
            self.assertEqual(params['fast_period'], 15)  # Average of 10 and 20
            self.assertEqual(params['slow_period'], 75)  # Average of 50 and 100
            
            # Other parameters should be selected from one parent
            self.assertIn(params['signal_type'], ['crossover', 'level'])
            self.assertIn(params['use_adaptive_sizing'], [True, False])
    
    def test_mutate(self):
        """Test mutation"""
        # Create an individual to mutate
        individual = {
            'name': 'TestIndividual',
            'parameters': {
                'fast_period': 10,
                'slow_period': 50,
                'signal_type': 'crossover',
                'use_adaptive_sizing': True
            }
        }
        
        # Set param ranges for mutation
        self.manager.param_ranges = self.param_ranges
        
        # Control the randomization
        with patch('random.sample', return_value=['fast_period']):  # Only mutate fast_period
            with patch('random.uniform', return_value=5.0):  # Add 5 to the value
                mutated = self.manager._mutate(individual)
                
                # Check the mutation occurred
                self.assertEqual(mutated['parameters']['fast_period'], 15)  # 10 + 5
                
                # Check other parameters remained the same
                self.assertEqual(mutated['parameters']['slow_period'], 50)
                self.assertEqual(mutated['parameters']['signal_type'], 'crossover')
                self.assertEqual(mutated['parameters']['use_adaptive_sizing'], True)
    
    def test_evolve(self):
        """Test evolution process"""
        # Initialize population
        self.manager.initialize_population(self.strategy_template, self.param_ranges)
        
        # Create mock evaluation function
        mock_evaluator = MagicMock(return_value={'sharpe_ratio': 1.0, 'win_rate': 0.5})
        
        # Run one generation of evolution
        with patch.object(self.manager, '_tournament_selection') as mock_tournament:
            # Make tournament selection return the first individual
            mock_tournament.return_value = self.manager.current_population[0]
            
            # Evolve
            result = self.manager.evolve(mock_evaluator)
            
            # Check evolution continued
            self.assertTrue(result)
            
            # Check generation incremented
            self.assertEqual(self.manager.current_generation, 1)
            
            # Check evaluator was called for each individual
            self.assertEqual(mock_evaluator.call_count, 10)  # population_size
    
    def test_check_termination(self):
        """Test termination conditions"""
        # Test generation limit
        self.manager.current_generation = self.manager.generation_limit
        self.assertTrue(self.manager._check_termination())
        
        # Test no improvement generations
        self.manager.current_generation = 0  # Reset
        self.manager.no_improvement_generations = self.manager.max_no_improvement
        self.assertTrue(self.manager._check_termination())
        
        # Test both conditions false
        self.manager.current_generation = 0
        self.manager.no_improvement_generations = 0
        self.assertFalse(self.manager._check_termination())
    
    def test_record_generation(self):
        """Test generation recording"""
        # Initialize and evaluate population
        self.manager.initialize_population(self.strategy_template, self.param_ranges)
        
        # Mock the best individual
        self.manager.best_individual = self.manager.current_population[0]
        self.manager.best_individual['id'] = 'best_id'
        self.manager.best_individual['fitness'] = 0.8
        
        # Record generation
        self.manager._record_generation()
        
        # Check generation was recorded
        self.assertEqual(len(self.manager.population_history), 1)
        
        # Check record structure
        record = self.manager.population_history[0]
        self.assertEqual(record['generation'], 0)
        self.assertEqual(record['best_individual'], 'best_id')
        self.assertEqual(record['best_fitness'], 0.8)
        self.assertEqual(len(record['population']), 10)  # population_size
    
    def test_save_load_state(self):
        """Test saving and loading state"""
        # Setup state to save
        self.manager.initialize_population(self.strategy_template, self.param_ranges)
        self.manager.current_generation = 3
        self.manager.best_individual = self.manager.current_population[0]
        
        # Create a temp file for state
        state_file = os.path.join(self.temp_dir, 'test_state.json')
        
        # Save state
        with patch('os.makedirs'):  # Mock directory creation
            with patch('json.dump') as mock_dump:
                self.manager.save_state(state_file)
                
                # Check json.dump was called
                mock_dump.assert_called_once()
                
                # Get the state that would be saved
                saved_state = mock_dump.call_args[0][0]
                
                # Check state structure
                self.assertEqual(saved_state['name'], 'test_manager')
                self.assertEqual(saved_state['current_generation'], 3)
                self.assertIn('best_individual', saved_state)
        
        # Load state
        with patch('os.path.exists', return_value=True):
            with patch('json.load', return_value={
                'name': 'loaded_manager',
                'current_generation': 5,
                'best_individual': {'id': 'loaded_best', 'fitness': 0.9}
            }):
                self.manager.load_state(state_file)
                
                # Check state was loaded
                self.assertEqual(self.manager.name, 'loaded_manager')
                self.assertEqual(self.manager.current_generation, 5)
                self.assertEqual(self.manager.best_individual['id'], 'loaded_best')
    
    def test_generate_evolution_report(self):
        """Test generating evolution report"""
        # Initialize and evolve population
        self.manager.initialize_population(self.strategy_template, self.param_ranges)
        
        # Mock population history
        self.manager.population_history = [
            {
                'generation': 0,
                'best_fitness': 0.5,
                'population': [{'fitness': 0.5}, {'fitness': 0.4}]
            },
            {
                'generation': 1,
                'best_fitness': 0.7,
                'population': [{'fitness': 0.7}, {'fitness': 0.6}]
            }
        ]
        
        # Mock best individual
        self.manager.best_individual = {
            'id': 'best_id',
            'fitness': 0.7,
            'performance': {'sharpe_ratio': 1.5, 'win_rate': 0.6}
        }
        
        # Generate report
        report = self.manager.generate_evolution_report()
        
        # Check report structure
        self.assertIn('status', report)
        self.assertIn('generations', report)
        self.assertIn('best_fitness', report)
        self.assertIn('improvement', report)
        self.assertIn('fitness_progression', report)
        
        # Check values
        self.assertEqual(report['best_fitness'], 0.7)
        self.assertEqual(report['best_strategy_id'], 'best_id')
        self.assertEqual(len(report['fitness_progression']['generations']), 2)
        
        # Check improvement calculation
        expected_improvement = (0.7 - 0.5) / 0.5 * 100  # 40%
        self.assertAlmostEqual(report['improvement'], expected_improvement)


if __name__ == '__main__':
    unittest.main() 