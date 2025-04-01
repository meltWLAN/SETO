#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test mode-specific configurations
"""

import unittest
import os
import json
from datetime import datetime, time
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from seto_versal.market.state import MarketState
from seto_versal.agents.factory import AgentFactory
from seto_versal.evolution.engine import EvolutionEngine
from seto_versal.risk.manager import RiskManager

class TestModeConfigs(unittest.TestCase):
    """Test mode-specific configurations"""
    
    def setUp(self):
        """Set up test environment"""
        # Create test configuration
        self.config = {
            'mode': 'backtest',
            'mode_configs': {
                'backtest': {
                    'update_interval': 300,
                    'use_cache': True,
                    'risk_level': 'high',
                    'max_drawdown': 0.05,
                    'max_position_size': 0.3,
                    'trailing_stop': True,
                    'mutation_rate': 0.3,
                    'breeding_threshold': 0.6,
                    'optimization_frequency': 'daily',
                    'agent_params': {
                        'trend': {
                            'confidence_threshold': 0.8,
                            'max_positions': 8,
                            'weight': 1.2,
                            'mode_restrictions': {
                                'backtest': True,
                                'paper': True,
                                'live': True
                            }
                        }
                    }
                },
                'paper': {
                    'update_interval': 60,
                    'use_cache': True,
                    'risk_level': 'medium',
                    'max_drawdown': 0.03,
                    'max_position_size': 0.25,
                    'trailing_stop': True,
                    'mutation_rate': 0.2,
                    'breeding_threshold': 0.5,
                    'optimization_frequency': 'weekly',
                    'agent_params': {
                        'trend': {
                            'confidence_threshold': 0.7,
                            'max_positions': 5,
                            'weight': 1.0,
                            'mode_restrictions': {
                                'backtest': True,
                                'paper': True,
                                'live': True
                            }
                        }
                    }
                },
                'live': {
                    'update_interval': 30,
                    'use_cache': False,
                    'risk_level': 'low',
                    'max_drawdown': 0.02,
                    'max_position_size': 0.2,
                    'trailing_stop': True,
                    'mutation_rate': 0.1,
                    'breeding_threshold': 0.4,
                    'optimization_frequency': 'monthly',
                    'agent_params': {
                        'trend': {
                            'confidence_threshold': 0.9,
                            'max_positions': 3,
                            'weight': 0.8,
                            'mode_restrictions': {
                                'backtest': True,
                                'paper': True,
                                'live': True
                            }
                        }
                    }
                }
            },
            'datasource': 'tushare',
            'universe': 'hs300',
            'trading_hours': {
                'open': '09:30',
                'close': '15:00',
                'lunch_start': '11:30',
                'lunch_end': '13:00'
            },
            'agents': [
                {
                    'type': 'trend',
                    'name': 'trend_agent',
                    'enabled': True,
                    'mode_restrictions': {
                        'backtest': True,
                        'paper': True,
                        'live': True
                    }
                }
            ]
        }
        
    def test_market_state(self):
        """Test MarketState mode-specific configurations"""
        # Test backtest mode
        self.config['mode'] = 'backtest'
        market_state = MarketState(self.config)
        self.assertEqual(market_state.update_interval, 300)
        self.assertTrue(market_state.use_cache)
        
        # Test paper mode
        self.config['mode'] = 'paper'
        market_state = MarketState(self.config)
        self.assertEqual(market_state.update_interval, 60)
        self.assertTrue(market_state.use_cache)
        
        # Test live mode
        self.config['mode'] = 'live'
        market_state = MarketState(self.config)
        self.assertEqual(market_state.update_interval, 30)
        self.assertFalse(market_state.use_cache)
        
    def test_agent_factory(self):
        """Test agent factory configuration"""
        # Test backtest mode
        backtest_config = {
            'mode': 'backtest',
            'agents': [
                {
                    'type': 'trend',
                    'name': 'trend_agent_1',
                    'confidence_threshold': 0.7,
                    'max_positions': 5,
                    'weight': 1.0,
                    'lookback_period': 20,
                    'trend_threshold': 0.05
                }
            ]
        }
        
        factory = AgentFactory(backtest_config)
        agent = factory.create_agent('trend', 'trend_agent_1')
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, 'trend_agent_1')
        self.assertEqual(agent.confidence_threshold, 0.7)
        self.assertEqual(agent.max_positions, 5)
        self.assertEqual(agent.weight, 1.0)
        self.assertEqual(agent.lookback_period, 20)
        self.assertEqual(agent.trend_threshold, 0.05)
        
    def test_evolution_engine(self):
        """Test EvolutionEngine mode-specific configurations"""
        # Test backtest mode
        self.config['mode'] = 'backtest'
        evolution_engine = EvolutionEngine(self.config)
        self.assertEqual(evolution_engine.mutation_rate, 0.3)
        self.assertEqual(evolution_engine.breeding_threshold, 0.6)
        self.assertEqual(evolution_engine.optimization_frequency, 'daily')
        
        # Test paper mode
        self.config['mode'] = 'paper'
        evolution_engine = EvolutionEngine(self.config)
        self.assertEqual(evolution_engine.mutation_rate, 0.2)
        self.assertEqual(evolution_engine.breeding_threshold, 0.5)
        self.assertEqual(evolution_engine.optimization_frequency, 'weekly')
        
        # Test live mode
        self.config['mode'] = 'live'
        evolution_engine = EvolutionEngine(self.config)
        self.assertEqual(evolution_engine.mutation_rate, 0.1)
        self.assertEqual(evolution_engine.breeding_threshold, 0.4)
        self.assertEqual(evolution_engine.optimization_frequency, 'monthly')
        
    def test_risk_manager(self):
        """Test RiskManager mode-specific configurations"""
        # Test backtest mode
        self.config['mode'] = 'backtest'
        risk_manager = RiskManager(self.config)
        self.assertEqual(risk_manager.risk_level, 'high')
        self.assertEqual(risk_manager.max_drawdown, 0.05)
        self.assertEqual(risk_manager.max_position_size, 0.3)
        self.assertTrue(risk_manager.trailing_stop)
        
        # Test paper mode
        self.config['mode'] = 'paper'
        risk_manager = RiskManager(self.config)
        self.assertEqual(risk_manager.risk_level, 'medium')
        self.assertEqual(risk_manager.max_drawdown, 0.03)
        self.assertEqual(risk_manager.max_position_size, 0.25)
        self.assertTrue(risk_manager.trailing_stop)
        
        # Test live mode
        self.config['mode'] = 'live'
        risk_manager = RiskManager(self.config)
        self.assertEqual(risk_manager.risk_level, 'low')
        self.assertEqual(risk_manager.max_drawdown, 0.02)
        self.assertEqual(risk_manager.max_position_size, 0.2)
        self.assertTrue(risk_manager.trailing_stop)
        
    def test_market_hours(self):
        """Test market hours configuration"""
        market_state = MarketState(self.config)
        
        # Test market open check
        self.assertTrue(market_state.is_market_open())  # Backtest mode always returns True
        
        # Test market hours parsing
        self.assertEqual(market_state.market_hours['open'], '09:30')
        self.assertEqual(market_state.market_hours['close'], '15:00')
        self.assertEqual(market_state.market_hours['lunch_start'], '11:30')
        self.assertEqual(market_state.market_hours['lunch_end'], '13:00')
        
    def test_risk_limits(self):
        """Test risk limits checking"""
        risk_manager = RiskManager(self.config)
        
        # Test position size limit
        portfolio_value = 100000
        positions = {
            '000001.SZ': {'value': 30000},  # 30% position
            '600519.SH': {'value': 20000}   # 20% position
        }
        
        # Backtest mode should allow larger positions
        self.config['mode'] = 'backtest'
        risk_manager = RiskManager(self.config)
        self.assertFalse(risk_manager.check_risk_limits(portfolio_value, positions))
        
        # Live mode should be more restrictive
        self.config['mode'] = 'live'
        risk_manager = RiskManager(self.config)
        self.assertTrue(risk_manager.check_risk_limits(portfolio_value, positions))
        
    def test_evolution_timing(self):
        """Test evolution timing based on mode"""
        # Test backtest mode
        self.config['mode'] = 'backtest'
        market_state = MarketState(self.config)
        # In backtest mode, evolution time is random with 10% chance
        evolution_times = [market_state.is_evolution_time() for _ in range(100)]
        self.assertTrue(any(evolution_times))  # At least one should be True
        
        # Test paper mode
        self.config['mode'] = 'paper'
        market_state = MarketState(self.config)
        # Paper mode also uses random evolution
        evolution_times = [market_state.is_evolution_time() for _ in range(100)]
        self.assertTrue(any(evolution_times))  # At least one should be True
        
        # Test live mode
        self.config['mode'] = 'live'
        market_state = MarketState(self.config)
        # Live mode checks market hours, which we can't easily test
        # Just verify it returns a boolean
        self.assertIsInstance(market_state.is_evolution_time(), bool)

if __name__ == '__main__':
    unittest.main() 