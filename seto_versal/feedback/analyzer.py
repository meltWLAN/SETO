#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feedback analyzer module for SETO-Versal
Tracks and analyzes trading performance for continuous improvement
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class FeedbackAnalyzer:
    """
    Feedback analyzer for SETO-Versal
    
    Records and analyzes trading activity to provide insights for system evolution
    """
    
    def __init__(self, config):
        """
        Initialize the feedback analyzer
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.record_trades = config.get('record_trades', True)
        self.attribution_analysis = config.get('attribution_analysis', True)
        self.performance_metrics = config.get('performance_metrics', 
                                             ['sharpe', 'sortino', 'win_rate', 'profit_factor'])
        
        # Data structures for tracking
        self.intentions = []
        self.decisions = []
        self.filtered_decisions = []
        self.execution_results = []
        self.daily_performance = []
        self.agent_performance = defaultdict(list)
        
        # Performance metrics
        self.current_portfolio_value = 0
        self.starting_portfolio_value = 0
        self.peak_portfolio_value = 0
        self.trades_won = 0
        self.trades_lost = 0
        self.total_profit = 0
        self.total_loss = 0
        
        # Strategy performance tracking
        self.strategy_performance = defaultdict(lambda: {
            'signals': 0,
            'executed': 0,
            'wins': 0,
            'losses': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'average_confidence': 0.0
        })
        
        # Market regime performance
        self.regime_performance = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_profit': 0.0,
            'total_loss': 0.0
        })
        
        logger.info("Feedback analyzer initialized")
    
    def record_cycle(self, intentions, decisions, filtered_decisions, results, market_state):
        """
        Record data from a single trading cycle
        
        Args:
            intentions (list): Trading intentions from agents
            decisions (list): Coordinated decisions
            filtered_decisions (list): Risk-filtered decisions
            results (list): Execution results
            market_state (MarketState): Current market state
        """
        # Record the data for analysis
        logger.debug(f"Recording cycle data: {len(intentions)} intentions, {len(decisions)} decisions")
        
        # Simple implementation for now
        timestamp = datetime.now()
        
        # Store data for analysis
        self.intentions.extend(intentions)
        self.decisions.extend(decisions)
        self.filtered_decisions.extend(filtered_decisions)
        self.execution_results.extend(results)
    
    def generate_daily_report(self, market_state, agents, positions):
        """
        Generate a daily performance report
        
        Args:
            market_state (MarketState): Current market state
            agents (list): List of agent instances
            positions (dict): Current portfolio positions
            
        Returns:
            dict: Daily performance report
        """
        # Calculate basic metrics
        num_trades = len(self.execution_results)
        
        # Create a simple report
        report = {
            'date': datetime.now().date().isoformat(),
            'market_regime': market_state.market_regime,
            'trades_executed': num_trades,
            'positions': len(positions),
            'agents_active': len(agents)
        }
        
        return report
        
    def _format_intention(self, intention, timestamp):
        """Format an agent intention for recording"""
        # Simple implementation - in reality would extract all relevant fields
        return {
            'timestamp': timestamp,
            'agent_name': intention.agent_name if hasattr(intention, 'agent_name') else 'unknown',
            'symbol': intention.symbol if hasattr(intention, 'symbol') else 'unknown',
            'direction': intention.direction if hasattr(intention, 'direction') else 'unknown',
            'confidence': intention.confidence if hasattr(intention, 'confidence') else 0,
            'type': 'intention'
        }
        
    def _format_decision(self, decision, timestamp, stage='coordinated'):
        """Format a trade decision for recording"""
        return {
            'timestamp': timestamp,
            'stage': stage,
            'id': decision.id if hasattr(decision, 'id') else 'unknown',
            'symbol': decision.symbol if hasattr(decision, 'symbol') else 'unknown',
            'decision_type': decision.decision_type if hasattr(decision, 'decision_type') else 'unknown',
            'quantity': decision.quantity if hasattr(decision, 'quantity') else 0,
            'confidence': decision.confidence if hasattr(decision, 'confidence') else 0,
            'agent_decisions': decision.agent_decisions if hasattr(decision, 'agent_decisions') else [],
            'type': 'decision'
        }
        
    def _format_result(self, result, timestamp):
        """Format an execution result for recording"""
        return {
            'timestamp': timestamp,
            'decision_id': result.get('decision_id', 'unknown'),
            'symbol': result.get('symbol', 'unknown'),
            'decision_type': result.get('decision_type', 'unknown'),
            'quantity': result.get('quantity', 0),
            'price': result.get('price', 0),
            'status': result.get('status', 'unknown'),
            'message': result.get('message', ''),
            'agent_name': result.get('agent_name', 'unknown'),
            'type': 'execution'
        }
        
    def _update_performance_metrics(self, results, market_state):
        """Update performance metrics based on execution results"""
        # Simple implementation - would be more comprehensive in reality
        for result in results:
            if result.get('status') != 'executed':
                continue
                
            price = result.get('price', 0)
            quantity = result.get('quantity', 0)
            trade_value = price * quantity
            
            # Update trade statistics
            if result.get('decision_type') == 'sell' and trade_value > 0:
                self.trades_won += 1
                self.total_profit += trade_value
            elif result.get('decision_type') == 'sell' and trade_value < 0:
                self.trades_lost += 1
                self.total_loss += abs(trade_value)
                
    def _calculate_portfolio_value(self, positions):
        """Calculate total portfolio value from positions"""
        total_value = 0
        for symbol, position in positions.items():
            total_value += position.get('market_value', 0)
            
        return total_value
        
    def _reset_daily_metrics(self, current_value):
        """Reset daily metrics for next day"""
        self.starting_portfolio_value = current_value
        self.current_portfolio_value = current_value
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
            
        # Clear daily tracking
        self.intentions = []
        self.decisions = []
        self.filtered_decisions = []
        self.execution_results = []
    
    def analyze_agent_performance(self):
        """
        Analyze agent performance
        
        Returns:
            dict: Agent performance statistics
        """
        results = {}
        
        for agent_name, stats in self.agent_performance.items():
            # Skip agents with no executions
            if stats['executions'] == 0:
                continue
            
            # Calculate key metrics
            total_trades = stats['wins'] + stats['losses']
            win_rate = stats['wins'] / total_trades if total_trades > 0 else 0
            avg_profit = stats['total_profit'] / stats['wins'] if stats['wins'] > 0 else 0
            avg_loss = stats['total_loss'] / stats['losses'] if stats['losses'] > 0 else 0
            profit_factor = stats['total_profit'] / stats['total_loss'] if stats['total_loss'] > 0 else float('inf')
            
            # Calculate execution rate
            execution_rate = stats['executions'] / stats['intentions'] if stats['intentions'] > 0 else 0
            
            # Calculate average confidence
            avg_confidence = stats['confidence_sum'] / stats['intentions'] if stats['intentions'] > 0 else 0
            
            results[agent_name] = {
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'execution_rate': execution_rate,
                'avg_confidence': avg_confidence,
                'total_profit': stats['total_profit'],
                'total_loss': stats['total_loss'],
                'net_profit': stats['total_profit'] - stats['total_loss'],
                'trades': total_trades
            }
        
        return results
    
    def analyze_strategy_performance(self):
        """
        Analyze strategy performance
        
        Returns:
            dict: Strategy performance statistics
        """
        results = {}
        
        for strategy_name, stats in self.strategy_performance.items():
            # Skip strategies with no executions
            if stats['executed'] == 0:
                continue
            
            # Calculate key metrics
            total_trades = stats['wins'] + stats['losses']
            win_rate = stats['wins'] / total_trades if total_trades > 0 else 0
            avg_profit = stats['total_profit'] / stats['wins'] if stats['wins'] > 0 else 0
            avg_loss = stats['total_loss'] / stats['losses'] if stats['losses'] > 0 else 0
            profit_factor = stats['total_profit'] / stats['total_loss'] if stats['total_loss'] > 0 else float('inf')
            
            # Calculate signal-to-execution rate
            execution_rate = stats['executed'] / stats['signals'] if stats['signals'] > 0 else 0
            
            results[strategy_name] = {
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'execution_rate': execution_rate,
                'total_profit': stats['total_profit'],
                'total_loss': stats['total_loss'],
                'net_profit': stats['total_profit'] - stats['total_loss'],
                'trades': total_trades
            }
        
        return results
    
    def analyze_market_regime_performance(self):
        """
        Analyze performance across different market regimes
        
        Returns:
            dict: Market regime performance statistics
        """
        results = {}
        
        for regime, stats in self.regime_performance.items():
            # Skip regimes with no trades
            if stats['trades'] == 0:
                continue
            
            # Calculate key metrics
            total_trades = stats['wins'] + stats['losses']
            win_rate = stats['wins'] / total_trades if total_trades > 0 else 0
            avg_profit = stats['total_profit'] / stats['wins'] if stats['wins'] > 0 else 0
            avg_loss = stats['total_loss'] / stats['losses'] if stats['losses'] > 0 else 0
            profit_factor = stats['total_profit'] / stats['total_loss'] if stats['total_loss'] > 0 else float('inf')
            
            results[regime] = {
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_profit': stats['total_profit'],
                'total_loss': stats['total_loss'],
                'net_profit': stats['total_profit'] - stats['total_loss'],
                'trades': total_trades
            }
        
        return results
    
    def get_performance_data(self):
        """
        Get comprehensive performance data for analysis
        
        Returns:
            dict: Performance data
        """
        return {
            'agent_performance': self.analyze_agent_performance(),
            'strategy_performance': self.analyze_strategy_performance(),
            'market_regime_performance': self.analyze_market_regime_performance(),
            'intention_count': len(self.intentions),
            'decision_count': len(self.decisions),
            'execution_count': len(self.execution_results)
        }
    
    def export_data(self, file_path):
        """
        Export feedback data to file
        
        Args:
            file_path (str): Path to output file
            
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            data = {
                'intentions': self.intentions,
                'decisions': self.decisions,
                'executions': self.execution_results,
                'market_states': self.daily_performance,
                'agent_performance': dict(self.agent_performance),
                'strategy_performance': dict(self.strategy_performance),
                'regime_performance': dict(self.regime_performance)
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Exported feedback data to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting feedback data: {e}")
            return False
    
    def import_data(self, file_path):
        """
        Import feedback data from file
        
        Args:
            file_path (str): Path to input file
            
        Returns:
            bool: True if import successful, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.intentions = data.get('intentions', [])
            self.decisions = data.get('decisions', [])
            self.execution_results = data.get('executions', [])
            self.daily_performance = data.get('market_states', [])
            
            # Convert defaultdicts
            for agent, stats in data.get('agent_performance', {}).items():
                self.agent_performance[agent].update(stats)
            
            for strategy, stats in data.get('strategy_performance', {}).items():
                self.strategy_performance[strategy].update(stats)
            
            for regime, stats in data.get('regime_performance', {}).items():
                self.regime_performance[regime].update(stats)
            
            logger.info(f"Imported feedback data from {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error importing feedback data: {e}")
            return False 