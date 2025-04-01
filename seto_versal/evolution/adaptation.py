#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptation engine module for SETO-Versal
Provides real-time adaptation of trading strategies based on changing market conditions
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
import json
import os
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class AdaptationTrigger(Enum):
    """Enum for adaptation triggers"""
    PERFORMANCE_DROP = "performance_drop"
    VOLATILITY_CHANGE = "volatility_change"
    REGIME_CHANGE = "regime_change"
    TIME_BASED = "time_based"
    MANUAL = "manual"

class AdaptationEngine:
    """
    Adaptation engine that dynamically adjusts strategies 
    based on market conditions and performance feedback
    """
    
    def __init__(self, config):
        """
        Initialize the adaptation engine
        
        Args:
            config (dict): Adaptation engine configuration
        """
        self.config = config
        self.name = config.get('name', 'adaptation_engine')
        
        # Adaptation settings
        self.check_interval = config.get('check_interval', 'daily')  # daily, hourly, etc.
        self.sensitivity = config.get('sensitivity', 0.5)  # 0.0-1.0, higher = more sensitive
        self.auto_adapt = config.get('auto_adapt', True)  # Automatically apply adaptations
        
        # Performance thresholds
        self.drawdown_threshold = config.get('drawdown_threshold', 0.08)  # 8% drawdown
        self.volatility_threshold = config.get('volatility_threshold', 0.3)  # 30% vol change
        self.win_rate_threshold = config.get('win_rate_threshold', 0.1)  # 10% win rate drop
        self.profit_factor_threshold = config.get('profit_factor_threshold', 0.3)  # 30% PF drop
        
        # Tracking and history
        self.adaptation_history = []
        self.strategy_states = {}
        self.market_regime_history = []
        self.last_check_time = None
        
        # Define the data path for saving adaptations
        self.data_path = config.get('data_path', 'seto_versal/data/adaptations')
        os.makedirs(self.data_path, exist_ok=True)
        
        logger.info(f"Adaptation engine '{self.name}' initialized")
    
    def check_adaptation_needed(self, market_state, feedback_data):
        """
        Check if adaptation is needed based on market conditions and feedback
        
        Args:
            market_state (MarketState): Current market state
            feedback_data (dict): Performance feedback data
            
        Returns:
            tuple: (needs_adaptation, trigger, adaptation_data)
                - needs_adaptation (bool): Whether adaptation is needed
                - trigger (AdaptationTrigger): The trigger type
                - adaptation_data (dict): Data related to the adaptation
        """
        # Check if enough time has passed since last check
        if not self._should_check_now():
            return False, None, None
        
        # Update last check time
        self.last_check_time = datetime.now()
        
        # Track current market regime
        current_regime = market_state.get_market_regime()
        self._update_regime_history(current_regime)
        
        # Check different adaptation triggers
        triggers = [
            self._check_performance_drop(feedback_data),
            self._check_volatility_change(market_state, feedback_data),
            self._check_regime_change(),
            self._check_time_based_adaptation()
        ]
        
        # Find the highest priority trigger (first non-None result)
        for trigger_result in triggers:
            if trigger_result[0]:  # If adaptation needed
                trigger, adaptation_data = trigger_result
                
                # Log the recommendation
                logger.info(f"Adaptation recommended: {trigger.value}")
                
                return True, trigger, adaptation_data
        
        # No adaptation needed
        return False, None, None
    
    def apply_adaptation(self, trigger, adaptation_data, strategies, market_state):
        """
        Apply adaptation to strategies based on trigger
        
        Args:
            trigger (AdaptationTrigger): What triggered the adaptation
            adaptation_data (dict): Data related to the adaptation
            strategies (list): List of strategies to adapt
            market_state (MarketState): Current market state
            
        Returns:
            dict: Adaptation results
        """
        if not trigger or not strategies:
            return {'success': False, 'message': 'Invalid trigger or strategies'}
        
        logger.info(f"Applying {trigger.value} adaptation to {len(strategies)} strategies")
        
        # Results tracking
        results = {
            'trigger': trigger.value,
            'timestamp': datetime.now().isoformat(),
            'adapted_strategies': [],
            'success': True,
            'message': ''
        }
        
        try:
            # Apply adaptation based on trigger type
            if trigger == AdaptationTrigger.PERFORMANCE_DROP:
                self._adapt_to_performance_drop(strategies, adaptation_data, results)
            
            elif trigger == AdaptationTrigger.VOLATILITY_CHANGE:
                self._adapt_to_volatility_change(strategies, adaptation_data, results)
            
            elif trigger == AdaptationTrigger.REGIME_CHANGE:
                self._adapt_to_regime_change(strategies, adaptation_data, market_state, results)
            
            elif trigger == AdaptationTrigger.TIME_BASED:
                self._apply_time_based_adaptation(strategies, adaptation_data, results)
            
            # Save adaptation history
            self._record_adaptation(trigger, adaptation_data, results)
            
            return results
        
        except Exception as e:
            error_msg = f"Error applying adaptation: {str(e)}"
            logger.error(error_msg)
            results['success'] = False
            results['message'] = error_msg
            return results
    
    def _should_check_now(self):
        """
        Determine if enough time has passed to check for adaptation
        
        Returns:
            bool: Whether to check for adaptation
        """
        if self.last_check_time is None:
            return True
        
        now = datetime.now()
        time_diff = now - self.last_check_time
        
        # Check based on interval
        if self.check_interval == 'hourly':
            return time_diff.total_seconds() >= 3600  # 1 hour
        elif self.check_interval == 'daily':
            return time_diff.total_seconds() >= 86400  # 24 hours
        elif self.check_interval == 'weekly':
            return time_diff.total_seconds() >= 604800  # 7 days
        
        # Default - check every hour
        return time_diff.total_seconds() >= 3600
    
    def _update_regime_history(self, current_regime):
        """
        Update the market regime history
        
        Args:
            current_regime (str): Current market regime
        """
        self.market_regime_history.append({
            'regime': current_regime,
            'timestamp': datetime.now().isoformat()
        })
        
        # Limit history size
        max_history = 100
        if len(self.market_regime_history) > max_history:
            self.market_regime_history = self.market_regime_history[-max_history:]
    
    def _check_performance_drop(self, feedback_data):
        """
        Check if performance has dropped significantly
        
        Args:
            feedback_data (dict): Performance feedback data
            
        Returns:
            tuple: (trigger, adaptation_data) or (None, None)
        """
        # Extract relevant metrics
        recent_metrics = feedback_data.get('recent_metrics', {})
        baseline_metrics = feedback_data.get('baseline_metrics', {})
        
        if not recent_metrics or not baseline_metrics:
            return None, None
        
        # Check various performance metrics
        drawdown_exceeded = False
        win_rate_drop = False
        profit_factor_drop = False
        sharpe_drop = False
        
        # Check drawdown
        recent_dd = recent_metrics.get('max_drawdown', 0)
        if recent_dd > self.drawdown_threshold * 100:  # Convert to percentage
            drawdown_exceeded = True
        
        # Check win rate
        recent_wr = recent_metrics.get('win_rate', 0)
        baseline_wr = baseline_metrics.get('win_rate', 0)
        if baseline_wr > 0 and (baseline_wr - recent_wr) / baseline_wr > self.win_rate_threshold:
            win_rate_drop = True
        
        # Check profit factor
        recent_pf = recent_metrics.get('profit_factor', 0)
        baseline_pf = baseline_metrics.get('profit_factor', 0)
        if baseline_pf > 1.0 and (baseline_pf - recent_pf) / baseline_pf > self.profit_factor_threshold:
            profit_factor_drop = True
        
        # Check Sharpe ratio
        recent_sharpe = recent_metrics.get('sharpe_ratio', 0)
        baseline_sharpe = baseline_metrics.get('sharpe_ratio', 0)
        if baseline_sharpe > 0.5 and (baseline_sharpe - recent_sharpe) / baseline_sharpe > 0.3:
            sharpe_drop = True
        
        # If enough indicators suggest a performance drop
        drop_indicators = sum([drawdown_exceeded, win_rate_drop, profit_factor_drop, sharpe_drop])
        if drop_indicators >= 2:  # At least 2 indicators
            adaptation_data = {
                'drawdown_exceeded': drawdown_exceeded,
                'win_rate_drop': win_rate_drop,
                'profit_factor_drop': profit_factor_drop,
                'sharpe_drop': sharpe_drop,
                'recent_metrics': recent_metrics,
                'baseline_metrics': baseline_metrics
            }
            return AdaptationTrigger.PERFORMANCE_DROP, adaptation_data
        
        return None, None
    
    def _check_volatility_change(self, market_state, feedback_data):
        """
        Check if market volatility has changed significantly
        
        Args:
            market_state (MarketState): Current market state
            feedback_data (dict): Performance feedback data
            
        Returns:
            tuple: (trigger, adaptation_data) or (None, None)
        """
        # Get current market volatility
        current_volatility = market_state.get_market_volatility()
        if current_volatility is None:
            return None, None
        
        # Get baseline volatility from feedback data
        baseline_volatility = feedback_data.get('baseline_metrics', {}).get('market_volatility')
        if baseline_volatility is None:
            return None, None
        
        # Calculate percent change
        volatility_change = abs(current_volatility - baseline_volatility) / baseline_volatility
        
        # If change exceeds threshold
        if volatility_change > self.volatility_threshold:
            adaptation_data = {
                'current_volatility': current_volatility,
                'baseline_volatility': baseline_volatility,
                'percent_change': volatility_change,
                'direction': 'increase' if current_volatility > baseline_volatility else 'decrease'
            }
            return AdaptationTrigger.VOLATILITY_CHANGE, adaptation_data
        
        return None, None
    
    def _check_regime_change(self):
        """
        Check if market regime has recently changed
        
        Returns:
            tuple: (trigger, adaptation_data) or (None, None)
        """
        # Need at least 2 regime points to detect a change
        if len(self.market_regime_history) < 2:
            return None, None
        
        # Get current and previous regimes
        current = self.market_regime_history[-1]
        previous = self.market_regime_history[-2]
        
        # Check if regime changed
        if current['regime'] != previous['regime']:
            # Calculate how long ago the change happened
            current_time = datetime.fromisoformat(current['timestamp'])
            previous_time = datetime.fromisoformat(previous['timestamp'])
            time_since_change = (datetime.now() - current_time).total_seconds() / 3600  # hours
            
            # Only trigger if change is recent (within 24 hours) and we haven't adapted yet
            if time_since_change < 24:
                # Check if we've already adapted to this regime change
                adapted = False
                for adaptation in self.adaptation_history:
                    if (adaptation['trigger'] == AdaptationTrigger.REGIME_CHANGE.value and
                        adaptation['timestamp'] > previous['timestamp']):
                        adapted = True
                        break
                
                if not adapted:
                    adaptation_data = {
                        'new_regime': current['regime'],
                        'previous_regime': previous['regime'],
                        'change_time': current['timestamp']
                    }
                    return AdaptationTrigger.REGIME_CHANGE, adaptation_data
        
        return None, None
    
    def _check_time_based_adaptation(self):
        """
        Check if regular time-based adaptation is needed
        
        Returns:
            tuple: (trigger, adaptation_data) or (None, None)
        """
        # Get the last time-based adaptation
        last_time_adaptation = None
        for adaptation in reversed(self.adaptation_history):
            if adaptation['trigger'] == AdaptationTrigger.TIME_BASED.value:
                last_time_adaptation = adaptation
                break
        
        # Determine if time-based adaptation is needed
        if last_time_adaptation is None:
            # No previous time-based adaptation
            adaptation_data = {'reason': 'initial_adaptation'}
            return AdaptationTrigger.TIME_BASED, adaptation_data
        else:
            # Check if enough time has passed since last adaptation
            last_time = datetime.fromisoformat(last_time_adaptation['timestamp'])
            time_diff = (datetime.now() - last_time).days
            
            adaptation_interval = self.config.get('time_adaptation_interval', 30)  # days
            if time_diff >= adaptation_interval:
                adaptation_data = {
                    'reason': 'periodic_adaptation',
                    'days_since_last': time_diff
                }
                return AdaptationTrigger.TIME_BASED, adaptation_data
        
        return None, None
    
    def _adapt_to_performance_drop(self, strategies, adaptation_data, results):
        """
        Apply adaptation for performance drop
        
        Args:
            strategies (list): Strategies to adapt
            adaptation_data (dict): Adaptation data
            results (dict): Results to update
        """
        for strategy in strategies:
            # Skip if strategy doesn't support adaptation
            if not hasattr(strategy, 'adapt') or not callable(getattr(strategy, 'adapt')):
                continue
            
            # Determine adaptation severity based on drops
            severity = 0
            if adaptation_data.get('drawdown_exceeded', False):
                severity += 3
            if adaptation_data.get('profit_factor_drop', False):
                severity += 2
            if adaptation_data.get('win_rate_drop', False):
                severity += 2
            if adaptation_data.get('sharpe_drop', False):
                severity += 1
            
            # Normalize severity to 0-1 range
            severity = min(1.0, severity / 8.0)
            
            # Apply more conservative settings for severe drops
            if severity > 0.6:
                changes = {
                    'risk_multiplier': max(0.5, 1.0 - severity),  # Reduce risk
                    'trade_frequency': max(0.6, 1.0 - severity/2),  # Slightly reduce frequency
                    'stop_loss_multiplier': 0.8,  # Tighter stops
                    'target_multiplier': 1.2,  # More conservative targets
                }
            else:
                changes = {
                    'risk_multiplier': max(0.7, 1.0 - severity/2),  # Slightly reduce risk
                    'stop_loss_multiplier': 0.9,  # Slightly tighter stops
                }
            
            # Apply adaptations
            adaptation_result = strategy.adapt(changes, reason="performance_drop")
            
            # Record in results
            results['adapted_strategies'].append({
                'strategy_id': getattr(strategy, 'id', str(strategy)),
                'severity': severity,
                'changes': changes,
                'result': adaptation_result
            })
    
    def _adapt_to_volatility_change(self, strategies, adaptation_data, results):
        """
        Apply adaptation for volatility change
        
        Args:
            strategies (list): Strategies to adapt
            adaptation_data (dict): Adaptation data
            results (dict): Results to update
        """
        # Get volatility change direction and magnitude
        direction = adaptation_data.get('direction', 'increase')
        volatility_change = adaptation_data.get('percent_change', 0.3)
        
        for strategy in strategies:
            # Skip if strategy doesn't support adaptation
            if not hasattr(strategy, 'adapt') or not callable(getattr(strategy, 'adapt')):
                continue
            
            # Different adaptations based on direction
            if direction == 'increase':
                # For increased volatility: reduce size, tighten stops, widen targets
                changes = {
                    'risk_multiplier': max(0.5, 1.0 - volatility_change/2),
                    'stop_loss_multiplier': 0.7,  # Tighter stops
                    'target_multiplier': 1.3,  # Wider targets
                    'volatility_adjustment': volatility_change
                }
            else:
                # For decreased volatility: can increase size slightly, adjust stops/targets
                changes = {
                    'risk_multiplier': min(1.2, 1.0 + volatility_change/4),
                    'stop_loss_multiplier': 0.9,  # Slightly tighter stops
                    'target_multiplier': 0.9,  # Tighter targets
                    'volatility_adjustment': -volatility_change
                }
            
            # Apply adaptations
            adaptation_result = strategy.adapt(changes, reason="volatility_change")
            
            # Record in results
            results['adapted_strategies'].append({
                'strategy_id': getattr(strategy, 'id', str(strategy)),
                'direction': direction,
                'volatility_change': volatility_change,
                'changes': changes,
                'result': adaptation_result
            })
    
    def _adapt_to_regime_change(self, strategies, adaptation_data, market_state, results):
        """
        Apply adaptation for market regime change
        
        Args:
            strategies (list): Strategies to adapt
            adaptation_data (dict): Adaptation data
            market_state (MarketState): Current market state
            results (dict): Results to update
        """
        # Get regime change details
        new_regime = adaptation_data.get('new_regime', 'unknown')
        previous_regime = adaptation_data.get('previous_regime', 'unknown')
        
        for strategy in strategies:
            # Skip if strategy doesn't support adaptation
            if not hasattr(strategy, 'adapt') or not callable(getattr(strategy, 'adapt')):
                continue
            
            # Determine if strategy works better or worse in new regime
            regime_fit = self._evaluate_strategy_regime_fit(strategy, new_regime, market_state)
            
            # Apply adaptations based on regime fit
            if regime_fit < 0.4:  # Poor fit
                changes = {
                    'active': False,  # Deactivate strategy
                    'reason': f"Poor fit for {new_regime} regime"
                }
            elif regime_fit < 0.7:  # Moderate fit
                changes = {
                    'risk_multiplier': 0.7,  # Reduce risk
                    'active': True,
                    'parameter_adjustments': self._get_regime_parameter_adjustments(strategy, new_regime)
                }
            else:  # Good fit
                changes = {
                    'risk_multiplier': 1.0,  # Normal risk
                    'active': True,
                    'parameter_adjustments': self._get_regime_parameter_adjustments(strategy, new_regime)
                }
            
            # Apply adaptations
            adaptation_result = strategy.adapt(changes, reason="regime_change")
            
            # Record in results
            results['adapted_strategies'].append({
                'strategy_id': getattr(strategy, 'id', str(strategy)),
                'regime_change': f"{previous_regime} -> {new_regime}",
                'regime_fit': regime_fit,
                'changes': changes,
                'result': adaptation_result
            })
    
    def _apply_time_based_adaptation(self, strategies, adaptation_data, results):
        """
        Apply regular time-based adaptation
        
        Args:
            strategies (list): Strategies to adapt
            adaptation_data (dict): Adaptation data
            results (dict): Results to update
        """
        reason = adaptation_data.get('reason', 'periodic_adaptation')
        
        for strategy in strategies:
            # Skip if strategy doesn't support adaptation
            if not hasattr(strategy, 'adapt') or not callable(getattr(strategy, 'adapt')):
                continue
            
            if reason == 'initial_adaptation':
                # Initial setup - just make sure strategy is active
                changes = {
                    'active': True,
                    'refresh_parameters': True
                }
            else:
                # Periodic adaptation - light refresh
                changes = {
                    'refresh_parameters': True,
                    'recalibrate': True
                }
            
            # Apply adaptations
            adaptation_result = strategy.adapt(changes, reason="time_based")
            
            # Record in results
            results['adapted_strategies'].append({
                'strategy_id': getattr(strategy, 'id', str(strategy)),
                'reason': reason,
                'changes': changes,
                'result': adaptation_result
            })
    
    def _evaluate_strategy_regime_fit(self, strategy, regime, market_state):
        """
        Evaluate how well a strategy fits the current market regime
        
        Args:
            strategy: Strategy to evaluate
            regime (str): Current market regime
            market_state (MarketState): Current market state
            
        Returns:
            float: Fit score (0.0-1.0)
        """
        # Default score
        default_score = 0.5
        
        # Try to get strategy type
        strategy_type = getattr(strategy, 'type', None)
        if not strategy_type:
            return default_score
        
        # Define regime affinities for different strategy types
        regime_affinities = {
            'trend': {
                'bull': 0.9,
                'bear': 0.6,
                'range': 0.3,
                'volatile': 0.4
            },
            'mean_reversion': {
                'bull': 0.5,
                'bear': 0.6,
                'range': 0.9,
                'volatile': 0.7
            },
            'breakout': {
                'bull': 0.8,
                'bear': 0.7,
                'range': 0.3,
                'volatile': 0.8
            },
            'momentum': {
                'bull': 0.9,
                'bear': 0.4,
                'range': 0.2,
                'volatile': 0.5
            },
            'value': {
                'bull': 0.6,
                'bear': 0.8,
                'range': 0.7,
                'volatile': 0.5
            }
        }
        
        # Get affinity score for this strategy type and regime
        if strategy_type in regime_affinities and regime in regime_affinities[strategy_type]:
            return regime_affinities[strategy_type][regime]
        
        # If historical performance data is available, use that
        strategy_stats = None
        if hasattr(market_state, 'get_strategy_stats'):
            strategy_id = getattr(strategy, 'id', str(strategy))
            strategy_stats = market_state.get_strategy_stats(strategy_id, regime=regime)
        
        if strategy_stats and 'win_rate' in strategy_stats:
            win_rate = strategy_stats['win_rate']
            profit_factor = strategy_stats.get('profit_factor', 1.0)
            
            # Calculate fit score from historical performance
            return min(1.0, (win_rate * 0.7) + (min(profit_factor, 3.0) / 3.0 * 0.3))
        
        return default_score
    
    def _get_regime_parameter_adjustments(self, strategy, regime):
        """
        Get parameter adjustments specific to a market regime
        
        Args:
            strategy: Strategy to adjust
            regime (str): Market regime
            
        Returns:
            dict: Parameter adjustments
        """
        # Get strategy type
        strategy_type = getattr(strategy, 'type', 'generic')
        
        # Define regime-specific parameter adjustments
        regime_params = {
            'trend': {
                'bull': {
                    'lookback_period': 1.2,  # Longer lookback
                    'entry_threshold': 0.9,  # Easier entry
                    'exit_threshold': 1.1,   # Later exit
                },
                'bear': {
                    'lookback_period': 0.8,  # Shorter lookback
                    'entry_threshold': 1.1,  # Harder entry
                    'exit_threshold': 0.9,   # Earlier exit
                },
                'range': {
                    'lookback_period': 0.7,  # Much shorter lookback
                    'entry_threshold': 1.2,  # Much harder entry
                },
                'volatile': {
                    'lookback_period': 0.6,  # Shortest lookback
                    'entry_threshold': 1.3,  # Hardest entry
                    'exit_threshold': 0.8,   # Earliest exit
                }
            },
            'mean_reversion': {
                'bull': {
                    'overbought_threshold': 1.1,  # Less sensitive
                    'oversold_threshold': 1.1,    # Less sensitive
                },
                'bear': {
                    'overbought_threshold': 0.9,  # More sensitive
                    'oversold_threshold': 0.9,    # More sensitive
                },
                'range': {
                    'overbought_threshold': 0.8,  # Most sensitive
                    'oversold_threshold': 0.8,    # Most sensitive
                },
                'volatile': {
                    'overbought_threshold': 1.2,  # Least sensitive
                    'oversold_threshold': 1.2,    # Least sensitive
                }
            }
            # Add other strategy types as needed
        }
        
        # Return adjustments if available
        if (strategy_type in regime_params and 
            regime in regime_params[strategy_type]):
            return regime_params[strategy_type][regime]
        
        # Generic adjustments
        return {
            'sensitivity': 1.0,  # No adjustment
        }
    
    def _record_adaptation(self, trigger, adaptation_data, results):
        """
        Record adaptation to history
        
        Args:
            trigger (AdaptationTrigger): Adaptation trigger
            adaptation_data (dict): Adaptation data
            results (dict): Adaptation results
        """
        # Create record
        adaptation_record = {
            'trigger': trigger.value,
            'timestamp': datetime.now().isoformat(),
            'adaptation_data': adaptation_data,
            'results': results
        }
        
        # Add to history
        self.adaptation_history.append(adaptation_record)
        
        # Limit history size
        max_history = 100
        if len(self.adaptation_history) > max_history:
            self.adaptation_history = self.adaptation_history[-max_history:]
        
        # Save to file
        self._save_adaptation_record(adaptation_record)
    
    def _save_adaptation_record(self, record):
        """
        Save adaptation record to file
        
        Args:
            record (dict): Adaptation record
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trigger = record['trigger']
        filename = f"{self.data_path}/adaptation_{trigger}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(record, f, indent=2)
            
            logger.info(f"Saved adaptation record to {filename}")
        except Exception as e:
            logger.warning(f"Failed to save adaptation record: {str(e)}")
    
    def get_adaptation_history(self, count=None):
        """
        Get adaptation history
        
        Args:
            count (int, optional): Number of records to return
            
        Returns:
            list: Adaptation history
        """
        if count is None:
            return self.adaptation_history
        
        return self.adaptation_history[-count:]
    
    def get_current_regime(self):
        """
        Get the current market regime
        
        Returns:
            str: Current market regime or None
        """
        if self.market_regime_history:
            return self.market_regime_history[-1]['regime']
        
        return None
    
    def manually_trigger_adaptation(self, trigger_type, trigger_data, strategies, market_state):
        """
        Manually trigger an adaptation
        
        Args:
            trigger_type (str): Type of trigger to simulate
            trigger_data (dict): Trigger data
            strategies (list): Strategies to adapt
            market_state (MarketState): Current market state
            
        Returns:
            dict: Adaptation results
        """
        # Convert trigger type to enum
        try:
            trigger = AdaptationTrigger(trigger_type)
        except ValueError:
            return {'success': False, 'message': f'Invalid trigger type: {trigger_type}'}
        
        # Always use MANUAL as the actual trigger
        actual_trigger = AdaptationTrigger.MANUAL
        
        # Combine trigger data
        adaptation_data = {
            'original_trigger': trigger.value,
            'manual_trigger': True,
            'trigger_data': trigger_data
        }
        
        # Apply the adaptation
        return self.apply_adaptation(actual_trigger, adaptation_data, strategies, market_state) 