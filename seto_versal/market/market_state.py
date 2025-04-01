#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Market state module for SETO-Versal
Provides market environment analysis and regime detection
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
import os
import json
import time

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"          # Strong uptrend
    BEAR = "bear"          # Strong downtrend
    RANGE = "range"        # Sideways/consolidation
    VOLATILE = "volatile"  # High volatility, direction unclear
    RECOVERY = "recovery"  # Early stage of recovery from bear market
    TOPPING = "topping"    # Late stage of bull market, signs of weakness
    UNKNOWN = "unknown"    # Not enough data to classify

class MarketState:
    """
    Tracks and analyzes the current state of the market
    Provides multiple timescale analysis and regime detection
    """
    
    def __init__(self, config: Dict[str, Any], data_source=None):
        """
        Initialize market state tracker
        
        Args:
            config (dict): Configuration for market state
            data_source (DataSource, optional): Data source to use
        """
        self.config = config
        self.name = config.get('name', 'market_state')
        self.data_source = data_source
        
        # Market indices to track
        self.primary_index = config.get('primary_index', '000001.SH')  # Default to Shanghai Composite
        self.secondary_indices = config.get('secondary_indices', ['399001.SZ', '399006.SZ'])  # Shenzhen Composite, GEM
        
        # Timeframes for analysis
        self.timeframes = config.get('timeframes', {
            'short_term': 20,    # Trading days (approximately 1 month)
            'medium_term': 60,   # Trading days (approximately 3 months)
            'long_term': 252     # Trading days (approximately 1 year)
        })
        
        # Thresholds for regime classification
        self.thresholds = config.get('thresholds', {
            'bull': {
                'trend_strength': 0.6,
                'percent_above_ma': 0.03,
                'recent_return': 0.05
            },
            'bear': {
                'trend_strength': -0.6,
                'percent_below_ma': -0.03,
                'recent_return': -0.05
            },
            'volatile': {
                'volatility_percentile': 80  # Percentile threshold for volatility
            },
            'recovery': {
                'percent_off_bottom': 0.1,
                'positive_days': 0.6
            },
            'topping': {
                'divergence_threshold': -0.2,
                'breadth_decline': -0.1
            }
        })
        
        # Current state tracking
        self.current_regime = MarketRegime.UNKNOWN
        self.last_update = None
        self.market_data = {}
        self.computed_metrics = {}
        self.regime_history = []
        self.volatility_history = []
        self.breadth_indicators = {}
        
        # Sector/industry tracking
        self.sectors = {}
        self.hot_sectors = []
        self.sector_rotation = []
        
        # Cache directory for metrics
        self.cache_dir = config.get('cache_dir', 'seto_versal/data/cache/market_state')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Market state '{self.name}' initialized, tracking {self.primary_index} as primary index")
    
    def update(self, force: bool = False) -> bool:
        """
        Update market state with latest data
        
        Args:
            force (bool): Force update even if recently updated
            
        Returns:
            bool: True if update was performed, False otherwise
        """
        # Check if we need to update
        current_time = datetime.now()
        if not force and self.last_update is not None:
            update_interval = self.config.get('update_interval', 3600)  # Default: 1 hour
            time_diff = (current_time - self.last_update).total_seconds()
            if time_diff < update_interval:
                logger.debug(f"Skipping update, last update was {time_diff:.1f}s ago")
                return False
        
        # Check if data source is available
        if self.data_source is None:
            logger.error("No data source available for market state update")
            return False
        
        try:
            # Calculate date ranges for different timeframes
            end_date = current_time.strftime("%Y-%m-%d")
            max_days = max(self.timeframes.values()) * 2  # Get more data than needed for calculations
            start_date = (current_time - timedelta(days=max_days)).strftime("%Y-%m-%d")
            
            # Get primary index data
            primary_data = self.data_source.get_market_index(
                self.primary_index, start_date, end_date, 'daily'
            )
            self.market_data['primary'] = primary_data
            
            # Get secondary indices data
            self.market_data['secondary'] = {}
            for index in self.secondary_indices:
                secondary_data = self.data_source.get_market_index(
                    index, start_date, end_date, 'daily'
                )
                self.market_data['secondary'][index] = secondary_data
            
            # Calculate metrics
            self._calculate_metrics()
            
            # Detect market regime
            self._detect_regime()
            
            # Update sector analysis
            self._update_sector_analysis()
            
            # Save state to history
            self._update_history()
            
            # Update timestamp
            self.last_update = current_time
            
            logger.info(f"Market state updated: {self.current_regime.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating market state: {str(e)}")
            return False
    
    def _calculate_metrics(self) -> None:
        """
        Calculate various market metrics from price data
        """
        # Get primary index data
        df = self.market_data['primary']
        if df is None or len(df) == 0:
            logger.warning("No primary index data available for calculating metrics")
            return
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        # Calculate metrics for each timeframe
        metrics = {}
        
        for timeframe, days in self.timeframes.items():
            period_data = df.iloc[-days:] if len(df) >= days else df
            
            # Return metrics
            returns = period_data['close'].pct_change().dropna()
            cumulative_return = (period_data['close'].iloc[-1] / period_data['close'].iloc[0]) - 1
            
            # Volatility metrics
            daily_volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Trend metrics
            period_data['ma20'] = period_data['close'].rolling(window=20).mean()
            period_data['ma50'] = period_data['close'].rolling(window=50).mean()
            period_data['ma200'] = period_data['close'].rolling(window=200).mean()
            
            # Calculate trend strength (percent of days trending in same direction)
            direction = np.sign(returns)
            trend_strength = direction.mean()  # Average direction (-1 to 1)
            
            # Calculate percentage above/below moving averages
            last_close = period_data['close'].iloc[-1]
            pct_above_ma20 = (last_close / period_data['ma20'].iloc[-1]) - 1 if not np.isnan(period_data['ma20'].iloc[-1]) else np.nan
            pct_above_ma50 = (last_close / period_data['ma50'].iloc[-1]) - 1 if not np.isnan(period_data['ma50'].iloc[-1]) else np.nan
            pct_above_ma200 = (last_close / period_data['ma200'].iloc[-1]) - 1 if not np.isnan(period_data['ma200'].iloc[-1]) else np.nan
            
            # Volume analysis
            if 'volume' in period_data.columns:
                avg_volume = period_data['volume'].mean()
                recent_volume = period_data['volume'].iloc[-5:].mean()  # Last 5 days
                volume_change = (recent_volume / avg_volume) - 1 if avg_volume > 0 else 0
            else:
                avg_volume = np.nan
                recent_volume = np.nan
                volume_change = np.nan
            
            # Store metrics for this timeframe
            metrics[timeframe] = {
                'cumulative_return': cumulative_return,
                'daily_volatility': daily_volatility,
                'trend_strength': trend_strength,
                'percent_positive_days': (returns > 0).mean(),
                'percent_above_ma20': pct_above_ma20,
                'percent_above_ma50': pct_above_ma50,
                'percent_above_ma200': pct_above_ma200,
                'volume_change': volume_change,
                'data_points': len(period_data),
                'last_close': last_close,
                'high': period_data['high'].max(),
                'low': period_data['low'].min(),
                'max_drawdown': self._calculate_drawdown(period_data['close'])
            }
        
        # Get volatility rank (percentile)
        historical_volatility = df['close'].pct_change().rolling(window=20).std().dropna() * np.sqrt(252)
        current_volatility = historical_volatility.iloc[-1] if len(historical_volatility) > 0 else np.nan
        
        if not np.isnan(current_volatility) and len(historical_volatility) > 20:
            volatility_percentile = percentileofscore(historical_volatility, current_volatility)
        else:
            volatility_percentile = np.nan
        
        # Breadth indicators (mock calculations - would be replaced with actual breadth data)
        advance_decline = np.random.normal(0, 0.5)  # Mock advance-decline line slope
        percent_above_ma50 = np.random.uniform(0.3, 0.7)  # Mock percent of stocks above 50-day MA
        
        # Store overall metrics
        self.computed_metrics = {
            'timeframes': metrics,
            'volatility_percentile': volatility_percentile,
            'current_volatility': current_volatility,
            'breadth': {
                'advance_decline': advance_decline,
                'percent_above_ma50': percent_above_ma50
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _detect_regime(self) -> None:
        """
        Detect current market regime based on calculated metrics
        """
        # Check if we have metrics
        if not self.computed_metrics or 'timeframes' not in self.computed_metrics:
            logger.warning("No metrics available for regime detection")
            self.current_regime = MarketRegime.UNKNOWN
            return
        
        # Get metrics for different timeframes
        short_term = self.computed_metrics['timeframes'].get('short_term', {})
        medium_term = self.computed_metrics['timeframes'].get('medium_term', {})
        long_term = self.computed_metrics['timeframes'].get('long_term', {})
        
        # Initialize regime scores
        regime_scores = {regime: 0 for regime in MarketRegime}
        
        # Check for bull market
        bull_threshold = self.thresholds['bull']
        if short_term.get('trend_strength', 0) > bull_threshold['trend_strength']:
            regime_scores[MarketRegime.BULL] += 1
        if medium_term.get('trend_strength', 0) > bull_threshold['trend_strength']:
            regime_scores[MarketRegime.BULL] += 1
        if short_term.get('percent_above_ma50', 0) > bull_threshold['percent_above_ma']:
            regime_scores[MarketRegime.BULL] += 1
        if medium_term.get('cumulative_return', 0) > bull_threshold['recent_return']:
            regime_scores[MarketRegime.BULL] += 1
        if long_term.get('cumulative_return', 0) > bull_threshold['recent_return']:
            regime_scores[MarketRegime.BULL] += 1
        
        # Check for bear market
        bear_threshold = self.thresholds['bear']
        if short_term.get('trend_strength', 0) < bear_threshold['trend_strength']:
            regime_scores[MarketRegime.BEAR] += 1
        if medium_term.get('trend_strength', 0) < bear_threshold['trend_strength']:
            regime_scores[MarketRegime.BEAR] += 1
        if short_term.get('percent_above_ma50', 0) < bear_threshold['percent_below_ma']:
            regime_scores[MarketRegime.BEAR] += 1
        if medium_term.get('cumulative_return', 0) < bear_threshold['recent_return']:
            regime_scores[MarketRegime.BEAR] += 1
        if long_term.get('cumulative_return', 0) < bear_threshold['recent_return']:
            regime_scores[MarketRegime.BEAR] += 1
        
        # Check for volatile market
        volatile_threshold = self.thresholds['volatile']
        volatility_percentile = self.computed_metrics.get('volatility_percentile', 0)
        if volatility_percentile > volatile_threshold['volatility_percentile']:
            regime_scores[MarketRegime.VOLATILE] += 2  # Give more weight to volatility
        
        # Check for sideways/range-bound market
        # Range is characterized by low directional strength but not high volatility
        if (abs(short_term.get('trend_strength', 0)) < 0.3 and 
            abs(medium_term.get('trend_strength', 0)) < 0.3 and
            volatility_percentile < 60):
            regime_scores[MarketRegime.RANGE] += 3
        
        # Check for recovery phase (coming out of bear market)
        recovery_threshold = self.thresholds['recovery']
        if (medium_term.get('cumulative_return', 0) < 0 and 
            short_term.get('cumulative_return', 0) > recovery_threshold['percent_off_bottom'] and
            short_term.get('percent_positive_days', 0) > recovery_threshold['positive_days']):
            regime_scores[MarketRegime.RECOVERY] += 2
        
        # Check for topping phase (late stage bull market)
        topping_threshold = self.thresholds['topping']
        breadth_decline = self.computed_metrics.get('breadth', {}).get('percent_above_ma50', 0)
        if (long_term.get('cumulative_return', 0) > 0.15 and 
            short_term.get('trend_strength', 0) < medium_term.get('trend_strength', 0) + topping_threshold['divergence_threshold'] and
            breadth_decline < topping_threshold['breadth_decline']):
            regime_scores[MarketRegime.TOPPING] += 2
        
        # Determine current regime (highest score wins)
        max_score = max(regime_scores.values())
        if max_score == 0:
            self.current_regime = MarketRegime.UNKNOWN
        else:
            # Get all regimes with the max score
            max_regimes = [r for r, s in regime_scores.items() if s == max_score]
            
            if len(max_regimes) == 1:
                self.current_regime = max_regimes[0]
            else:
                # If tie, use precedence rules
                precedence = [
                    MarketRegime.VOLATILE,  # Volatility takes precedence
                    MarketRegime.BEAR,      # Bear markets are important to detect
                    MarketRegime.RECOVERY,  # Recovery is an important transition
                    MarketRegime.TOPPING,   # Topping is an important warning
                    MarketRegime.BULL,      # Bull markets are common
                    MarketRegime.RANGE,     # Range is a default when no strong signals
                    MarketRegime.UNKNOWN    # Last resort
                ]
                
                for regime in precedence:
                    if regime in max_regimes:
                        self.current_regime = regime
                        break
        
        logger.debug(f"Regime detection scores: {regime_scores}")
        logger.info(f"Detected market regime: {self.current_regime.value}")
    
    def _update_sector_analysis(self) -> None:
        """
        Analyze sector performance and rotation
        Note: This is a placeholder - would be replaced with actual sector data
        """
        # Mock sector data for now
        self.sectors = {
            'technology': {'return': 0.05, 'momentum': 0.8, 'volatility': 0.2},
            'healthcare': {'return': 0.02, 'momentum': 0.3, 'volatility': 0.15},
            'finance': {'return': -0.01, 'momentum': -0.2, 'volatility': 0.18},
            'consumer': {'return': 0.03, 'momentum': 0.5, 'volatility': 0.12},
            'energy': {'return': -0.02, 'momentum': -0.4, 'volatility': 0.25}
        }
        
        # Determine hot sectors (top performing)
        sector_momentum = [(s, d['momentum']) for s, d in self.sectors.items()]
        sector_momentum.sort(key=lambda x: x[1], reverse=True)
        self.hot_sectors = [s for s, m in sector_momentum if m > 0.2]
        
        # Track sector rotation
        if not self.sector_rotation:
            self.sector_rotation = [{'timestamp': datetime.now().isoformat(), 'top_sectors': self.hot_sectors}]
        else:
            last_rotation = self.sector_rotation[-1]
            if set(last_rotation['top_sectors']) != set(self.hot_sectors):
                self.sector_rotation.append({
                    'timestamp': datetime.now().isoformat(),
                    'top_sectors': self.hot_sectors,
                    'previous': last_rotation['top_sectors']
                })
    
    def _update_history(self) -> None:
        """
        Update regime and volatility history
        """
        # Add current regime to history if it changed
        if not self.regime_history or self.regime_history[-1]['regime'] != self.current_regime.value:
            self.regime_history.append({
                'timestamp': datetime.now().isoformat(),
                'regime': self.current_regime.value
            })
            
            # Limit history size
            max_history = 100
            if len(self.regime_history) > max_history:
                self.regime_history = self.regime_history[-max_history:]
        
        # Add volatility to history
        current_volatility = self.computed_metrics.get('current_volatility')
        if current_volatility is not None and not np.isnan(current_volatility):
            self.volatility_history.append({
                'timestamp': datetime.now().isoformat(),
                'value': current_volatility
            })
            
            # Limit history size
            max_history = 100
            if len(self.volatility_history) > max_history:
                self.volatility_history = self.volatility_history[-max_history:]
    
    def _calculate_drawdown(self, prices: pd.Series) -> float:
        """
        Calculate maximum drawdown for a price series
        
        Args:
            prices (pd.Series): Series of prices
            
        Returns:
            float: Maximum drawdown as a percentage
        """
        if prices.empty:
            return 0.0
            
        # Calculate the running maximum
        running_max = prices.cummax()
        
        # Calculate the drawdown
        drawdown = (prices / running_max) - 1
        
        # Get the maximum drawdown
        max_drawdown = drawdown.min()
        
        return max_drawdown if not np.isnan(max_drawdown) else 0.0
    
    def get_market_regime(self) -> str:
        """
        Get current market regime
        
        Returns:
            str: Current market regime
        """
        return self.current_regime.value
    
    def get_market_volatility(self) -> float:
        """
        Get current market volatility
        
        Returns:
            float: Current volatility level
        """
        return self.computed_metrics.get('current_volatility')
    
    def get_market_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of current market state
        
        Returns:
            dict: Summary of market state
        """
        # Check if we need to update first
        if self.last_update is None or (datetime.now() - self.last_update).total_seconds() > 3600:
            self.update()
        
        summary = {
            'regime': self.current_regime.value,
            'volatility': self.get_market_volatility(),
            'hot_sectors': self.hot_sectors,
            'short_term': self.computed_metrics.get('timeframes', {}).get('short_term', {}),
            'medium_term': self.computed_metrics.get('timeframes', {}).get('medium_term', {}),
            'long_term': self.computed_metrics.get('timeframes', {}).get('long_term', {}),
            'breadth': self.computed_metrics.get('breadth', {}),
            'timestamp': datetime.now().isoformat(),
            'primary_index': self.primary_index
        }
        
        return summary
    
    def get_strategy_stats(self, strategy_id: str, regime: str = None) -> Dict[str, Any]:
        """
        Get strategy performance statistics for specific market regime
        
        Args:
            strategy_id (str): Strategy identifier
            regime (str, optional): Market regime to get stats for (default: current regime)
            
        Returns:
            dict: Strategy statistics
        """
        # This would be replaced with actual strategy performance data by regime
        # Mock data for now
        if regime is None:
            regime = self.current_regime.value
            
        # Mock performance data by regime type
        regime_performance = {
            'bull': {'win_rate': 0.65, 'profit_factor': 2.1, 'avg_return': 0.02},
            'bear': {'win_rate': 0.35, 'profit_factor': 0.8, 'avg_return': -0.01},
            'range': {'win_rate': 0.55, 'profit_factor': 1.5, 'avg_return': 0.01},
            'volatile': {'win_rate': 0.48, 'profit_factor': 1.2, 'avg_return': 0.005},
            'recovery': {'win_rate': 0.60, 'profit_factor': 1.8, 'avg_return': 0.015},
            'topping': {'win_rate': 0.40, 'profit_factor': 0.9, 'avg_return': -0.005},
            'unknown': {'win_rate': 0.50, 'profit_factor': 1.0, 'avg_return': 0.0}
        }
        
        # Strategy type influences performance in different regimes
        strategy_type_modifier = {
            'trend': {'bull': 1.2, 'bear': 0.8, 'range': 0.7, 'volatile': 0.9},
            'mean_reversion': {'bull': 0.8, 'bear': 0.9, 'range': 1.3, 'volatile': 1.1},
            'breakout': {'bull': 1.1, 'bear': 0.9, 'range': 0.7, 'volatile': 1.2},
            'value': {'bull': 0.9, 'bear': 1.1, 'range': 1.0, 'volatile': 0.9}
        }
        
        # Derive strategy type from ID (mock)
        strategy_type = None
        if 'trend' in strategy_id:
            strategy_type = 'trend'
        elif 'reversion' in strategy_id:
            strategy_type = 'mean_reversion'
        elif 'breakout' in strategy_id:
            strategy_type = 'breakout'
        elif 'value' in strategy_id:
            strategy_type = 'value'
        
        # Get base stats for the regime
        stats = regime_performance.get(regime, regime_performance['unknown']).copy()
        
        # Apply strategy type modifier if applicable
        if strategy_type and strategy_type in strategy_type_modifier:
            modifiers = strategy_type_modifier[strategy_type]
            if regime in modifiers:
                modifier = modifiers[regime]
                stats['win_rate'] *= modifier
                stats['profit_factor'] *= modifier
                stats['avg_return'] *= modifier
        
        # Add additional stats
        stats['regime'] = regime
        stats['strategy_id'] = strategy_id
        stats['strategy_type'] = strategy_type
        stats['trades_count'] = np.random.randint(20, 100)
        
        return stats
    
    def save_state(self, filename: str = None) -> bool:
        """
        Save current market state to file
        
        Args:
            filename (str, optional): Filename to save to
            
        Returns:
            bool: True if successful, False otherwise
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.cache_dir}/market_state_{timestamp}.json"
        
        try:
            # Prepare data for serialization
            # Convert DataFrame to dict for JSON serialization
            market_data_serial = {}
            for key, value in self.market_data.items():
                if isinstance(value, pd.DataFrame):
                    market_data_serial[key] = {
                        'data': value.reset_index().to_dict(orient='records')
                    }
                elif isinstance(value, dict):
                    market_data_serial[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, pd.DataFrame):
                            market_data_serial[key][sub_key] = {
                                'data': sub_value.reset_index().to_dict(orient='records')
                            }
                        else:
                            market_data_serial[key][sub_key] = sub_value
                else:
                    market_data_serial[key] = value
            
            state = {
                'current_regime': self.current_regime.value,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'computed_metrics': self.computed_metrics,
                'regime_history': self.regime_history,
                'volatility_history': self.volatility_history,
                'sectors': self.sectors,
                'hot_sectors': self.hot_sectors,
                'sector_rotation': self.sector_rotation,
                'config': {
                    'primary_index': self.primary_index,
                    'secondary_indices': self.secondary_indices,
                    'timeframes': self.timeframes,
                    'thresholds': self.thresholds
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Market state saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving market state: {str(e)}")
            return False
    
    def load_state(self, filename: str) -> bool:
        """
        Load market state from file
        
        Args:
            filename (str): Filename to load from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            # Restore state
            self.current_regime = MarketRegime(state.get('current_regime', 'unknown'))
            self.last_update = datetime.fromisoformat(state['last_update']) if state.get('last_update') else None
            self.computed_metrics = state.get('computed_metrics', {})
            self.regime_history = state.get('regime_history', [])
            self.volatility_history = state.get('volatility_history', [])
            self.sectors = state.get('sectors', {})
            self.hot_sectors = state.get('hot_sectors', [])
            self.sector_rotation = state.get('sector_rotation', [])
            
            # Restore configuration if present
            config = state.get('config', {})
            if config:
                self.primary_index = config.get('primary_index', self.primary_index)
                self.secondary_indices = config.get('secondary_indices', self.secondary_indices)
                self.timeframes = config.get('timeframes', self.timeframes)
                self.thresholds = config.get('thresholds', self.thresholds)
            
            logger.info(f"Market state loaded from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading market state: {str(e)}")
            return False

# Function to fix missing 'percentileofscore' function which might not be imported
def percentileofscore(a, score):
    """
    Calculate the percentile rank of a score relative to an array
    
    Args:
        a (array-like): Array of values
        score (float): Score to find percentile for
        
    Returns:
        float: Percentile (0-100)
    """
    a = np.asarray(a)
    n = len(a)
    
    if n == 0:
        return np.nan
    
    # Count values below score
    count = np.sum(a < score)
    
    # Calculate percentile
    percentile = (count / n) * 100
    
    return percentile 