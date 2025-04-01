#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Portfolio optimizer module for SETO-Versal
Calculates optimal asset allocation based on various optimization methods
"""

import logging
import numpy as np
from enum import Enum
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """Enum for portfolio optimization methods"""
    EQUAL_WEIGHT = "equal_weight"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    TARGET_RISK = "target_risk"
    TARGET_RETURN = "target_return"

class PortfolioOptimizer:
    """
    Portfolio optimizer that calculates optimal asset allocations
    based on risk-return profiles and constraints
    """
    
    def __init__(self, config):
        """
        Initialize the portfolio optimizer
        
        Args:
            config (dict): Optimizer configuration
        """
        self.config = config
        
        # Default configuration
        self.default_method = OptimizationMethod(config.get('default_method', 'equal_weight'))
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)
        self.min_asset_weight = config.get('min_asset_weight', 0.0)
        self.max_asset_weight = config.get('max_asset_weight', 0.3)
        self.target_volatility = config.get('target_volatility', 0.15)
        self.target_return = config.get('target_return', 0.10)
        
        # Portfolio constraints
        self.constraints = config.get('constraints', {})
        self.asset_classes = config.get('asset_classes', {})
        self.class_limits = config.get('class_limits', {})
        
        # Optimization settings
        self.num_simulations = config.get('num_simulations', 10000)
        self.covariance_method = config.get('covariance_method', 'standard')
        self.use_views = config.get('use_views', False)
        
        # Internal storage
        self.last_optimization = None
        self.views = {}
        
        # Data directory
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'data', 
            'portfolios'
        )
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info(f"Portfolio optimizer initialized with {self.default_method.value} method")
    
    def optimize_portfolio(self, assets, returns, covariance=None, method=None, constraints=None):
        """
        Optimize portfolio allocation
        
        Args:
            assets (list): List of asset symbols
            returns (dict): Expected returns for each asset
            covariance (np.ndarray, optional): Covariance matrix of returns
            method (OptimizationMethod, optional): Optimization method to use
            constraints (dict, optional): Additional constraints to apply
            
        Returns:
            dict: Optimized portfolio including:
                - weights: Dict of asset weights
                - expected_return: Expected portfolio return
                - expected_volatility: Expected portfolio volatility
                - sharpe_ratio: Expected Sharpe ratio
                - method: Method used for optimization
        """
        if len(assets) == 0:
            logger.warning("No assets provided for optimization")
            return None
        
        # Make sure we have returns for all assets
        for asset in assets:
            if asset not in returns:
                logger.warning(f"Missing return data for {asset}, excluding from optimization")
                assets.remove(asset)
        
        if len(assets) == 0:
            logger.warning("No valid assets with return data")
            return None
        
        # Convert returns to numpy array
        returns_array = np.array([returns[asset] for asset in assets])
        
        # Determine optimization method
        if method is None:
            method = self.default_method
        
        # Generate covariance matrix if not provided
        if covariance is None:
            logger.info("Covariance matrix not provided, using identity matrix")
            covariance = np.identity(len(assets))
        
        # Combine constraints
        all_constraints = self._build_constraints(assets, constraints)
        
        # Perform optimization based on method
        if method == OptimizationMethod.EQUAL_WEIGHT:
            weights = self._equal_weight(assets)
        
        elif method == OptimizationMethod.MINIMUM_VARIANCE:
            weights = self._minimum_variance(assets, covariance, all_constraints)
        
        elif method == OptimizationMethod.MAXIMUM_SHARPE:
            weights = self._maximum_sharpe(assets, returns_array, covariance, all_constraints)
        
        elif method == OptimizationMethod.RISK_PARITY:
            weights = self._risk_parity(assets, covariance, all_constraints)
        
        elif method == OptimizationMethod.BLACK_LITTERMAN:
            weights = self._black_litterman(assets, returns_array, covariance, all_constraints)
        
        elif method == OptimizationMethod.TARGET_RISK:
            weights = self._target_risk(assets, returns_array, covariance, all_constraints)
        
        elif method == OptimizationMethod.TARGET_RETURN:
            weights = self._target_return(assets, returns_array, covariance, all_constraints)
        
        else:
            logger.warning(f"Unknown optimization method {method}, using equal weight")
            weights = self._equal_weight(assets)
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(returns_array * np.array(list(weights.values())))
        portfolio_variance = self._calculate_portfolio_variance(weights, covariance, assets)
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Create result
        result = {
            'weights': weights,
            'expected_return': float(portfolio_return),
            'expected_volatility': float(portfolio_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'method': method.value,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store last optimization
        self.last_optimization = result
        
        return result
    
    def set_views(self, views):
        """
        Set investor views for Black-Litterman model
        
        Args:
            views (dict): Asset views (asset -> expected return)
        """
        self.views = views
        logger.info(f"Set investor views for {len(views)} assets")
    
    def check_rebalance_needed(self, current_weights, optimal_weights):
        """
        Check if portfolio rebalancing is needed
        
        Args:
            current_weights (dict): Current asset weights
            optimal_weights (dict): Optimal asset weights
            
        Returns:
            bool: True if rebalance is needed
        """
        # Check if any weight deviation exceeds threshold
        for asset, weight in optimal_weights.items():
            current = current_weights.get(asset, 0.0)
            if abs(current - weight) > self.rebalance_threshold:
                return True
        
        return False
    
    def calculate_rebalance_trades(self, current_holdings, optimal_weights, total_value):
        """
        Calculate trades needed to rebalance portfolio
        
        Args:
            current_holdings (dict): Current asset holdings {asset: shares}
            optimal_weights (dict): Optimal asset weights
            total_value (float): Total portfolio value
            
        Returns:
            dict: Trades needed {asset: shares_delta}
        """
        trades = {}
        
        # Calculate current values and weights
        current_values = {}
        current_weights = {}
        current_prices = {}
        
        # Assume we have a market_state or similar to get prices
        for asset, shares in current_holdings.items():
            price = self._get_asset_price(asset)
            if price is None:
                logger.warning(f"Missing price for {asset}")
                continue
                
            current_prices[asset] = price
            value = shares * price
            current_values[asset] = value
            current_weights[asset] = value / total_value if total_value > 0 else 0
        
        # Calculate target values
        target_values = {asset: total_value * weight for asset, weight in optimal_weights.items()}
        
        # Calculate deltas in values
        value_deltas = {}
        for asset in set(list(current_values.keys()) + list(target_values.keys())):
            current = current_values.get(asset, 0.0)
            target = target_values.get(asset, 0.0)
            value_deltas[asset] = target - current
        
        # Convert value deltas to share deltas
        for asset, value_delta in value_deltas.items():
            if abs(value_delta) < 1.0:  # Ignore very small changes
                continue
                
            price = current_prices.get(asset)
            if price is None or price <= 0:
                logger.warning(f"Missing or invalid price for {asset}")
                continue
                
            # Calculate share delta and round to whole shares
            share_delta = value_delta / price
            share_delta = int(share_delta)
            
            if share_delta != 0:
                trades[asset] = share_delta
        
        return trades
    
    def save_optimization(self, optimization, filename=None):
        """
        Save optimization result to file
        
        Args:
            optimization (dict): Optimization result
            filename (str, optional): Filename to save to
            
        Returns:
            str: Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"portfolio_optimization_{timestamp}.json"
        
        file_path = os.path.join(self.data_dir, filename)
        
        with open(file_path, 'w') as f:
            json.dump(optimization, f, indent=2)
        
        logger.info(f"Saved optimization to {file_path}")
        return file_path
    
    def load_optimization(self, filename):
        """
        Load optimization result from file
        
        Args:
            filename (str): Filename to load from
            
        Returns:
            dict: Loaded optimization result or None
        """
        file_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"Optimization file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                optimization = json.load(f)
            
            logger.info(f"Loaded optimization from {file_path}")
            return optimization
        except Exception as e:
            logger.error(f"Error loading optimization: {e}")
            return None
    
    def _build_constraints(self, assets, additional_constraints=None):
        """
        Build constraint set for optimization
        
        Args:
            assets (list): List of asset symbols
            additional_constraints (dict, optional): Additional constraints
            
        Returns:
            dict: Complete set of constraints
        """
        # Start with base constraints
        constraints = {
            'min_weights': {asset: self.min_asset_weight for asset in assets},
            'max_weights': {asset: self.max_asset_weight for asset in assets}
        }
        
        # Add class limits
        class_constraints = {}
        for asset_class, limit in self.class_limits.items():
            class_assets = [a for a in assets if self.asset_classes.get(a) == asset_class]
            if class_assets:
                class_constraints[asset_class] = {
                    'assets': class_assets,
                    'min': limit.get('min', 0.0),
                    'max': limit.get('max', 1.0)
                }
        
        if class_constraints:
            constraints['class_limits'] = class_constraints
        
        # Add configured constraints
        for key, value in self.constraints.items():
            if key not in constraints:
                constraints[key] = value
        
        # Add additional constraints
        if additional_constraints:
            for key, value in additional_constraints.items():
                if key == 'min_weights' or key == 'max_weights':
                    # Merge with existing min/max constraints
                    constraints[key].update(value)
                else:
                    constraints[key] = value
        
        return constraints
    
    def _equal_weight(self, assets):
        """
        Equal weight portfolio allocation
        
        Args:
            assets (list): List of asset symbols
            
        Returns:
            dict: Asset weights
        """
        weight = 1.0 / len(assets)
        return {asset: weight for asset in assets}
    
    def _minimum_variance(self, assets, covariance, constraints):
        """
        Minimum variance portfolio allocation
        
        Args:
            assets (list): List of asset symbols
            covariance (np.ndarray): Covariance matrix
            constraints (dict): Optimization constraints
            
        Returns:
            dict: Asset weights
        """
        # Simplified implementation (in reality would use quadratic programming)
        # This is a placeholder for the actual optimization algorithm
        
        # In a real implementation, would solve:
        # min w^T Σ w
        # subject to Σw_i = 1, w_i >= min_weight, w_i <= max_weight
        
        # For demonstration, generate random weights that sum to 1
        # and are within the min/max constraints
        min_weights = constraints.get('min_weights', {})
        max_weights = constraints.get('max_weights', {})
        
        # Start with minimum weights
        weights = {asset: min_weights.get(asset, 0.0) for asset in assets}
        
        # Allocate remaining weight proportional to inverse variance
        remaining = 1.0 - sum(weights.values())
        if remaining > 0:
            # Get diagonal elements (variances)
            variances = np.diag(covariance)
            
            # Handle zero variances
            variances = np.maximum(variances, 1e-8)
            
            # Calculate inverse variance
            inv_var = 1.0 / variances
            
            # Normalize to get weights
            inv_var_weights = inv_var / np.sum(inv_var)
            
            # Distribute remaining weight
            for i, asset in enumerate(assets):
                # Calculate additional weight
                additional = remaining * inv_var_weights[i]
                
                # Apply max constraint
                max_additional = max_weights.get(asset, 1.0) - weights[asset]
                additional = min(additional, max_additional)
                
                weights[asset] += additional
        
        # Normalize to ensure sum is 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {asset: weight / total for asset, weight in weights.items()}
        
        return weights
    
    def _maximum_sharpe(self, assets, returns, covariance, constraints):
        """
        Maximum Sharpe ratio portfolio allocation
        
        Args:
            assets (list): List of asset symbols
            returns (np.ndarray): Expected returns
            covariance (np.ndarray): Covariance matrix
            constraints (dict): Optimization constraints
            
        Returns:
            dict: Asset weights
        """
        # Simplified implementation
        # This is a placeholder for the actual optimization algorithm
        
        # In a real implementation, would use optimization to maximize:
        # (r_p - r_f) / σ_p
        
        # For demonstration, use a Monte Carlo approach
        min_weights = constraints.get('min_weights', {})
        max_weights = constraints.get('max_weights', {})
        
        n_assets = len(assets)
        best_sharpe = -np.inf
        best_weights = self._equal_weight(assets)
        
        # Monte Carlo simulation
        for _ in range(self.num_simulations):
            # Generate random weights
            weights_array = np.random.random(n_assets)
            weights_array = weights_array / np.sum(weights_array)
            
            # Apply min/max constraints
            for i, asset in enumerate(assets):
                min_w = min_weights.get(asset, 0.0)
                max_w = max_weights.get(asset, 1.0)
                weights_array[i] = max(min_w, min(max_w, weights_array[i]))
            
            # Normalize again after constraints
            weights_array = weights_array / np.sum(weights_array)
            
            # Calculate portfolio return and risk
            port_return = np.sum(returns * weights_array)
            port_variance = np.dot(weights_array.T, np.dot(covariance, weights_array))
            port_volatility = np.sqrt(port_variance)
            
            # Calculate Sharpe ratio
            sharpe = (port_return - self.risk_free_rate) / port_volatility if port_volatility > 0 else -np.inf
            
            # Update if better Sharpe ratio found
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = {asset: float(weights_array[i]) for i, asset in enumerate(assets)}
        
        return best_weights
    
    def _risk_parity(self, assets, covariance, constraints):
        """
        Risk parity portfolio allocation
        
        Args:
            assets (list): List of asset symbols
            covariance (np.ndarray): Covariance matrix
            constraints (dict): Optimization constraints
            
        Returns:
            dict: Asset weights
        """
        # Simplified implementation
        # In a real implementation, would solve for weights such that
        # the risk contribution of each asset is equal
        
        # Inverse volatility as a simple approximation of risk parity
        n_assets = len(assets)
        vols = np.sqrt(np.diag(covariance))
        
        # Handle zero volatilities
        vols = np.maximum(vols, 1e-8)
        
        # Inverse volatility weights
        inv_vols = 1.0 / vols
        weights_array = inv_vols / np.sum(inv_vols)
        
        # Apply constraints
        min_weights = constraints.get('min_weights', {})
        max_weights = constraints.get('max_weights', {})
        
        for i, asset in enumerate(assets):
            min_w = min_weights.get(asset, 0.0)
            max_w = max_weights.get(asset, 1.0)
            weights_array[i] = max(min_w, min(max_w, weights_array[i]))
        
        # Normalize again after constraints
        weights_array = weights_array / np.sum(weights_array)
        
        return {asset: float(weights_array[i]) for i, asset in enumerate(assets)}
    
    def _black_litterman(self, assets, returns, covariance, constraints):
        """
        Black-Litterman portfolio allocation
        
        Args:
            assets (list): List of asset symbols
            returns (np.ndarray): Expected returns
            covariance (np.ndarray): Covariance matrix
            constraints (dict): Optimization constraints
            
        Returns:
            dict: Asset weights
        """
        # Simplified Black-Litterman implementation
        # In a real implementation, would adjust market implied returns
        # based on investor views
        
        # If no views provided, use maximum Sharpe
        if not self.use_views or not self.views:
            return self._maximum_sharpe(assets, returns, covariance, constraints)
        
        # Get views for the assets we have
        asset_views = {asset: self.views.get(asset, returns[i]) 
                      for i, asset in enumerate(assets) if asset in self.views}
        
        if not asset_views:
            return self._maximum_sharpe(assets, returns, covariance, constraints)
        
        # Adjust expected returns based on views
        # This is a simplified implementation
        adjusted_returns = returns.copy()
        for i, asset in enumerate(assets):
            if asset in asset_views:
                # Blend market-implied return with view
                confidence = 0.5  # Could be configured
                adjusted_returns[i] = (1 - confidence) * returns[i] + confidence * asset_views[asset]
        
        # Use adjusted returns with maximum Sharpe
        return self._maximum_sharpe(assets, adjusted_returns, covariance, constraints)
    
    def _target_risk(self, assets, returns, covariance, constraints):
        """
        Target risk portfolio allocation
        
        Args:
            assets (list): List of asset symbols
            returns (np.ndarray): Expected returns
            covariance (np.ndarray): Covariance matrix
            constraints (dict): Optimization constraints
            
        Returns:
            dict: Asset weights
        """
        # Simplified implementation
        # In a real implementation, would find weights that produce
        # portfolio with target volatility and maximum return
        
        # Generate efficient frontier
        n_points = 100
        target_vol = self.target_volatility
        
        # Start with minimum variance portfolio
        min_var_weights = self._minimum_variance(assets, covariance, constraints)
        min_var_vol = np.sqrt(self._calculate_portfolio_variance(min_var_weights, covariance, assets))
        
        # If target vol is below min var, use min var
        if target_vol <= min_var_vol:
            return min_var_weights
        
        # Get maximum Sharpe portfolio
        max_sharpe_weights = self._maximum_sharpe(assets, returns, covariance, constraints)
        max_sharpe_vol = np.sqrt(self._calculate_portfolio_variance(max_sharpe_weights, covariance, assets))
        
        # If target vol is above max Sharpe vol, blend with maximum return portfolio
        if target_vol > max_sharpe_vol:
            # Find maximum return portfolio
            max_return_weights = self._maximum_return(assets, returns, constraints)
            max_return_vol = np.sqrt(self._calculate_portfolio_variance(max_return_weights, covariance, assets))
            
            # If target vol is above max return vol, use max return
            if target_vol >= max_return_vol:
                return max_return_weights
            
            # Blend max Sharpe and max return
            blend_ratio = (target_vol - max_sharpe_vol) / (max_return_vol - max_sharpe_vol)
            
            weights = {}
            for asset in assets:
                w1 = max_sharpe_weights.get(asset, 0.0)
                w2 = max_return_weights.get(asset, 0.0)
                weights[asset] = (1 - blend_ratio) * w1 + blend_ratio * w2
            
            return weights
        
        # Target vol is between min var and max Sharpe
        # Blend min var and max Sharpe
        blend_ratio = (target_vol - min_var_vol) / (max_sharpe_vol - min_var_vol)
        
        weights = {}
        for asset in assets:
            w1 = min_var_weights.get(asset, 0.0)
            w2 = max_sharpe_weights.get(asset, 0.0)
            weights[asset] = (1 - blend_ratio) * w1 + blend_ratio * w2
        
        return weights
    
    def _target_return(self, assets, returns, covariance, constraints):
        """
        Target return portfolio allocation
        
        Args:
            assets (list): List of asset symbols
            returns (np.ndarray): Expected returns
            covariance (np.ndarray): Covariance matrix
            constraints (dict): Optimization constraints
            
        Returns:
            dict: Asset weights
        """
        # Simplified implementation
        # In a real implementation, would find weights that produce
        # portfolio with target return and minimum risk
        
        target_ret = self.target_return
        
        # Calculate min and max possible returns
        min_var_weights = self._minimum_variance(assets, covariance, constraints)
        min_var_ret = np.sum(returns * np.array([min_var_weights.get(asset, 0.0) for asset in assets]))
        
        max_return_weights = self._maximum_return(assets, returns, constraints)
        max_return = np.sum(returns * np.array([max_return_weights.get(asset, 0.0) for asset in assets]))
        
        # If target return is below min var return, use min var
        if target_ret <= min_var_ret:
            return min_var_weights
        
        # If target return is above max return, use max return
        if target_ret >= max_return:
            return max_return_weights
        
        # Get maximum Sharpe portfolio
        max_sharpe_weights = self._maximum_sharpe(assets, returns, covariance, constraints)
        max_sharpe_ret = np.sum(returns * np.array([max_sharpe_weights.get(asset, 0.0) for asset in assets]))
        
        # If target return is close to max Sharpe return, use max Sharpe
        if abs(target_ret - max_sharpe_ret) < 0.001:
            return max_sharpe_weights
        
        # If target return is between min var and max Sharpe, blend them
        if target_ret < max_sharpe_ret:
            blend_ratio = (target_ret - min_var_ret) / (max_sharpe_ret - min_var_ret)
            
            weights = {}
            for asset in assets:
                w1 = min_var_weights.get(asset, 0.0)
                w2 = max_sharpe_weights.get(asset, 0.0)
                weights[asset] = (1 - blend_ratio) * w1 + blend_ratio * w2
            
            return weights
        
        # If target return is between max Sharpe and max return, blend them
        blend_ratio = (target_ret - max_sharpe_ret) / (max_return - max_sharpe_ret)
        
        weights = {}
        for asset in assets:
            w1 = max_sharpe_weights.get(asset, 0.0)
            w2 = max_return_weights.get(asset, 0.0)
            weights[asset] = (1 - blend_ratio) * w1 + blend_ratio * w2
        
        return weights
    
    def _maximum_return(self, assets, returns, constraints):
        """
        Maximum return portfolio allocation
        
        Args:
            assets (list): List of asset symbols
            returns (np.ndarray): Expected returns
            constraints (dict): Optimization constraints
            
        Returns:
            dict: Asset weights
        """
        # Find asset with highest return
        max_return_asset = assets[np.argmax(returns)]
        
        # Get max weight for this asset
        max_weights = constraints.get('max_weights', {})
        max_weight = max_weights.get(max_return_asset, 1.0)
        
        # Allocate maximum to highest return asset
        weights = {asset: 0.0 for asset in assets}
        weights[max_return_asset] = max_weight
        
        # Distribute remaining to next highest returns
        remaining = 1.0 - max_weight
        sorted_indices = np.argsort(-returns)  # Sort in descending order
        
        for i in sorted_indices:
            if remaining <= 0:
                break
                
            asset = assets[i]
            if asset == max_return_asset:
                continue
                
            max_w = max_weights.get(asset, 1.0)
            w = min(max_w, remaining)
            weights[asset] = w
            remaining -= w
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {asset: weight / total for asset, weight in weights.items()}
        
        return weights
    
    def _calculate_portfolio_variance(self, weights, covariance, assets):
        """
        Calculate portfolio variance
        
        Args:
            weights (dict): Asset weights
            covariance (np.ndarray): Covariance matrix
            assets (list): List of asset symbols
            
        Returns:
            float: Portfolio variance
        """
        weights_array = np.array([weights.get(asset, 0.0) for asset in assets])
        return np.dot(weights_array.T, np.dot(covariance, weights_array))
    
    def _get_asset_price(self, asset):
        """
        Get current price for an asset
        
        Args:
            asset (str): Asset symbol
            
        Returns:
            float: Current price or None
        """
        # This is a placeholder - in a real implementation,
        # would get the price from market data
        # For now, just return a random price
        return 100.0 