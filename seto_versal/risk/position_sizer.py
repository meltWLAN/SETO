#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Position sizer module for SETO-Versal
Calculates optimal position sizes based on various methods and risk parameters
"""

import logging
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class SizingMethod(Enum):
    """Enum for position sizing methods"""
    FIXED_PERCENT = "fixed_percent"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY = "kelly"
    OPTIMAL_F = "optimal_f"
    FIXED_RISK = "fixed_risk"

class PositionSizer:
    """
    Position sizer that calculates optimal trade sizes based on account size,
    market conditions, and risk tolerance
    """
    
    def __init__(self, config):
        """
        Initialize the position sizer
        
        Args:
            config (dict): Position sizer configuration
        """
        self.config = config
        
        # Default configuration
        self.default_method = SizingMethod(config.get('default_method', 'fixed_percent'))
        self.default_size = config.get('default_size', 0.05)
        self.max_position_size = config.get('max_position_size', 0.15)
        self.min_position_size = config.get('min_position_size', 0.01)
        
        # Method-specific parameters
        self.fixed_percent = config.get('fixed_percent', 0.05)
        self.risk_per_trade = config.get('risk_per_trade', 0.01)
        self.volatility_factor = config.get('volatility_factor', 0.2)
        self.kelly_fraction = config.get('kelly_fraction', 0.5)  # Half-Kelly for safety
        
        # Account parameters
        self.account_size = config.get('account_size', 1000000)
        
        # Symbol-specific settings
        self.symbol_settings = config.get('symbol_settings', {})
        
        logger.info(f"Position sizer initialized with {self.default_method.value} method")
    
    def calculate_position_size(self, symbol, price, stop_loss=None, target=None, 
                               risk_level=1.0, method=None, market_state=None):
        """
        Calculate position size for a trade
        
        Args:
            symbol (str): Symbol to trade
            price (float): Current price
            stop_loss (float, optional): Stop loss price
            target (float, optional): Target price
            risk_level (float): Risk level multiplier (0.5 = half risk, 2.0 = double risk)
            method (SizingMethod, optional): Sizing method to use, defaults to configured method
            market_state (MarketState, optional): Current market state
            
        Returns:
            dict: Position sizing details including:
                - size_percent: Position size as percentage of account
                - size_shares: Position size in number of shares/contracts
                - size_value: Position size in currency value
                - method: Method used for calculation
        """
        # Get symbol-specific settings if available
        symbol_config = self.symbol_settings.get(symbol, {})
        
        # Determine sizing method
        if method is None:
            method = SizingMethod(symbol_config.get('method', self.default_method.value))
        
        # Apply symbol-specific limits
        max_size = min(
            symbol_config.get('max_position_size', self.max_position_size),
            self.max_position_size
        )
        min_size = max(
            symbol_config.get('min_position_size', self.min_position_size),
            self.min_position_size
        )
        
        # Calculate size based on method
        size_percent = 0.0
        
        if method == SizingMethod.FIXED_PERCENT:
            size_percent = self._fixed_percent_size(symbol, symbol_config)
        
        elif method == SizingMethod.VOLATILITY_ADJUSTED:
            size_percent = self._volatility_adjusted_size(symbol, market_state, symbol_config)
        
        elif method == SizingMethod.FIXED_RISK:
            if stop_loss is None:
                logger.warning(f"Stop loss required for FIXED_RISK sizing, using default method for {symbol}")
                size_percent = self._fixed_percent_size(symbol, symbol_config)
            else:
                size_percent = self._fixed_risk_size(price, stop_loss, symbol_config)
        
        elif method == SizingMethod.KELLY:
            if target is None or stop_loss is None:
                logger.warning(f"Target and stop loss required for KELLY sizing, using default method for {symbol}")
                size_percent = self._fixed_percent_size(symbol, symbol_config)
            else:
                size_percent = self._kelly_size(price, stop_loss, target, market_state, symbol, symbol_config)
        
        elif method == SizingMethod.OPTIMAL_F:
            size_percent = self._optimal_f_size(symbol, market_state, symbol_config)
        
        else:
            logger.warning(f"Unknown sizing method {method}, using default")
            size_percent = self.default_size
        
        # Apply risk level multiplier
        size_percent *= risk_level
        
        # Apply min/max constraints
        size_percent = max(min_size, min(max_size, size_percent))
        
        # Calculate position size in base currency and shares
        position_value = self.account_size * size_percent
        position_shares = position_value / price if price > 0 else 0
        
        # Round to whole shares
        position_shares = int(position_shares)
        position_value = position_shares * price
        
        # Recalculate actual size percentage
        actual_size_percent = position_value / self.account_size if self.account_size > 0 else 0
        
        return {
            'size_percent': actual_size_percent,
            'size_shares': position_shares,
            'size_value': position_value,
            'method': method.value
        }
    
    def update_account_size(self, account_size):
        """
        Update the account size
        
        Args:
            account_size (float): New account size
        """
        self.account_size = account_size
        logger.debug(f"Account size updated to {account_size}")
    
    def _fixed_percent_size(self, symbol, symbol_config):
        """
        Calculate position size based on fixed percentage
        
        Args:
            symbol (str): Symbol to trade
            symbol_config (dict): Symbol-specific configuration
            
        Returns:
            float: Position size as percentage of account
        """
        # Use symbol-specific percentage if available
        return symbol_config.get('fixed_percent', self.fixed_percent)
    
    def _volatility_adjusted_size(self, symbol, market_state, symbol_config):
        """
        Calculate position size based on volatility
        
        Args:
            symbol (str): Symbol to trade
            market_state (MarketState): Current market state
            symbol_config (dict): Symbol-specific configuration
            
        Returns:
            float: Position size as percentage of account
        """
        # Get base size
        base_size = symbol_config.get('fixed_percent', self.fixed_percent)
        
        # If no market state available, return base size
        if market_state is None:
            return base_size
        
        # Get symbol volatility
        volatility = self._get_symbol_volatility(symbol, market_state)
        if volatility is None:
            return base_size
        
        # Get market volatility for reference
        market_volatility = self._get_market_volatility(market_state)
        if market_volatility is None or market_volatility == 0:
            return base_size
        
        # Calculate relative volatility
        relative_volatility = volatility / market_volatility
        
        # Apply volatility adjustment (higher volatility = smaller position)
        volatility_factor = symbol_config.get('volatility_factor', self.volatility_factor)
        size_percent = base_size * (1.0 / max(0.5, relative_volatility * volatility_factor))
        
        return size_percent
    
    def _fixed_risk_size(self, price, stop_loss, symbol_config):
        """
        Calculate position size based on fixed risk amount
        
        Args:
            price (float): Current price
            stop_loss (float): Stop loss price
            symbol_config (dict): Symbol-specific configuration
            
        Returns:
            float: Position size as percentage of account
        """
        # Calculate risk percentage per share
        if price <= 0 or stop_loss <= 0:
            return self.default_size
        
        # Calculate risk per share
        risk_percent = abs(price - stop_loss) / price
        
        if risk_percent <= 0:
            return self.default_size
        
        # Get risk per trade setting
        risk_per_trade = symbol_config.get('risk_per_trade', self.risk_per_trade)
        
        # Calculate position size that risks exactly risk_per_trade of account
        size_percent = risk_per_trade / risk_percent
        
        return size_percent
    
    def _kelly_size(self, price, stop_loss, target, market_state, symbol, symbol_config):
        """
        Calculate position size based on Kelly criterion
        
        Args:
            price (float): Current price
            stop_loss (float): Stop loss price
            target (float): Target price
            market_state (MarketState): Current market state
            symbol (str): Symbol to trade
            symbol_config (dict): Symbol-specific configuration
            
        Returns:
            float: Position size as percentage of account
        """
        # Calculate potential gain and loss
        gain_ratio = (target - price) / price
        loss_ratio = (price - stop_loss) / price
        
        if loss_ratio <= 0:
            return self.default_size
        
        # Estimate win probability
        win_prob = self._estimate_win_probability(symbol, market_state, symbol_config)
        
        # Calculate Kelly fraction
        kelly = (win_prob * (1 + gain_ratio) - (1 - win_prob)) / gain_ratio
        
        # Apply Kelly fraction for safety
        kelly_fraction = symbol_config.get('kelly_fraction', self.kelly_fraction)
        kelly *= kelly_fraction
        
        # Ensure it's not negative
        return max(0, kelly)
    
    def _optimal_f_size(self, symbol, market_state, symbol_config):
        """
        Calculate position size based on optimal f
        
        Args:
            symbol (str): Symbol to trade
            market_state (MarketState): Current market state
            symbol_config (dict): Symbol-specific configuration
            
        Returns:
            float: Position size as percentage of account
        """
        # This is a simplified implementation of optimal f
        # In practice, would need historical trade data
        
        # Use half-Kelly as approximation of optimal f
        win_prob = self._estimate_win_probability(symbol, market_state, symbol_config)
        avg_win_loss_ratio = symbol_config.get('avg_win_loss_ratio', 1.5)
        
        optimal_f = win_prob - ((1 - win_prob) / avg_win_loss_ratio)
        
        # Apply safety factor
        safety_factor = symbol_config.get('optimal_f_safety', 0.5)
        optimal_f *= safety_factor
        
        return max(0, optimal_f)
    
    def _estimate_win_probability(self, symbol, market_state, symbol_config):
        """
        Estimate win probability for a trade
        
        Args:
            symbol (str): Symbol to trade
            market_state (MarketState): Current market state
            symbol_config (dict): Symbol-specific configuration
            
        Returns:
            float: Estimated win probability
        """
        # If symbol has historical win rate, use it
        if 'historical_win_rate' in symbol_config:
            return symbol_config['historical_win_rate']
        
        # If market state is available, try to get from there
        if market_state is not None:
            # Try to get win rate from strategy stats if available
            strategy = market_state.get_strategy_stats(symbol)
            if strategy and 'win_rate' in strategy:
                return strategy['win_rate']
        
        # Default win probability
        return symbol_config.get('default_win_probability', 0.55)
    
    def _get_symbol_volatility(self, symbol, market_state):
        """
        Get volatility for a symbol
        
        Args:
            symbol (str): Symbol to get volatility for
            market_state (MarketState): Current market state
            
        Returns:
            float: Volatility estimate or None
        """
        if market_state is None:
            return None
            
        # Try to get from market state indicators
        if hasattr(market_state, 'indicators') and symbol in market_state.indicators:
            indicators = market_state.indicators[symbol]
            if 'volatility' in indicators:
                return indicators['volatility']
        
        # Calculate from price history
        if hasattr(market_state, 'get_history'):
            history = market_state.get_history(symbol, 20)
            if history and len(history) > 5:
                closes = [bar['close'] for bar in history]
                returns = [np.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
                return np.std(returns) * np.sqrt(252)  # Annualized
        
        return None
    
    def _get_market_volatility(self, market_state):
        """
        Get market volatility
        
        Args:
            market_state (MarketState): Current market state
            
        Returns:
            float: Market volatility or None
        """
        if market_state is None:
            return None
            
        # Try to get from market state indicators
        if hasattr(market_state, 'indicators') and 'market' in market_state.indicators:
            indicators = market_state.indicators['market']
            if 'volatility' in indicators:
                return indicators['volatility']
        
        # Use index volatility as proxy
        index_symbol = getattr(market_state, 'index_symbol', '000001.SH')
        return self._get_symbol_volatility(index_symbol, market_state) 