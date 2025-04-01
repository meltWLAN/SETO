#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Risk control module for SETO-Versal
Implements the philosophy-driven risk management system
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Philosophy-driven risk management system
    
    Enforces risk constraints based on predefined principles:
    - Maximum drawdown limits
    - Position size constraints
    - Consecutive loss management
    - Stop loss and profit taking rules
    - Exposure limits by sector/market
    """
    
    def __init__(self, config):
        """
        Initialize the risk manager
        
        Args:
            config (dict): Risk control configuration
        """
        self.config = config
        
        # Extract risk control parameters
        risk_config = config.get('risk_control', {})
        self.max_drawdown = risk_config.get('max_drawdown', 0.03)
        self.max_position_size = risk_config.get('max_position_size', 0.25)
        self.consecutive_loss_limit = risk_config.get('consecutive_loss_limit', 3)
        self.fixed_stop_loss = risk_config.get('fixed_stop_loss', 0.05)
        self.trailing_stop = risk_config.get('trailing_stop', True)
        self.trailing_percentage = risk_config.get('trailing_percentage', 0.03)
        
        # Extract position management parameters
        pos_config = config.get('position_management', {})
        self.entry_scaling = pos_config.get('entry_scaling', [1.0])  # Default to full position
        self.exit_scaling = pos_config.get('exit_scaling', [1.0])  # Default to full exit
        
        # Extract discipline parameters
        discipline_config = config.get('discipline', {})
        self.forbidden_strategies = discipline_config.get('forbidden_strategies', [])
        self.cooling_period = discipline_config.get('cooling_period', 3)
        self.max_open_trades = discipline_config.get('max_open_trades', 15)
        
        # Internal state
        self.consecutive_losses = 0
        self.in_cooling_mode = False
        self.cooling_until = None
        self.loss_history = []
        self.risk_violations = []
        
        logger.info("Risk Manager initialized with philosophical principles")
    
    def filter_decisions(self, decisions, portfolio):
        """
        Filter trading decisions based on risk management rules
        
        Args:
            decisions (list): List of TradeDecision objects
            portfolio (Portfolio): Current portfolio state
            
        Returns:
            list: Filtered list of TradeDecision objects
        """
        if not decisions:
            return []
        
        logger.info(f"Applying risk management filters to {len(decisions)} decisions")
        
        # Check if we're in cooling mode
        if self.in_cooling_mode:
            if datetime.now() > self.cooling_until:
                logger.info("Cooling period ended, resuming normal operations")
                self.in_cooling_mode = False
                self.cooling_until = None
            else:
                logger.info(f"In cooling mode until {self.cooling_until}, limiting decisions")
                # During cooling, only allow selling decisions or very high confidence buys
                decisions = [d for d in decisions if d.direction == 'SELL' or d.confidence > 0.85]
        
        # Apply various filters
        decisions = self._apply_drawdown_filter(decisions, portfolio)
        decisions = self._apply_position_size_filter(decisions, portfolio)
        decisions = self._apply_exposure_filters(decisions, portfolio)
        decisions = self._apply_stop_loss_rules(decisions, portfolio)
        decisions = self._apply_forbidden_strategies_filter(decisions)
        
        logger.info(f"After risk management, {len(decisions)} decisions remain")
        return decisions
    
    def _apply_drawdown_filter(self, decisions, portfolio):
        """
        Filter decisions based on maximum drawdown limits
        
        Args:
            decisions (list): List of TradeDecision objects
            portfolio (Portfolio): Current portfolio state
            
        Returns:
            list: Filtered list of TradeDecision objects
        """
        # Calculate current drawdown
        current_drawdown = portfolio.get_current_drawdown()
        
        # If drawdown exceeds threshold, only allow sell decisions
        if current_drawdown > self.max_drawdown:
            logger.warning(f"Drawdown ({current_drawdown:.2%}) exceeds maximum ({self.max_drawdown:.2%})")
            self.risk_violations.append({
                'timestamp': datetime.now(),
                'type': 'drawdown_exceeded',
                'details': {'current': current_drawdown, 'max': self.max_drawdown}
            })
            
            # Only allow sell decisions when drawdown exceeds threshold
            return [d for d in decisions if d.direction == 'SELL']
        
        return decisions
    
    def _apply_position_size_filter(self, decisions, portfolio):
        """
        Apply position size limits to decisions
        
        Args:
            decisions (list): List of TradeDecision objects
            portfolio (Portfolio): Current portfolio state
            
        Returns:
            list: Filtered list of TradeDecision objects with adjusted quantities
        """
        total_value = portfolio.total_value
        
        for decision in decisions:
            if decision.direction == 'BUY':
                # Calculate maximum position value
                max_position_value = total_value * self.max_position_size
                
                # Get current position value if any
                current_position_value = portfolio.get_position_value(decision.stock_code)
                
                # Calculate available position value
                available_position_value = max_position_value - current_position_value
                
                if available_position_value <= 0:
                    # Position already at maximum size, reject decision
                    logger.info(f"Position size for {decision.stock_code} already at maximum, rejecting")
                    decisions.remove(decision)
                    continue
                
                # Calculate scaled entry if needed
                if len(self.entry_scaling) > 1:
                    # Use first entry scale factor for initial position
                    scale_factor = self.entry_scaling[0]
                    if current_position_value > 0:
                        # If position exists, use next scale factor if available
                        position_count = portfolio.get_position_entries_count(decision.stock_code)
                        if position_count < len(self.entry_scaling):
                            scale_factor = self.entry_scaling[position_count]
                        else:
                            # All scale entries used, reject decision
                            logger.info(f"All scale entries used for {decision.stock_code}, rejecting")
                            decisions.remove(decision)
                            continue
                    
                    # Apply scale factor to available position value
                    available_position_value *= scale_factor
                
                # Update decision with position size constraint
                if decision.quantity is None:
                    # This will be properly calculated later with current price
                    decision.quantity = -1  # Placeholder for "calculate later"
                else:
                    # Adjust quantity if it exceeds max position size
                    stock_price = portfolio.get_latest_price(decision.stock_code)
                    if stock_price:
                        max_shares = int(available_position_value / stock_price)
                        decision.quantity = min(decision.quantity, max_shares)
        
        return decisions
    
    def _apply_exposure_filters(self, decisions, portfolio):
        """
        Apply exposure limits by sector and total positions
        
        Args:
            decisions (list): List of TradeDecision objects
            portfolio (Portfolio): Current portfolio state
            
        Returns:
            list: Filtered list of TradeDecision objects
        """
        # Check if we've reached maximum open positions
        current_positions = portfolio.get_open_position_count()
        
        if current_positions >= self.max_open_trades:
            logger.info(f"Maximum open trades ({self.max_open_trades}) reached, rejecting buy decisions")
            # Only allow sell decisions
            return [d for d in decisions if d.direction == 'SELL']
        
        # Otherwise, limit new buys to available slots
        available_slots = self.max_open_trades - current_positions
        
        if available_slots > 0:
            buy_decisions = [d for d in decisions if d.direction == 'BUY']
            sell_decisions = [d for d in decisions if d.direction == 'SELL']
            
            # Sort buy decisions by confidence and take top N
            buy_decisions = sorted(buy_decisions, key=lambda d: d.confidence, reverse=True)
            filtered_buy_decisions = buy_decisions[:available_slots]
            
            return filtered_buy_decisions + sell_decisions
        
        return decisions
    
    def _apply_stop_loss_rules(self, decisions, portfolio):
        """
        Apply stop loss and profit taking rules
        
        Args:
            decisions (list): List of TradeDecision objects
            portfolio (Portfolio): Current portfolio state
            
        Returns:
            list: Filtered list of TradeDecision objects with adjusted stop loss
        """
        for decision in decisions:
            if decision.direction == 'BUY':
                # Apply fixed stop loss if not provided
                if decision.stop_loss is None:
                    stock_price = portfolio.get_latest_price(decision.stock_code)
                    if stock_price:
                        decision.stop_loss = stock_price * (1 - self.fixed_stop_loss)
                        logger.debug(f"Applied fixed stop loss for {decision.stock_code}: {decision.stop_loss:.2f}")
        
        return decisions
    
    def _apply_forbidden_strategies_filter(self, decisions):
        """
        Filter decisions based on forbidden strategies
        
        Args:
            decisions (list): List of TradeDecision objects
            
        Returns:
            list: Filtered list of TradeDecision objects
        """
        # Currently a placeholder - in a real implementation, this would analyze the decision
        # to detect martingale, revenge trading, etc.
        return decisions
    
    def update_loss_count(self, trade_result):
        """
        Update consecutive loss counter based on trade result
        
        Args:
            trade_result (dict): Trade result with profit/loss info
            
        Returns:
            bool: True if cooling mode entered, False otherwise
        """
        self.loss_history.append({
            'timestamp': datetime.now(),
            'profit_pct': trade_result.get('profit_pct', 0),
            'stock_code': trade_result.get('stock_code', 'unknown')
        })
        
        # Keep history manageable
        if len(self.loss_history) > 100:
            self.loss_history = self.loss_history[-100:]
        
        # Check if this was a loss
        if trade_result.get('profit_pct', 0) < 0:
            self.consecutive_losses += 1
            logger.info(f"Loss recorded ({self.consecutive_losses} consecutive losses)")
            
            # Check if we need to enter cooling mode
            if self.consecutive_losses >= self.consecutive_loss_limit:
                self._enter_cooling_mode()
                return True
        else:
            # Reset loss counter on a win
            self.consecutive_losses = 0
            logger.info("Win recorded, resetting consecutive loss counter")
        
        return False
    
    def _enter_cooling_mode(self):
        """Enter cooling mode to reduce trading activity after consecutive losses"""
        self.in_cooling_mode = True
        self.cooling_until = datetime.now() + timedelta(days=self.cooling_period)
        
        logger.warning(f"Entering cooling mode until {self.cooling_until}")
        
        self.risk_violations.append({
            'timestamp': datetime.now(),
            'type': 'consecutive_losses',
            'details': {'count': self.consecutive_losses, 'cooling_days': self.cooling_period}
        })
    
    def get_risk_stats(self):
        """
        Get current risk management statistics
        
        Returns:
            dict: Risk statistics
        """
        return {
            'consecutive_losses': self.consecutive_losses,
            'in_cooling_mode': self.in_cooling_mode,
            'cooling_until': self.cooling_until,
            'risk_violations': len(self.risk_violations),
            'recent_violations': self.risk_violations[-5:] if self.risk_violations else []
        } 