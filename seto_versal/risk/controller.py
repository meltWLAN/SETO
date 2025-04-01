#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Risk controller for SETO-Versal trading system.

This module implements risk controls, rule validation, and
adaptive risk level management to protect capital and
enforce trading discipline.
"""

import os
import enum
import yaml
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union


class RiskLevel(enum.Enum):
    """
    Risk level enumeration representing the current
    trading risk profile.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    def __str__(self):
        return self.value


class RiskController:
    """
    Risk management controller that enforces trading rules
    and adapts risk levels based on performance and market conditions.
    
    The controller:
    - Validates trades against position size, sector exposure, and other limits
    - Adapts risk levels based on drawdown, win/loss streaks, and volatility
    - Triggers circuit breakers when needed
    - Tracks rule violations and generates risk reports
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk controller with configuration.
        
        Args:
            config: Configuration dictionary containing risk parameters
                   and file locations
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing risk controller")
        
        # Basic configuration
        self.name = config.get('name', 'risk_controller')
        self.risk_rules_file = config.get('risk_rules_file', 'config/risk_rules.yaml')
        
        # Load default rules
        self.rules = self._load_risk_rules()
        
        # Set current risk level
        initial_level = config.get('initial_risk_level', 'medium')
        self.current_risk_level = getattr(RiskLevel, initial_level.upper())
        
        # Override defaults with config values if provided
        self.max_drawdown_percent = config.get('max_drawdown_percent', 
                                               self.rules.get('max_drawdown_percent', 3.0))
        self.max_position_percent = config.get('max_position_percent', 
                                               self.rules.get('max_position_percent', 5.0))
        self.max_sector_percent = config.get('max_sector_percent', 
                                               self.rules.get('max_sector_percent', 20.0))
        self.min_cash_percent = config.get('min_cash_percent', 
                                           self.rules.get('min_cash_percent', 10.0))
        self.consecutive_loss_limit = config.get('consecutive_loss_limit', 
                                                 self.rules.get('consecutive_loss_limit', 3))
        
        # Initialize state variables
        self.current_consecutive_losses = 0
        self.cooling_until = None
        self.drawdown_history = []
        self.rule_violations = []
        
        # Market conditions cache
        self.market_regime_cache = {}
        
        self.logger.info(f"Risk controller initialized with level: {self.current_risk_level}")
    
    def _load_risk_rules(self) -> Dict[str, Any]:
        """
        Load risk rules from YAML configuration file.
        
        Returns:
            Dictionary of risk rules
        """
        default_rules = {
            'max_drawdown_percent': 3.0,
            'max_position_percent': 5.0,
            'max_sector_percent': 20.0,
            'min_cash_percent': 10.0,
            'consecutive_loss_limit': 3,
            'rules_by_risk_level': {
                'low': {'max_position_percent': 5.0},
                'medium': {'max_position_percent': 3.0},
                'high': {'max_position_percent': 2.0},
                'critical': {'max_position_percent': 1.0},
            }
        }
        
        try:
            if os.path.exists(self.risk_rules_file):
                with open(self.risk_rules_file, 'r') as f:
                    rules = yaml.safe_load(f)
                    self.logger.info(f"Loaded risk rules from {self.risk_rules_file}")
                    return rules
            else:
                self.logger.warning(f"Risk rules file {self.risk_rules_file} not found. Using defaults.")
                return default_rules
        except Exception as e:
            self.logger.error(f"Error loading risk rules: {str(e)}. Using defaults.")
            return default_rules
    
    def validate_trade(self, 
                      trade: Dict[str, Any], 
                      portfolio: Any, 
                      market: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a proposed trade against risk rules.
        
        Args:
            trade: Trade dictionary containing symbol, direction, quantity, etc.
            portfolio: Portfolio object to check current positions and exposure
            market: Market object to check conditions
            
        Returns:
            Tuple of (is_valid, reason) where reason is None if valid
        """
        # Check if in cooling period
        if self.cooling_until and datetime.now() < self.cooling_until:
            cooling_time_left = (self.cooling_until - datetime.now()).total_seconds() / 60
            return False, f"In cooling period for {cooling_time_left:.1f} more minutes"
        
        # Get current risk parameters for risk level
        level_rules = self.rules.get('rules_by_risk_level', {}).get(str(self.current_risk_level), {})
        max_position_pct = level_rules.get('max_position_percent', self.max_position_percent)
        
        # Get position details
        symbol = trade.get('symbol')
        direction = trade.get('direction', 'buy')
        quantity = trade.get('quantity', 0)
        
        # Validate required fields
        if not symbol or not quantity:
            return False, "Missing required trade fields (symbol, quantity)"
        
        # Validate stop loss exists for buy trades
        if direction.lower() == 'buy' and 'stop_loss' not in trade:
            return False, "Buy trade missing required stop loss"
        
        # Get current position if any
        current_position = portfolio.get_position(symbol)
        current_quantity = current_position.get('quantity', 0) if current_position else 0
        
        # Get price (implementation depends on your system)
        price = self._get_price(symbol, market)
        
        # Calculate position size impact
        portfolio_value = portfolio.get_total_value()
        new_position_value = (current_quantity + quantity) * price if direction.lower() == 'buy' else \
                             (current_quantity - quantity) * price
        
        position_percent = (new_position_value / portfolio_value) * 100 if portfolio_value else 0
        
        # Check position size limit
        if position_percent > max_position_pct:
            self._record_rule_violation('max_position_percent', symbol, 
                                       f"Position size {position_percent:.1f}% exceeds {max_position_pct:.1f}% limit")
            return False, f"Position size {position_percent:.1f}% exceeds {max_position_pct:.1f}% limit"
        
        # Check sector exposure if buying
        if direction.lower() == 'buy':
            # Get sector for the symbol (implementation depends on your system)
            sector = self._get_sector(symbol)
            
            # Calculate new sector exposure
            current_sector_exposure = self._get_sector_exposure(sector, portfolio)
            new_sector_exposure = current_sector_exposure + (quantity * price / portfolio_value * 100)
            
            # Check sector exposure limit
            if new_sector_exposure > self.max_sector_percent:
                self._record_rule_violation('max_sector_percent', sector, 
                                          f"Sector {sector} exposure {new_sector_exposure:.1f}% exceeds {self.max_sector_percent:.1f}% limit")
                return False, f"Sector {sector} exposure {new_sector_exposure:.1f}% exceeds {self.max_sector_percent:.1f}% limit"
            
            # Check cash reserves
            cash = portfolio.get_cash()
            trade_cost = quantity * price # Simplified, should include commission
            cash_after_trade = cash - trade_cost
            cash_percent_after = (cash_after_trade / portfolio_value) * 100
            
            if cash_percent_after < self.min_cash_percent:
                self._record_rule_violation('min_cash_percent', None,
                                          f"Cash after trade {cash_percent_after:.1f}% below {self.min_cash_percent:.1f}% minimum")
                return False, f"Cash after trade {cash_percent_after:.1f}% below {self.min_cash_percent:.1f}% minimum"
        
        return True, None
    
    def _get_price(self, symbol: str, market: Any = None) -> float:
        """
        Get current price for a symbol.
        Implementation depends on your market data provider.
        
        Args:
            symbol: Ticker symbol
            market: Optional market object for price lookup
            
        Returns:
            Current price as float
        """
        # This is a placeholder - implement based on your system
        if market:
            return market.get_current_price(symbol)
        return 100.0  # Default placeholder price
    
    def _get_sector(self, symbol: str) -> str:
        """
        Get sector for a symbol.
        Implementation depends on your data provider.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Sector name
        """
        # This is a placeholder - implement based on your system
        # Could use a lookup table, API call, or other data source
        sectors = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'JPM': 'Financials',
            'XOM': 'Energy'
        }
        return sectors.get(symbol, 'Unknown')
    
    def _get_sector_exposure(self, sector: str, portfolio: Any) -> float:
        """
        Calculate current exposure to a specific sector.
        
        Args:
            sector: Sector name
            portfolio: Portfolio object
            
        Returns:
            Sector exposure as percentage of portfolio
        """
        # This is a placeholder - implement based on your system
        # Could be calculated by summing positions for that sector
        # and dividing by portfolio value
        return 5.0  # Default placeholder percentage
    
    def _record_rule_violation(self, rule: str, entity: Optional[str], reason: str):
        """
        Record a rule violation for tracking and reporting.
        
        Args:
            rule: The risk rule that was violated
            entity: The symbol or sector involved 
            reason: Description of the violation
        """
        violation = {
            'timestamp': datetime.now().isoformat(),
            'rule': rule,
            'symbol': entity,
            'reason': reason
        }
        self.rule_violations.append(violation)
        self.logger.warning(f"Risk rule violation: {reason}")
    
    def update_risk_level(self, portfolio: Any, market: Any):
        """
        Update risk level based on portfolio drawdown and market conditions.
        
        Args:
            portfolio: Portfolio object to check drawdown
            market: Market object to check conditions
        """
        # Get current drawdown
        current_drawdown = portfolio.get_current_drawdown()
        
        # Record drawdown history
        self.drawdown_history.append({
            'timestamp': datetime.now().isoformat(),
            'drawdown': current_drawdown
        })
        
        # Keep limited history
        if len(self.drawdown_history) > 100:
            self.drawdown_history = self.drawdown_history[-100:]
        
        # Get market regime
        market_regime = market.get_market_regime()
        self.market_regime_cache[datetime.now().date().isoformat()] = market_regime
        
        # Determine risk level based on drawdown percentage of max allowed
        drawdown_percent_of_max = (current_drawdown / self.max_drawdown_percent) * 100
        
        previous_level = self.current_risk_level
        
        if drawdown_percent_of_max > 80:
            self.current_risk_level = RiskLevel.CRITICAL
        elif drawdown_percent_of_max > 60:
            self.current_risk_level = RiskLevel.HIGH
        elif drawdown_percent_of_max > 30:
            self.current_risk_level = RiskLevel.MEDIUM
        else:
            self.current_risk_level = RiskLevel.LOW
        
        # Apply additional rules based on market regime
        if market_regime == 'bear':
            # In bear market, increase risk level by one step if not already critical
            if self.current_risk_level != RiskLevel.CRITICAL:
                levels = list(RiskLevel)
                current_index = levels.index(self.current_risk_level)
                if current_index < len(levels) - 1:
                    self.current_risk_level = levels[current_index + 1]
        
        if previous_level != self.current_risk_level:
            self.logger.info(f"Risk level changed from {previous_level} to {self.current_risk_level}")
            self.logger.info(f"Drawdown: {current_drawdown:.2f}% ({drawdown_percent_of_max:.1f}% of max)")
    
    def record_trade_result(self, trade_result: Dict[str, Any]):
        """
        Record the result of a trade and update risk state.
        
        Args:
            trade_result: Dictionary containing trade result info
                          (must contain 'pnl' key)
        """
        # Extract PnL from trade result
        pnl = trade_result.get('pnl', 0.0)
        
        if pnl < 0:
            # Losing trade
            self.current_consecutive_losses += 1
            
            # If we've hit the consecutive loss limit, enter cooling period
            if self.current_consecutive_losses >= self.consecutive_loss_limit:
                cooling_hours = 2 * self.current_consecutive_losses  # Scales with loss streak
                self.cooling_until = datetime.now() + timedelta(hours=cooling_hours)
                self.logger.warning(f"Entered {cooling_hours} hour cooling period after "
                                 f"{self.current_consecutive_losses} consecutive losses")
        else:
            # Winning trade, reset consecutive losses
            self.current_consecutive_losses = 0
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """
        Generate a risk report with current status and history.
        
        Returns:
            Dictionary containing risk information
        """
        # Summarize rule violations by type
        violation_counts = {}
        for violation in self.rule_violations[-50:]:  # Last 50 violations
            rule = violation.get('rule')
            violation_counts[rule] = violation_counts.get(rule, 0) + 1
        
        # Recent drawdowns (last 10)
        recent_drawdowns = self.drawdown_history[-10:] if self.drawdown_history else []
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'risk_level': str(self.current_risk_level),
            'consecutive_losses': self.current_consecutive_losses,
            'cooling_until': self.cooling_until.isoformat() if self.cooling_until else None,
            'recent_drawdowns': recent_drawdowns,
            'rule_violations': violation_counts,
            'market_regimes': dict(list(self.market_regime_cache.items())[-5:])  # Last 5 days
        }
        
        return report
    
    def save_state(self, filepath: str = None):
        """
        Save risk controller state to file.
        
        Args:
            filepath: Path to save state file (optional)
        """
        if filepath is None:
            filepath = f"data/risk/{self.name}_state.json"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare state data
        state = {
            'name': self.name,
            'current_risk_level': str(self.current_risk_level),
            'current_consecutive_losses': self.current_consecutive_losses,
            'cooling_until': self.cooling_until.isoformat() if self.cooling_until else None,
            'drawdown_history': self.drawdown_history[-100:],  # Last 100 entries
            'rule_violations': self.rule_violations[-100:],  # Last 100 violations
            'market_regime_cache': self.market_regime_cache,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            self.logger.info(f"Risk controller state saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving risk controller state: {str(e)}")
            return False
    
    def load_state(self, filepath: str = None):
        """
        Load risk controller state from file.
        
        Args:
            filepath: Path to state file (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if filepath is None:
            filepath = f"data/risk/{self.name}_state.json"
        
        if not os.path.exists(filepath):
            self.logger.warning(f"Risk controller state file {filepath} not found")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore state
            self.name = state.get('name', self.name)
            self.current_risk_level = getattr(RiskLevel, state.get('current_risk_level', 'MEDIUM').upper())
            self.current_consecutive_losses = state.get('current_consecutive_losses', 0)
            
            cooling_until = state.get('cooling_until')
            self.cooling_until = datetime.fromisoformat(cooling_until) if cooling_until else None
            
            self.drawdown_history = state.get('drawdown_history', [])
            self.rule_violations = state.get('rule_violations', [])
            self.market_regime_cache = state.get('market_regime_cache', {})
            
            self.logger.info(f"Risk controller state loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading risk controller state: {str(e)}")
            return False
