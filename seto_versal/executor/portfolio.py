#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Portfolio module for SETO-Versal
Manages portfolio positions and tracks performance metrics
"""

import logging
import uuid
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class Portfolio:
    """
    Portfolio management class
    
    Tracks positions, cash balance, and performance metrics
    """
    
    def __init__(self, initial_capital):
        """
        Initialize the portfolio
        
        Args:
            initial_capital (float): Initial cash balance
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # Stock code -> Position dictionary
        self.position_history = []  # Track position changes over time
        self.price_cache = {}  # Stock code -> latest price
        
        # Track portfolio value over time
        self.values = pd.DataFrame(columns=['timestamp', 'cash', 'positions_value', 'total_value'])
        self._record_value()
        
        # Track trades
        self.trades = []
        
        # Performance tracking
        self.high_watermark = initial_capital
        self.max_drawdown = 0.0
        self.total_return = 0.0
        self.annualized_return = 0.0
        self.sharpe_ratio = 0.0
        self.win_count = 0
        self.loss_count = 0
        
        logger.info(f"Portfolio initialized with {initial_capital} capital")
    
    def add_position(self, stock_code, quantity, price, commission=0.0):
        """
        Add a new position or increase an existing position
        
        Args:
            stock_code (str): Stock code
            quantity (int): Quantity to add
            price (float): Price per share
            commission (float): Commission amount
            
        Returns:
            dict: Updated position
        """
        total_cost = quantity * price + commission
        
        # Check if we have enough cash
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash to add position: {stock_code}")
            return None
        
        # Update cash
        self.cash -= total_cost
        
        # Create or update position
        if stock_code in self.positions:
            position = self.positions[stock_code]
            
            # Calculate new average price
            old_value = position['quantity'] * position['avg_price']
            new_value = quantity * price
            total_quantity = position['quantity'] + quantity
            
            position['avg_price'] = (old_value + new_value) / total_quantity
            position['quantity'] += quantity
            position['entries'].append({
                'timestamp': datetime.now(),
                'quantity': quantity,
                'price': price,
                'commission': commission
            })
            
            logger.info(f"Added to position: {stock_code}, {quantity} shares @ {price}")
        else:
            self.positions[stock_code] = {
                'stock_code': stock_code,
                'quantity': quantity,
                'avg_price': price,
                'current_price': price,  # Initialize with purchase price
                'entries': [{
                    'timestamp': datetime.now(),
                    'quantity': quantity,
                    'price': price,
                    'commission': commission
                }],
                'exits': []
            }
            
            logger.info(f"New position: {stock_code}, {quantity} shares @ {price}")
        
        # Update price cache
        self.price_cache[stock_code] = price
        
        # Record position change
        self.position_history.append({
            'timestamp': datetime.now(),
            'action': 'BUY',
            'stock_code': stock_code,
            'quantity': quantity,
            'price': price,
            'commission': commission
        })
        
        # Record portfolio value
        self._record_value()
        
        return self.positions[stock_code]
    
    def remove_position(self, stock_code, quantity, price, commission=0.0, tax=0.0):
        """
        Reduce or close a position
        
        Args:
            stock_code (str): Stock code
            quantity (int): Quantity to remove
            price (float): Price per share
            commission (float): Commission amount
            tax (float): Tax amount
            
        Returns:
            dict: Exit information or None if position doesn't exist
        """
        if stock_code not in self.positions:
            logger.warning(f"Cannot remove position: {stock_code} not in portfolio")
            return None
        
        position = self.positions[stock_code]
        
        # Check if we have enough shares
        if quantity > position['quantity']:
            logger.warning(f"Insufficient shares to remove: {stock_code} ({quantity} > {position['quantity']})")
            return None
        
        # Calculate proceeds and profit/loss
        proceeds = quantity * price - commission - tax
        cost_basis = quantity * position['avg_price']
        profit_loss = proceeds - cost_basis
        profit_pct = profit_loss / cost_basis if cost_basis > 0 else 0.0
        
        # Update cash
        self.cash += proceeds
        
        # Record the trade
        trade = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now(),
            'stock_code': stock_code,
            'action': 'SELL',
            'quantity': quantity,
            'entry_price': position['avg_price'],
            'exit_price': price,
            'cost_basis': cost_basis,
            'proceeds': proceeds,
            'profit_loss': profit_loss,
            'profit_pct': profit_pct,
            'commission': commission,
            'tax': tax
        }
        self.trades.append(trade)
        
        # Update win/loss counts
        if profit_loss > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        # Update position or remove if fully closed
        if quantity < position['quantity']:
            # Reduce position
            position['quantity'] -= quantity
            position['current_price'] = price  # Update current price
            
            position['exits'].append({
                'timestamp': datetime.now(),
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'tax': tax,
                'profit_loss': profit_loss,
                'profit_pct': profit_pct
            })
            
            logger.info(f"Reduced position: {stock_code}, {quantity} shares @ {price}, P/L: {profit_loss:.2f} ({profit_pct:.2%})")
        else:
            # Close position
            position['exits'].append({
                'timestamp': datetime.now(),
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'tax': tax,
                'profit_loss': profit_loss,
                'profit_pct': profit_pct
            })
            
            logger.info(f"Closed position: {stock_code}, {quantity} shares @ {price}, P/L: {profit_loss:.2f} ({profit_pct:.2%})")
            
            # Remove from positions
            del self.positions[stock_code]
        
        # Update price cache
        self.price_cache[stock_code] = price
        
        # Record position change
        self.position_history.append({
            'timestamp': datetime.now(),
            'action': 'SELL',
            'stock_code': stock_code,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'tax': tax,
            'profit_loss': profit_loss,
            'profit_pct': profit_pct
        })
        
        # Record portfolio value
        self._record_value()
        
        return trade
    
    def update_position_price(self, stock_code, price):
        """
        Update the current price of a position
        
        Args:
            stock_code (str): Stock code
            price (float): New price
            
        Returns:
            bool: True if price was updated, False otherwise
        """
        # Update price cache
        self.price_cache[stock_code] = price
        
        # Update position if it exists
        if stock_code in self.positions:
            self.positions[stock_code]['current_price'] = price
            return True
        
        return False
    
    def update_prices(self, prices_dict):
        """
        Update prices for multiple positions
        
        Args:
            prices_dict (dict): Dictionary of stock_code -> price
            
        Returns:
            int: Number of prices updated
        """
        updated = 0
        
        for stock_code, price in prices_dict.items():
            if self.update_position_price(stock_code, price):
                updated += 1
        
        # Record portfolio value after updates
        self._record_value()
        
        return updated
    
    def get_position(self, stock_code):
        """
        Get a specific position
        
        Args:
            stock_code (str): Stock code
            
        Returns:
            dict: Position information or None if not found
        """
        return self.positions.get(stock_code)
    
    def get_position_value(self, stock_code):
        """
        Get the current value of a position
        
        Args:
            stock_code (str): Stock code
            
        Returns:
            float: Position value or 0 if not found
        """
        position = self.get_position(stock_code)
        if position is None:
            return 0.0
        
        return position['quantity'] * position['current_price']
    
    def get_latest_price(self, stock_code):
        """
        Get the latest price for a stock
        
        Args:
            stock_code (str): Stock code
            
        Returns:
            float: Latest price or None if not available
        """
        return self.price_cache.get(stock_code)
    
    def get_all_positions(self):
        """
        Get all current positions
        
        Returns:
            list: List of position dictionaries
        """
        return list(self.positions.values())
    
    def get_open_position_count(self):
        """
        Get the number of open positions
        
        Returns:
            int: Number of open positions
        """
        return len(self.positions)
    
    def get_position_entries_count(self, stock_code):
        """
        Get the number of entries for a position
        
        Args:
            stock_code (str): Stock code
            
        Returns:
            int: Number of entries or 0 if position not found
        """
        position = self.get_position(stock_code)
        if position is None:
            return 0
        
        return len(position['entries'])
    
    def _record_value(self):
        """Record the current portfolio value"""
        positions_value = sum(p['quantity'] * p['current_price'] for p in self.positions.values())
        total_value = self.cash + positions_value
        
        # Record value
        self.values = self.values.append({
            'timestamp': datetime.now(),
            'cash': self.cash,
            'positions_value': positions_value,
            'total_value': total_value
        }, ignore_index=True)
        
        # Update performance metrics
        self._update_performance_metrics(total_value)
    
    def _update_performance_metrics(self, current_value):
        """
        Update performance metrics
        
        Args:
            current_value (float): Current portfolio value
        """
        # Update high watermark and max drawdown
        if current_value > self.high_watermark:
            self.high_watermark = current_value
        
        # Calculate current drawdown
        current_drawdown = 1.0 - current_value / self.high_watermark
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Calculate total return
        self.total_return = (current_value / self.initial_capital) - 1.0
        
        # Calculate annualized return (if we have enough history)
        if len(self.values) > 1:
            first_date = self.values.iloc[0]['timestamp']
            days = (datetime.now() - first_date).days
            if days > 0:
                self.annualized_return = ((1 + self.total_return) ** (365 / days)) - 1
        
        # Calculate Sharpe ratio if we have enough history
        if len(self.values) > 30:  # Need at least 30 data points
            returns = self.values['total_value'].pct_change().dropna()
            if len(returns) > 0:
                avg_return = returns.mean()
                std_return = returns.std()
                if std_return > 0:
                    self.sharpe_ratio = avg_return / std_return * np.sqrt(252)  # Annualized
    
    def get_current_drawdown(self):
        """
        Get the current drawdown
        
        Returns:
            float: Current drawdown as a percentage
        """
        if len(self.values) == 0:
            return 0.0
        
        current_value = self.values.iloc[-1]['total_value']
        return 1.0 - current_value / self.high_watermark
    
    def get_performance_summary(self):
        """
        Get a summary of portfolio performance
        
        Returns:
            dict: Performance summary
        """
        win_rate = self.win_count / (self.win_count + self.loss_count) if (self.win_count + self.loss_count) > 0 else 0.0
        
        return {
            'total_value': self.values.iloc[-1]['total_value'] if len(self.values) > 0 else self.initial_capital,
            'cash': self.cash,
            'positions_value': self.values.iloc[-1]['positions_value'] if len(self.values) > 0 else 0.0,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': win_rate,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'trade_count': self.win_count + self.loss_count,
            'positions_count': len(self.positions)
        }
    
    def get_equity_curve(self):
        """
        Get the equity curve data
        
        Returns:
            pandas.DataFrame: Equity curve data
        """
        return self.values 