#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trade executor module for SETO-Versal
Responsible for converting trading decisions to actual orders
"""

import logging
import json
import os
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Enum for order statuses"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class TradeExecutor:
    """
    Trade executor that converts trading decisions to orders
    
    Responsible for:
    - Validating trading decisions against risk controls
    - Converting decisions to orders for the broker
    - Tracking order execution status
    - Handling trade settlement and position updates
    """
    
    def __init__(self, config, risk_controller=None):
        """
        Initialize the trade executor
        
        Args:
            config (dict): Executor configuration
            risk_controller (object, optional): Risk controller instance
        """
        self.config = config
        self.risk_controller = risk_controller
        self.name = config.get('name', 'main_executor')
        
        # Mode: live, paper, backtest
        self.mode = config.get('mode', 'paper')
        
        # Broker settings
        self.broker_name = config.get('broker', 'simulator')
        self.broker_config = config.get('broker_config', {})
        
        # Execution settings
        self.default_order_type = config.get('default_order_type', 'market')
        self.slippage_model = config.get('slippage_model', 'fixed')
        self.slippage_value = config.get('slippage_value', 0.001)  # 0.1% by default
        
        # T+1 settlement settings for Chinese market
        self.t_plus_one = config.get('t_plus_one', True)
        
        # Order and position tracking
        self.open_orders = {}  # order_id -> order_details
        self.filled_orders = []
        self.positions = {}  # symbol -> position_details
        self.available_funds = config.get('initial_capital', 1000000)
        self.frozen_funds = 0
        
        # Assets available for T+1 selling (for Chinese A-shares)
        self.t1_available_assets = {}  # symbol -> quantity
        
        # Data directory
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'data', 
            'executor'
        )
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize broker connection
        self._initialize_broker()
        
        logger.info(f"Trade executor '{self.name}' initialized in {self.mode} mode")
    
    def _initialize_broker(self):
        """
        Initialize connection with broker
        """
        # In a real implementation, this would connect to an actual broker
        # For now, we'll use a simulated broker
        
        if self.mode == 'live':
            logger.warning("Live trading mode active - executing real orders")
            # Here we would initialize connection to a real broker
        else:
            logger.info(f"Using simulated broker in {self.mode} mode")
        
        # Load existing positions if any
        self._load_positions()
    
    def execute_decisions(self, decisions, market_state):
        """
        Execute trading decisions
        
        Args:
            decisions (list): Trading decisions from director
            market_state (MarketState): Current market state
            
        Returns:
            list: Execution results
        """
        execution_results = []
        
        # Update order statuses before processing new decisions
        self._update_order_status()
        
        # Update T+1 available assets
        if self.t_plus_one:
            self._update_t1_available_assets()
        
        # Process each decision
        for decision in decisions:
            # Skip invalid decisions
            if not self._validate_decision(decision):
                continue
            
            # Apply risk controls if available
            if self.risk_controller:
                if not self.risk_controller.check_decision(decision, self.positions, market_state):
                    logger.warning(f"Decision rejected by risk control: {decision['symbol']} {decision['direction']}")
                    continue
            
            # Execute the decision
            result = self._execute_decision(decision, market_state)
            execution_results.append(result)
        
        # Save updated positions
        self._save_positions()
        
        return execution_results
    
    def _validate_decision(self, decision):
        """
        Validate if a decision has all required fields
        
        Args:
            decision (dict): Trading decision
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_fields = ['symbol', 'direction', 'size']
        
        for field in required_fields:
            if field not in decision:
                logger.warning(f"Invalid decision: missing {field}")
                return False
        
        # Validate direction
        if decision['direction'] not in ['buy', 'sell']:
            logger.warning(f"Invalid direction: {decision['direction']}")
            return False
        
        # Validate size
        if decision['size'] <= 0:
            logger.warning(f"Invalid size: {decision['size']}")
            return False
        
        return True
    
    def _execute_decision(self, decision, market_state):
        """
        Execute a trading decision
        
        Args:
            decision (dict): Trading decision
            market_state (MarketState): Current market state
            
        Returns:
            dict: Execution result
        """
        symbol = decision['symbol']
        direction = decision['direction']
        size = decision['size']
        
        # Get current market data
        current_data = market_state.get_ohlcv(symbol)
        if not current_data:
            logger.warning(f"No market data for {symbol}, cannot execute")
            return self._create_execution_result(decision, False, "No market data")
        
        # Check if the market is open
        if not market_state.is_market_open():
            logger.warning(f"Market is closed, cannot execute {symbol} {direction}")
            return self._create_execution_result(decision, False, "Market closed")
        
        # Check T+1 restrictions for sell orders
        if self.t_plus_one and direction == 'sell':
            if not self._check_t1_availability(symbol, size):
                logger.warning(f"T+1 restriction: cannot sell {symbol}, not enough available shares")
                return self._create_execution_result(decision, False, "T+1 restriction")
        
        # Calculate order quantity based on position size (fraction of capital)
        current_price = current_data['close']
        capital = self.available_funds + sum(
            pos['quantity'] * market_state.get_ohlcv(sym)['close']
            for sym, pos in self.positions.items()
            if market_state.get_ohlcv(sym)
        )
        
        # Calculate quantity to trade (rounded to whole shares)
        quantity = int((capital * size) / current_price)
        
        # Minimum quantity check
        if quantity <= 0:
            logger.warning(f"Calculated quantity too small for {symbol} {direction}")
            return self._create_execution_result(decision, False, "Quantity too small")
        
        # Check available funds for buy orders
        if direction == 'buy':
            required_funds = quantity * current_price * (1 + 0.003)  # Add fees
            if required_funds > self.available_funds:
                logger.warning(f"Not enough funds to buy {symbol}, need {required_funds}, have {self.available_funds}")
                # Try with maximum possible quantity
                quantity = int(self.available_funds / (current_price * 1.003))
                if quantity <= 0:
                    return self._create_execution_result(decision, False, "Insufficient funds")
        
        # Generate order
        order = {
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'order_type': self.default_order_type,
            'limit_price': None,
            'stop_price': None,
            'order_id': f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.open_orders)}",
            'status': OrderStatus.PENDING.value,
            'decision_id': decision.get('id'),
            'timestamp': datetime.now(),
            'filled_quantity': 0,
            'filled_price': None,
            'filled_timestamp': None
        }
        
        # Add stop loss if provided
        if 'stop_loss' in decision and decision['stop_loss']:
            order['stop_price'] = decision['stop_loss']
        
        # Add limit price if using limit orders
        if self.default_order_type == 'limit':
            # Set limit price slightly better than current price
            if direction == 'buy':
                order['limit_price'] = current_price * (1 + self.slippage_value)
            else:
                order['limit_price'] = current_price * (1 - self.slippage_value)
        
        # Place the order
        success, message = self._place_order(order, market_state)
        
        # Update order status
        if success:
            order['status'] = OrderStatus.SUBMITTED.value
            self.open_orders[order['order_id']] = order
            
            # Update available funds for buy orders
            if direction == 'buy':
                self.frozen_funds += quantity * current_price * (1 + 0.003)
                self.available_funds -= quantity * current_price * (1 + 0.003)
        
        # Create and return execution result
        return self._create_execution_result(
            decision, 
            success, 
            message, 
            order_id=order['order_id'] if success else None,
            quantity=quantity
        )
    
    def _place_order(self, order, market_state):
        """
        Place an order with the broker
        
        Args:
            order (dict): Order details
            market_state (MarketState): Current market state
            
        Returns:
            tuple: (success, message)
        """
        # In a real implementation, this would send the order to a broker
        # For simulation, we'll assume immediate execution
        
        if self.mode == 'paper' or self.mode == 'backtest':
            # Simulate order placement
            # For paper trading, we'll assume the order is accepted
            
            # In a real implementation, this would communicate with an actual broker
            # For now, we'll just log the order and simulate acceptance
            logger.info(f"Simulated order placed: {order['direction']} {order['quantity']} {order['symbol']}")
            
            # If in backtest mode, immediately fill the order
            if self.mode == 'backtest':
                self._fill_simulated_order(order, market_state)
            
            return True, "Order placed successfully"
            
        elif self.mode == 'live':
            # In live mode, place the order with the real broker
            # This is a placeholder - would need real broker API integration
            logger.info(f"LIVE ORDER placed: {order['direction']} {order['quantity']} {order['symbol']}")
            return True, "Order placed with broker"
        
        return False, "Unknown trading mode"
    
    def _fill_simulated_order(self, order, market_state):
        """
        Simulate order filling for backtest mode
        
        Args:
            order (dict): Order details
            market_state (MarketState): Current market state
        """
        # Get current price
        current_data = market_state.get_ohlcv(order['symbol'])
        if not current_data:
            return
        
        current_price = current_data['close']
        
        # Apply slippage
        if order['direction'] == 'buy':
            fill_price = current_price * (1 + self.slippage_value)
        else:
            fill_price = current_price * (1 - self.slippage_value)
        
        # Update order details
        order['status'] = OrderStatus.FILLED.value
        order['filled_quantity'] = order['quantity']
        order['filled_price'] = fill_price
        order['filled_timestamp'] = datetime.now()
        
        # Update positions
        self._update_positions_after_fill(order)
        
        # Move from open to filled orders
        if order['order_id'] in self.open_orders:
            del self.open_orders[order['order_id']]
            self.filled_orders.append(order)
    
    def _update_order_status(self):
        """
        Update status of open orders
        """
        # In a real implementation, this would query the broker for updates
        # For paper trading, we'll simulate fills
        
        if self.mode != 'paper':
            return
        
        # Loop through open orders
        for order_id, order in list(self.open_orders.items()):
            # Skip orders that are already completed
            if order['status'] in [
                OrderStatus.FILLED.value,
                OrderStatus.CANCELLED.value,
                OrderStatus.REJECTED.value,
                OrderStatus.EXPIRED.value
            ]:
                continue
            
            # For paper trading, assume orders fill after some time
            order_age = datetime.now() - order['timestamp']
            
            # Assume market orders fill immediately, limit orders may take time
            if order['order_type'] == 'market' or order_age > timedelta(minutes=5):
                # Simulate a fill
                order['status'] = OrderStatus.FILLED.value
                order['filled_quantity'] = order['quantity']
                order['filled_price'] = order.get('limit_price') or order.get('price_at_submission')
                order['filled_timestamp'] = datetime.now()
                
                # Update positions
                self._update_positions_after_fill(order)
                
                # Move from open to filled orders
                del self.open_orders[order_id]
                self.filled_orders.append(order)
                
                logger.info(f"Order filled: {order['direction']} {order['quantity']} {order['symbol']} at {order['filled_price']}")
    
    def _update_positions_after_fill(self, order):
        """
        Update positions after an order is filled
        
        Args:
            order (dict): Filled order details
        """
        symbol = order['symbol']
        direction = order['direction']
        quantity = order['filled_quantity']
        price = order['filled_price']
        
        # Update positions
        if direction == 'buy':
            if symbol in self.positions:
                # Update existing position
                pos = self.positions[symbol]
                new_quantity = pos['quantity'] + quantity
                new_cost = (pos['avg_cost'] * pos['quantity'] + price * quantity) / new_quantity
                
                pos['quantity'] = new_quantity
                pos['avg_cost'] = new_cost
                pos['last_updated'] = datetime.now()
            else:
                # Create new position
                self.positions[symbol] = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'avg_cost': price,
                    'open_date': datetime.now(),
                    'last_updated': datetime.now()
                }
                
            # Update funds
            total_cost = quantity * price * (1 + 0.003)  # Include fees
            self.frozen_funds -= total_cost
            
            # Update T+1 available (will be available next day)
            if self.t_plus_one:
                self.t1_available_assets[symbol] = self.t1_available_assets.get(symbol, 0)
            
        elif direction == 'sell':
            if symbol in self.positions:
                # Update existing position
                pos = self.positions[symbol]
                
                if quantity >= pos['quantity']:
                    # Selling entire position
                    del self.positions[symbol]
                else:
                    # Partial sell
                    pos['quantity'] -= quantity
                    pos['last_updated'] = datetime.now()
                
                # Update funds
                proceeds = quantity * price * (1 - 0.003)  # Account for fees
                self.available_funds += proceeds
                
                # Update T+1 available
                if self.t_plus_one:
                    self.t1_available_assets[symbol] = max(0, self.t1_available_assets.get(symbol, 0) - quantity)
            else:
                logger.warning(f"Attempted to sell {symbol} but no position exists")
    
    def _check_t1_availability(self, symbol, size):
        """
        Check if there are enough shares available to sell under T+1 rules
        
        Args:
            symbol (str): Symbol to check
            size (float): Size of position to sell (as fraction of capital)
            
        Returns:
            bool: True if enough shares are available, False otherwise
        """
        if not self.t_plus_one:
            return True
            
        # Get available quantity
        available_qty = self.t1_available_assets.get(symbol, 0)
        
        # If we have a position but it's not available for T+1, reject
        if (symbol in self.positions and 
            self.positions[symbol]['quantity'] > 0 and 
            available_qty <= 0):
            return False
            
        # For simplicity, if we have any shares available, allow selling
        return available_qty > 0
    
    def _update_t1_available_assets(self):
        """
        Update T+1 available assets (for Chinese A-shares)
        
        In T+1 settlement, shares bought today can only be sold tomorrow
        """
        # In a real implementation, this would be updated based on actual settlement
        # For simulation, we'll update based on positions and time
        
        # Simple approach - assume positions held overnight are available to sell
        for symbol, position in self.positions.items():
            # If the position was last updated yesterday or earlier, it's available to sell
            if (datetime.now() - position['last_updated']).days >= 1:
                self.t1_available_assets[symbol] = position['quantity']
    
    def _create_execution_result(self, decision, success, message, order_id=None, quantity=0):
        """
        Create standardized execution result
        
        Args:
            decision (dict): Original trading decision
            success (bool): Whether execution was successful
            message (str): Result message
            order_id (str, optional): Order ID if successful
            quantity (int): Order quantity
            
        Returns:
            dict: Execution result
        """
        return {
            'decision_id': decision.get('id'),
            'symbol': decision['symbol'],
            'direction': decision['direction'],
            'size': decision['size'],
            'quantity': quantity,
            'success': success,
            'message': message,
            'order_id': order_id,
            'timestamp': datetime.now()
        }
    
    def _load_positions(self):
        """
        Load positions from file
        """
        positions_file = os.path.join(self.data_dir, 'positions.json')
        funds_file = os.path.join(self.data_dir, 't1_assets.json')
        
        # Load positions
        if os.path.exists(positions_file):
            try:
                with open(positions_file, 'r') as f:
                    positions_data = json.load(f)
                    
                # Convert date strings back to datetime objects
                for symbol, pos in positions_data.items():
                    if 'open_date' in pos:
                        pos['open_date'] = datetime.fromisoformat(pos['open_date'])
                    if 'last_updated' in pos:
                        pos['last_updated'] = datetime.fromisoformat(pos['last_updated'])
                
                self.positions = positions_data
                logger.info(f"Loaded {len(self.positions)} positions")
            except Exception as e:
                logger.error(f"Error loading positions: {e}")
        
        # Load T+1 available assets
        if os.path.exists(funds_file):
            try:
                with open(funds_file, 'r') as f:
                    self.t1_available_assets = json.load(f)
                logger.info(f"Loaded T+1 available assets")
            except Exception as e:
                logger.error(f"Error loading T+1 assets: {e}")
    
    def _save_positions(self):
        """
        Save positions to file
        """
        positions_file = os.path.join(self.data_dir, 'positions.json')
        funds_file = os.path.join(self.data_dir, 't1_assets.json')
        
        # Convert positions to serializable format
        serializable_positions = {}
        for symbol, pos in self.positions.items():
            serializable_pos = pos.copy()
            if 'open_date' in serializable_pos:
                serializable_pos['open_date'] = serializable_pos['open_date'].isoformat()
            if 'last_updated' in serializable_pos:
                serializable_pos['last_updated'] = serializable_pos['last_updated'].isoformat()
            serializable_positions[symbol] = serializable_pos
        
        # Save positions
        try:
            with open(positions_file, 'w') as f:
                json.dump(serializable_positions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving positions: {e}")
        
        # Save T+1 available assets
        try:
            with open(funds_file, 'w') as f:
                json.dump(self.t1_available_assets, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving T+1 assets: {e}")
    
    def get_positions(self):
        """
        Get current positions
        
        Returns:
            dict: Current positions
        """
        return self.positions
    
    def get_funds(self):
        """
        Get available and frozen funds
        
        Returns:
            dict: Fund information
        """
        return {
            'available': self.available_funds,
            'frozen': self.frozen_funds,
            'total': self.available_funds + self.frozen_funds
        }
    
    def get_order_history(self):
        """
        Get order history
        
        Returns:
            list: Order history
        """
        # Return open orders and filled orders
        return {
            'open': list(self.open_orders.values()),
            'filled': self.filled_orders
        }
    
    def get_t1_available(self):
        """
        Get T+1 available assets
        
        Returns:
            dict: T+1 available assets
        """
        return self.t1_available_assets
    
    def cancel_order(self, order_id):
        """
        Cancel an open order
        
        Args:
            order_id (str): Order ID to cancel
            
        Returns:
            bool: True if cancelled, False otherwise
        """
        # Check if order exists
        if order_id not in self.open_orders:
            logger.warning(f"Cannot cancel order {order_id}: not found")
            return False
        
        order = self.open_orders[order_id]
        
        # Check if order can be cancelled
        if order['status'] not in [OrderStatus.PENDING.value, OrderStatus.SUBMITTED.value]:
            logger.warning(f"Cannot cancel order {order_id}: status is {order['status']}")
            return False
        
        # Cancel the order
        order['status'] = OrderStatus.CANCELLED.value
        
        # Return funds if it was a buy order
        if order['direction'] == 'buy':
            # Estimate the funds that were reserved
            reserved_funds = order['quantity'] * order.get('limit_price', 0) * (1 + 0.003)
            self.frozen_funds -= reserved_funds
            self.available_funds += reserved_funds
        
        # Move to filled orders (for tracking)
        del self.open_orders[order_id]
        self.filled_orders.append(order)
        
        logger.info(f"Cancelled order {order_id}")
        
        return True 