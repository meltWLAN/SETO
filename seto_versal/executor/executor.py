#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trade executor module for SETO-Versal
Implements trade execution with T+1 constraints for Chinese A-shares
"""

import logging
import uuid
from datetime import datetime, timedelta, time
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
import json
import os
import pandas as pd

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order status enum"""
    PENDING = "pending"           # Order waiting to be submitted
    SUBMITTED = "submitted"       # Order submitted to broker
    PARTIAL = "partial"           # Order partially filled
    FILLED = "filled"             # Order fully filled
    CANCELLED = "cancelled"       # Order cancelled
    REJECTED = "rejected"         # Order rejected
    EXPIRED = "expired"           # Order expired (e.g., day order)
    T1_LOCKED = "t1_locked"       # T+1 locked position (bought today)

class OrderType(Enum):
    """Order type enum"""
    MARKET = "market"             # Market order
    LIMIT = "limit"               # Limit order
    STOP = "stop"                 # Stop order
    STOP_LIMIT = "stop_limit"     # Stop limit order

class ExecutionResult:
    """
    Represents the result of an order execution
    """
    
    def __init__(self, 
                order_id: str,
                symbol: str,
                direction: str,
                quantity: int,
                price: float,
                status: OrderStatus,
                timestamp: datetime,
                commission: float = 0.0,
                error_message: Optional[str] = None):
        """
        Initialize execution result
        
        Args:
            order_id (str): ID of the order
            symbol (str): Trading symbol
            direction (str): Buy/Sell
            quantity (int): Quantity executed
            price (float): Execution price
            status (OrderStatus): Status of the order
            timestamp (datetime): Execution timestamp
            commission (float): Commission paid
            error_message (str, optional): Error message if any
        """
        self.order_id = order_id
        self.symbol = symbol
        self.direction = direction.lower()
        self.quantity = quantity
        self.price = price
        self.status = status
        self.timestamp = timestamp
        self.commission = commission
        self.error_message = error_message
        self.value = price * quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert execution result to dictionary
        
        Returns:
            dict: Dictionary representation
        """
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'commission': self.commission,
            'error_message': self.error_message,
            'value': self.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResult':
        """
        Create ExecutionResult from dictionary
        
        Args:
            data (dict): Dictionary representation
            
        Returns:
            ExecutionResult: Instance from dictionary
        """
        return cls(
            order_id=data['order_id'],
            symbol=data['symbol'],
            direction=data['direction'],
            quantity=data['quantity'],
            price=data['price'],
            status=OrderStatus(data['status']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            commission=data.get('commission', 0.0),
            error_message=data.get('error_message')
        )
    
    def __str__(self) -> str:
        """String representation of the execution result"""
        return (f"{self.direction.upper()} {self.quantity} {self.symbol} @ {self.price:.2f} "
                f"[{self.status.value.upper()}]")

class TradeExecutor:
    """
    Trade executor that handles order submission and execution
    Implements T+1 rule for Chinese A-shares
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trade executor with configuration
        
        Args:
            config (dict): Executor configuration
        """
        self.config = config
        self.name = config.get('name', 'trade_executor')
        
        # Broker and account settings
        self.broker_id = config.get('broker_id', 'simulated')
        self.account_id = config.get('account_id', 'default')
        self.is_simulation = config.get('simulation', True)
        
        # Commission settings
        self.commission_rate = config.get('commission_rate', 0.0003)  # 0.03%
        self.min_commission = config.get('min_commission', 5.0)  # Minimum commission per trade
        self.stamp_duty = config.get('stamp_duty', 0.001)  # 0.1% stamp duty on sells
        
        # Order book and position tracking
        self.orders = {}  # Order ID to order details
        self.positions = {}  # Symbol to position details
        self.t1_locked_positions = {}  # Positions bought today (T+0) 
        self.execution_history = []
        
        # Trade constraints
        self.t1_enabled = config.get('t1_enabled', True)  # Enable T+1 rule
        self.trading_hours = config.get('trading_hours', {
            'morning_start': time(9, 30),
            'morning_end': time(11, 30),
            'afternoon_start': time(13, 0),
            'afternoon_end': time(15, 0)
        })
        
        # Initial capital
        self.initial_capital = config.get('system', {}).get('initial_capital', 1000000.0)
        self.cash = self.initial_capital
        
        # Execution settings
        self.max_slippage = config.get('max_slippage', 0.002)  # Maximum slippage allowed
        self.default_order_type = OrderType(config.get('default_order_type', 'limit'))
        
        # Data paths
        self.data_path = config.get('data_path', 'seto_versal/data/execution')
        os.makedirs(self.data_path, exist_ok=True)
        
        # Load positions if available
        self._load_positions()
        
        logger.info(f"Trade executor '{self.name}' initialized with T+1 {'enabled' if self.t1_enabled else 'disabled'}")
    
    def execute_decision(self, decision, market_state) -> ExecutionResult:
        """
        Execute a coordinated decision
        
        Args:
            decision: Coordinated decision to execute
            market_state: Current market state
            
        Returns:
            ExecutionResult: Result of the execution
        """
        # Create an order ID
        order_id = str(uuid.uuid4())
        
        # Get current price
        try:
            current_price = self._get_current_price(decision.symbol, market_state)
        except Exception as e:
            logger.error(f"Error getting price for {decision.symbol}: {str(e)}")
            return ExecutionResult(
                order_id=order_id,
                symbol=decision.symbol,
                direction=decision.decision_type,
                quantity=0,
                price=0.0,
                status=OrderStatus.REJECTED,
                timestamp=datetime.now(),
                error_message=f"Error getting price: {str(e)}"
            )
        
        # Check T+1 restrictions for sells
        if decision.decision_type == 'sell' and self.t1_enabled:
            if decision.symbol in self.t1_locked_positions:
                logger.warning(f"Cannot sell {decision.symbol} due to T+1 restriction")
                return ExecutionResult(
                    order_id=order_id,
                    symbol=decision.symbol,
                    direction=decision.decision_type,
                    quantity=0,
                    price=current_price,
                    status=OrderStatus.REJECTED,
                    timestamp=datetime.now(),
                    error_message="T+1 restriction: position bought today"
                )
        
        # Check if we have the position to sell
        if decision.decision_type == 'sell':
            position = self.positions.get(decision.symbol, {'quantity': 0})
            if position['quantity'] <= 0:
                logger.warning(f"Cannot sell {decision.symbol} - no position held")
                return ExecutionResult(
                    order_id=order_id,
                    symbol=decision.symbol,
                    direction=decision.decision_type,
                    quantity=0,
                    price=current_price,
                    status=OrderStatus.REJECTED,
                    timestamp=datetime.now(),
                    error_message="No position to sell"
                )
        
        # Determine quantity if not specified
        quantity = decision.quantity
        if quantity is None:
            # This would involve position sizing logic
            # For now use a placeholder
            quantity = 100  # Default lot size
        
        # Execute the trade (simulated)
        if self.is_simulation:
            execution_price = self._simulate_execution_price(current_price, decision.decision_type)
            status = OrderStatus.FILLED
            commission = self._calculate_commission(execution_price * quantity, decision.decision_type)
            
            # Update positions
            self._update_positions(decision.symbol, decision.decision_type, quantity, execution_price)
            
            # Log execution
            logger.info(f"Executed {decision.decision_type} order for {quantity} {decision.symbol} @ {execution_price:.2f}")
        else:
            # In a real implementation, this would submit to broker API
            # For now simulate a successful execution
            execution_price = current_price
            status = OrderStatus.FILLED
            commission = self._calculate_commission(execution_price * quantity, decision.decision_type)
            
            # Update positions
            self._update_positions(decision.symbol, decision.decision_type, quantity, execution_price)
            
            logger.info(f"Executed {decision.decision_type} order via broker for {quantity} {decision.symbol}")
        
        # Create execution result
        result = ExecutionResult(
            order_id=order_id,
            symbol=decision.symbol,
            direction=decision.decision_type,
            quantity=quantity,
            price=execution_price,
            status=status,
            timestamp=datetime.now(),
            commission=commission
        )
        
        # Record in history
        self.execution_history.append(result.to_dict())
        
        # Update order book
        self.orders[order_id] = {
            'order_id': order_id,
            'symbol': decision.symbol,
            'direction': decision.decision_type,
            'quantity': quantity,
            'price': execution_price,
            'status': status.value,
            'timestamp': datetime.now().isoformat(),
            'decision_id': getattr(decision, 'id', None),
            'commission': commission
        }
        
        # Mark decision as executed if it has that attribute
        if hasattr(decision, 'executed'):
            decision.executed = True
            if hasattr(decision, 'execution_info'):
                decision.execution_info = result.to_dict()
        
        # Save positions to file
        self._save_positions()
        
        return result
    
    def execute_trade_plan(self, trade_plan: List[Dict[str, Any]], market_state) -> List[ExecutionResult]:
        """
        Execute a plan of trades
        
        Args:
            trade_plan (list): List of trade specifications
            market_state: Current market state
            
        Returns:
            list: List of execution results
        """
        results = []
        
        for trade in trade_plan:
            # Basic validation
            required_fields = ['symbol', 'direction', 'quantity']
            missing_fields = [f for f in required_fields if f not in trade]
            
            if missing_fields:
                logger.error(f"Trade missing required fields: {missing_fields}")
                continue
            
            # Create a simple decision-like object
            class SimpleTrade:
                pass
            
            decision = SimpleTrade()
            decision.symbol = trade['symbol']
            decision.decision_type = trade['direction']
            decision.quantity = trade['quantity']
            decision.id = trade.get('id', str(uuid.uuid4()))
            
            # Execute the trade
            result = self.execute_decision(decision, market_state)
            results.append(result)
        
        return results
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get current position for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            dict: Position details or empty position
        """
        return self.positions.get(symbol, {
            'symbol': symbol,
            'quantity': 0,
            'avg_price': 0.0,
            'value': 0.0,
            'last_updated': None
        })
    
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all current positions
        
        Returns:
            dict: Dictionary of positions by symbol
        """
        return self.positions
    
    def get_t1_locked_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all T+1 locked positions (bought today)
        
        Returns:
            dict: Dictionary of locked positions by symbol
        """
        return self.t1_locked_positions
    
    def process_day_end(self) -> None:
        """
        Process end of trading day
        - Clear T+1 locked positions
        - Update position data
        """
        logger.info("Processing end of trading day")
        
        # Clear T+1 locks
        self.t1_locked_positions = {}
        
        # Save positions
        self._save_positions()
        
        # Save execution history
        self._save_execution_history()
        
        logger.info("End of day processing complete - T+1 locks cleared")
    
    def _simulate_execution_price(self, current_price: float, direction: str) -> float:
        """
        Simulate execution price with slippage
        
        Args:
            current_price (float): Current market price
            direction (str): Buy/Sell direction
            
        Returns:
            float: Simulated execution price
        """
        import random
        
        # Random slippage between 0 and max_slippage
        slippage = random.uniform(0, self.max_slippage)
        
        # Apply slippage in the unfavorable direction
        if direction.lower() == 'buy':
            # Higher price for buys
            price = current_price * (1 + slippage)
        else:
            # Lower price for sells
            price = current_price * (1 - slippage)
        
        return price
    
    def _calculate_commission(self, trade_value: float, direction: str) -> float:
        """
        Calculate trade commission
        
        Args:
            trade_value (float): Value of the trade
            direction (str): Buy/Sell direction
            
        Returns:
            float: Commission amount
        """
        # Base commission
        commission = trade_value * self.commission_rate
        
        # Apply minimum commission
        commission = max(commission, self.min_commission)
        
        # Add stamp duty for sells
        if direction.lower() == 'sell':
            commission += trade_value * self.stamp_duty
        
        return commission
    
    def _update_positions(self, symbol: str, direction: str, quantity: int, price: float) -> None:
        """
        Update positions after a trade
        
        Args:
            symbol (str): Trading symbol
            direction (str): Buy/Sell direction
            quantity (int): Trade quantity
            price (float): Execution price
        """
        direction = direction.lower()
        
        # Get current position
        position = self.positions.get(symbol, {
            'symbol': symbol,
            'quantity': 0,
            'avg_price': 0.0,
            'value': 0.0,
            'last_updated': None
        })
        
        if direction == 'buy':
            # Calculate new average price and quantity
            current_quantity = position['quantity']
            current_value = current_quantity * position.get('avg_price', 0.0)
            new_value = price * quantity
            
            if current_quantity + quantity > 0:
                new_avg_price = (current_value + new_value) / (current_quantity + quantity)
            else:
                new_avg_price = price
            
            # Update position
            position['quantity'] = current_quantity + quantity
            position['avg_price'] = new_avg_price
            position['value'] = position['quantity'] * new_avg_price
            position['last_updated'] = datetime.now().isoformat()
            
            # Record in T+1 locked positions if enabled
            if self.t1_enabled:
                t1_position = self.t1_locked_positions.get(symbol, {
                    'symbol': symbol,
                    'quantity': 0,
                    'avg_price': 0.0,
                    'value': 0.0,
                    'last_updated': None
                })
                
                t1_position['quantity'] = t1_position.get('quantity', 0) + quantity
                t1_position['avg_price'] = price
                t1_position['value'] = t1_position['quantity'] * price
                t1_position['last_updated'] = datetime.now().isoformat()
                
                self.t1_locked_positions[symbol] = t1_position
        
        elif direction == 'sell':
            # Update position
            current_quantity = position['quantity']
            new_quantity = current_quantity - quantity
            
            if new_quantity < 0:
                new_quantity = 0  # Prevent negative positions
                logger.warning(f"Attempted to sell more {symbol} than held, capped at {current_quantity}")
            
            position['quantity'] = new_quantity
            position['value'] = new_quantity * position.get('avg_price', 0.0)
            position['last_updated'] = datetime.now().isoformat()
            
            # If position is zero, adjust average price
            if new_quantity == 0:
                position['avg_price'] = 0.0
        
        # Update positions dictionary
        self.positions[symbol] = position
    
    def _get_current_price(self, symbol: str, market_state) -> float:
        """
        Get current price for a symbol
        
        Args:
            symbol (str): Trading symbol
            market_state: Current market state
            
        Returns:
            float: Current price
        """
        # In a real implementation, this would get the current market price
        # For simulation, get price from market state if available
        try:
            if hasattr(market_state, 'data_source') and market_state.data_source is not None:
                # Get latest price from data source
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
                
                # Try to get price data
                df = market_state.data_source.get_price_data(symbol, start_date, end_date, 'daily')
                
                # Return last close price
                if df is not None and not df.empty:
                    return df['close'].iloc[-1]
            
            # Fallback to random price if no data
            import random
            return random.uniform(50, 500)
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {str(e)}")
            # Fallback to random price for simulation
            import random
            return random.uniform(50, 500)
    
    def _save_positions(self) -> None:
        """Save positions to file"""
        try:
            positions_file = os.path.join(self.data_path, 'positions.json')
            
            data = {
                'positions': self.positions,
                't1_locked_positions': self.t1_locked_positions,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(positions_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Positions saved to {positions_file}")
        except Exception as e:
            logger.error(f"Error saving positions: {str(e)}")
    
    def _load_positions(self) -> None:
        """Load positions from file"""
        try:
            positions_file = os.path.join(self.data_path, 'positions.json')
            
            if os.path.exists(positions_file):
                with open(positions_file, 'r') as f:
                    data = json.load(f)
                
                self.positions = data.get('positions', {})
                self.t1_locked_positions = data.get('t1_locked_positions', {})
                
                logger.info(f"Loaded {len(self.positions)} positions from {positions_file}")
        except Exception as e:
            logger.error(f"Error loading positions: {str(e)}")
    
    def _save_execution_history(self) -> None:
        """Save execution history to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            history_file = os.path.join(self.data_path, f'execution_history_{timestamp}.json')
            
            with open(history_file, 'w') as f:
                json.dump(self.execution_history, f, indent=2)
            
            logger.info(f"Execution history ({len(self.execution_history)} trades) saved to {history_file}")
            
            # Clear history after saving
            self.execution_history = []
        except Exception as e:
            logger.error(f"Error saving execution history: {str(e)}")
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order if possible
        
        Args:
            order_id (str): ID of the order to cancel
            
        Returns:
            bool: True if cancelled, False otherwise
        """
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found for cancellation")
            return False
        
        order = self.orders[order_id]
        
        # Check if order can be cancelled
        cancellable_statuses = [OrderStatus.PENDING.value, OrderStatus.SUBMITTED.value]
        if order['status'] not in cancellable_statuses:
            logger.warning(f"Cannot cancel order {order_id} with status {order['status']}")
            return False
        
        # Update order status
        order['status'] = OrderStatus.CANCELLED.value
        self.orders[order_id] = order
        
        logger.info(f"Cancelled order {order_id} for {order['symbol']}")
        return True
    
    def generate_trade_report(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Generate trade report for a date range
        
        Args:
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Trade report
        """
        # Default to last 30 days if not specified
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Convert to datetime for comparison
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Get all execution history files
        history_files = []
        for file in os.listdir(self.data_path):
            if file.startswith('execution_history_') and file.endswith('.json'):
                date_str = file.replace('execution_history_', '').replace('.json', '')
                try:
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                    if start_dt <= file_date <= end_dt:
                        history_files.append(os.path.join(self.data_path, file))
                except Exception:
                    pass
        
        # Load all relevant history files
        all_executions = []
        for file in history_files:
            try:
                with open(file, 'r') as f:
                    executions = json.load(f)
                    all_executions.extend(executions)
            except Exception as e:
                logger.error(f"Error loading execution history from {file}: {str(e)}")
        
        # Convert to DataFrame
        if all_executions:
            df = pd.DataFrame(all_executions)
            
            # Convert timestamp string to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by date range
            df = df[(df['timestamp'].dt.date >= start_dt.date()) & 
                   (df['timestamp'].dt.date <= end_dt.date())]
            
            return df
        else:
            # Return empty DataFrame if no executions
            return pd.DataFrame(columns=[
                'order_id', 'symbol', 'direction', 'quantity', 'price', 
                'status', 'timestamp', 'commission', 'error_message', 'value'
            ])
    
    def calculate_daily_pnl(self, date: str = None) -> Dict[str, Any]:
        """
        Calculate profit and loss for a specific day
        
        Args:
            date (str, optional): Date in YYYY-MM-DD format, defaults to today
            
        Returns:
            dict: PnL summary
        """
        # Default to today if not specified
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Get executions for the day
        trades_df = self.generate_trade_report(date, date)
        
        if trades_df.empty:
            return {
                'date': date,
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'total_value': 0.0,
                'total_commission': 0.0,
                'symbols_traded': [],
                'net_position_change': {}
            }
        
        # Calculate summary statistics
        total_trades = len(trades_df)
        buy_trades = len(trades_df[trades_df['direction'] == 'buy'])
        sell_trades = len(trades_df[trades_df['direction'] == 'sell'])
        total_value = trades_df['value'].sum()
        total_commission = trades_df['commission'].sum()
        symbols_traded = trades_df['symbol'].unique().tolist()
        
        # Calculate net position change by symbol
        net_position_change = {}
        for symbol in symbols_traded:
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            buys = symbol_trades[symbol_trades['direction'] == 'buy']
            sells = symbol_trades[symbol_trades['direction'] == 'sell']
            
            buy_quantity = buys['quantity'].sum() if not buys.empty else 0
            sell_quantity = sells['quantity'].sum() if not sells.empty else 0
            
            net_position_change[symbol] = {
                'net_change': buy_quantity - sell_quantity,
                'buy_quantity': buy_quantity,
                'sell_quantity': sell_quantity,
                'buy_value': buys['value'].sum() if not buys.empty else 0,
                'sell_value': sells['value'].sum() if not sells.empty else 0,
                'average_buy_price': buys['price'].mean() if not buys.empty else 0,
                'average_sell_price': sells['price'].mean() if not sells.empty else 0
            }
        
        # Return summary
        return {
            'date': date,
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_value': total_value,
            'total_commission': total_commission,
            'symbols_traded': symbols_traded,
            'net_position_change': net_position_change
        }

    def execute_decisions(self, decisions, market_state):
        """
        Execute trade decisions from the coordinator
        
        Args:
            decisions (list): List of trade decisions to execute
            market_state (MarketState): Current market state
            
        Returns:
            list: List of execution results
        """
        if not decisions:
            logger.info("No decisions to execute")
            return []
            
        logger.info(f"Executing {len(decisions)} trade decisions")
        results = []
        
        for decision in decisions:
            # Get current market price
            symbol = decision.symbol
            price = market_state.get_price(symbol)
            
            if not price:
                logger.warning(f"No price available for {symbol}, skipping execution")
                continue
                
            # Create execution result
            result = {
                'decision_id': decision.id if hasattr(decision, 'id') else str(uuid.uuid4()),
                'symbol': symbol,
                'decision_type': decision.decision_type if hasattr(decision, 'decision_type') else 'unknown',
                'quantity': decision.quantity if hasattr(decision, 'quantity') else 0,
                'price': price,
                'timestamp': datetime.now(),
                'status': 'executed',
                'message': 'Simulated execution in backtest mode'
            }
            
            # Update positions (placeholder implementation)
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'symbol': symbol,
                    'quantity': 0,
                    'avg_price': 0,
                    'cost_basis': 0,
                    'current_price': price,
                    'market_value': 0,
                    'last_update': datetime.now()
                }
                
            # Simple position update logic
            position = self.positions[symbol]
            quantity = result['quantity']
            
            # Update position based on decision type
            if result['decision_type'] == 'buy':
                position['quantity'] += quantity
                position['avg_price'] = (position['avg_price'] * (position['quantity'] - quantity) + price * quantity) / position['quantity']
                position['cost_basis'] = position['avg_price'] * position['quantity']
                
                # Apply T+1 rule for buy orders
                if self.t1_enabled:
                    if symbol not in self.t1_locked_positions:
                        self.t1_locked_positions[symbol] = 0
                    self.t1_locked_positions[symbol] += quantity
                    
            elif result['decision_type'] == 'sell':
                # Check T+1 rule
                if self.t1_enabled and symbol in self.t1_locked_positions and self.t1_locked_positions[symbol] > 0:
                    sellable_quantity = position['quantity'] - self.t1_locked_positions[symbol]
                    if quantity > sellable_quantity:
                        logger.warning(f"T+1 rule violation for {symbol}: trying to sell {quantity} but only {sellable_quantity} available, reducing order")
                        quantity = max(0, sellable_quantity)
                        result['quantity'] = quantity
                        result['message'] = 'Reduced quantity due to T+1 rule'
                        
                if quantity <= 0:
                    result['status'] = 'rejected'
                    result['message'] = 'No sellable quantity available due to T+1 rule'
                else:
                    position['quantity'] -= quantity
                    # If position is now zero, reset avg price
                    if position['quantity'] <= 0:
                        position['avg_price'] = 0
                        position['cost_basis'] = 0
                        position['quantity'] = 0
                    else:
                        position['cost_basis'] = position['avg_price'] * position['quantity']
            
            # Update current market value
            position['current_price'] = price
            position['market_value'] = position['quantity'] * price
            position['last_update'] = datetime.now()
            
            # Add to results
            results.append(result)
            
        return results 