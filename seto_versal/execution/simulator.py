#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模拟交易所执行客户端，用于回测和模拟交易。
"""

import os
import time
import logging
import json
import random
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

from seto_versal.execution.client import ExecutionClient, Order, OrderType, OrderStatus


class SimulatedExecutionClient(ExecutionClient):
    """
    模拟交易所执行客户端，用于回测和模拟交易
    """
    
    def __init__(self, name: str = "simulator", config: Dict[str, Any] = None):
        """
        初始化模拟交易所执行客户端
        
        Args:
            name: 客户端名称
            config: 配置参数
        """
        super().__init__(name, config)
        
        # 默认配置
        self.default_config = {
            "initial_balance": 100000.0,  # 初始资金
            "commission_rate": 0.0003,    # 手续费率
            "slippage_model": "fixed",    # 滑点模型: fixed, random, proportional
            "slippage_value": 0.0001,     # 滑点值
            "order_latency_ms": 50,       # 订单延迟（毫秒）
            "fill_latency_ms": 100,       # 成交延迟（毫秒）
            "fill_probability": 0.98,     # 市价单成交概率
            "partial_fill_probability": 0.2,  # 部分成交概率
            "reject_probability": 0.01,   # 拒绝概率
            "price_impact_factor": 0.0,   # 价格影响因子
            "tick_size": 0.01,            # 最小价格变动单位
            "lot_size": 1.0,              # 最小交易数量单位
            "market_hours": {             # 市场交易时间（可选）
                "open_time": "09:30",
                "close_time": "16:00",
                "timezone": "America/New_York"
            },
            "symbols": {                 # 交易品种配置
                "AAPL": {"price": 150.0, "bid": 149.95, "ask": 150.05, "volume": 1000000},
                "MSFT": {"price": 250.0, "bid": 249.95, "ask": 250.05, "volume": 800000},
                "GOOG": {"price": 2500.0, "bid": 2499.5, "ask": 2500.5, "volume": 500000}
            }
        }
        
        # 合并配置
        self.config = {**self.default_config, **(config or {})}
        
        # 账户信息
        self.account = {
            "balance": self.config["initial_balance"],
            "equity": self.config["initial_balance"],
            "margin_used": 0.0,
            "available": self.config["initial_balance"]
        }
        
        # 持仓信息
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # 市场数据
        self.market_data = self.config["symbols"].copy()
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 后台任务线程
        self.background_thread = None
        self.running = False
        
        # 订单处理队列
        self.order_queue: List[str] = []
        
        self.logger.info(f"模拟交易所执行客户端已初始化，初始资金: {self.config['initial_balance']}")
    
    def connect(self) -> bool:
        """
        连接到模拟交易所
        
        Returns:
            连接是否成功
        """
        with self.lock:
            if self.is_connected:
                return True
            
            self.is_connected = True
            self.running = True
            
            # 启动后台任务线程
            self.background_thread = threading.Thread(target=self._background_task)
            self.background_thread.daemon = True
            self.background_thread.start()
            
            self.logger.info("已连接到模拟交易所")
            return True
    
    def disconnect(self) -> bool:
        """
        断开与模拟交易所的连接
        
        Returns:
            断开连接是否成功
        """
        with self.lock:
            if not self.is_connected:
                return True
            
            self.running = False
            if self.background_thread and self.background_thread.is_alive():
                self.background_thread.join(timeout=2.0)
            
            self.is_connected = False
            self.logger.info("已断开与模拟交易所的连接")
            return True
    
    def is_authenticated(self) -> bool:
        """
        检查是否已认证
        
        Returns:
            是否已认证，模拟交易所总是返回True
        """
        return self.is_connected
    
    def place_order(self, order: Order) -> Tuple[bool, Optional[str]]:
        """
        下单
        
        Args:
            order: 订单对象
            
        Returns:
            元组(是否成功, 失败原因)
        """
        if not self.is_connected:
            return False, "未连接到交易所"
        
        with self.lock:
            # 检查订单参数
            if order.order_type != OrderType.MARKET and order.price is None:
                return False, "限价单必须指定价格"
            
            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
                return False, "止损单必须指定止损价格"
            
            if order.symbol not in self.market_data:
                return False, f"不支持的交易品种: {order.symbol}"
            
            # 检查余额
            symbol_data = self.market_data[order.symbol]
            estimated_price = order.price or symbol_data["price"]
            estimated_cost = estimated_price * order.quantity
            
            if order.direction == "BUY" and estimated_cost > self.account["available"]:
                return False, "可用资金不足"
            
            # 随机拒绝一些订单（模拟）
            if random.random() < self.config["reject_probability"]:
                order.update_status(OrderStatus.REJECTED, "订单被随机拒绝（模拟）")
                self.order_manager.add_order(order)
                return False, "订单被随机拒绝（模拟）"
            
            # 更新订单状态为已提交
            order.update_status(OrderStatus.SUBMITTED)
            self.order_manager.add_order(order)
            
            # 添加到订单处理队列
            self.order_queue.append(order.order_id)
            
            self.logger.info(f"订单已提交: {order}")
            
            # 如果有订单延迟，模拟异步处理
            if self.config["order_latency_ms"] > 0:
                return True, None
            
            # 否则立即处理订单
            self._process_order(order.order_id)
            return True, None
    
    def cancel_order(self, order_id: str) -> Tuple[bool, Optional[str]]:
        """
        取消订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            元组(是否成功, 失败原因)
        """
        if not self.is_connected:
            return False, "未连接到交易所"
        
        with self.lock:
            order = self.order_manager.get_order(order_id)
            if not order:
                return False, f"订单不存在: {order_id}"
            
            if order.status.is_complete:
                return False, f"订单已完成，无法取消: {order_id}"
            
            # 更新订单状态为已取消
            success = self.order_manager.cancel_order(order_id, "用户取消")
            
            if success:
                self.logger.info(f"订单已取消: {order}")
                
                # 如果订单在处理队列中，移除它
                if order_id in self.order_queue:
                    self.order_queue.remove(order_id)
            
            return success, None if success else "无法取消订单"
    
    def get_order_status(self, order_id: str) -> Tuple[Optional[OrderStatus], Dict[str, Any]]:
        """
        获取订单状态
        
        Args:
            order_id: 订单ID
            
        Returns:
            元组(订单状态, 额外信息)
        """
        if not self.is_connected:
            return None, {"error": "未连接到交易所"}
        
        order = self.order_manager.get_order(order_id)
        if not order:
            return None, {"error": f"订单不存在: {order_id}"}
        
        extra_info = {
            "filled_quantity": order.filled_quantity,
            "average_fill_price": order.average_fill_price,
            "commission": order.commission,
            "status_updates": order.status_updates
        }
        
        return order.status, extra_info
    
    def get_open_orders(self) -> List[Order]:
        """
        获取未成交订单
        
        Returns:
            未成交订单列表
        """
        if not self.is_connected:
            return []
        
        return self.order_manager.get_active_orders()
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        获取持仓
        
        Returns:
            持仓信息映射（品种到持仓信息）
        """
        if not self.is_connected:
            return {}
        
        with self.lock:
            return self.positions.copy()
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        获取账户信息
        
        Returns:
            账户信息
        """
        if not self.is_connected:
            return {}
        
        with self.lock:
            return self.account.copy()
    
    def _background_task(self) -> None:
        """
        后台任务线程，用于处理订单和更新市场数据
        """
        self.logger.info("后台任务线程已启动")
        
        while self.running:
            try:
                # 处理订单队列
                self._process_order_queue()
                
                # 更新市场数据
                self._update_market_data()
                
                # 检查止损单
                self._check_stop_orders()
                
                # 更新账户信息
                self._update_account_info()
                
                # 休眠一段时间
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"后台任务异常: {str(e)}")
        
        self.logger.info("后台任务线程已停止")
    
    def _process_order_queue(self) -> None:
        """
        处理订单队列
        """
        with self.lock:
            # 复制队列，避免处理时修改
            order_ids = list(self.order_queue)
        
        for order_id in order_ids:
            try:
                # 模拟订单延迟
                if self.config["order_latency_ms"] > 0:
                    time.sleep(self.config["order_latency_ms"] / 1000.0)
                
                self._process_order(order_id)
                
                with self.lock:
                    if order_id in self.order_queue:
                        self.order_queue.remove(order_id)
            except Exception as e:
                self.logger.error(f"处理订单 {order_id} 异常: {str(e)}")
    
    def _process_order(self, order_id: str) -> None:
        """
        处理单个订单
        
        Args:
            order_id: 订单ID
        """
        with self.lock:
            order = self.order_manager.get_order(order_id)
            if not order or order.status.is_complete:
                return
            
            # 更新订单状态为已接受
            if order.status == OrderStatus.SUBMITTED:
                updates = {"status": OrderStatus.ACCEPTED}
                self.order_manager.update_order(order_id, updates)
                self._handle_order_update(order_id, updates)
            
            # 处理不同类型的订单
            if order.order_type == OrderType.MARKET:
                self._process_market_order(order)
            elif order.order_type == OrderType.LIMIT:
                self._process_limit_order(order)
            elif order.order_type == OrderType.STOP:
                # 止损单在市场数据更新时检查触发
                pass
            elif order.order_type == OrderType.STOP_LIMIT:
                # 止损限价单在市场数据更新时检查触发
                pass
    
    def _process_market_order(self, order: Order) -> None:
        """
        处理市价单
        
        Args:
            order: 订单对象
        """
        if not order or order.status.is_complete:
            return
        
        symbol_data = self.market_data.get(order.symbol)
        if not symbol_data:
            order.update_status(OrderStatus.REJECTED, f"不支持的交易品种: {order.symbol}")
            return
        
        # 模拟成交延迟
        if self.config["fill_latency_ms"] > 0:
            time.sleep(self.config["fill_latency_ms"] / 1000.0)
        
        # 随机决定是否成交
        if random.random() > self.config["fill_probability"]:
            order.update_status(OrderStatus.REJECTED, "订单被随机拒绝（模拟成交失败）")
            return
        
        # 计算成交价格（加入滑点）
        price = self._calculate_execution_price(order)
        
        # 随机决定是否部分成交
        quantity = order.quantity
        if random.random() < self.config["partial_fill_probability"] and not order.status == OrderStatus.PARTIALLY_FILLED:
            # 部分成交
            filled_ratio = random.uniform(0.1, 0.9)
            quantity = order.quantity * filled_ratio
        
        # 计算手续费
        commission = price * quantity * self.config["commission_rate"]
        
        # 添加成交记录
        order.add_fill(quantity, price, datetime.now(), commission)
        
        # 更新持仓和账户信息
        self._update_position_from_order(order, quantity, price, commission)
        
        # 如果是部分成交，稍后再处理剩余部分
        if order.status == OrderStatus.PARTIALLY_FILLED:
            # 将订单重新加入队列
            self.order_queue.append(order.order_id)
    
    def _process_limit_order(self, order: Order) -> None:
        """
        处理限价单
        
        Args:
            order: 订单对象
        """
        if not order or order.status.is_complete:
            return
        
        symbol_data = self.market_data.get(order.symbol)
        if not symbol_data:
            order.update_status(OrderStatus.REJECTED, f"不支持的交易品种: {order.symbol}")
            return
        
        # 检查限价条件是否满足
        can_execute = False
        if order.direction == "BUY" and symbol_data["ask"] <= order.price:
            can_execute = True
        elif order.direction == "SELL" and symbol_data["bid"] >= order.price:
            can_execute = True
        
        if not can_execute:
            # 限价条件未满足，保持订单活跃
            return
        
        # 模拟成交延迟
        if self.config["fill_latency_ms"] > 0:
            time.sleep(self.config["fill_latency_ms"] / 1000.0)
        
        # 计算成交价格（按限价成交）
        price = order.price
        
        # 随机决定是否部分成交
        quantity = order.quantity
        if random.random() < self.config["partial_fill_probability"] and not order.status == OrderStatus.PARTIALLY_FILLED:
            # 部分成交
            filled_ratio = random.uniform(0.1, 0.9)
            quantity = order.quantity * filled_ratio
        
        # 计算手续费
        commission = price * quantity * self.config["commission_rate"]
        
        # 添加成交记录
        order.add_fill(quantity, price, datetime.now(), commission)
        
        # 更新持仓和账户信息
        self._update_position_from_order(order, quantity, price, commission)
    
    def _check_stop_orders(self) -> None:
        """
        检查止损单是否触发
        """
        active_orders = self.order_manager.get_active_orders()
        stop_orders = [o for o in active_orders if o.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]]
        
        for order in stop_orders:
            symbol_data = self.market_data.get(order.symbol)
            if not symbol_data:
                continue
            
            # 检查止损条件是否触发
            triggered = False
            if order.direction == "BUY" and symbol_data["price"] >= order.stop_price:
                triggered = True
            elif order.direction == "SELL" and symbol_data["price"] <= order.stop_price:
                triggered = True
            
            if triggered:
                self.logger.info(f"止损单已触发: {order}")
                
                if order.order_type == OrderType.STOP:
                    # 转换为市价单
                    order.order_type = OrderType.MARKET
                    self.order_queue.append(order.order_id)
                elif order.order_type == OrderType.STOP_LIMIT:
                    # 转换为限价单
                    order.order_type = OrderType.LIMIT
                    self.order_queue.append(order.order_id)
    
    def _update_position_from_order(self, order: Order, filled_quantity: float, price: float, commission: float) -> None:
        """
        根据订单更新持仓
        
        Args:
            order: 订单对象
            filled_quantity: 成交数量
            price: 成交价格
            commission: 手续费
        """
        with self.lock:
            symbol = order.symbol
            
            # 确保持仓对象存在
            if symbol not in self.positions:
                self.positions[symbol] = {
                    "symbol": symbol,
                    "quantity": 0.0,
                    "average_price": 0.0,
                    "market_price": price,
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
            
            position = self.positions[symbol]
            current_quantity = position["quantity"]
            
            # 根据交易方向更新持仓
            if order.direction == "BUY":
                # 买入
                new_quantity = current_quantity + filled_quantity
                
                if current_quantity >= 0:
                    # 增加多头持仓
                    new_cost = position["average_price"] * current_quantity + price * filled_quantity
                    position["average_price"] = new_cost / new_quantity if new_quantity > 0 else 0.0
                else:
                    # 减少空头持仓
                    realized_pnl = (position["average_price"] - price) * min(abs(current_quantity), filled_quantity)
                    position["realized_pnl"] += realized_pnl
                    
                    if abs(current_quantity) < filled_quantity:
                        # 持仓方向改变
                        remaining_quantity = filled_quantity - abs(current_quantity)
                        position["average_price"] = price
                    else:
                        # 持仓方向不变
                        remaining_quantity = abs(current_quantity) - filled_quantity
                        if remaining_quantity > 0:
                            # 还有剩余空头
                            pass
                        else:
                            # 完全平仓
                            position["average_price"] = 0.0
                
                position["quantity"] = new_quantity
                
                # 更新账户余额
                self.account["balance"] -= price * filled_quantity + commission
            
            elif order.direction == "SELL":
                # 卖出
                new_quantity = current_quantity - filled_quantity
                
                if current_quantity <= 0:
                    # 增加空头持仓
                    new_cost = position["average_price"] * abs(current_quantity) + price * filled_quantity
                    position["average_price"] = new_cost / abs(new_quantity) if new_quantity < 0 else 0.0
                else:
                    # 减少多头持仓
                    realized_pnl = (price - position["average_price"]) * min(current_quantity, filled_quantity)
                    position["realized_pnl"] += realized_pnl
                    
                    if current_quantity < filled_quantity:
                        # 持仓方向改变
                        remaining_quantity = filled_quantity - current_quantity
                        position["average_price"] = price
                    else:
                        # 持仓方向不变
                        remaining_quantity = current_quantity - filled_quantity
                        if remaining_quantity > 0:
                            # 还有剩余多头
                            pass
                        else:
                            # 完全平仓
                            position["average_price"] = 0.0
                
                position["quantity"] = new_quantity
                
                # 更新账户余额
                self.account["balance"] += price * filled_quantity - commission
            
            # 更新持仓市场价格和未实现盈亏
            position["market_price"] = price
            if position["quantity"] > 0:
                position["unrealized_pnl"] = (price - position["average_price"]) * position["quantity"]
            elif position["quantity"] < 0:
                position["unrealized_pnl"] = (position["average_price"] - price) * abs(position["quantity"])
            else:
                position["unrealized_pnl"] = 0.0
            
            position["timestamp"] = datetime.now().isoformat()
            
            # 发送持仓更新通知
            self._handle_position_update(position)
    
    def _update_market_data(self) -> None:
        """
        更新市场数据（模拟）
        """
        with self.lock:
            for symbol, data in self.market_data.items():
                # 模拟随机价格波动
                price_change = random.normalvariate(0, 0.001) * data["price"]
                
                # 应用价格变动
                data["price"] = max(0.01, data["price"] + price_change)
                data["bid"] = max(0.01, data["price"] - random.uniform(0, 0.5))
                data["ask"] = data["price"] + random.uniform(0, 0.5)
                
                # 应用价格取整（按照最小价格变动单位）
                tick_size = self.config["tick_size"]
                data["price"] = round(data["price"] / tick_size) * tick_size
                data["bid"] = round(data["bid"] / tick_size) * tick_size
                data["ask"] = round(data["ask"] / tick_size) * tick_size
                
                # 更新时间戳
                data["timestamp"] = datetime.now().isoformat()
    
    def _update_account_info(self) -> None:
        """
        更新账户信息
        """
        with self.lock:
            # 计算持仓价值和未实现盈亏
            positions_value = 0.0
            unrealized_pnl = 0.0
            
            for position in self.positions.values():
                if position["quantity"] != 0:
                    symbol_price = self.market_data[position["symbol"]]["price"]
                    position_value = abs(position["quantity"]) * symbol_price
                    positions_value += position_value
                    
                    # 更新持仓市场价格和未实现盈亏
                    position["market_price"] = symbol_price
                    if position["quantity"] > 0:
                        position["unrealized_pnl"] = (symbol_price - position["average_price"]) * position["quantity"]
                    else:
                        position["unrealized_pnl"] = (position["average_price"] - symbol_price) * abs(position["quantity"])
                    
                    unrealized_pnl += position["unrealized_pnl"]
            
            # 更新账户信息
            self.account["equity"] = self.account["balance"] + unrealized_pnl
            self.account["margin_used"] = positions_value * 0.1  # 模拟10%保证金要求
            self.account["available"] = self.account["equity"] - self.account["margin_used"]
            self.account["timestamp"] = datetime.now().isoformat()
            
            # 发送账户更新通知
            self._handle_account_update(self.account)
    
    def _calculate_execution_price(self, order: Order) -> float:
        """
        计算订单的执行价格（包含滑点）
        
        Args:
            order: 订单对象
            
        Returns:
            执行价格
        """
        symbol_data = self.market_data[order.symbol]
        
        # 基础价格
        if order.direction == "BUY":
            base_price = symbol_data["ask"]
        else:
            base_price = symbol_data["bid"]
        
        # 应用滑点
        slippage_model = self.config["slippage_model"]
        slippage_value = self.config["slippage_value"]
        
        if slippage_model == "fixed":
            # 固定滑点（绝对值）
            if order.direction == "BUY":
                price = base_price + slippage_value
            else:
                price = base_price - slippage_value
        elif slippage_model == "proportional":
            # 比例滑点
            if order.direction == "BUY":
                price = base_price * (1 + slippage_value)
            else:
                price = base_price * (1 - slippage_value)
        elif slippage_model == "random":
            # 随机滑点
            random_factor = random.uniform(0, slippage_value)
            if order.direction == "BUY":
                price = base_price * (1 + random_factor)
            else:
                price = base_price * (1 - random_factor)
        else:
            # 默认无滑点
            price = base_price
        
        # 应用价格影响因子（大订单会影响价格）
        impact_factor = self.config["price_impact_factor"]
        if impact_factor > 0:
            normalized_quantity = order.quantity / symbol_data["volume"]
            price_impact = normalized_quantity * impact_factor * base_price
            
            if order.direction == "BUY":
                price += price_impact
            else:
                price -= price_impact
        
        # 应用价格取整（按照最小价格变动单位）
        tick_size = self.config["tick_size"]
        price = round(price / tick_size) * tick_size
        
        return max(0.01, price)  # 确保价格为正 