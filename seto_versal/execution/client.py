#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
执行客户端模块，定义了与交易所和经纪商通信的接口。
"""

import os
import time
import logging
import json
import enum
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Type, Set
from abc import ABC, abstractmethod


class OrderType(enum.Enum):
    """订单类型枚举"""
    MARKET = "MARKET"  # 市价单
    LIMIT = "LIMIT"    # 限价单
    STOP = "STOP"      # 止损单
    STOP_LIMIT = "STOP_LIMIT"  # 止损限价单
    TRAILING_STOP = "TRAILING_STOP"  # 追踪止损单
    
    def __str__(self):
        return self.value


class OrderStatus(enum.Enum):
    """订单状态枚举"""
    CREATED = "CREATED"        # 已创建
    SUBMITTED = "SUBMITTED"    # 已提交
    ACCEPTED = "ACCEPTED"      # 已接受
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # 部分成交
    FILLED = "FILLED"          # 全部成交
    CANCELED = "CANCELED"      # 已取消
    REJECTED = "REJECTED"      # 已拒绝
    EXPIRED = "EXPIRED"        # 已过期
    ERROR = "ERROR"            # 出错
    
    def __str__(self):
        return self.value
    
    @property
    def is_active(self) -> bool:
        """
        判断订单是否处于活跃状态
        
        Returns:
            订单是否活跃
        """
        return self in [
            OrderStatus.CREATED,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED
        ]
    
    @property
    def is_complete(self) -> bool:
        """
        判断订单是否已完成
        
        Returns:
            订单是否完成
        """
        return self in [
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.ERROR
        ]


class Order:
    """
    订单类，表示一个交易订单
    """
    
    def __init__(self, 
                 symbol: str, 
                 order_type: OrderType, 
                 direction: str, 
                 quantity: float, 
                 price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 order_id: Optional[str] = None,
                 client_order_id: Optional[str] = None,
                 strategy_id: Optional[str] = None,
                 timestamp: Optional[datetime] = None,
                 tags: Optional[Dict[str, Any]] = None):
        """
        初始化订单
        
        Args:
            symbol: 交易品种代码
            order_type: 订单类型
            direction: 交易方向，"BUY"或"SELL"
            quantity: 交易数量
            price: 交易价格，对于市价单可以为None
            stop_price: 止损价格，对于止损单和止损限价单必须提供
            order_id: 订单ID，如果为None则自动生成
            client_order_id: 客户端订单ID，用于跟踪订单
            strategy_id: 策略ID
            timestamp: 订单创建时间
            tags: 订单标签
        """
        self.symbol = symbol
        self.order_type = order_type if isinstance(order_type, OrderType) else OrderType(order_type)
        self.direction = direction.upper()
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.order_id = order_id or str(uuid.uuid4())
        self.client_order_id = client_order_id or f"{symbol}_{int(time.time())}"
        self.strategy_id = strategy_id
        self.timestamp = timestamp or datetime.now()
        self.tags = tags or {}
        
        # 订单状态
        self.status = OrderStatus.CREATED
        
        # 成交信息
        self.filled_quantity = 0.0
        self.average_fill_price = 0.0
        self.commission = 0.0
        self.fills: List[Dict[str, Any]] = []
        
        # 订单更新历史
        self.status_updates: List[Dict[str, Any]] = [{
            "timestamp": self.timestamp,
            "status": self.status.value
        }]
        
        # 订单有效期
        self.time_in_force = "GTC"  # Good Till Canceled
        self.expire_time: Optional[datetime] = None
        
        # 错误信息
        self.error_message: Optional[str] = None
    
    def update_status(self, new_status: OrderStatus, message: Optional[str] = None) -> None:
        """
        更新订单状态
        
        Args:
            new_status: 新状态
            message: 状态更新消息
        """
        if isinstance(new_status, str):
            new_status = OrderStatus(new_status)
        
        old_status = self.status
        self.status = new_status
        
        # 记录状态更新
        update = {
            "timestamp": datetime.now(),
            "old_status": old_status.value,
            "new_status": new_status.value
        }
        
        if message:
            update["message"] = message
            if new_status == OrderStatus.ERROR or new_status == OrderStatus.REJECTED:
                self.error_message = message
        
        self.status_updates.append(update)
    
    def add_fill(self, quantity: float, price: float, timestamp: Optional[datetime] = None, commission: float = 0.0) -> None:
        """
        添加成交记录
        
        Args:
            quantity: 成交数量
            price: 成交价格
            timestamp: 成交时间
            commission: 手续费
        """
        fill_timestamp = timestamp or datetime.now()
        
        # 创建成交记录
        fill = {
            "timestamp": fill_timestamp,
            "quantity": quantity,
            "price": price,
            "commission": commission
        }
        
        self.fills.append(fill)
        
        # 更新订单成交状态
        self.filled_quantity += quantity
        total_cost = sum(f["price"] * f["quantity"] for f in self.fills)
        self.average_fill_price = total_cost / self.filled_quantity if self.filled_quantity > 0 else 0.0
        self.commission += commission
        
        # 检查是否完全成交
        if abs(self.filled_quantity - self.quantity) < 0.000001:
            self.update_status(OrderStatus.FILLED)
        elif self.filled_quantity > 0:
            self.update_status(OrderStatus.PARTIALLY_FILLED)
    
    def calculate_notional(self) -> float:
        """
        计算订单名义价值
        
        Returns:
            订单名义价值
        """
        if self.price is not None:
            return self.quantity * self.price
        elif self.filled_quantity > 0:
            return self.filled_quantity * self.average_fill_price
        return 0.0
    
    def cancel(self, message: Optional[str] = None) -> bool:
        """
        取消订单
        
        Args:
            message: 取消原因
            
        Returns:
            是否成功取消
        """
        if self.status.is_complete:
            return False
        
        self.update_status(OrderStatus.CANCELED, message)
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将订单转换为字典
        
        Returns:
            订单信息的字典表示
        """
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "order_type": self.order_type.value,
            "direction": self.direction,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "strategy_id": self.strategy_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "commission": self.commission,
            "fills": self.fills,
            "status_updates": self.status_updates,
            "time_in_force": self.time_in_force,
            "expire_time": self.expire_time.isoformat() if self.expire_time else None,
            "error_message": self.error_message,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """
        从字典创建订单
        
        Args:
            data: 订单信息字典
            
        Returns:
            创建的订单对象
        """
        timestamp = datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"]
        
        order = cls(
            symbol=data["symbol"],
            order_type=data["order_type"],
            direction=data["direction"],
            quantity=data["quantity"],
            price=data.get("price"),
            stop_price=data.get("stop_price"),
            order_id=data.get("order_id"),
            client_order_id=data.get("client_order_id"),
            strategy_id=data.get("strategy_id"),
            timestamp=timestamp,
            tags=data.get("tags", {})
        )
        
        # 恢复订单状态
        order.status = OrderStatus(data["status"])
        order.filled_quantity = data["filled_quantity"]
        order.average_fill_price = data["average_fill_price"]
        order.commission = data["commission"]
        order.fills = data["fills"]
        order.status_updates = data["status_updates"]
        order.time_in_force = data["time_in_force"]
        if data.get("expire_time"):
            order.expire_time = datetime.fromisoformat(data["expire_time"]) if isinstance(data["expire_time"], str) else data["expire_time"]
        order.error_message = data.get("error_message")
        
        return order
    
    def __str__(self) -> str:
        """
        返回订单的字符串表示
        
        Returns:
            订单信息字符串
        """
        price_str = f", 价格: {self.price}" if self.price is not None else ""
        stop_str = f", 止损价: {self.stop_price}" if self.stop_price is not None else ""
        fill_str = f", 已成交: {self.filled_quantity}/{self.quantity}" if self.filled_quantity > 0 else ""
        
        return f"订单[{self.order_id}]: {self.symbol}, {self.direction}, {self.order_type.value}, 数量: {self.quantity}{price_str}{stop_str}{fill_str}, 状态: {self.status.value}" 


class OrderManager:
    """
    订单管理器，负责管理订单的生命周期
    """
    
    def __init__(self):
        """
        初始化订单管理器
        """
        self.logger = logging.getLogger(__name__)
        self.orders: Dict[str, Order] = {}  # 订单ID到订单对象的映射
        self.client_order_map: Dict[str, str] = {}  # 客户端订单ID到订单ID的映射
        self.strategy_orders: Dict[str, List[str]] = {}  # 策略ID到订单ID列表的映射
        self.active_orders: List[str] = []  # 活跃订单ID列表
        self.filled_orders: List[str] = []  # 已成交订单ID列表
        self.canceled_orders: List[str] = []  # 已取消订单ID列表
        self.rejected_orders: List[str] = []  # 已拒绝订单ID列表
    
    def add_order(self, order: Order) -> str:
        """
        添加订单
        
        Args:
            order: 订单对象
            
        Returns:
            订单ID
        """
        if order.order_id in self.orders:
            self.logger.warning(f"订单ID {order.order_id} 已存在，将被覆盖")
        
        self.orders[order.order_id] = order
        self.client_order_map[order.client_order_id] = order.order_id
        
        # 添加到策略订单映射
        if order.strategy_id:
            if order.strategy_id not in self.strategy_orders:
                self.strategy_orders[order.strategy_id] = []
            self.strategy_orders[order.strategy_id].append(order.order_id)
        
        # 根据订单状态添加到相应列表
        if order.status.is_active:
            self.active_orders.append(order.order_id)
        elif order.status == OrderStatus.FILLED:
            self.filled_orders.append(order.order_id)
        elif order.status == OrderStatus.CANCELED:
            self.canceled_orders.append(order.order_id)
        elif order.status in [OrderStatus.REJECTED, OrderStatus.ERROR]:
            self.rejected_orders.append(order.order_id)
        
        self.logger.info(f"添加订单: {order}")
        
        return order.order_id
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        获取订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            订单对象，如果不存在则返回None
        """
        return self.orders.get(order_id)
    
    def get_order_by_client_id(self, client_order_id: str) -> Optional[Order]:
        """
        根据客户端订单ID获取订单
        
        Args:
            client_order_id: 客户端订单ID
            
        Returns:
            订单对象，如果不存在则返回None
        """
        order_id = self.client_order_map.get(client_order_id)
        if order_id:
            return self.orders.get(order_id)
        return None
    
    def update_order(self, order_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新订单
        
        Args:
            order_id: 订单ID
            updates: 更新的字段和值
            
        Returns:
            更新是否成功
        """
        order = self.get_order(order_id)
        if not order:
            self.logger.warning(f"订单ID {order_id} 不存在，无法更新")
            return False
        
        # 更新订单字段
        for key, value in updates.items():
            if key == "status":
                # 特殊处理状态更新
                old_status = order.status
                new_status = OrderStatus(value) if isinstance(value, str) else value
                
                # 更新状态列表
                if old_status.is_active and not new_status.is_active:
                    if order.order_id in self.active_orders:
                        self.active_orders.remove(order.order_id)
                
                if new_status == OrderStatus.FILLED:
                    if order.order_id not in self.filled_orders:
                        self.filled_orders.append(order.order_id)
                elif new_status == OrderStatus.CANCELED:
                    if order.order_id not in self.canceled_orders:
                        self.canceled_orders.append(order.order_id)
                elif new_status in [OrderStatus.REJECTED, OrderStatus.ERROR]:
                    if order.order_id not in self.rejected_orders:
                        self.rejected_orders.append(order.order_id)
                
                # 调用订单的状态更新方法
                order.update_status(new_status, updates.get("message"))
            elif key == "fill":
                # 特殊处理成交更新
                fill_data = value
                order.add_fill(
                    quantity=fill_data["quantity"],
                    price=fill_data["price"],
                    timestamp=fill_data.get("timestamp"),
                    commission=fill_data.get("commission", 0.0)
                )
            else:
                # 普通字段更新
                setattr(order, key, value)
        
        self.logger.info(f"更新订单: {order}")
        
        return True
    
    def cancel_order(self, order_id: str, reason: Optional[str] = None) -> bool:
        """
        取消订单
        
        Args:
            order_id: 订单ID
            reason: 取消原因
            
        Returns:
            取消是否成功
        """
        order = self.get_order(order_id)
        if not order:
            self.logger.warning(f"订单ID {order_id} 不存在，无法取消")
            return False
        
        if order.status.is_complete:
            self.logger.warning(f"订单 {order_id} 已完成，无法取消")
            return False
        
        success = order.cancel(reason)
        
        if success:
            if order.order_id in self.active_orders:
                self.active_orders.remove(order.order_id)
            if order.order_id not in self.canceled_orders:
                self.canceled_orders.append(order.order_id)
                
            self.logger.info(f"订单已取消: {order}")
        
        return success
    
    def get_strategy_orders(self, strategy_id: str) -> List[Order]:
        """
        获取策略的订单
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            订单列表
        """
        order_ids = self.strategy_orders.get(strategy_id, [])
        return [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
    
    def get_active_orders(self) -> List[Order]:
        """
        获取所有活跃订单
        
        Returns:
            活跃订单列表
        """
        return [self.orders[order_id] for order_id in self.active_orders if order_id in self.orders]
    
    def get_filled_orders(self) -> List[Order]:
        """
        获取所有已成交订单
        
        Returns:
            已成交订单列表
        """
        return [self.orders[order_id] for order_id in self.filled_orders if order_id in self.orders]
    
    def get_canceled_orders(self) -> List[Order]:
        """
        获取所有已取消订单
        
        Returns:
            已取消订单列表
        """
        return [self.orders[order_id] for order_id in self.canceled_orders if order_id in self.orders]
    
    def get_rejected_orders(self) -> List[Order]:
        """
        获取所有已拒绝订单
        
        Returns:
            已拒绝订单列表
        """
        return [self.orders[order_id] for order_id in self.rejected_orders if order_id in self.orders]
    
    def clear_completed_orders(self, days_to_keep: int = 30) -> int:
        """
        清除已完成的订单
        
        Args:
            days_to_keep: 保留最近多少天的订单
            
        Returns:
            清除的订单数量
        """
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        orders_to_remove = []
        
        for order_id, order in self.orders.items():
            if order.status.is_complete and order.timestamp < cutoff_time:
                orders_to_remove.append(order_id)
        
        for order_id in orders_to_remove:
            order = self.orders[order_id]
            
            # 从各种列表中移除
            if order_id in self.filled_orders:
                self.filled_orders.remove(order_id)
            if order_id in self.canceled_orders:
                self.canceled_orders.remove(order_id)
            if order_id in self.rejected_orders:
                self.rejected_orders.remove(order_id)
            
            # 从客户端订单映射中移除
            if order.client_order_id in self.client_order_map:
                del self.client_order_map[order.client_order_id]
            
            # 从策略订单映射中移除
            if order.strategy_id and order.strategy_id in self.strategy_orders:
                if order_id in self.strategy_orders[order.strategy_id]:
                    self.strategy_orders[order.strategy_id].remove(order_id)
            
            # 从订单字典中移除
            del self.orders[order_id]
        
        self.logger.info(f"清除了 {len(orders_to_remove)} 个已完成订单")
        
        return len(orders_to_remove)
    
    def save_orders(self, file_path: str) -> bool:
        """
        将订单保存到文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            保存是否成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 转换订单为字典
            orders_data = {order_id: order.to_dict() for order_id, order in self.orders.items()}
            
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(orders_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"订单已保存到文件: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存订单失败: {str(e)}")
            return False
    
    def load_orders(self, file_path: str) -> bool:
        """
        从文件加载订单
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载是否成功
        """
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"文件不存在: {file_path}")
                return False
            
            # 从文件加载订单数据
            with open(file_path, 'r', encoding='utf-8') as f:
                orders_data = json.load(f)
            
            # 清空当前订单
            self.orders = {}
            self.client_order_map = {}
            self.strategy_orders = {}
            self.active_orders = []
            self.filled_orders = []
            self.canceled_orders = []
            self.rejected_orders = []
            
            # 重新添加订单
            for order_id, order_data in orders_data.items():
                order = Order.from_dict(order_data)
                self.add_order(order)
            
            self.logger.info(f"从文件加载了 {len(self.orders)} 个订单: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"加载订单失败: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取订单统计信息
        
        Returns:
            订单统计信息
        """
        return {
            "total_orders": len(self.orders),
            "active_orders": len(self.active_orders),
            "filled_orders": len(self.filled_orders),
            "canceled_orders": len(self.canceled_orders),
            "rejected_orders": len(self.rejected_orders),
            "strategies": len(self.strategy_orders)
        }


class ExecutionClient(ABC):
    """
    执行客户端抽象基类，定义了与交易所和经纪商通信的接口
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化执行客户端
        
        Args:
            name: 客户端名称
            config: 配置参数
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"execution.{name}")
        self.logger.info(f"初始化执行客户端: {name}")
        
        # 订单管理器
        self.order_manager = OrderManager()
        
        # 客户端状态
        self.is_connected = False
        self.last_error: Optional[str] = None
        
        # 回调函数
        self.on_order_update: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self.on_trade_update: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self.on_position_update: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_account_update: Optional[Callable[[Dict[str, Any]], None]] = None
    
    @abstractmethod
    def connect(self) -> bool:
        """
        连接到交易所或经纪商
        
        Returns:
            连接是否成功
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开与交易所或经纪商的连接
        
        Returns:
            断开连接是否成功
        """
        pass
    
    @abstractmethod
    def is_authenticated(self) -> bool:
        """
        检查是否已认证
        
        Returns:
            是否已认证
        """
        pass
    
    @abstractmethod
    def place_order(self, order: Order) -> Tuple[bool, Optional[str]]:
        """
        下单
        
        Args:
            order: 订单对象
            
        Returns:
            元组(是否成功, 失败原因)
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> Tuple[bool, Optional[str]]:
        """
        取消订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            元组(是否成功, 失败原因)
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Tuple[Optional[OrderStatus], Dict[str, Any]]:
        """
        获取订单状态
        
        Args:
            order_id: 订单ID
            
        Returns:
            元组(订单状态, 额外信息)
        """
        pass
    
    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """
        获取未成交订单
        
        Returns:
            未成交订单列表
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        获取持仓
        
        Returns:
            持仓信息映射（品种到持仓信息）
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        获取账户信息
        
        Returns:
            账户信息
        """
        pass
    
    def register_callbacks(self, 
                         on_order_update: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                         on_trade_update: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                         on_position_update: Optional[Callable[[Dict[str, Any]], None]] = None,
                         on_account_update: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        """
        注册回调函数
        
        Args:
            on_order_update: 订单更新回调
            on_trade_update: 成交更新回调
            on_position_update: 持仓更新回调
            on_account_update: 账户更新回调
        """
        if on_order_update:
            self.on_order_update = on_order_update
        if on_trade_update:
            self.on_trade_update = on_trade_update
        if on_position_update:
            self.on_position_update = on_position_update
        if on_account_update:
            self.on_account_update = on_account_update
    
    def _handle_order_update(self, order_id: str, updates: Dict[str, Any]) -> None:
        """
        处理订单更新
        
        Args:
            order_id: 订单ID
            updates: 更新的字段和值
        """
        # 更新订单管理器中的订单
        self.order_manager.update_order(order_id, updates)
        
        # 调用订单更新回调
        if self.on_order_update:
            self.on_order_update(order_id, updates)
    
    def _handle_trade_update(self, order_id: str, trade_data: Dict[str, Any]) -> None:
        """
        处理成交更新
        
        Args:
            order_id: 订单ID
            trade_data: 成交数据
        """
        # 更新订单管理器中的订单
        updates = {"fill": trade_data}
        self.order_manager.update_order(order_id, updates)
        
        # 调用成交更新回调
        if self.on_trade_update:
            self.on_trade_update(order_id, trade_data)
    
    def _handle_position_update(self, position_data: Dict[str, Any]) -> None:
        """
        处理持仓更新
        
        Args:
            position_data: 持仓数据
        """
        # 调用持仓更新回调
        if self.on_position_update:
            self.on_position_update(position_data)
    
    def _handle_account_update(self, account_data: Dict[str, Any]) -> None:
        """
        处理账户更新
        
        Args:
            account_data: 账户数据
        """
        # 调用账户更新回调
        if self.on_account_update:
            self.on_account_update(account_data)
    
    def save_state(self, directory: str = "data/execution") -> bool:
        """
        保存客户端状态
        
        Args:
            directory: 保存目录
            
        Returns:
            保存是否成功
        """
        os.makedirs(directory, exist_ok=True)
        orders_file = os.path.join(directory, f"{self.name}_orders.json")
        return self.order_manager.save_orders(orders_file)
    
    def load_state(self, directory: str = "data/execution") -> bool:
        """
        加载客户端状态
        
        Args:
            directory: 加载目录
            
        Returns:
            加载是否成功
        """
        orders_file = os.path.join(directory, f"{self.name}_orders.json")
        if os.path.exists(orders_file):
            return self.order_manager.load_orders(orders_file)
        return False 