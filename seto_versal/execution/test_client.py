#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
执行模块的测试文件
"""

import unittest
import time
import uuid
import json
import tempfile
import os
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from seto_versal.execution.client import (
    OrderType, OrderStatus, Order, OrderManager, ExecutionClient
)
from seto_versal.execution.simulator import SimulatedExecutionClient


class TestOrderStatus(unittest.TestCase):
    """测试订单状态枚举类"""
    
    def test_order_status_properties(self):
        """测试订单状态属性"""
        # 活跃状态
        self.assertTrue(OrderStatus.CREATED.is_active)
        self.assertTrue(OrderStatus.SUBMITTED.is_active)
        self.assertTrue(OrderStatus.ACCEPTED.is_active)
        self.assertTrue(OrderStatus.PARTIALLY_FILLED.is_active)
        
        # 完成状态
        self.assertTrue(OrderStatus.FILLED.is_complete)
        self.assertTrue(OrderStatus.CANCELED.is_complete)
        self.assertTrue(OrderStatus.REJECTED.is_complete)
        self.assertTrue(OrderStatus.EXPIRED.is_complete)
        self.assertTrue(OrderStatus.ERROR.is_complete)
        
        # 非活跃状态
        self.assertFalse(OrderStatus.FILLED.is_active)
        self.assertFalse(OrderStatus.CANCELED.is_active)
        
        # 非完成状态
        self.assertFalse(OrderStatus.CREATED.is_complete)
        self.assertFalse(OrderStatus.ACCEPTED.is_complete)


class TestOrder(unittest.TestCase):
    """测试订单类"""
    
    def setUp(self):
        """设置测试环境"""
        self.order_id = str(uuid.uuid4())
        self.client_order_id = "test_order_123"
        self.strategy_id = "test_strategy"
        self.symbol = "AAPL"
        self.quantity = 100
        self.price = 150.5
        self.stop_price = 148.0
        self.direction = "BUY"
        self.order_type = OrderType.LIMIT
        
        self.order = Order(
            order_id=self.order_id,
            client_order_id=self.client_order_id,
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            quantity=self.quantity,
            price=self.price,
            stop_price=self.stop_price,
            direction=self.direction,
            order_type=self.order_type
        )
    
    def test_order_creation(self):
        """测试创建订单"""
        self.assertEqual(self.order.order_id, self.order_id)
        self.assertEqual(self.order.client_order_id, self.client_order_id)
        self.assertEqual(self.order.strategy_id, self.strategy_id)
        self.assertEqual(self.order.symbol, self.symbol)
        self.assertEqual(self.order.quantity, self.quantity)
        self.assertEqual(self.order.price, self.price)
        self.assertEqual(self.order.stop_price, self.stop_price)
        self.assertEqual(self.order.direction, self.direction)
        self.assertEqual(self.order.order_type, self.order_type)
        self.assertEqual(self.order.status, OrderStatus.CREATED)
        self.assertEqual(self.order.filled_quantity, 0.0)
        self.assertEqual(self.order.remaining_quantity, self.quantity)
        self.assertEqual(self.order.average_fill_price, 0.0)
        self.assertEqual(self.order.commission, 0.0)
        self.assertEqual(len(self.order.fills), 0)
        self.assertEqual(len(self.order.status_updates), 1)
    
    def test_update_status(self):
        """测试更新订单状态"""
        reason = "测试状态更新"
        self.order.update_status(OrderStatus.SUBMITTED, reason)
        
        self.assertEqual(self.order.status, OrderStatus.SUBMITTED)
        self.assertEqual(len(self.order.status_updates), 2)
        
        last_update = self.order.status_updates[-1]
        self.assertEqual(last_update["status"], OrderStatus.SUBMITTED)
        self.assertEqual(last_update["reason"], reason)
        self.assertIsInstance(last_update["timestamp"], str)
    
    def test_add_fill(self):
        """测试添加成交记录"""
        timestamp = datetime.now()
        quantity = 50
        price = 151.0
        commission = 2.5
        
        self.order.add_fill(quantity, price, timestamp, commission)
        
        self.assertEqual(len(self.order.fills), 1)
        self.assertEqual(self.order.filled_quantity, quantity)
        self.assertEqual(self.order.remaining_quantity, self.quantity - quantity)
        self.assertEqual(self.order.average_fill_price, price)
        self.assertEqual(self.order.commission, commission)
        self.assertEqual(self.order.status, OrderStatus.PARTIALLY_FILLED)
        
        # 添加第二个成交记录
        timestamp2 = datetime.now() + timedelta(seconds=10)
        quantity2 = 50
        price2 = 151.5
        commission2 = 2.7
        
        self.order.add_fill(quantity2, price2, timestamp2, commission2)
        
        self.assertEqual(len(self.order.fills), 2)
        self.assertEqual(self.order.filled_quantity, quantity + quantity2)
        self.assertEqual(self.order.remaining_quantity, 0)
        
        # 验证平均成交价格
        expected_avg_price = (price * quantity + price2 * quantity2) / (quantity + quantity2)
        self.assertAlmostEqual(self.order.average_fill_price, expected_avg_price)
        
        self.assertEqual(self.order.commission, commission + commission2)
        self.assertEqual(self.order.status, OrderStatus.FILLED)
    
    def test_notional_value(self):
        """测试订单名义价值计算"""
        # 未成交时的名义价值
        expected_notional = self.quantity * self.price
        self.assertEqual(self.order.notional_value, expected_notional)
        
        # 部分成交后的名义价值
        quantity = 40
        price = 152.0
        commission = 1.5
        timestamp = datetime.now()
        
        self.order.add_fill(quantity, price, timestamp, commission)
        
        expected_filled_notional = quantity * price
        expected_remaining_notional = (self.quantity - quantity) * self.price
        expected_total_notional = expected_filled_notional + expected_remaining_notional
        
        self.assertEqual(self.order.filled_notional, expected_filled_notional)
        self.assertEqual(self.order.notional_value, expected_total_notional)
    
    def test_to_dict(self):
        """测试转换为字典"""
        order_dict = self.order.to_dict()
        
        self.assertEqual(order_dict["order_id"], self.order_id)
        self.assertEqual(order_dict["client_order_id"], self.client_order_id)
        self.assertEqual(order_dict["strategy_id"], self.strategy_id)
        self.assertEqual(order_dict["symbol"], self.symbol)
        self.assertEqual(order_dict["quantity"], self.quantity)
        self.assertEqual(order_dict["price"], self.price)
        self.assertEqual(order_dict["stop_price"], self.stop_price)
        self.assertEqual(order_dict["direction"], self.direction)
        self.assertEqual(order_dict["order_type"], self.order_type.name)
        self.assertEqual(order_dict["status"], OrderStatus.CREATED.name)
        self.assertEqual(order_dict["filled_quantity"], 0.0)
        self.assertEqual(order_dict["average_fill_price"], 0.0)
        self.assertEqual(order_dict["commission"], 0.0)
        self.assertEqual(len(order_dict["fills"]), 0)
        self.assertEqual(len(order_dict["status_updates"]), 1)
    
    def test_from_dict(self):
        """测试从字典创建订单"""
        # 先转换为字典
        order_dict = self.order.to_dict()
        
        # 再从字典创建订单
        new_order = Order.from_dict(order_dict)
        
        self.assertEqual(new_order.order_id, self.order_id)
        self.assertEqual(new_order.client_order_id, self.client_order_id)
        self.assertEqual(new_order.strategy_id, self.strategy_id)
        self.assertEqual(new_order.symbol, self.symbol)
        self.assertEqual(new_order.quantity, self.quantity)
        self.assertEqual(new_order.price, self.price)
        self.assertEqual(new_order.stop_price, self.stop_price)
        self.assertEqual(new_order.direction, self.direction)
        self.assertEqual(new_order.order_type, self.order_type)
        self.assertEqual(new_order.status, OrderStatus.CREATED)
        self.assertEqual(new_order.filled_quantity, 0.0)
        self.assertEqual(new_order.average_fill_price, 0.0)
        self.assertEqual(new_order.commission, 0.0)
        self.assertEqual(len(new_order.fills), 0)
        self.assertEqual(len(new_order.status_updates), 1)


class TestOrderManager(unittest.TestCase):
    """测试订单管理器类"""
    
    def setUp(self):
        """设置测试环境"""
        self.order_manager = OrderManager()
        
        # 创建一些测试订单
        self.orders = []
        for i in range(5):
            order = Order(
                order_id=f"order_{i}",
                client_order_id=f"client_order_{i}",
                strategy_id="test_strategy",
                symbol="AAPL",
                quantity=100,
                price=150.0 + i,
                direction="BUY",
                order_type=OrderType.LIMIT
            )
            self.orders.append(order)
    
    def test_add_order(self):
        """测试添加订单"""
        for order in self.orders:
            self.order_manager.add_order(order)
        
        self.assertEqual(len(self.order_manager.get_all_orders()), len(self.orders))
        
        # 检查查询功能
        for i, order in enumerate(self.orders):
            retrieved_order = self.order_manager.get_order(f"order_{i}")
            self.assertEqual(retrieved_order.order_id, order.order_id)
            
            client_retrieved_order = self.order_manager.get_order_by_client_id(f"client_order_{i}")
            self.assertEqual(client_retrieved_order.client_order_id, order.client_order_id)
    
    def test_update_order(self):
        """测试更新订单"""
        order = self.orders[0]
        self.order_manager.add_order(order)
        
        # 更新状态
        updates = {"status": OrderStatus.SUBMITTED, "reason": "测试更新"}
        self.order_manager.update_order(order.order_id, updates)
        
        updated_order = self.order_manager.get_order(order.order_id)
        self.assertEqual(updated_order.status, OrderStatus.SUBMITTED)
        self.assertEqual(updated_order.status_updates[-1]["reason"], "测试更新")
        
        # 更新价格
        updates = {"price": 160.0}
        self.order_manager.update_order(order.order_id, updates)
        
        updated_order = self.order_manager.get_order(order.order_id)
        self.assertEqual(updated_order.price, 160.0)
    
    def test_cancel_order(self):
        """测试取消订单"""
        order = self.orders[0]
        self.order_manager.add_order(order)
        
        # 取消订单
        reason = "测试取消"
        success = self.order_manager.cancel_order(order.order_id, reason)
        
        self.assertTrue(success)
        canceled_order = self.order_manager.get_order(order.order_id)
        self.assertEqual(canceled_order.status, OrderStatus.CANCELED)
        self.assertEqual(canceled_order.status_updates[-1]["reason"], reason)
        
        # 尝试取消已完成的订单
        success = self.order_manager.cancel_order(order.order_id, "再次取消")
        self.assertFalse(success)
    
    def test_get_orders_by_status(self):
        """测试按状态获取订单"""
        for order in self.orders:
            self.order_manager.add_order(order)
        
        # 所有订单默认都是CREATED状态
        created_orders = self.order_manager.get_orders_by_status(OrderStatus.CREATED)
        self.assertEqual(len(created_orders), len(self.orders))
        
        # 将部分订单状态更新为SUBMITTED
        for i in range(2):
            self.order_manager.update_order(f"order_{i}", {"status": OrderStatus.SUBMITTED})
        
        submitted_orders = self.order_manager.get_orders_by_status(OrderStatus.SUBMITTED)
        self.assertEqual(len(submitted_orders), 2)
        
        # 将一个订单状态更新为FILLED
        self.order_manager.update_order("order_0", {"status": OrderStatus.FILLED})
        
        filled_orders = self.order_manager.get_orders_by_status(OrderStatus.FILLED)
        self.assertEqual(len(filled_orders), 1)
    
    def test_get_active_orders(self):
        """测试获取活跃订单"""
        for order in self.orders:
            self.order_manager.add_order(order)
        
        # 所有订单默认都是活跃的
        active_orders = self.order_manager.get_active_orders()
        self.assertEqual(len(active_orders), len(self.orders))
        
        # 将部分订单状态更新为已完成
        self.order_manager.update_order("order_0", {"status": OrderStatus.FILLED})
        self.order_manager.update_order("order_1", {"status": OrderStatus.CANCELED})
        
        active_orders = self.order_manager.get_active_orders()
        self.assertEqual(len(active_orders), len(self.orders) - 2)
    
    def test_get_orders_by_strategy(self):
        """测试按策略获取订单"""
        # 添加默认策略的订单
        for order in self.orders:
            self.order_manager.add_order(order)
        
        # 添加另一个策略的订单
        other_strategy_order = Order(
            order_id="other_strategy_order",
            client_order_id="other_client_order",
            strategy_id="other_strategy",
            symbol="MSFT",
            quantity=200,
            price=250.0,
            direction="SELL",
            order_type=OrderType.MARKET
        )
        self.order_manager.add_order(other_strategy_order)
        
        # 验证按策略查询
        default_strategy_orders = self.order_manager.get_orders_by_strategy("test_strategy")
        self.assertEqual(len(default_strategy_orders), len(self.orders))
        
        other_strategy_orders = self.order_manager.get_orders_by_strategy("other_strategy")
        self.assertEqual(len(other_strategy_orders), 1)
        self.assertEqual(other_strategy_orders[0].order_id, "other_strategy_order")
    
    def test_clear_completed_orders(self):
        """测试清理已完成订单"""
        for order in self.orders:
            self.order_manager.add_order(order)
        
        # 将部分订单状态更新为已完成
        self.order_manager.update_order("order_0", {"status": OrderStatus.FILLED})
        self.order_manager.update_order("order_1", {"status": OrderStatus.CANCELED})
        self.order_manager.update_order("order_2", {"status": OrderStatus.REJECTED})
        
        # 设置完成时间
        for i in range(3):
            order = self.order_manager.get_order(f"order_{i}")
            order.status_updates[-1]["timestamp"] = (datetime.now() - timedelta(hours=25)).isoformat()
        
        # 清理超过24小时的已完成订单
        cleared_count = self.order_manager.clear_completed_orders(retention_hours=24)
        self.assertEqual(cleared_count, 3)
        
        # 验证剩余订单数量
        all_orders = self.order_manager.get_all_orders()
        self.assertEqual(len(all_orders), len(self.orders) - 3)
    
    def test_save_and_load_orders(self):
        """测试保存和加载订单"""
        # 添加订单
        for order in self.orders:
            self.order_manager.add_order(order)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file_path = tmp.name
        
        try:
            # 保存订单
            self.order_manager.save_orders(file_path)
            
            # 创建新的订单管理器并加载订单
            new_manager = OrderManager()
            loaded_count = new_manager.load_orders(file_path)
            
            # 验证加载结果
            self.assertEqual(loaded_count, len(self.orders))
            self.assertEqual(len(new_manager.get_all_orders()), len(self.orders))
            
            # 验证订单信息完整性
            for i in range(len(self.orders)):
                original_order = self.order_manager.get_order(f"order_{i}")
                loaded_order = new_manager.get_order(f"order_{i}")
                
                self.assertEqual(loaded_order.order_id, original_order.order_id)
                self.assertEqual(loaded_order.client_order_id, original_order.client_order_id)
                self.assertEqual(loaded_order.strategy_id, original_order.strategy_id)
                self.assertEqual(loaded_order.symbol, original_order.symbol)
                self.assertEqual(loaded_order.quantity, original_order.quantity)
                self.assertEqual(loaded_order.price, original_order.price)
                self.assertEqual(loaded_order.direction, original_order.direction)
                self.assertEqual(loaded_order.order_type, original_order.order_type)
                self.assertEqual(loaded_order.status, original_order.status)
        finally:
            # 清理临时文件
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def test_get_stats(self):
        """测试获取订单统计信息"""
        for order in self.orders:
            self.order_manager.add_order(order)
        
        # 更新部分订单状态
        self.order_manager.update_order("order_0", {"status": OrderStatus.FILLED})
        self.order_manager.update_order("order_1", {"status": OrderStatus.CANCELED})
        self.order_manager.update_order("order_2", {"status": OrderStatus.REJECTED})
        self.order_manager.update_order("order_3", {"status": OrderStatus.SUBMITTED})
        
        # 获取统计信息
        stats = self.order_manager.get_stats()
        
        self.assertEqual(stats["total"], len(self.orders))
        self.assertEqual(stats["active"], 2)  # CREATED + SUBMITTED
        self.assertEqual(stats["completed"], 3)  # FILLED + CANCELED + REJECTED
        self.assertEqual(stats["by_status"][OrderStatus.CREATED.name], 1)
        self.assertEqual(stats["by_status"][OrderStatus.SUBMITTED.name], 1)
        self.assertEqual(stats["by_status"][OrderStatus.FILLED.name], 1)
        self.assertEqual(stats["by_status"][OrderStatus.CANCELED.name], 1)
        self.assertEqual(stats["by_status"][OrderStatus.REJECTED.name], 1)


class TestSimulatedExecutionClient(unittest.TestCase):
    """测试模拟交易所执行客户端"""
    
    def setUp(self):
        """设置测试环境"""
        # 配置模拟交易所
        config = {
            "initial_balance": 10000.0,
            "order_latency_ms": 0,  # 设置为0以便同步测试
            "fill_latency_ms": 0,
            "fill_probability": 1.0,  # 确保市价单总是成交
            "partial_fill_probability": 0.0,  # 禁用部分成交
            "reject_probability": 0.0,  # 禁用随机拒绝
        }
        
        self.client = SimulatedExecutionClient(name="test_simulator", config=config)
        
        # 连接模拟交易所
        self.client.connect()
        
        # 回调记录
        self.order_updates = []
        self.trade_updates = []
        self.position_updates = []
        self.account_updates = []
        
        # 注册回调
        self.client.register_order_callback(self._on_order_update)
        self.client.register_trade_callback(self._on_trade_update)
        self.client.register_position_callback(self._on_position_update)
        self.client.register_account_callback(self._on_account_update)
    
    def tearDown(self):
        """清理测试环境"""
        self.client.disconnect()
        self.order_updates.clear()
        self.trade_updates.clear()
        self.position_updates.clear()
        self.account_updates.clear()
    
    def _on_order_update(self, order_id, updates):
        """订单更新回调"""
        self.order_updates.append((order_id, updates))
    
    def _on_trade_update(self, trade):
        """成交更新回调"""
        self.trade_updates.append(trade)
    
    def _on_position_update(self, position):
        """持仓更新回调"""
        self.position_updates.append(position)
    
    def _on_account_update(self, account):
        """账户更新回调"""
        self.account_updates.append(account)
    
    def test_connection(self):
        """测试连接和断开连接"""
        # 已在setUp中连接
        self.assertTrue(self.client.is_connected)
        self.assertTrue(self.client.is_authenticated())
        
        # 测试断开连接
        success = self.client.disconnect()
        self.assertTrue(success)
        self.assertFalse(self.client.is_connected)
        
        # 重新连接
        success = self.client.connect()
        self.assertTrue(success)
        self.assertTrue(self.client.is_connected)
    
    def test_place_market_order(self):
        """测试下市价单"""
        # 创建市价买单
        order = Order(
            order_id="test_market_buy",
            client_order_id="client_market_buy",
            strategy_id="test_strategy",
            symbol="AAPL",
            quantity=10,
            direction="BUY",
            order_type=OrderType.MARKET
        )
        
        # 下单
        success, msg = self.client.place_order(order)
        self.assertTrue(success)
        self.assertIsNone(msg)
        
        # 等待订单处理（模拟交易所处理订单）
        time.sleep(0.2)
        
        # 验证订单状态
        status, info = self.client.get_order_status("test_market_buy")
        self.assertEqual(status, OrderStatus.FILLED)
        self.assertEqual(info["filled_quantity"], 10)
        
        # 获取持仓
        positions = self.client.get_positions()
        self.assertIn("AAPL", positions)
        self.assertEqual(positions["AAPL"]["quantity"], 10)
        
        # 测试卖单
        order = Order(
            order_id="test_market_sell",
            client_order_id="client_market_sell",
            strategy_id="test_strategy",
            symbol="AAPL",
            quantity=5,
            direction="SELL",
            order_type=OrderType.MARKET
        )
        
        # 下单
        success, msg = self.client.place_order(order)
        self.assertTrue(success)
        self.assertIsNone(msg)
        
        # 等待订单处理
        time.sleep(0.2)
        
        # 验证订单状态
        status, info = self.client.get_order_status("test_market_sell")
        self.assertEqual(status, OrderStatus.FILLED)
        
        # 更新后的持仓
        positions = self.client.get_positions()
        self.assertEqual(positions["AAPL"]["quantity"], 5)
        
        # 验证回调次数
        self.assertGreater(len(self.order_updates), 0)
        self.assertGreater(len(self.position_updates), 0)
        self.assertGreater(len(self.account_updates), 0)
    
    def test_place_limit_order(self):
        """测试下限价单"""
        # 获取当前市场价格
        market_data = self.client.market_data["AAPL"]
        current_price = market_data["price"]
        
        # 创建低于市场价的买单（通常会立即成交）
        buy_price = current_price * 1.1  # 高于市场价10%
        order = Order(
            order_id="test_limit_buy",
            client_order_id="client_limit_buy",
            strategy_id="test_strategy",
            symbol="AAPL",
            quantity=10,
            price=buy_price,
            direction="BUY",
            order_type=OrderType.LIMIT
        )
        
        # 下单
        success, msg = self.client.place_order(order)
        self.assertTrue(success)
        self.assertIsNone(msg)
        
        # 等待订单处理
        time.sleep(0.2)
        
        # 验证订单状态
        status, info = self.client.get_order_status("test_limit_buy")
        self.assertEqual(status, OrderStatus.FILLED)
        
        # 创建高于市场价的卖单
        sell_price = current_price * 0.9  # 低于市场价10%
        order = Order(
            order_id="test_limit_sell",
            client_order_id="client_limit_sell",
            strategy_id="test_strategy",
            symbol="AAPL",
            quantity=5,
            price=sell_price,
            direction="SELL",
            order_type=OrderType.LIMIT
        )
        
        # 下单
        success, msg = self.client.place_order(order)
        self.assertTrue(success)
        self.assertIsNone(msg)
        
        # 等待订单处理
        time.sleep(0.2)
        
        # 验证订单状态
        status, info = self.client.get_order_status("test_limit_sell")
        self.assertEqual(status, OrderStatus.FILLED)
    
    def test_cancel_order(self):
        """测试取消订单"""
        # 创建一个不会立即成交的限价单
        market_data = self.client.market_data["AAPL"]
        current_price = market_data["price"]
        
        # 买入限价单，价格低于市场价50%（不会立即成交）
        buy_price = current_price * 0.5
        order = Order(
            order_id="test_cancel_order",
            client_order_id="client_cancel_order",
            strategy_id="test_strategy",
            symbol="AAPL",
            quantity=10,
            price=buy_price,
            direction="BUY",
            order_type=OrderType.LIMIT
        )
        
        # 下单
        success, msg = self.client.place_order(order)
        self.assertTrue(success)
        self.assertIsNone(msg)
        
        # 等待订单被接受
        time.sleep(0.1)
        
        # 取消订单
        success, msg = self.client.cancel_order("test_cancel_order")
        self.assertTrue(success)
        self.assertIsNone(msg)
        
        # 验证订单状态
        status, info = self.client.get_order_status("test_cancel_order")
        self.assertEqual(status, OrderStatus.CANCELED)
    
    def test_get_open_orders(self):
        """测试获取未成交订单"""
        # 创建几个不会立即成交的限价单
        market_data = self.client.market_data["AAPL"]
        current_price = market_data["price"]
        
        # 下单时价格低于市场价50%，不会立即成交
        for i in range(3):
            order = Order(
                order_id=f"open_order_{i}",
                client_order_id=f"client_open_order_{i}",
                strategy_id="test_strategy",
                symbol="AAPL",
                quantity=10,
                price=current_price * 0.5,
                direction="BUY",
                order_type=OrderType.LIMIT
            )
            self.client.place_order(order)
        
        # 等待订单被接受
        time.sleep(0.1)
        
        # 获取未成交订单
        open_orders = self.client.get_open_orders()
        self.assertEqual(len(open_orders), 3)
    
    def test_get_account_info(self):
        """测试获取账户信息"""
        # 初始账户信息
        initial_info = self.client.get_account_info()
        self.assertEqual(initial_info["balance"], 10000.0)
        self.assertEqual(initial_info["equity"], 10000.0)
        self.assertEqual(initial_info["margin_used"], 0.0)
        self.assertEqual(initial_info["available"], 10000.0)
        
        # 下一个市价买单
        order = Order(
            order_id="account_test_order",
            client_order_id="client_account_test",
            strategy_id="test_strategy",
            symbol="AAPL",
            quantity=10,
            direction="BUY",
            order_type=OrderType.MARKET
        )
        self.client.place_order(order)
        
        # 等待订单处理
        time.sleep(0.2)
        
        # 验证账户信息变化
        updated_info = self.client.get_account_info()
        self.assertLess(updated_info["balance"], 10000.0)  # 余额减少


if __name__ == "__main__":
    unittest.main() 