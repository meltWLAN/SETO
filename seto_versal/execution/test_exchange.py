#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易所适配器的测试文件
"""

import unittest
import json
import time
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock
import requests
import responses
import websocket

from seto_versal.execution.client import OrderType, OrderStatus, Order
from seto_versal.execution.exchange import BinanceExecutionClient


class TestBinanceExecutionClient(unittest.TestCase):
    """测试币安执行客户端适配器"""
    
    def setUp(self):
        """设置测试环境"""
        # 配置并创建币安客户端
        self.config = {
            "api_key": "test_api_key",
            "secret_key": "test_secret_key",
            "use_testnet": True,
            "timeout": 2
        }
        
        # 开启响应模拟
        responses.start()
        
        # 创建币安客户端
        with patch('websocket.WebSocketApp'):
            with patch.object(BinanceExecutionClient, '_connect_websocket'):
                self.client = BinanceExecutionClient(name="test_binance", config=self.config)
    
    def tearDown(self):
        """清理测试环境"""
        # 停止响应模拟
        responses.stop()
        responses.reset()
    
    @responses.activate
    def test_connect(self):
        """测试连接到币安交易所"""
        # 模拟ping端点响应
        responses.add(
            responses.GET,
            "https://testnet.binance.vision/api/v3/ping",
            json={},
            status=200
        )
        
        # 使用模拟WebSocket连接
        with patch.object(BinanceExecutionClient, '_connect_websocket') as mock_connect_ws:
            # 连接到交易所
            success = self.client.connect()
            
            # 验证结果
            self.assertTrue(success)
            self.assertTrue(self.client.is_connected)
            mock_connect_ws.assert_called_once()
    
    @responses.activate
    def test_disconnect(self):
        """测试断开与币安交易所的连接"""
        # 准备：先连接
        responses.add(
            responses.GET,
            "https://testnet.binance.vision/api/v3/ping",
            json={},
            status=200
        )
        
        with patch.object(BinanceExecutionClient, '_connect_websocket'):
            self.client.connect()
        
        # 模拟断开连接
        with patch.object(BinanceExecutionClient, '_disconnect_websocket') as mock_disconnect_ws:
            # 断开连接
            success = self.client.disconnect()
            
            # 验证结果
            self.assertTrue(success)
            self.assertFalse(self.client.is_connected)
            mock_disconnect_ws.assert_called_once()
    
    @responses.activate
    def test_is_authenticated(self):
        """测试身份验证检查"""
        # 准备：先连接
        responses.add(
            responses.GET,
            "https://testnet.binance.vision/api/v3/ping",
            json={},
            status=200
        )
        
        with patch.object(BinanceExecutionClient, '_connect_websocket'):
            self.client.connect()
        
        # 模拟账户信息响应（成功）
        responses.add(
            responses.GET,
            "https://testnet.binance.vision/api/v3/account",
            json={"makerCommission": 10, "takerCommission": 10, "balances": []},
            status=200
        )
        
        # 使用mock替换签名方法
        with patch.object(BinanceExecutionClient, '_send_signed_request', return_value=Mock(status_code=200)) as mock_signed_request:
            # 检查是否已认证
            authenticated = self.client.is_authenticated()
            
            # 验证结果
            self.assertTrue(authenticated)
            mock_signed_request.assert_called_once_with("GET", "/api/v3/account")
    
    @responses.activate
    def test_place_order_market(self):
        """测试下市价单"""
        # 准备：先连接
        responses.add(
            responses.GET,
            "https://testnet.binance.vision/api/v3/ping",
            json={},
            status=200
        )
        
        with patch.object(BinanceExecutionClient, '_connect_websocket'):
            self.client.connect()
        
        # 创建订单对象
        order = Order(
            order_id="test_market_order",
            client_order_id="client_test_market",
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            quantity=0.01,
            direction="BUY",
            order_type=OrderType.MARKET
        )
        
        # 模拟下单响应
        order_response = {
            "symbol": "BTCUSDT",
            "orderId": 12345678,
            "orderListId": -1,
            "clientOrderId": "client_test_market",
            "transactTime": 1507725176595,
            "price": "0.00000000",
            "origQty": "0.01000000",
            "executedQty": "0.01000000",
            "cummulativeQuoteQty": "10.00000000",
            "status": "FILLED",
            "timeInForce": "GTC",
            "type": "MARKET",
            "side": "BUY"
        }
        
        # 使用mock替换签名方法
        with patch.object(BinanceExecutionClient, '_send_signed_request') as mock_signed_request:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = order_response
            mock_signed_request.return_value = mock_response
            
            # 模拟订单管理器的方法
            self.client.order_manager.add_order = MagicMock()
            
            # 下单
            success, msg = self.client.place_order(order)
            
            # 验证结果
            self.assertTrue(success)
            self.assertIsNone(msg)
            self.assertEqual(order.status, OrderStatus.SUBMITTED)
            self.client.order_manager.add_order.assert_called_once_with(order)
            
            # 验证请求参数
            _, _, kwargs = mock_signed_request.mock_calls[0]
            params = kwargs.get('params', {})
            self.assertEqual(params["symbol"], "BTCUSDT")
            self.assertEqual(params["side"], "BUY")
            self.assertEqual(params["type"], "MARKET")
            self.assertEqual(params["quantity"], "0.01")
    
    @responses.activate
    def test_place_order_limit(self):
        """测试下限价单"""
        # 准备：先连接
        responses.add(
            responses.GET,
            "https://testnet.binance.vision/api/v3/ping",
            json={},
            status=200
        )
        
        with patch.object(BinanceExecutionClient, '_connect_websocket'):
            self.client.connect()
        
        # 创建订单对象
        order = Order(
            order_id="test_limit_order",
            client_order_id="client_test_limit",
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            quantity=0.01,
            price=20000.0,
            direction="BUY",
            order_type=OrderType.LIMIT
        )
        
        # 模拟下单响应
        order_response = {
            "symbol": "BTCUSDT",
            "orderId": 12345678,
            "orderListId": -1,
            "clientOrderId": "client_test_limit",
            "transactTime": 1507725176595,
            "price": "20000.00000000",
            "origQty": "0.01000000",
            "executedQty": "0.00000000",
            "cummulativeQuoteQty": "0.00000000",
            "status": "NEW",
            "timeInForce": "GTC",
            "type": "LIMIT",
            "side": "BUY"
        }
        
        # 使用mock替换签名方法
        with patch.object(BinanceExecutionClient, '_send_signed_request') as mock_signed_request:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = order_response
            mock_signed_request.return_value = mock_response
            
            # 模拟订单管理器的方法
            self.client.order_manager.add_order = MagicMock()
            
            # 下单
            success, msg = self.client.place_order(order)
            
            # 验证结果
            self.assertTrue(success)
            self.assertIsNone(msg)
            self.assertEqual(order.status, OrderStatus.SUBMITTED)
            self.client.order_manager.add_order.assert_called_once_with(order)
            
            # 验证请求参数
            _, _, kwargs = mock_signed_request.mock_calls[0]
            params = kwargs.get('params', {})
            self.assertEqual(params["symbol"], "BTCUSDT")
            self.assertEqual(params["side"], "BUY")
            self.assertEqual(params["type"], "LIMIT")
            self.assertEqual(params["quantity"], "0.01")
            self.assertEqual(params["price"], "20000.0")
            self.assertEqual(params["timeInForce"], "GTC")
    
    @responses.activate
    def test_cancel_order(self):
        """测试取消订单"""
        # 准备：先连接
        responses.add(
            responses.GET,
            "https://testnet.binance.vision/api/v3/ping",
            json={},
            status=200
        )
        
        with patch.object(BinanceExecutionClient, '_connect_websocket'):
            self.client.connect()
        
        # 创建订单对象
        order = Order(
            order_id="test_cancel_order",
            client_order_id="client_test_cancel",
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            quantity=0.01,
            price=20000.0,
            direction="BUY",
            order_type=OrderType.LIMIT
        )
        
        # 模拟取消订单响应
        cancel_response = {
            "symbol": "BTCUSDT",
            "origClientOrderId": "client_test_cancel",
            "orderId": 12345678,
            "orderListId": -1,
            "clientOrderId": "cancelOrder123",
            "price": "20000.00000000",
            "origQty": "0.01000000",
            "executedQty": "0.00000000",
            "cummulativeQuoteQty": "0.00000000",
            "status": "CANCELED",
            "timeInForce": "GTC",
            "type": "LIMIT",
            "side": "BUY"
        }
        
        # 使用mock替换签名方法
        with patch.object(BinanceExecutionClient, '_send_signed_request') as mock_signed_request:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = cancel_response
            mock_signed_request.return_value = mock_response
            
            # 模拟订单管理器的方法
            self.client.order_manager.get_order = MagicMock(return_value=order)
            self.client.order_manager.cancel_order = MagicMock(return_value=True)
            
            # 取消订单
            success, msg = self.client.cancel_order("test_cancel_order")
            
            # 验证结果
            self.assertTrue(success)
            self.assertIsNone(msg)
            self.client.order_manager.cancel_order.assert_called_once_with("test_cancel_order", "用户取消")
            
            # 验证请求参数
            _, _, kwargs = mock_signed_request.mock_calls[0]
            params = kwargs.get('params', {})
            self.assertEqual(params["symbol"], "BTCUSDT")
            self.assertEqual(params["origClientOrderId"], "client_test_cancel")
    
    @responses.activate
    def test_get_order_status(self):
        """测试获取订单状态"""
        # 准备：先连接
        responses.add(
            responses.GET,
            "https://testnet.binance.vision/api/v3/ping",
            json={},
            status=200
        )
        
        with patch.object(BinanceExecutionClient, '_connect_websocket'):
            self.client.connect()
        
        # 创建订单对象
        order = Order(
            order_id="test_status_order",
            client_order_id="client_test_status",
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            quantity=0.01,
            price=20000.0,
            direction="BUY",
            order_type=OrderType.LIMIT
        )
        
        # 模拟订单状态响应
        status_response = {
            "symbol": "BTCUSDT",
            "orderId": 12345678,
            "orderListId": -1,
            "clientOrderId": "client_test_status",
            "price": "20000.00000000",
            "origQty": "0.01000000",
            "executedQty": "0.00500000",
            "cummulativeQuoteQty": "100.00000000",
            "status": "PARTIALLY_FILLED",
            "timeInForce": "GTC",
            "type": "LIMIT",
            "side": "BUY",
            "stopPrice": "0.00000000",
            "icebergQty": "0.00000000",
            "time": 1507725176595,
            "updateTime": 1507725176595,
            "isWorking": True,
            "origQuoteOrderQty": "200.00000000"
        }
        
        # 使用mock替换签名方法
        with patch.object(BinanceExecutionClient, '_send_signed_request') as mock_signed_request:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = status_response
            mock_signed_request.return_value = mock_response
            
            # 模拟订单管理器的方法
            self.client.order_manager.get_order = MagicMock(return_value=order)
            self.client.order_manager.update_order = MagicMock()
            
            # 获取订单状态
            status, info = self.client.get_order_status("test_status_order")
            
            # 验证结果
            self.assertEqual(status, OrderStatus.PARTIALLY_FILLED)
            self.assertEqual(info["binance_order_id"], 12345678)
            self.assertEqual(info["filled_quantity"], 0.005)
            self.assertEqual(info["status"], "PARTIALLY_FILLED")
            
            # 验证更新订单状态
            self.client.order_manager.update_order.assert_called_once()
            
            # 验证请求参数
            _, _, kwargs = mock_signed_request.mock_calls[0]
            params = kwargs.get('params', {})
            self.assertEqual(params["symbol"], "BTCUSDT")
            self.assertEqual(params["origClientOrderId"], "client_test_status")
    
    @responses.activate
    def test_get_open_orders(self):
        """测试获取未成交订单"""
        # 准备：先连接
        responses.add(
            responses.GET,
            "https://testnet.binance.vision/api/v3/ping",
            json={},
            status=200
        )
        
        with patch.object(BinanceExecutionClient, '_connect_websocket'):
            self.client.connect()
        
        # 模拟未成交订单响应
        open_orders_response = [
            {
                "symbol": "BTCUSDT",
                "orderId": 12345678,
                "orderListId": -1,
                "clientOrderId": "client_order_1",
                "price": "20000.00000000",
                "origQty": "0.01000000",
                "executedQty": "0.00000000",
                "cummulativeQuoteQty": "0.00000000",
                "status": "NEW",
                "timeInForce": "GTC",
                "type": "LIMIT",
                "side": "BUY",
                "stopPrice": "0.00000000",
                "icebergQty": "0.00000000",
                "time": 1507725176595,
                "updateTime": 1507725176595,
                "isWorking": True,
                "origQuoteOrderQty": "200.00000000"
            },
            {
                "symbol": "ETHUSDT",
                "orderId": 87654321,
                "orderListId": -1,
                "clientOrderId": "client_order_2",
                "price": "1500.00000000",
                "origQty": "0.1000000",
                "executedQty": "0.05000000",
                "cummulativeQuoteQty": "75.00000000",
                "status": "PARTIALLY_FILLED",
                "timeInForce": "GTC",
                "type": "LIMIT",
                "side": "BUY",
                "stopPrice": "0.00000000",
                "icebergQty": "0.00000000",
                "time": 1507725176595,
                "updateTime": 1507725176595,
                "isWorking": True,
                "origQuoteOrderQty": "150.00000000"
            }
        ]
        
        # 使用mock替换签名方法
        with patch.object(BinanceExecutionClient, '_send_signed_request') as mock_signed_request:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = open_orders_response
            mock_signed_request.return_value = mock_response
            
            # 模拟订单管理器的方法
            order1 = Order(
                order_id="order_1",
                client_order_id="client_order_1",
                strategy_id="test_strategy",
                symbol="BTCUSDT",
                quantity=0.01,
                price=20000.0,
                direction="BUY",
                order_type=OrderType.LIMIT
            )
            
            # 模拟第一个订单存在，第二个订单不存在
            def mock_get_order_by_client_id(client_id):
                if client_id == "client_order_1":
                    return order1
                return None
            
            self.client.order_manager.get_order_by_client_id = MagicMock(side_effect=mock_get_order_by_client_id)
            self.client.order_manager.update_order = MagicMock()
            self.client.order_manager.add_order = MagicMock()
            
            # 获取未成交订单
            orders = self.client.get_open_orders()
            
            # 验证结果
            self.assertEqual(len(orders), 2)
            self.assertEqual(orders[0].client_order_id, "client_order_1")
            self.assertEqual(orders[1].client_order_id, "client_order_2")
            
            # 验证更新和添加订单
            self.client.order_manager.update_order.assert_called_once()
            self.client.order_manager.add_order.assert_called_once()
            
            # 验证请求参数
            mock_signed_request.assert_called_once_with("GET", "/api/v3/openOrders")
    
    def test_convert_binance_status(self):
        """测试转换币安订单状态"""
        test_cases = [
            ("NEW", OrderStatus.ACCEPTED),
            ("PARTIALLY_FILLED", OrderStatus.PARTIALLY_FILLED),
            ("FILLED", OrderStatus.FILLED),
            ("CANCELED", OrderStatus.CANCELED),
            ("PENDING_CANCEL", OrderStatus.SUBMITTED),
            ("REJECTED", OrderStatus.REJECTED),
            ("EXPIRED", OrderStatus.EXPIRED),
            ("UNKNOWN", OrderStatus.CREATED)  # 默认值
        ]
        
        for binance_status, expected_status in test_cases:
            actual_status = self.client._convert_binance_status(binance_status)
            self.assertEqual(actual_status, expected_status)
    
    def test_convert_binance_order_type(self):
        """测试转换币安订单类型"""
        test_cases = [
            ("LIMIT", OrderType.LIMIT),
            ("MARKET", OrderType.MARKET),
            ("STOP_LOSS", OrderType.STOP),
            ("STOP_LOSS_LIMIT", OrderType.STOP_LIMIT),
            ("TAKE_PROFIT", OrderType.STOP),
            ("TAKE_PROFIT_LIMIT", OrderType.STOP_LIMIT),
            ("LIMIT_MAKER", OrderType.LIMIT),
            ("UNKNOWN", OrderType.MARKET)  # 默认值
        ]
        
        for binance_type, expected_type in test_cases:
            actual_type = self.client._convert_binance_order_type(binance_type)
            self.assertEqual(actual_type, expected_type)
    
    @responses.activate
    def test_get_account_info(self):
        """测试获取账户信息"""
        # 准备：先连接
        responses.add(
            responses.GET,
            "https://testnet.binance.vision/api/v3/ping",
            json={},
            status=200
        )
        
        with patch.object(BinanceExecutionClient, '_connect_websocket'):
            self.client.connect()
        
        # 模拟账户信息响应
        account_response = {
            "makerCommission": 10,
            "takerCommission": 10,
            "buyerCommission": 0,
            "sellerCommission": 0,
            "canTrade": True,
            "canWithdraw": True,
            "canDeposit": True,
            "updateTime": 1507725176595,
            "accountType": "SPOT",
            "balances": [
                {
                    "asset": "BTC",
                    "free": "0.10000000",
                    "locked": "0.02000000"
                },
                {
                    "asset": "ETH",
                    "free": "2.00000000",
                    "locked": "0.00000000"
                },
                {
                    "asset": "USDT",
                    "free": "5000.00000000",
                    "locked": "500.00000000"
                }
            ]
        }
        
        # 使用mock替换更新账户信息方法
        with patch.object(BinanceExecutionClient, '_update_account_info') as mock_update_account:
            # 设置模拟账户信息
            self.client.account_info = {
                "makerCommission": 10,
                "takerCommission": 10,
                "balances": {
                    "BTC": {"free": 0.1, "locked": 0.02, "total": 0.12},
                    "USDT": {"free": 5000.0, "locked": 500.0, "total": 5500.0}
                }
            }
            
            # 获取账户信息
            account_info = self.client.get_account_info()
            
            # 验证结果
            self.assertEqual(account_info["makerCommission"], 10)
            self.assertEqual(account_info["balances"]["BTC"]["total"], 0.12)
            self.assertEqual(account_info["balances"]["USDT"]["free"], 5000.0)
            
            # 验证更新账户信息方法被调用
            mock_update_account.assert_called_once()
    
    @responses.activate
    def test_get_positions(self):
        """测试获取持仓信息"""
        # 准备：先连接
        responses.add(
            responses.GET,
            "https://testnet.binance.vision/api/v3/ping",
            json={},
            status=200
        )
        
        with patch.object(BinanceExecutionClient, '_connect_websocket'):
            self.client.connect()
        
        # 使用mock替换更新账户信息方法
        with patch.object(BinanceExecutionClient, '_update_account_info') as mock_update_account:
            # 设置模拟持仓信息
            self.client.positions = {
                "BTC": {
                    "symbol": "BTC",
                    "quantity": 0.12,
                    "available": 0.1,
                    "locked": 0.02,
                    "timestamp": datetime.now().isoformat()
                },
                "ETH": {
                    "symbol": "ETH",
                    "quantity": 2.0,
                    "available": 2.0,
                    "locked": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # 获取持仓信息
            positions = self.client.get_positions()
            
            # 验证结果
            self.assertEqual(len(positions), 2)
            self.assertEqual(positions["BTC"]["quantity"], 0.12)
            self.assertEqual(positions["ETH"]["available"], 2.0)
            
            # 验证更新账户信息方法被调用
            mock_update_account.assert_called_once()
    
    @responses.activate
    def test_handle_websocket_message_execution_report(self):
        """测试处理WebSocket订单执行报告消息"""
        # 准备执行报告消息
        execution_report = {
            "e": "executionReport",
            "E": 1499405658658,
            "s": "BTCUSDT",
            "c": "client_order_ws",
            "S": "BUY",
            "o": "LIMIT",
            "f": "GTC",
            "q": "0.01000000",
            "p": "20000.00000000",
            "P": "0.00000000",
            "F": "0.00000000",
            "g": -1,
            "C": "",
            "x": "NEW",
            "X": "NEW",
            "r": "NONE",
            "i": 123456789,
            "l": "0.00000000",
            "z": "0.00000000",
            "L": "0.00000000",
            "n": "0.00000000",
            "N": None,
            "T": 1499405658657,
            "t": -1,
            "I": 123456789,
            "w": True,
            "m": False,
            "M": False,
            "O": 1499405658657,
            "Z": "0.00000000",
            "Y": "0.00000000",
            "Q": "0.00000000"
        }
        
        # 模拟订单管理器方法
        self.client.order_manager.get_order_by_client_id = MagicMock(return_value=None)
        self.client.order_manager.add_order = MagicMock()
        
        # 模拟处理订单更新回调
        self.client._handle_order_update = MagicMock()
        
        # 处理消息
        self.client._handle_websocket_message(execution_report)
        
        # 验证结果
        self.client.order_manager.get_order_by_client_id.assert_called_once_with("client_order_ws")
        self.client.order_manager.add_order.assert_called_once()
        self.client._handle_order_update.assert_called_once()


if __name__ == "__main__":
    unittest.main() 