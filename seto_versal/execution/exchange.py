#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易所适配器实现，具体交易所的接口实现
"""

import os
import time
import logging
import json
import requests
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import threading
import uuid
import websocket

from seto_versal.execution.client import ExecutionClient, Order, OrderType, OrderStatus


class BinanceExecutionClient(ExecutionClient):
    """
    币安交易所执行客户端
    此实现仅作为示例，实际使用时需要根据币安API文档完善
    """
    
    def __init__(self, name: str = "binance", config: Dict[str, Any] = None):
        """
        初始化币安交易所执行客户端
        
        Args:
            name: 客户端名称
            config: 配置参数
        """
        super().__init__(name, config)
        
        # 默认配置
        self.default_config = {
            "api_key": "",
            "secret_key": "",
            "base_url": "https://api.binance.com",
            "ws_url": "wss://stream.binance.com:9443/ws",
            "timeout": 10,
            "recvWindow": 5000,
            "use_testnet": False
        }
        
        # 合并配置
        self.config = {**self.default_config, **(config or {})}
        
        # 使用测试网络
        if self.config["use_testnet"]:
            self.config["base_url"] = "https://testnet.binance.vision"
            self.config["ws_url"] = "wss://testnet.binance.vision/ws"
        
        # API密钥检查
        if not self.config["api_key"] or not self.config["secret_key"]:
            self.logger.warning("API密钥未设置，无法进行身份验证操作")
        
        # HTTP会话
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.config["api_key"],
            "Content-Type": "application/json"
        })
        
        # WebSocket连接
        self.ws = None
        self.ws_thread = None
        self.ws_connected = False
        self.ws_subscriptions = set()
        
        # 映射Binance订单ID到内部订单ID
        self.binance_order_id_map = {}
        
        # 账户信息缓存
        self.account_info = {}
        self.positions = {}
        self.last_account_update = 0
        
        # 线程锁
        self.lock = threading.RLock()
        
        self.logger.info(f"币安执行客户端已初始化")
    
    def connect(self) -> bool:
        """
        连接到币安交易所
        
        Returns:
            连接是否成功
        """
        if self.is_connected:
            return True
        
        try:
            # 测试API连接
            response = self.session.get(
                f"{self.config['base_url']}/api/v3/ping",
                timeout=self.config["timeout"]
            )
            response.raise_for_status()
            
            # 连接WebSocket（用于订单和账户更新）
            self._connect_websocket()
            
            self.is_connected = True
            self.logger.info("已连接到币安交易所")
            return True
        except Exception as e:
            self.logger.error(f"连接币安交易所失败: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        断开与币安交易所的连接
        
        Returns:
            断开连接是否成功
        """
        if not self.is_connected:
            return True
        
        try:
            # 关闭WebSocket连接
            self._disconnect_websocket()
            
            # 关闭HTTP会话
            self.session.close()
            
            self.is_connected = False
            self.logger.info("已断开与币安交易所的连接")
            return True
        except Exception as e:
            self.logger.error(f"断开币安交易所连接失败: {str(e)}")
            return False
    
    def is_authenticated(self) -> bool:
        """
        检查是否已认证
        
        Returns:
            是否已认证
        """
        if not self.is_connected:
            return False
        
        try:
            # 尝试获取账户信息
            response = self._send_signed_request("GET", "/api/v3/account")
            return response.status_code == 200
        except Exception:
            return False
    
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
        
        if not self.config["api_key"] or not self.config["secret_key"]:
            return False, "API密钥未设置"
        
        try:
            # 构建参数
            params = {
                "symbol": order.symbol,
                "side": order.direction,
                "newClientOrderId": order.client_order_id,
                "quantity": str(order.quantity),
            }
            
            # 根据订单类型设置参数
            if order.order_type == OrderType.MARKET:
                params["type"] = "MARKET"
            elif order.order_type == OrderType.LIMIT:
                params["type"] = "LIMIT"
                params["price"] = str(order.price)
                params["timeInForce"] = "GTC"  # Good Till Cancel
            elif order.order_type == OrderType.STOP:
                params["type"] = "STOP_LOSS"
                params["stopPrice"] = str(order.stop_price)
            elif order.order_type == OrderType.STOP_LIMIT:
                params["type"] = "STOP_LOSS_LIMIT"
                params["price"] = str(order.price)
                params["stopPrice"] = str(order.stop_price)
                params["timeInForce"] = "GTC"
            
            # 发送请求
            response = self._send_signed_request("POST", "/api/v3/order", params)
            response.raise_for_status()
            result = response.json()
            
            # 更新订单状态
            binance_order_id = str(result["orderId"])
            self.binance_order_id_map[binance_order_id] = order.order_id
            
            # 保存订单并更新状态
            order.update_status(OrderStatus.SUBMITTED, "订单已提交到交易所")
            self.order_manager.add_order(order)
            
            self.logger.info(f"订单已提交: {order}, 交易所订单ID: {binance_order_id}")
            return True, None
        except requests.exceptions.RequestException as e:
            error_msg = f"下单请求失败: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = f"下单请求失败: {error_data.get('msg', str(e))}"
                except Exception:
                    pass
            
            self.logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"下单异常: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
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
        
        order = self.order_manager.get_order(order_id)
        if not order:
            return False, f"订单不存在: {order_id}"
        
        if order.status.is_complete:
            return False, f"订单已完成，无法取消: {order_id}"
        
        try:
            # 构建参数
            params = {
                "symbol": order.symbol,
                "origClientOrderId": order.client_order_id
            }
            
            # 发送请求
            response = self._send_signed_request("DELETE", "/api/v3/order", params)
            response.raise_for_status()
            
            # 更新订单状态
            success = self.order_manager.cancel_order(order_id, "用户取消")
            if success:
                self.logger.info(f"订单已取消: {order}")
            
            return success, None
        except requests.exceptions.RequestException as e:
            error_msg = f"取消订单请求失败: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = f"取消订单请求失败: {error_data.get('msg', str(e))}"
                except Exception:
                    pass
            
            self.logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"取消订单异常: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
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
        
        try:
            # 构建参数
            params = {
                "symbol": order.symbol,
                "origClientOrderId": order.client_order_id
            }
            
            # 发送请求
            response = self._send_signed_request("GET", "/api/v3/order", params)
            response.raise_for_status()
            result = response.json()
            
            # 解析订单状态
            binance_status = result["status"]
            status = self._convert_binance_status(binance_status)
            
            # 更新订单信息
            updates = {
                "status": status,
                "filled_quantity": float(result["executedQty"]),
                "price": float(result["price"]) if result["price"] != "0" else None,
                "average_fill_price": float(result["price"]) if result["price"] != "0" else None
            }
            
            self.order_manager.update_order(order_id, updates)
            
            extra_info = {
                "binance_order_id": result["orderId"],
                "filled_quantity": float(result["executedQty"]),
                "cummulative_quote_qty": float(result["cummulativeQuoteQty"]),
                "status": binance_status,
                "time_in_force": result["timeInForce"],
                "order_type": result["type"],
                "side": result["side"],
                "update_time": result["updateTime"]
            }
            
            return status, extra_info
        except requests.exceptions.RequestException as e:
            error_msg = f"获取订单状态请求失败: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = f"获取订单状态请求失败: {error_data.get('msg', str(e))}"
                except Exception:
                    pass
            
            self.logger.error(error_msg)
            return order.status, {"error": error_msg}
        except Exception as e:
            error_msg = f"获取订单状态异常: {str(e)}"
            self.logger.error(error_msg)
            return order.status, {"error": error_msg}
    
    def get_open_orders(self) -> List[Order]:
        """
        获取未成交订单
        
        Returns:
            未成交订单列表
        """
        if not self.is_connected:
            return []
        
        try:
            # 发送请求获取所有未成交订单
            response = self._send_signed_request("GET", "/api/v3/openOrders")
            response.raise_for_status()
            results = response.json()
            
            # 处理结果
            open_orders = []
            for result in results:
                client_order_id = result["clientOrderId"]
                order = self.order_manager.get_order_by_client_id(client_order_id)
                
                if order:
                    # 更新订单状态
                    binance_status = result["status"]
                    status = self._convert_binance_status(binance_status)
                    
                    updates = {
                        "status": status,
                        "filled_quantity": float(result["executedQty"]),
                        "price": float(result["price"]) if result["price"] != "0" else None,
                        "average_fill_price": float(result["price"]) if result["price"] != "0" else None
                    }
                    
                    self.order_manager.update_order(order.order_id, updates)
                    open_orders.append(order)
                else:
                    # 创建新订单对象
                    new_order = Order(
                        order_id=str(uuid.uuid4()),
                        client_order_id=client_order_id,
                        strategy_id="unknown",
                        symbol=result["symbol"],
                        quantity=float(result["origQty"]),
                        price=float(result["price"]) if result["price"] != "0" else None,
                        direction=result["side"],
                        order_type=self._convert_binance_order_type(result["type"])
                    )
                    
                    binance_status = result["status"]
                    status = self._convert_binance_status(binance_status)
                    new_order.status = status
                    new_order.filled_quantity = float(result["executedQty"])
                    
                    # 保存订单
                    self.order_manager.add_order(new_order)
                    open_orders.append(new_order)
                    
                    # 映射币安订单ID
                    binance_order_id = str(result["orderId"])
                    self.binance_order_id_map[binance_order_id] = new_order.order_id
            
            return open_orders
        except Exception as e:
            self.logger.error(f"获取未成交订单异常: {str(e)}")
            return []
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        获取持仓
        
        Returns:
            持仓信息映射（品种到持仓信息）
        """
        if not self.is_connected:
            return {}
        
        # 刷新账户信息
        self._update_account_info()
        return self.positions
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        获取账户信息
        
        Returns:
            账户信息
        """
        if not self.is_connected:
            return {}
        
        # 刷新账户信息
        self._update_account_info()
        return self.account_info
    
    def _update_account_info(self) -> None:
        """
        更新账户信息
        """
        # 限制请求频率，不要太频繁请求账户信息
        current_time = time.time()
        if current_time - self.last_account_update < 1.0:  # 最多每秒请求一次
            return
        
        try:
            # 发送请求
            response = self._send_signed_request("GET", "/api/v3/account")
            response.raise_for_status()
            result = response.json()
            
            # 更新账户信息
            balances = {}
            for balance in result["balances"]:
                asset = balance["asset"]
                free = float(balance["free"])
                locked = float(balance["locked"])
                total = free + locked
                if total > 0:
                    balances[asset] = {
                        "free": free,
                        "locked": locked,
                        "total": total
                    }
            
            self.account_info = {
                "makerCommission": result["makerCommission"],
                "takerCommission": result["takerCommission"],
                "buyerCommission": result["buyerCommission"],
                "sellerCommission": result["sellerCommission"],
                "canTrade": result["canTrade"],
                "canWithdraw": result["canWithdraw"],
                "canDeposit": result["canDeposit"],
                "updateTime": result["updateTime"],
                "balances": balances
            }
            
            # 更新持仓信息（对于现货交易所，持仓就是余额）
            self.positions = {}
            for asset, balance in balances.items():
                if balance["total"] > 0:
                    self.positions[asset] = {
                        "symbol": asset,
                        "quantity": balance["total"],
                        "available": balance["free"],
                        "locked": balance["locked"],
                        "timestamp": datetime.now().isoformat()
                    }
            
            self.last_account_update = current_time
            
            # 发送账户更新通知
            self._handle_account_update(self.account_info)
        except Exception as e:
            self.logger.error(f"更新账户信息异常: {str(e)}")
    
    def _send_signed_request(self, method: str, endpoint: str, params: Dict = None) -> requests.Response:
        """
        发送签名请求
        
        Args:
            method: HTTP方法
            endpoint: API端点
            params: 请求参数
            
        Returns:
            响应对象
        """
        url = f"{self.config['base_url']}{endpoint}"
        
        # 构建请求参数
        request_params = params.copy() if params else {}
        request_params["timestamp"] = int(time.time() * 1000)
        request_params["recvWindow"] = self.config["recvWindow"]
        
        # 生成签名
        query_string = "&".join([f"{key}={val}" for key, val in request_params.items()])
        signature = hmac.new(
            self.config["secret_key"].encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        request_params["signature"] = signature
        
        # 发送请求
        if method == "GET":
            return self.session.get(url, params=request_params, timeout=self.config["timeout"])
        elif method == "POST":
            return self.session.post(url, params=request_params, timeout=self.config["timeout"])
        elif method == "DELETE":
            return self.session.delete(url, params=request_params, timeout=self.config["timeout"])
        else:
            raise ValueError(f"不支持的HTTP方法: {method}")
    
    def _connect_websocket(self) -> None:
        """
        连接WebSocket
        """
        if self.ws_connected:
            return
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._handle_websocket_message(data)
            except Exception as e:
                self.logger.error(f"处理WebSocket消息异常: {str(e)}")
        
        def on_error(ws, error):
            self.logger.error(f"WebSocket错误: {str(error)}")
        
        def on_close(ws, close_status_code, close_msg):
            self.ws_connected = False
            self.logger.info(f"WebSocket连接已关闭: {close_status_code} - {close_msg}")
        
        def on_open(ws):
            self.ws_connected = True
            self.logger.info("WebSocket连接已打开")
            
            # 订阅用户数据流
            self._subscribe_user_data_stream()
        
        # 创建WebSocket连接
        self.ws = websocket.WebSocketApp(
            self.config["ws_url"],
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # 在后台线程中运行WebSocket
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def _disconnect_websocket(self) -> None:
        """
        断开WebSocket连接
        """
        if self.ws:
            # 关闭WebSocket连接
            self.ws.close()
            
            # 等待线程结束
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=2.0)
            
            self.ws = None
            self.ws_thread = None
            self.ws_connected = False
    
    def _subscribe_user_data_stream(self) -> None:
        """
        订阅用户数据流
        """
        try:
            # 创建用户数据流
            response = self.session.post(
                f"{self.config['base_url']}/api/v3/userDataStream",
                timeout=self.config["timeout"]
            )
            response.raise_for_status()
            result = response.json()
            listen_key = result["listenKey"]
            
            # 订阅用户数据流
            if self.ws_connected:
                self.ws.send(json.dumps({
                    "method": "SUBSCRIBE",
                    "params": [
                        f"{listen_key}"
                    ],
                    "id": int(time.time() * 1000)
                }))
            
            # 启动一个线程定期延长listenKey的有效期
            def keep_alive_listen_key():
                while self.is_connected and self.ws_connected:
                    try:
                        time.sleep(30 * 60)  # 每30分钟ping一次
                        self.session.put(
                            f"{self.config['base_url']}/api/v3/userDataStream",
                            params={"listenKey": listen_key},
                            timeout=self.config["timeout"]
                        )
                    except Exception as e:
                        self.logger.error(f"延长listenKey有效期异常: {str(e)}")
            
            keep_alive_thread = threading.Thread(target=keep_alive_listen_key)
            keep_alive_thread.daemon = True
            keep_alive_thread.start()
        except Exception as e:
            self.logger.error(f"订阅用户数据流异常: {str(e)}")
    
    def _handle_websocket_message(self, data: Dict) -> None:
        """
        处理WebSocket消息
        
        Args:
            data: 消息数据
        """
        if "e" not in data:
            return
        
        event_type = data["e"]
        
        if event_type == "executionReport":
            # 订单更新事件
            self._handle_execution_report(data)
        elif event_type == "outboundAccountPosition":
            # 账户更新事件
            self._handle_account_position(data)
        elif event_type == "balanceUpdate":
            # 余额更新事件
            self._handle_balance_update(data)
    
    def _handle_execution_report(self, data: Dict) -> None:
        """
        处理订单执行报告
        
        Args:
            data: 执行报告数据
        """
        symbol = data["s"]
        client_order_id = data["c"]
        binance_order_id = str(data["i"])
        order_status = data["X"]
        executed_qty = float(data["l"])
        executed_price = float(data["L"])
        order_type = data["o"]
        side = data["S"]
        
        # 获取订单对象
        order = self.order_manager.get_order_by_client_id(client_order_id)
        
        if not order:
            # 如果找不到订单，创建一个新的订单对象
            order_id = str(uuid.uuid4())
            self.binance_order_id_map[binance_order_id] = order_id
            
            order = Order(
                order_id=order_id,
                client_order_id=client_order_id,
                strategy_id="unknown",
                symbol=symbol,
                quantity=float(data["q"]),
                price=float(data["p"]) if data["p"] != "0" else None,
                direction=side,
                order_type=self._convert_binance_order_type(order_type)
            )
            
            self.order_manager.add_order(order)
        
        # 更新订单状态
        status = self._convert_binance_status(order_status)
        
        # 处理成交信息
        if executed_qty > 0 and executed_price > 0:
            # 添加成交记录
            timestamp = datetime.fromtimestamp(data["T"] / 1000)
            commission = float(data["n"]) if "n" in data else 0.0
            
            order.add_fill(executed_qty, executed_price, timestamp, commission)
        
        # 更新订单状态
        if order.status != status:
            reason = f"交易所订单状态: {order_status}"
            order.update_status(status, reason)
            
            # 处理订单回调
            self._handle_order_update(order.order_id, {"status": status})
    
    def _handle_account_position(self, data: Dict) -> None:
        """
        处理账户持仓更新
        
        Args:
            data: 持仓更新数据
        """
        # 更新账户余额
        for balance in data["B"]:
            asset = balance["a"]
            free = float(balance["f"])
            locked = float(balance["l"])
            total = free + locked
            
            if asset not in self.account_info.get("balances", {}):
                self.account_info.setdefault("balances", {})[asset] = {}
            
            self.account_info["balances"][asset] = {
                "free": free,
                "locked": locked,
                "total": total
            }
            
            # 更新持仓信息
            self.positions[asset] = {
                "symbol": asset,
                "quantity": total,
                "available": free,
                "locked": locked,
                "timestamp": datetime.fromtimestamp(data["u"] / 1000).isoformat()
            }
            
            # 处理持仓回调
            self._handle_position_update(self.positions[asset])
        
        # 处理账户回调
        self._handle_account_update(self.account_info)
    
    def _handle_balance_update(self, data: Dict) -> None:
        """
        处理余额更新
        
        Args:
            data: 余额更新数据
        """
        asset = data["a"]
        delta = float(data["d"])
        
        if asset in self.account_info.get("balances", {}):
            balance = self.account_info["balances"][asset]
            balance["total"] += delta
            balance["free"] += delta  # 假设是free余额变化
            
            # 更新持仓信息
            if asset in self.positions:
                self.positions[asset]["quantity"] += delta
                self.positions[asset]["available"] += delta
                self.positions[asset]["timestamp"] = datetime.fromtimestamp(data["T"] / 1000).isoformat()
                
                # 处理持仓回调
                self._handle_position_update(self.positions[asset])
        
        # 处理账户回调
        self._handle_account_update(self.account_info)
    
    def _convert_binance_status(self, binance_status: str) -> OrderStatus:
        """
        转换币安订单状态到内部订单状态
        
        Args:
            binance_status: 币安订单状态
            
        Returns:
            内部订单状态
        """
        status_map = {
            "NEW": OrderStatus.ACCEPTED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELED,
            "PENDING_CANCEL": OrderStatus.SUBMITTED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED
        }
        return status_map.get(binance_status, OrderStatus.CREATED)
    
    def _convert_binance_order_type(self, binance_type: str) -> OrderType:
        """
        转换币安订单类型到内部订单类型
        
        Args:
            binance_type: 币安订单类型
            
        Returns:
            内部订单类型
        """
        type_map = {
            "LIMIT": OrderType.LIMIT,
            "MARKET": OrderType.MARKET,
            "STOP_LOSS": OrderType.STOP,
            "STOP_LOSS_LIMIT": OrderType.STOP_LIMIT,
            "TAKE_PROFIT": OrderType.STOP,
            "TAKE_PROFIT_LIMIT": OrderType.STOP_LIMIT,
            "LIMIT_MAKER": OrderType.LIMIT
        }
        return type_map.get(binance_type, OrderType.MARKET) 