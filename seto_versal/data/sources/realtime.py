#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实时数据源模块，提供各种市场数据的实时接入
支持雅虎财经、东方财富、同花顺等数据源
"""

import logging
import time
import threading
import json
import websocket
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from queue import Queue
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class RealTimeDataSource(ABC):
    """实时数据源基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化实时数据源
        
        Args:
            name: 数据源名称
            config: 配置参数
        """
        self.name = name
        self.config = config
        self.streaming_connections = {}  # 保存所有连接
        self.subscription_callbacks = {}  # 回调函数映射
        self.is_connected = False
        self.stream_thread = None
        self.stream_should_stop = False
        self.data_buffer = {}  # 每个符号的数据缓冲区
        self.buffer_size = config.get("buffer_size", 1000)
        
        # 数据质量统计
        self.statistics = {
            "total_messages": 0,
            "valid_messages": 0,
            "error_messages": 0,
            "reconnects": 0,
            "start_time": None,
            "last_message_time": None
        }
        
    @abstractmethod
    def connect(self) -> bool:
        """连接到数据源"""
        pass
        
    @abstractmethod
    def disconnect(self) -> bool:
        """断开数据源连接"""
        pass
        
    @abstractmethod
    def is_available(self) -> bool:
        """检查数据源是否可用"""
        pass
        
    @abstractmethod
    def subscribe(self, symbol: str, callback: Optional[Callable] = None) -> bool:
        """
        订阅实时数据
        
        Args:
            symbol: 交易品种代码
            callback: 数据更新回调函数
            
        Returns:
            订阅是否成功
        """
        pass
        
    @abstractmethod
    def unsubscribe(self, symbol: str) -> bool:
        """
        取消订阅
        
        Args:
            symbol: 交易品种代码
            
        Returns:
            取消订阅是否成功
        """
        pass
        
    @abstractmethod
    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """
        获取实时快照
        
        Args:
            symbol: 交易品种代码
            
        Returns:
            包含最新数据的字典
        """
        pass
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据源统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.statistics.copy()
        
        # 计算每秒消息数
        if stats["start_time"] and stats["total_messages"] > 0:
            runtime = (datetime.now() - stats["start_time"]).total_seconds()
            stats["messages_per_second"] = stats["total_messages"] / max(1, runtime)
            
        # 计算成功率
        if stats["total_messages"] > 0:
            stats["success_rate"] = stats["valid_messages"] / stats["total_messages"]
            
        return stats
        
    def _start_streaming_thread(self):
        """启动后台数据流线程"""
        if self.stream_thread is None or not self.stream_thread.is_alive():
            self.stream_should_stop = False
            self.stream_thread = threading.Thread(
                target=self._streaming_worker,
                daemon=True
            )
            self.stream_thread.start()
            logger.info(f"已启动{self.name}数据流线程")
            
    def _stop_streaming_thread(self):
        """停止后台数据流线程"""
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_should_stop = True
            self.stream_thread.join(timeout=5)
            logger.info(f"已停止{self.name}数据流线程")
            
    def _streaming_worker(self):
        """数据流处理工作线程"""
        if not hasattr(self, "_process_stream_data"):
            logger.error(f"{self.name}未实现_process_stream_data方法")
            return
            
        self.statistics["start_time"] = datetime.now()
        
        while not self.stream_should_stop:
            try:
                self._process_stream_data()
            except Exception as e:
                logger.error(f"{self.name}数据流处理异常: {e}")
                time.sleep(5)  # 异常后暂停一段时间
                
                # 尝试重连
                if self.is_connected:
                    self.disconnect()
                    
                if self.connect():
                    self.statistics["reconnects"] += 1
                    
                    # 重新订阅所有符号
                    for symbol in list(self.subscription_callbacks.keys()):
                        self.subscribe(symbol)
                        
            time.sleep(0.01)  # 避免CPU占用过高
            
    def _add_to_buffer(self, symbol: str, data: Dict[str, Any]):
        """添加数据到缓冲区"""
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = []
            
        buffer = self.data_buffer[symbol]
        buffer.append(data)
        
        # 限制缓冲区大小
        if len(buffer) > self.buffer_size:
            self.data_buffer[symbol] = buffer[-self.buffer_size:]
            
    def get_buffer(self, symbol: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取数据缓冲区内容
        
        Args:
            symbol: 交易品种代码
            limit: 返回的最大数据条数
            
        Returns:
            缓冲区数据列表
        """
        if symbol not in self.data_buffer:
            return []
            
        buffer = self.data_buffer[symbol]
        if limit is not None:
            return buffer[-limit:]
        return buffer.copy()
        
    def to_dataframe(self, symbol: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        将缓冲区数据转换为DataFrame
        
        Args:
            symbol: 交易品种代码
            limit: 返回的最大数据条数
            
        Returns:
            包含数据的DataFrame
        """
        buffer = self.get_buffer(symbol, limit)
        if not buffer:
            return pd.DataFrame()
            
        return pd.DataFrame(buffer)


class YahooFinanceRealTime(RealTimeDataSource):
    """雅虎财经实时数据源"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("yahoo_finance", config)
        self.api_key = config.get("api_key", "")
        self.api_base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
        self.websocket = None
        self.message_queue = Queue()
        
    def connect(self) -> bool:
        """连接到雅虎财经API"""
        try:
            # 测试连接
            test_symbol = "AAPL"
            response = requests.get(f"{self.api_base_url}{test_symbol}", 
                                    params={"interval": "1m", "range": "1d"})
            
            if response.status_code == 200:
                self.is_connected = True
                self._start_streaming_thread()
                logger.info("成功连接到雅虎财经API")
                return True
            else:
                logger.error(f"连接雅虎财经API失败: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"连接雅虎财经API异常: {e}")
            return False
            
    def disconnect(self) -> bool:
        """断开与雅虎财经API的连接"""
        self._stop_streaming_thread()
        self.is_connected = False
        logger.info("已断开与雅虎财经API的连接")
        return True
        
    def is_available(self) -> bool:
        """检查雅虎财经API是否可用"""
        return self.is_connected
        
    def subscribe(self, symbol: str, callback: Optional[Callable] = None) -> bool:
        """订阅雅虎财经实时数据"""
        if not self.is_connected:
            logger.warning(f"尚未连接到雅虎财经API，无法订阅{symbol}")
            return False
            
        self.subscription_callbacks[symbol] = callback
        logger.info(f"已订阅{symbol}的雅虎财经实时数据")
        return True
        
    def unsubscribe(self, symbol: str) -> bool:
        """取消订阅雅虎财经实时数据"""
        if symbol in self.subscription_callbacks:
            del self.subscription_callbacks[symbol]
            logger.info(f"已取消订阅{symbol}的雅虎财经实时数据")
            return True
        return False
        
    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """获取雅虎财经实时快照"""
        try:
            response = requests.get(f"{self.api_base_url}{symbol}", 
                                   params={"interval": "1m", "range": "1d"})
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("chart", {}).get("result", [])
                
                if result:
                    quote = result[0].get("indicators", {}).get("quote", [{}])[0]
                    timestamp = result[0].get("timestamp", [])[-1]
                    
                    snapshot = {
                        "symbol": symbol,
                        "timestamp": datetime.fromtimestamp(timestamp),
                        "price": quote.get("close", [])[-1],
                        "open": quote.get("open", [])[-1],
                        "high": quote.get("high", [])[-1],
                        "low": quote.get("low", [])[-1],
                        "volume": quote.get("volume", [])[-1]
                    }
                    
                    # 添加到缓冲区
                    self._add_to_buffer(symbol, snapshot)
                    return snapshot
            
            return {}
            
        except Exception as e:
            logger.error(f"获取{symbol}的雅虎财经实时快照异常: {e}")
            return {}
            
    def _process_stream_data(self):
        """处理数据流"""
        # 雅虎财经不提供Websocket API，使用轮询方式
        for symbol in list(self.subscription_callbacks.keys()):
            try:
                snapshot = self.get_snapshot(symbol)
                
                if snapshot:
                    self.statistics["total_messages"] += 1
                    self.statistics["valid_messages"] += 1
                    self.statistics["last_message_time"] = datetime.now()
                    
                    # 调用回调函数
                    callback = self.subscription_callbacks.get(symbol)
                    if callback:
                        callback(snapshot)
            except Exception as e:
                self.statistics["error_messages"] += 1
                logger.error(f"处理{symbol}的雅虎财经数据流异常: {e}")
                
            time.sleep(1)  # 避免请求过于频繁


class EastMoneyRealTime(RealTimeDataSource):
    """东方财富实时数据源"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("east_money", config)
        self.user_id = config.get("user_id", "")
        self.password = config.get("password", "")
        self.token = None
        self.api_base_url = "https://push2.eastmoney.com/api/qt/stock/get"
        self.websocket = None
        
    def connect(self) -> bool:
        """连接到东方财富API"""
        try:
            # 模拟登录获取token
            if self.user_id and self.password:
                # 实际项目中实现真实的登录逻辑
                self.token = "sample_token"  # 示例token
                
            # 测试连接
            test_symbol = "000001"  # 平安银行
            response = requests.get(self.api_base_url, 
                                   params={"secid": f"0.{test_symbol}", "fields": "f43,f44,f45,f46,f47"})
            
            if response.status_code == 200:
                self.is_connected = True
                self._start_streaming_thread()
                logger.info("成功连接到东方财富API")
                return True
            else:
                logger.error(f"连接东方财富API失败: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"连接东方财富API异常: {e}")
            return False
            
    def disconnect(self) -> bool:
        """断开与东方财富API的连接"""
        self._stop_streaming_thread()
        if self.websocket:
            self.websocket.close()
            self.websocket = None
            
        self.is_connected = False
        self.token = None
        logger.info("已断开与东方财富API的连接")
        return True
        
    def is_available(self) -> bool:
        """检查东方财富API是否可用"""
        return self.is_connected and self.token is not None
        
    def subscribe(self, symbol: str, callback: Optional[Callable] = None) -> bool:
        """订阅东方财富实时数据"""
        if not self.is_connected:
            logger.warning(f"尚未连接到东方财富API，无法订阅{symbol}")
            return False
            
        # 判断是上海还是深圳
        market = "0" if symbol.startswith("6") else "1"
        secid = f"{market}.{symbol}"
        
        self.subscription_callbacks[symbol] = callback
        logger.info(f"已订阅{symbol}的东方财富实时数据")
        return True
        
    def unsubscribe(self, symbol: str) -> bool:
        """取消订阅东方财富实时数据"""
        if symbol in self.subscription_callbacks:
            del self.subscription_callbacks[symbol]
            logger.info(f"已取消订阅{symbol}的东方财富实时数据")
            return True
        return False
        
    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """获取东方财富实时快照"""
        try:
            # 判断是上海还是深圳
            market = "0" if symbol.startswith("6") else "1"
            secid = f"{market}.{symbol}"
            
            response = requests.get(self.api_base_url, 
                                   params={"secid": secid, 
                                          "fields": "f43,f44,f45,f46,f47,f48,f49,f50,f51,f52"})
            
            if response.status_code == 200:
                data = response.json()
                if data["rc"] == 0 and "data" in data:
                    quote = data["data"]
                    
                    snapshot = {
                        "symbol": symbol,
                        "timestamp": datetime.now(),
                        "price": quote.get("f43", 0) / 100.0,  # 价格统一以元为单位
                        "open": quote.get("f46", 0) / 100.0,
                        "high": quote.get("f44", 0) / 100.0,
                        "low": quote.get("f45", 0) / 100.0,
                        "volume": quote.get("f47", 0),
                        "amount": quote.get("f48", 0) / 100.0,  # 成交额
                        "bid_price": quote.get("f50", 0) / 100.0,  # 买一价
                        "ask_price": quote.get("f51", 0) / 100.0,  # 卖一价
                        "bid_volume": quote.get("f52", 0),  # 买一量
                        "ask_volume": quote.get("f53", 0)   # 卖一量
                    }
                    
                    # 添加到缓冲区
                    self._add_to_buffer(symbol, snapshot)
                    return snapshot
            
            return {}
            
        except Exception as e:
            logger.error(f"获取{symbol}的东方财富实时快照异常: {e}")
            return {}
            
    def _process_stream_data(self):
        """处理数据流"""
        # 东方财富数据轮询
        for symbol in list(self.subscription_callbacks.keys()):
            try:
                snapshot = self.get_snapshot(symbol)
                
                if snapshot:
                    self.statistics["total_messages"] += 1
                    self.statistics["valid_messages"] += 1
                    self.statistics["last_message_time"] = datetime.now()
                    
                    # 调用回调函数
                    callback = self.subscription_callbacks.get(symbol)
                    if callback:
                        callback(snapshot)
            except Exception as e:
                self.statistics["error_messages"] += 1
                logger.error(f"处理{symbol}的东方财富数据流异常: {e}")
                
            time.sleep(0.5)  # 避免请求过于频繁


# 注册所有实现的实时数据源
REALTIME_SOURCES = {
    "yahoo_finance": YahooFinanceRealTime,
    "east_money": EastMoneyRealTime
}

def get_realtime_source(source_name: str, config: Dict[str, Any]) -> Optional[RealTimeDataSource]:
    """
    获取实时数据源实例
    
    Args:
        source_name: 数据源名称
        config: 配置参数
        
    Returns:
        实时数据源实例，如果不存在则返回None
    """
    if source_name in REALTIME_SOURCES:
        return REALTIME_SOURCES[source_name](config)
    else:
        logger.error(f"未知的实时数据源: {source_name}")
        return None 