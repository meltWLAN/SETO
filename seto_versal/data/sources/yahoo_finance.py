#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
雅虎财经数据源实现

用于获取国际市场股票、ETF、期货、加密货币等数据
"""

import os
import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import json

from seto_versal.data.manager import DataSource, TimeFrame

# 尝试导入yfinance
try:
    import yfinance as yf
except ImportError:
    yf = None
    logging.warning("yfinance模块未安装，YahooFinanceDataSource将不可用")

logger = logging.getLogger(__name__)

class YahooFinanceDataSource(DataSource):
    """
    雅虎财经数据源实现
    
    用于获取国际市场的股票、ETF、期货、加密货币等数据
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化雅虎财经数据源
        
        Args:
            name: 数据源名称
            config: 配置参数
        """
        super().__init__(name, config)
        
        if yf is None:
            raise ImportError("yfinance模块未安装，请先安装: pip install yfinance")
        
        self.connected = False
        self.rate_limit = self.config.get('rate_limit', {'minute': 2000})
        self.request_count = 0
        self.last_request_time = datetime.now()
        
        # 数据缓存
        self.data_cache = {}
        self.cache_dir = self.config.get('cache_dir', 'data/cache/yahoo')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 支持的时间周期
        self.supported_timeframes = {
            TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15, 
            TimeFrame.MINUTE_30, TimeFrame.HOUR_1, TimeFrame.DAY_1, 
            TimeFrame.WEEK_1, TimeFrame.MONTH_1
        }
        
        # 市场列表
        self.markets = self.config.get('markets', ['US', 'HK', 'UK', 'JP'])
        
        # 热门股票列表
        self.popular_symbols = {
            'US': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'PG'],
            'HK': ['9988.HK', '0700.HK', '3690.HK', '9999.HK', '1810.HK'],
            'UK': ['HSBA.L', 'BP.L', 'GSK.L', 'AZN.L', 'ULVR.L'],
            'JP': ['7203.T', '9432.T', '9984.T', '6758.T', '6861.T'],
            'CRYPTO': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD'],
            'ETF': ['SPY', 'QQQ', 'VTI', 'ARKK', 'GLD']
        }
        
        # 符号映射：将内部符号映射到Yahoo Finance符号
        self.symbol_mapping = {}
        
    def connect(self) -> bool:
        """
        连接到雅虎财经API
        
        Returns:
            连接是否成功
        """
        try:
            # 测试连接，获取一个简单的数据
            test_symbol = 'AAPL'
            test_data = yf.download(test_symbol, period='1d', auto_adjust=True, progress=False)
            
            self.connected = not test_data.empty
            
            if self.connected:
                self.logger.info(f"成功连接到雅虎财经API")
            else:
                self.logger.error("连接雅虎财经API失败，返回的数据为空")
                
            return self.connected
            
        except Exception as e:
            self.connected = False
            self.logger.error(f"连接雅虎财经API失败: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        断开雅虎财经API连接
        
        Returns:
            断开是否成功
        """
        self.connected = False
        return True
    
    def is_connected(self) -> bool:
        """
        检查是否已连接到雅虎财经API
        
        Returns:
            是否已连接
        """
        return self.connected
    
    def _check_rate_limit(self) -> None:
        """
        检查和管理API请求频率限制
        """
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # 重置计数器（如果已过一分钟）
        if self.last_request_time < minute_ago:
            self.request_count = 0
            self.last_request_time = now
        
        # 检查是否超过限制
        if self.request_count >= self.rate_limit.get('minute', 2000):
            wait_time = 60 - (now - minute_ago).total_seconds()
            self.logger.warning(f"达到API请求频率限制，等待 {wait_time:.1f} 秒")
            time.sleep(max(1, wait_time))
            self.request_count = 0
            self.last_request_time = datetime.now()
        
        self.request_count += 1
        
    def _map_symbol(self, symbol: str) -> str:
        """
        将内部符号映射到雅虎财经符号
        
        Args:
            symbol: 内部符号
            
        Returns:
            雅虎财经符号
        """
        # 如果已有映射，直接返回
        if symbol in self.symbol_mapping:
            return self.symbol_mapping[symbol]
            
        # 默认假设符号格式已经兼容
        return symbol
        
    def _timeframe_to_yahoo(self, timeframe: TimeFrame) -> str:
        """
        转换TimeFrame到雅虎财经API参数
        
        Args:
            timeframe: 时间周期
            
        Returns:
            雅虎财经API时间周期参数
        """
        mapping = {
            TimeFrame.MINUTE_1: "1m",
            TimeFrame.MINUTE_5: "5m",
            TimeFrame.MINUTE_15: "15m",
            TimeFrame.MINUTE_30: "30m",
            TimeFrame.HOUR_1: "1h",
            TimeFrame.DAY_1: "1d",
            TimeFrame.WEEK_1: "1wk",
            TimeFrame.MONTH_1: "1mo"
        }
        
        return mapping.get(timeframe, "1d")
        
    def _period_for_timeframe(self, timeframe: TimeFrame, days: int) -> str:
        """
        根据时间周期确定API的period参数
        
        Args:
            timeframe: 时间周期
            days: 历史数据天数
            
        Returns:
            period参数值
        """
        # 对于分钟级别的数据，雅虎只提供最近7天
        if timeframe in [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15, TimeFrame.MINUTE_30]:
            return "7d"
            
        # 对于小时级别的数据，提供最近60天
        if timeframe == TimeFrame.HOUR_1:
            return "60d"
            
        # 对于日级别及以上的数据，根据请求的天数决定
        if days <= 30:
            return "1mo"
        elif days <= 90:
            return "3mo"
        elif days <= 180:
            return "6mo"
        elif days <= 365:
            return "1y"
        elif days <= 730:
            return "2y"
        elif days <= 1825:
            return "5y"
        else:
            return "max"
    
    def _standardize_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        标准化雅虎财经数据为统一格式
        
        Args:
            data: 原始数据
            symbol: 交易品种代码
            
        Returns:
            标准化后的数据
        """
        if data.empty:
            return pd.DataFrame()
            
        # 雅虎财经的列名：Open, High, Low, Close, Adj Close, Volume
        # 标准化列名为：timestamp, open, high, low, close, volume
        
        # 重命名列
        data = data.reset_index()
        data.columns = [col.lower() for col in data.columns]
        
        # 如果有adj close列，用它替换close
        if 'adj close' in data.columns:
            data['close'] = data['adj close']
            data = data.drop(columns=['adj close'])
            
        # 确保时间列名为timestamp
        if 'date' in data.columns:
            data = data.rename(columns={'date': 'timestamp'})
            
        # 添加symbol列
        data['symbol'] = symbol
        
        # 确保所有必要的列都存在
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        for col in required_columns:
            if col not in data.columns:
                if col == 'volume':
                    data[col] = 0
                elif col != 'timestamp' and col != 'symbol':
                    data[col] = data['close']
                    
        # 确保timestamp是datetime类型
        if data['timestamp'].dtype != 'datetime64[ns]':
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
        # 按时间排序
        data = data.sort_values('timestamp')
        
        # 删除重复的时间戳
        data = data.drop_duplicates(subset=['timestamp'])
        
        # 设置索引为timestamp
        data = data.set_index('timestamp')
        
        return data
    
    def get_symbols(self) -> List[str]:
        """
        获取数据源支持的所有交易品种
        
        Returns:
            交易品种列表
        """
        # 这里我们只返回一些热门的股票作为示例
        # 实际应用中，可以使用雅虎财经的API获取更完整的列表
        all_symbols = []
        for market, symbols in self.popular_symbols.items():
            all_symbols.extend(symbols)
            
        return all_symbols
    
    def get_historical_data(self, 
                          symbol: str, 
                          timeframe: TimeFrame, 
                          start_time: datetime, 
                          end_time: Optional[datetime] = None, 
                          limit: Optional[int] = None) -> pd.DataFrame:
        """
        获取历史市场数据
        
        Args:
            symbol: 交易品种代码
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间，默认为当前时间
            limit: 返回的最大数据条数
            
        Returns:
            包含历史数据的DataFrame
        """
        if not self.is_connected():
            self.connect()
            
        if not self.is_connected():
            self.logger.error(f"获取历史数据失败: 未连接到雅虎财经API")
            return pd.DataFrame()
            
        if not self.validate_timeframe(timeframe):
            self.logger.error(f"获取历史数据失败: 不支持的时间周期 {timeframe}")
            return pd.DataFrame()
            
        # 检查缓存中是否有最近的数据
        cache_key = f"{symbol}_{timeframe.value}"
        if cache_key in self.data_cache:
            cached_data = self.data_cache[cache_key]
            
            # 如果缓存数据覆盖了请求的时间范围，直接返回
            if (cached_data.index.min() <= start_time and 
                (end_time is None or cached_data.index.max() >= end_time)):
                
                # 过滤所需的时间范围
                filtered_data = cached_data[(cached_data.index >= start_time)]
                if end_time is not None:
                    filtered_data = filtered_data[(filtered_data.index <= end_time)]
                    
                # 应用数量限制
                if limit is not None and len(filtered_data) > limit:
                    filtered_data = filtered_data.iloc[-limit:]
                    
                return filtered_data.reset_index()
                
        # 检查频率限制
        self._check_rate_limit()
        
        try:
            yahoo_symbol = self._map_symbol(symbol)
            yahoo_interval = self._timeframe_to_yahoo(timeframe)
            
            # 对于end_time，如果未提供，使用当前时间
            if end_time is None:
                end_time = datetime.now()
                
            # 计算请求数据的天数
            days = (end_time - start_time).days + 1
            
            # 根据时间周期确定period参数
            period = self._period_for_timeframe(timeframe, days)
            
            # 根据period的限制，可能需要使用start和end参数
            if timeframe in [TimeFrame.DAY_1, TimeFrame.WEEK_1, TimeFrame.MONTH_1]:
                # 这些时间周期支持明确的开始和结束日期
                data = yf.download(
                    yahoo_symbol,
                    start=start_time.strftime('%Y-%m-%d'),
                    end=end_time.strftime('%Y-%m-%d'),
                    interval=yahoo_interval,
                    auto_adjust=True,
                    progress=False
                )
            else:
                # 分钟和小时级别只能使用period，可能无法获取特定日期范围
                data = yf.download(
                    yahoo_symbol,
                    period=period,
                    interval=yahoo_interval,
                    auto_adjust=True,
                    progress=False
                )
                
                # 过滤指定的时间范围
                data = data[(data.index >= start_time) & (data.index <= end_time)]
            
            # 标准化数据
            standardized_data = self._standardize_data(data, symbol)
            
            # 应用数量限制
            if limit is not None and len(standardized_data) > limit:
                standardized_data = standardized_data.iloc[-limit:]
                
            # 更新缓存
            self.data_cache[cache_key] = standardized_data
            
            # 缓存到文件
            self._save_to_cache(symbol, timeframe, standardized_data)
            
            return standardized_data.reset_index()
            
        except Exception as e:
            self.logger.error(f"获取历史数据失败: {symbol} {timeframe}, 错误: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_data(self, 
                      symbol: str, 
                      timeframe: TimeFrame) -> pd.DataFrame:
        """
        获取最新的市场数据
        
        Args:
            symbol: 交易品种代码
            timeframe: 时间周期
            
        Returns:
            包含最新数据的DataFrame
        """
        # 对于最新数据，我们获取最近一段时间的数据并返回最新的几条
        now = datetime.now()
        
        if timeframe == TimeFrame.MINUTE_1 or timeframe == TimeFrame.MINUTE_5:
            # 对于1分钟和5分钟数据，获取最近1天
            start_time = now - timedelta(days=1)
        elif timeframe == TimeFrame.MINUTE_15 or timeframe == TimeFrame.MINUTE_30:
            # 对于15分钟和30分钟数据，获取最近2天
            start_time = now - timedelta(days=2)
        elif timeframe == TimeFrame.HOUR_1:
            # 对于1小时数据，获取最近5天
            start_time = now - timedelta(days=5)
        else:
            # 对于更长周期，获取最近30天
            start_time = now - timedelta(days=30)
            
        # 获取历史数据
        data = self.get_historical_data(symbol, timeframe, start_time, now)
        
        # 如果数据为空，返回空DataFrame
        if data.empty:
            return pd.DataFrame()
            
        # 返回最新的几条数据
        limit = 10  # 返回最新10条
        if len(data) > limit:
            return data.iloc[-limit:]
        else:
            return data
            
    def _save_to_cache(self, symbol: str, timeframe: TimeFrame, data: pd.DataFrame) -> None:
        """
        保存数据到缓存文件
        
        Args:
            symbol: 交易品种代码
            timeframe: 时间周期
            data: 要缓存的数据
        """
        if data.empty:
            return
            
        try:
            # 构造缓存文件路径
            cache_file = os.path.join(self.cache_dir, f"{symbol}_{timeframe.value}.csv")
            
            # 保存到CSV文件
            data.to_csv(cache_file)
            
        except Exception as e:
            self.logger.error(f"保存缓存数据失败: {e}") 