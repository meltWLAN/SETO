#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal 增强型数据管理器

提供高性能、高可扩展性的数据管理功能，集成数据源、缓存和处理流水线。
支持多数据源访问、智能缓存管理和数据处理流水线。
"""

import os
import logging
import pandas as pd
import numpy as np
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from seto_versal.data.manager import DataManager, DataSource, TimeFrame
from seto_versal.data.cache import DataCache, CacheLevel
from seto_versal.data.processor import ProcessingPipeline, DataCleaner, FeatureGenerator

# 导入数据源模块
try:
    import tushare as ts
except ImportError:
    ts = None
    logging.warning("TuShare模块未安装，TuShare数据源将不可用")

try:
    import akshare as ak
except ImportError:
    ak = None
    logging.warning("AKShare模块未安装，AKShare数据源将不可用")

# 导入新的数据源
try:
    from seto_versal.data.sources.yahoo_finance import YahooFinanceDataSource
except ImportError:
    YahooFinanceDataSource = None

logger = logging.getLogger(__name__)

class TuShareDataSource(DataSource):
    """
    TuShare数据源实现
    
    用于从TuShare API获取中国股票市场数据
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化TuShare数据源
        
        Args:
            name: 数据源名称
            config: 配置参数，必须包含api_token
        """
        super().__init__(name, config)
        
        if ts is None:
            raise ImportError("TuShare模块未安装，请先安装: pip install tushare")
        
        self.api_token = self.config.get('api_token')
        if not self.api_token:
            raise ValueError("TuShare数据源需要提供API Token")
        
        self.api = None
        self.connected = False
        self.rate_limit = self.config.get('rate_limit', {'minute': 500})
        self.request_count = 0
        self.last_request_time = datetime.now()
        
        # 数据缓存
        self.data_cache = {}
        self.cache_dir = self.config.get('cache_dir', 'data/cache/tushare')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 支持的时间周期
        self.supported_timeframes = {
            TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15, 
            TimeFrame.MINUTE_30, TimeFrame.HOUR_1, TimeFrame.DAY_1, 
            TimeFrame.WEEK_1, TimeFrame.MONTH_1
        }
        
        # 市场列表
        self.markets = ['SSE', 'SZSE']  # 上交所、深交所
    
    def connect(self) -> bool:
        """
        连接到TuShare API
        
        Returns:
            连接是否成功
        """
        try:
            self.api = ts.pro_api(self.api_token)
            self.connected = True
            self.logger.info(f"成功连接到TuShare API")
            
            # 尝试获取部分股票列表以验证连接
            self._get_stock_list()
            
            return True
        except Exception as e:
            self.connected = False
            self.logger.error(f"连接TuShare API失败: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        断开TuShare API连接
        
        Returns:
            断开是否成功
        """
        self.connected = False
        self.api = None
        return True
    
    def is_connected(self) -> bool:
        """
        检查是否已连接到TuShare API
        
        Returns:
            是否已连接
        """
        return self.connected and self.api is not None
    
    def _get_stock_list(self) -> List[str]:
        """
        获取股票列表
        
        Returns:
            股票代码列表
        """
        if not self.is_connected():
            return []
        
        try:
            # 获取上市公司股票列表
            data = self.api.stock_basic(exchange='', list_status='L', 
                                     fields='ts_code,name,area,industry,list_date')
            self.available_symbols = data['ts_code'].tolist()
            return self.available_symbols
        except Exception as e:
            self.logger.error(f"获取股票列表失败: {e}")
            return []
    
    def get_symbols(self) -> List[str]:
        """
        获取数据源支持的所有交易品种
        
        Returns:
            交易品种列表
        """
        if not self.available_symbols:
            self._get_stock_list()
        return self.available_symbols
    
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
        if self.request_count >= self.rate_limit.get('minute', 500):
            wait_time = 60 - (now - minute_ago).total_seconds()
            self.logger.warning(f"达到API请求频率限制，等待 {wait_time:.1f} 秒")
            time.sleep(max(1, wait_time))
            self.request_count = 0
            self.last_request_time = datetime.now()
        
        self.request_count += 1
    
    def _format_time(self, dt: datetime) -> str:
        """
        格式化时间为TuShare API要求的格式
        
        Args:
            dt: 要格式化的时间
            
        Returns:
            格式化后的时间字符串
        """
        return dt.strftime("%Y%m%d")
    
    def _timeframe_to_tushare(self, timeframe: TimeFrame) -> str:
        """
        转换TimeFrame到TuShare频率
        
        Args:
            timeframe: 时间周期
            
        Returns:
            TuShare频率代码
        """
        mapping = {
            TimeFrame.MINUTE_1: '1min',
            TimeFrame.MINUTE_5: '5min',
            TimeFrame.MINUTE_15: '15min',
            TimeFrame.MINUTE_30: '30min',
            TimeFrame.HOUR_1: '60min',
            TimeFrame.DAY_1: 'D',
            TimeFrame.WEEK_1: 'W',
            TimeFrame.MONTH_1: 'M'
        }
        return mapping.get(timeframe, 'D')
    
    def _standardize_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        将TuShare返回的数据标准化为统一格式
        
        Args:
            data: TuShare API返回的原始数据
            symbol: 交易品种代码
            
        Returns:
            标准化后的DataFrame
        """
        if data.empty:
            return pd.DataFrame()
        
        # 重命名列以匹配标准格式
        column_map = {
            'trade_date': 'datetime',
            'ts_code': 'symbol',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'amount': 'amount',
            'change': 'change',
            'pct_chg': 'pct_change'
        }
        
        # 日期时间处理
        if 'trade_date' in data.columns:
            data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
        elif 'trade_time' in data.columns:
            data['trade_time'] = pd.to_datetime(data['trade_time'])
            data.rename(columns={'trade_time': 'datetime'}, inplace=True)
        
        # 重命名列
        df = data.rename(columns={k: v for k, v in column_map.items() if k in data.columns})
        
        # 确保有symbol列
        if 'symbol' not in df.columns:
            df['symbol'] = symbol
        
        # 确保必需的列存在
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'datetime' and 'trade_date' in data.columns:
                    df['datetime'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
                else:
                    df[col] = np.nan
        
        # 设置索引
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
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
            if not self.connect():
                return pd.DataFrame()
        
        if not self.validate_timeframe(timeframe):
            self.logger.error(f"不支持的时间周期: {timeframe}")
            return pd.DataFrame()
        
        if not self.validate_symbol(symbol):
            self.logger.error(f"不支持的交易品种: {symbol}")
            return pd.DataFrame()
        
        # 设置默认结束时间
        if end_time is None:
            end_time = datetime.now()
        
        # 检查请求频率限制
        self._check_rate_limit()
        
        try:
            # 日频及以上周期使用daily接口
            if timeframe in [TimeFrame.DAY_1, TimeFrame.WEEK_1, TimeFrame.MONTH_1]:
                data = self.api.daily(ts_code=symbol, 
                                   start_date=self._format_time(start_time),
                                   end_date=self._format_time(end_time))
            # 分钟级别使用分钟接口
            else:
                freq = self._timeframe_to_tushare(timeframe)
                data = ts.pro_bar(ts_code=symbol, 
                                freq=freq,
                                start_date=self._format_time(start_time),
                                end_date=self._format_time(end_time))
            
            if data is None or data.empty:
                self.logger.warning(f"未获取到数据: {symbol}, {timeframe}, {start_time} - {end_time}")
                return pd.DataFrame()
            
            # 标准化数据
            df = self._standardize_data(data, symbol)
            
            # 应用limit限制
            if limit is not None:
                df = df.iloc[-limit:]
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取历史数据失败: {symbol}, {timeframe}, {e}")
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
        # 对于日频及以上，获取最近的10个周期
        if timeframe in [TimeFrame.DAY_1, TimeFrame.WEEK_1, TimeFrame.MONTH_1]:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)  # 获取最近30天的数据
            df = self.get_historical_data(symbol, timeframe, start_time, end_time, 10)
            return df.iloc[-1:] if not df.empty else df
        
        # 对于分钟级别，使用实时行情接口
        try:
            # 检查请求频率限制
            self._check_rate_limit()
            
            # 获取实时行情
            data = ts.get_realtime_quotes(symbol.split('.')[0])
            
            if data is None or data.empty:
                self.logger.warning(f"未获取到实时数据: {symbol}")
                return pd.DataFrame()
            
            # 转换实时行情为OHLCV格式
            df = pd.DataFrame({
                'datetime': [datetime.now()],
                'symbol': [symbol],
                'open': [float(data.iloc[0]['open'])],
                'high': [float(data.iloc[0]['high'])],
                'low': [float(data.iloc[0]['low'])],
                'close': [float(data.iloc[0]['price'])],
                'volume': [float(data.iloc[0]['volume'])]
            })
            
            df.set_index('datetime', inplace=True)
            return df
            
        except Exception as e:
            self.logger.error(f"获取最新数据失败: {symbol}, {e}")
            return pd.DataFrame()

class AKShareDataSource(DataSource):
    """
    AKShare数据源实现
    
    用于从AKShare库获取中国股票市场数据，作为TuShare的备选
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化AKShare数据源
        
        Args:
            name: 数据源名称
            config: 配置参数
        """
        super().__init__(name, config)
        
        if ak is None:
            raise ImportError("AKShare模块未安装，请先安装: pip install akshare")
        
        self.connected = False
        
        # 请求速率限制
        self.rate_limit = self.config.get('rate_limit', {'minute': 300})
        self.request_count = 0
        self.last_request_time = datetime.now()
        
        # 数据缓存
        self.data_cache = {}
        self.cache_dir = self.config.get('cache_dir', 'data/cache/akshare')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 支持的时间周期
        self.supported_timeframes = {
            TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15, 
            TimeFrame.MINUTE_30, TimeFrame.HOUR_1, TimeFrame.DAY_1, 
            TimeFrame.WEEK_1, TimeFrame.MONTH_1
        }
    
    def connect(self) -> bool:
        """
        连接到AKShare
        
        Returns:
            连接是否成功
        """
        try:
            # 获取股票列表以验证连接
            self._get_stock_list()
            self.connected = True
            self.logger.info(f"成功连接到AKShare")
            return True
        except Exception as e:
            self.connected = False
            self.logger.error(f"连接AKShare失败: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        断开AKShare连接
        
        Returns:
            断开是否成功
        """
        self.connected = False
        return True
    
    def is_connected(self) -> bool:
        """
        检查是否已连接到AKShare
        
        Returns:
            是否已连接
        """
        return self.connected
    
    def _get_stock_list(self) -> List[str]:
        """
        获取股票列表
        
        Returns:
            股票代码列表
        """
        try:
            # 获取A股股票列表
            a_stock = ak.stock_info_a_code_name()
            # 转换为标准格式
            symbols = []
            for _, row in a_stock.iterrows():
                code = row['code']
                # 转换为 tushare 格式的代码（带市场后缀）
                if code.startswith('6'):
                    symbols.append(f"{code}.SH")
                else:
                    symbols.append(f"{code}.SZ")
            
            self.available_symbols = symbols
            return symbols
        except Exception as e:
            self.logger.error(f"获取股票列表失败: {e}")
            return []
    
    def get_symbols(self) -> List[str]:
        """
        获取数据源支持的所有交易品种
        
        Returns:
            交易品种列表
        """
        if not self.available_symbols:
            self._get_stock_list()
        return self.available_symbols
    
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
        if self.request_count >= self.rate_limit.get('minute', 300):
            wait_time = 60 - (now - minute_ago).total_seconds()
            self.logger.warning(f"达到API请求频率限制，等待 {wait_time:.1f} 秒")
            time.sleep(max(1, wait_time))
            self.request_count = 0
            self.last_request_time = datetime.now()
        
        self.request_count += 1
    
    def _format_symbol(self, symbol: str) -> str:
        """
        将统一格式的股票代码转换为AKShare格式
        
        Args:
            symbol: 统一格式的股票代码（如600000.SH）
            
        Returns:
            AKShare格式的股票代码
        """
        if '.' not in symbol:
            return symbol
        
        code, market = symbol.split('.')
        
        if market == 'SH':
            return f"sh{code}"
        elif market == 'SZ':
            return f"sz{code}"
        return symbol
    
    def _timeframe_to_akshare(self, timeframe: TimeFrame) -> str:
        """
        转换TimeFrame到AKShare频率
        
        Args:
            timeframe: 时间周期
            
        Returns:
            AKShare频率代码
        """
        mapping = {
            TimeFrame.MINUTE_1: '1',
            TimeFrame.MINUTE_5: '5',
            TimeFrame.MINUTE_15: '15',
            TimeFrame.MINUTE_30: '30',
            TimeFrame.HOUR_1: '60',
            TimeFrame.DAY_1: 'daily',
            TimeFrame.WEEK_1: 'weekly',
            TimeFrame.MONTH_1: 'monthly'
        }
        return mapping.get(timeframe, 'daily')
    
    def _standardize_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        将AKShare返回的数据标准化为统一格式
        
        Args:
            data: AKShare返回的原始数据
            symbol: 交易品种代码
            
        Returns:
            标准化后的DataFrame
        """
        if data.empty:
            return pd.DataFrame()
        
        # 重命名列以匹配标准格式
        column_map = {
            '日期': 'datetime',
            'date': 'datetime',
            '开盘': 'open',
            'open': 'open',
            '最高': 'high',
            'high': 'high',
            '最低': 'low',
            'low': 'low',
            '收盘': 'close',
            'close': 'close',
            '成交量': 'volume',
            'volume': 'volume',
            '成交额': 'amount',
            'amount': 'amount',
            '涨跌幅': 'pct_change',
            'change': 'change'
        }
        
        # 重命名列
        df = data.rename(columns={k: v for k, v in column_map.items() if k in data.columns})
        
        # 确保有symbol列
        if 'symbol' not in df.columns:
            df['symbol'] = symbol
        
        # 确保必需的列存在
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'datetime' and '日期' in data.columns:
                    df['datetime'] = pd.to_datetime(data['日期'])
                else:
                    df[col] = np.nan
        
        # 日期时间处理
        if 'datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 设置索引
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
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
            if not self.connect():
                return pd.DataFrame()
        
        if not self.validate_timeframe(timeframe):
            self.logger.error(f"不支持的时间周期: {timeframe}")
            return pd.DataFrame()
        
        # 设置默认结束时间
        if end_time is None:
            end_time = datetime.now()
        
        # 检查请求频率限制
        self._check_rate_limit()
        
        try:
            ak_symbol = self._format_symbol(symbol)
            period = self._timeframe_to_akshare(timeframe)
            
            # 日线及以上数据
            if timeframe in [TimeFrame.DAY_1, TimeFrame.WEEK_1, TimeFrame.MONTH_1]:
                # 获取日K线数据
                if timeframe == TimeFrame.DAY_1:
                    data = ak.stock_zh_a_hist(symbol=ak_symbol.replace('sh', '').replace('sz', ''), 
                                           period="daily",
                                           start_date=start_time.strftime('%Y%m%d'),
                                           end_date=end_time.strftime('%Y%m%d'),
                                           adjust="qfq")
                # 周线数据
                elif timeframe == TimeFrame.WEEK_1:
                    data = ak.stock_zh_a_hist(symbol=ak_symbol.replace('sh', '').replace('sz', ''), 
                                           period="weekly",
                                           start_date=start_time.strftime('%Y%m%d'),
                                           end_date=end_time.strftime('%Y%m%d'),
                                           adjust="qfq")
                # 月线数据
                else:
                    data = ak.stock_zh_a_hist(symbol=ak_symbol.replace('sh', '').replace('sz', ''), 
                                           period="monthly",
                                           start_date=start_time.strftime('%Y%m%d'),
                                           end_date=end_time.strftime('%Y%m%d'),
                                           adjust="qfq")
            # 分钟数据
            else:
                # AKShare对分钟数据的限制较多，只能获取最近的数据
                data = ak.stock_zh_a_minute(symbol=ak_symbol, period=period)
            
            if data is None or data.empty:
                self.logger.warning(f"未获取到数据: {symbol}, {timeframe}, {start_time} - {end_time}")
                return pd.DataFrame()
            
            # 标准化数据
            df = self._standardize_data(data, symbol)
            
            # 筛选时间范围
            df = df[(df.index >= pd.Timestamp(start_time)) & (df.index <= pd.Timestamp(end_time))]
            
            # 应用limit限制
            if limit is not None:
                df = df.iloc[-limit:]
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取历史数据失败: {symbol}, {timeframe}, {e}")
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
        try:
            # 检查请求频率限制
            self._check_rate_limit()
            
            ak_symbol = self._format_symbol(symbol)
            
            # 获取实时行情
            data = ak.stock_zh_a_spot_em()
            
            # 筛选指定股票
            symbol_code = symbol.split('.')[0]
            stock_data = data[data['代码'] == symbol_code]
            
            if stock_data.empty:
                self.logger.warning(f"未获取到实时数据: {symbol}")
                return pd.DataFrame()
            
            # 转换为标准格式
            now = datetime.now()
            df = pd.DataFrame({
                'datetime': [now],
                'symbol': [symbol],
                'open': [float(stock_data.iloc[0]['今开'])],
                'high': [float(stock_data.iloc[0]['最高'])],
                'low': [float(stock_data.iloc[0]['最低'])],
                'close': [float(stock_data.iloc[0]['最新价'])],
                'volume': [float(stock_data.iloc[0]['成交量'])],
                'amount': [float(stock_data.iloc[0]['成交额'])],
                'pct_change': [float(stock_data.iloc[0]['涨跌幅'])],
            })
            
            df.set_index('datetime', inplace=True)
            return df
            
        except Exception as e:
            self.logger.error(f"获取最新数据失败: {symbol}, {e}")
            return pd.DataFrame()

class EnhancedDataManager:
    """
    增强型数据管理器，提供高性能数据访问和处理
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化增强型数据管理器
        
        Args:
            config: 配置参数，包括数据源、缓存、处理流水线等配置
        """
        self.config = config or {}
        
        # 初始化基础数据管理器
        self.data_manager = DataManager(self.config.get('data_manager', {}))
        
        # 初始化数据缓存
        self.cache = DataCache(self.config.get('cache', {}))
        
        # 初始化处理流水线
        self.pipelines: Dict[str, ProcessingPipeline] = {}
        
        # 创建默认流水线
        self._create_default_pipeline()
        
        # 加载已配置的流水线
        self._load_pipelines()
        
        # 多线程支持
        self.max_workers = self.config.get('max_workers', 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # 记录请求的数据集以便预加载
        self.frequently_accessed = {}
        self.last_preload_time = datetime.now()
        self.preload_interval = self.config.get('preload_interval_seconds', 3600)
        
        # 统计信息
        self.stats = {
            'data_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_time': 0.0,
            'total_time': 0.0,
            'source_switches': 0
        }
        
        # 主要数据源和备选数据源
        self.primary_source = None
        self.backup_source = None
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 设置数据源
        self._setup_data_sources()
        
        # 初始化数据源
        self.data_sources: Dict[str, DataSource] = {}
        self._init_data_sources()
        
        logger.info("增强型数据管理器初始化完成")
    
    def _setup_data_sources(self) -> None:
        """设置主要数据源和备选数据源"""
        tushare_config = self.config.get('tushare', {})
        akshare_config = self.config.get('akshare', {})
        
        # 尝试创建TuShare数据源
        if ts is not None and 'api_token' in tushare_config:
            try:
                tushare_source = TuShareDataSource('tushare', tushare_config)
                if tushare_source.connect():
                    self.add_data_source(tushare_source, is_default=True)
                    self.primary_source = 'tushare'
                    logger.info("TuShare数据源设置为主要数据源")
                else:
                    logger.warning("TuShare数据源连接失败")
            except Exception as e:
                logger.error(f"创建TuShare数据源失败: {e}")
        
        # 尝试创建AKShare数据源
        if ak is not None:
            try:
                akshare_source = AKShareDataSource('akshare', akshare_config)
                if akshare_source.connect():
                    self.add_data_source(akshare_source)
                    if self.primary_source is None:
                        self.primary_source = 'akshare'
                        self.data_manager.add_data_source(akshare_source, is_default=True)
                        logger.info("AKShare数据源设置为主要数据源")
                    else:
                        self.backup_source = 'akshare'
                        logger.info("AKShare数据源设置为备选数据源")
                else:
                    logger.warning("AKShare数据源连接失败")
            except Exception as e:
                logger.error(f"创建AKShare数据源失败: {e}")
        
        # 如果没有可用数据源，创建CSV数据源作为最后的备选
        if self.primary_source is None:
            try:
                csv_path = self.config.get('csv_path', 'data/market')
                os.makedirs(csv_path, exist_ok=True)
                csv_source = self.data_manager.create_csv_data_source('csv', csv_path, is_default=True)
                if csv_source:
                    self.primary_source = 'csv'
                    logger.info("CSV数据源设置为主要数据源")
            except Exception as e:
                logger.error(f"创建CSV数据源失败: {e}")
    
    def _create_default_pipeline(self) -> None:
        """创建默认数据处理流水线"""
        default_pipeline = ProcessingPipeline("default")
        
        # 添加默认清洗处理器
        cleaner = DataCleaner("default_cleaner", {
            'fill_method': 'ffill',
            'drop_na': False,
            'remove_outliers': True,
            'outlier_std_threshold': 3.0
        })
        default_pipeline.add_processor(cleaner)
        
        # 添加默认特征生成处理器
        feature_generator = FeatureGenerator("default_features", {
            'features': [
                {'type': 'sma', 'window': 20},
                {'type': 'ema', 'window': 9},
                {'type': 'rsi', 'window': 14},
                {'type': 'bollinger', 'window': 20, 'std_dev': 2}
            ],
            'price_col': 'close',
            'volume_col': 'volume',
            'inplace': True
        })
        default_pipeline.add_processor(feature_generator)
        
        # 添加到流水线字典
        self.pipelines["default"] = default_pipeline
    
    def _load_pipelines(self) -> None:
        """从配置加载处理流水线"""
        pipeline_configs = self.config.get('pipelines', {})
        
        for name, config in pipeline_configs.items():
            if name == "default" and "default" in self.pipelines:
                # 跳过默认流水线，因为已经创建
                continue
                
            pipeline = ProcessingPipeline(name)
            
            # 加载处理器
            for processor_config in config.get('processors', []):
                processor_type = processor_config.get('type')
                processor_name = processor_config.get('name')
                
                if processor_type == 'cleaner':
                    processor = DataCleaner(processor_name, processor_config)
                elif processor_type == 'feature':
                    processor = FeatureGenerator(processor_name, processor_config)
                else:
                    logger.warning(f"未知处理器类型: {processor_type}")
                    continue
                
                pipeline.add_processor(processor)
            
            self.pipelines[name] = pipeline
            logger.info(f"已加载流水线: {name} 包含 {len(pipeline.processors)} 个处理器")
    
    def add_data_source(self, source: DataSource, is_default: bool = False) -> bool:
        """
        添加数据源
        
        Args:
            source: 要添加的数据源
            is_default: 是否设为默认数据源
            
        Returns:
            是否成功添加
        """
        return self.data_manager.add_data_source(source, is_default)
    
    def add_pipeline(self, pipeline: ProcessingPipeline) -> None:
        """
        添加处理流水线
        
        Args:
            pipeline: 要添加的处理流水线
        """
        with self._lock:
            self.pipelines[pipeline.name] = pipeline
            logger.info(f"添加流水线: {pipeline.name}")
    
    def get_historical_data(self, 
                          symbol: str, 
                          timeframe: TimeFrame, 
                          start_time: datetime, 
                          end_time: Optional[datetime] = None, 
                          limit: Optional[int] = None,
                          source_name: Optional[str] = None,
                          pipeline_name: str = "default",
                          use_cache: bool = True) -> pd.DataFrame:
        """
        获取历史市场数据，支持自动数据源切换和缓存
        
        Args:
            symbol: 交易品种代码
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间，默认为当前时间
            limit: 返回的最大数据条数
            source_name: 数据源名称，如果为None则使用默认数据源
            pipeline_name: 处理流水线名称，如果为None则不应用处理
            use_cache: 是否使用缓存
            
        Returns:
            处理后的历史市场数据
        """
        with self._lock:
            self.stats['data_requests'] += 1
        
        total_start_time = datetime.now()
        
        # 记录频繁访问的数据
        self._record_access(symbol, timeframe)
        
        # 设置默认结束时间
        if end_time is None:
            end_time = datetime.now()
        
        # 缓存键
        cache_key = f"{symbol}_{timeframe}_{start_time.isoformat()}_{end_time.isoformat()}_{limit}"
        
        # 检查缓存
        if use_cache:
            cached_data = self.cache.get(cache_key, CacheLevel.MEMORY)
            if cached_data is not None:
                with self._lock:
                    self.stats['cache_hits'] += 1
                logger.debug(f"缓存命中: {cache_key}")
                
                # 应用处理流水线
                if pipeline_name and pipeline_name in self.pipelines:
                    pipeline_start = datetime.now()
                    processed_data = self.pipelines[pipeline_name].process(cached_data.copy())
                    with self._lock:
                        self.stats['processing_time'] += (datetime.now() - pipeline_start).total_seconds()
                    return processed_data
                return cached_data
        
        with self._lock:
            self.stats['cache_misses'] += 1
        
        # 尝试从指定数据源获取数据
        if source_name:
            data = self._get_data_from_source(symbol, timeframe, start_time, end_time, limit, source_name)
        # 否则先尝试主数据源，如果失败则尝试备选数据源
        else:
            data = self._get_data_with_fallback(symbol, timeframe, start_time, end_time, limit)
        
        if data.empty:
            logger.warning(f"未能获取数据: {symbol} {timeframe} {start_time}-{end_time}")
            return pd.DataFrame()
        
        # 保存到缓存
        if use_cache and not data.empty:
            self.cache.set(cache_key, data, CacheLevel.MEMORY)
            # 对于日线数据，还保存到磁盘缓存
            if timeframe in [TimeFrame.DAY_1, TimeFrame.WEEK_1, TimeFrame.MONTH_1]:
                self.cache.set(cache_key, data, CacheLevel.DISK)
        
        # 应用处理流水线
        if pipeline_name and pipeline_name in self.pipelines:
            pipeline_start = datetime.now()
            processed_data = self.pipelines[pipeline_name].process(data.copy())
            with self._lock:
                self.stats['processing_time'] += (datetime.now() - pipeline_start).total_seconds()
            result = processed_data
        else:
            result = data
            
        # 更新统计信息
        with self._lock:
            self.stats['total_time'] += (datetime.now() - total_start_time).total_seconds()
            
        # 检查是否需要预加载数据
        self._check_preload()
            
        return result
    
    def _get_data_from_source(self, symbol, timeframe, start_time, end_time, limit, source_name):
        """从指定数据源获取数据"""
        try:
            source = self.data_manager.get_data_source(source_name)
            if not source:
                logger.error(f"数据源不存在: {source_name}")
                return pd.DataFrame()
                
            return self.data_manager.get_historical_data(
                symbol, timeframe, start_time, end_time, limit, source_name
            )
        except Exception as e:
            logger.error(f"从{source_name}获取数据失败: {e}")
            return pd.DataFrame()
    
    def _get_data_with_fallback(self, symbol, timeframe, start_time, end_time, limit):
        """尝试主数据源，失败时切换到备选数据源"""
        # 先尝试主数据源
        if self.primary_source:
            data = self._get_data_from_source(symbol, timeframe, start_time, end_time, limit, self.primary_source)
            if not data.empty:
                return data
        
        # 如果主数据源失败且有备选数据源，尝试备选数据源
        if self.backup_source:
            logger.warning(f"主数据源{self.primary_source}获取失败，尝试备选数据源{self.backup_source}")
            with self._lock:
                self.stats['source_switches'] += 1
            return self._get_data_from_source(symbol, timeframe, start_time, end_time, limit, self.backup_source)
        
        return pd.DataFrame()
    
    def preload_market_data(self, symbols: List[str], timeframe: TimeFrame, days: int = 90) -> None:
        """
        预加载市场数据到缓存
        
        Args:
            symbols: 要预加载的交易品种列表
            timeframe: 时间周期
            days: 预加载的天数
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        logger.info(f"开始预加载 {len(symbols)} 个交易品种的 {timeframe} 数据...")
        
        futures = []
        for symbol in symbols:
            future = self.executor.submit(
                self.get_historical_data,
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                use_cache=False  # 强制从数据源获取新数据
            )
            futures.append((symbol, future))
        
        # 等待所有预加载任务完成
        success_count = 0
        for symbol, future in futures:
            try:
                data = future.result()
                if not data.empty:
                    success_count += 1
            except Exception as e:
                logger.error(f"预加载数据失败: {symbol}, {e}")
        
        logger.info(f"预加载完成. 成功: {success_count}/{len(symbols)}")
    
    def preload_fundamentals(self, symbols: List[str]) -> None:
        """
        预加载基本面数据
        
        Args:
            symbols: 要预加载的交易品种列表
        """
        logger.info(f"开始预加载 {len(symbols)} 个交易品种的基本面数据...")
        
        # 这里可以实现基本面数据的预加载逻辑
        # TODO: 实现基本面数据预加载
        
        logger.info(f"基本面数据预加载完成")
        
    def get_market_status(self) -> Dict[str, Any]:
        """
        获取市场和数据源状态
        
        Returns:
            包含市场和数据源状态的字典
        """
        status = {
            'primary_source': self.primary_source,
            'backup_source': self.backup_source,
            'stats': self.get_manager_stats(),
            'cache_stats': self.get_cache_stats(),
            'pipelines': self.get_pipelines(),
        }
        return status
    
    def get_latest_data(self, 
                      symbol: str, 
                      timeframe: TimeFrame,
                      source_name: Optional[str] = None,
                      pipeline_name: str = "default",
                      use_cache: bool = True) -> pd.DataFrame:
        """
        获取最新的市场数据，应用缓存和处理流水线
        
        Args:
            symbol: 交易品种代码
            timeframe: 时间周期
            source_name: 数据源名称，如果为None则使用默认数据源
            pipeline_name: 处理流水线名称，如果为None则不应用处理
            use_cache: 是否使用缓存
            
        Returns:
            处理后的最新市场数据
        """
        # 从数据管理器获取原始数据
        data = self.data_manager.get_latest_data(symbol, timeframe, source_name)
        
        # 记录频繁访问的数据
        self._record_access(symbol, timeframe)
        
        # 应用处理流水线
        if data is not None and not data.empty and pipeline_name in self.pipelines:
            pipeline = self.pipelines[pipeline_name]
            data = pipeline.process(data)
        
        return data
    
    def get_data_batch(self, 
                     requests: List[Dict[str, Any]],
                     pipeline_name: str = "default",
                     use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        批量获取多个数据请求，使用多线程并行处理
        
        Args:
            requests: 数据请求列表，每个请求是一个包含以下键的字典：
                     symbol, timeframe, start_time, end_time, limit, source_name
            pipeline_name: 处理流水线名称
            use_cache: 是否使用缓存
            
        Returns:
            字典，键为请求ID，值为对应的DataFrame
        """
        results = {}
        futures = {}
        
        # 提交所有请求到线程池
        for i, request in enumerate(requests):
            request_id = request.get('id', str(i))
            
            future = self.executor.submit(
                self.get_historical_data,
                symbol=request['symbol'],
                timeframe=request['timeframe'],
                start_time=request['start_time'],
                end_time=request.get('end_time'),
                limit=request.get('limit'),
                source_name=request.get('source_name'),
                pipeline_name=pipeline_name,
                use_cache=use_cache
            )
            
            futures[future] = request_id
        
        # 收集结果
        for future in as_completed(futures):
            request_id = futures[future]
            try:
                data = future.result()
                results[request_id] = data
            except Exception as e:
                logger.error(f"获取数据请求 {request_id} 失败: {e}")
                results[request_id] = None
        
        return results
    
    def _record_access(self, symbol: str, timeframe: TimeFrame) -> None:
        """记录数据访问以便预加载"""
        key = f"{symbol}_{timeframe.value}"
        
        with self._lock:
            if key in self.frequently_accessed:
                self.frequently_accessed[key] += 1
            else:
                self.frequently_accessed[key] = 1
    
    def _check_preload(self) -> None:
        """检查是否需要触发预加载"""
        now = datetime.now()
        
        # 如果距离上次预加载时间不够长，则跳过
        if (now - self.last_preload_time).total_seconds() < self.preload_interval:
            return
        
        # 在单独线程中执行预加载
        threading.Thread(target=self._preload_frequently_accessed).start()
        self.last_preload_time = now
    
    def _preload_frequently_accessed(self) -> None:
        """预加载频繁访问的数据"""
        with self._lock:
            # 复制数据以避免在处理过程中修改
            items = self.frequently_accessed.copy()
            # 重置计数
            self.frequently_accessed = {}
        
        # 如果没有频繁访问的数据，跳过
        if not items:
            return
        
        # 排序找出最常访问的数据
        sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:5]  # 最多预加载5项
        
        logger.info(f"开始预加载频繁访问的数据: {len(top_items)}项")
        
        for key, count in top_items:
            symbol, timeframe_str = key.split('_')
            timeframe = TimeFrame.from_string(timeframe_str)
            
            try:
                # 计算预加载的时间范围
                end_time = datetime.now()
                start_time = end_time - timedelta(days=30)  # 预加载最近30天的数据
                
                # 检查是否已经在缓存中
                if self.cache.get(symbol, timeframe, start_time, end_time) is None:
                    # 获取数据并缓存
                    data = self.data_manager.get_historical_data(
                        symbol, timeframe, start_time, end_time
                    )
                    
                    if data is not None and not data.empty:
                        self.cache.set(symbol, timeframe, data, start_time, end_time)
                        logger.info(f"预加载完成: {symbol} {timeframe_str}, {len(data)}行")
            except Exception as e:
                logger.error(f"预加载数据失败: {symbol} {timeframe_str}, 错误: {e}")
    
    def get_available_symbols(self, source_name: Optional[str] = None) -> List[str]:
        """
        获取可用的交易品种列表
        
        Args:
            source_name: 数据源名称，如果为None则使用默认数据源
            
        Returns:
            交易品种列表
        """
        return self.data_manager.get_available_symbols(source_name)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        return self.cache.get_stats()
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """
        获取数据管理器统计信息
        
        Returns:
            数据管理器统计信息
        """
        with self._lock:
            stats = self.stats.copy()
            
            # 计算命中率
            if stats['data_requests'] > 0:
                stats['cache_hit_rate'] = stats['cache_hits'] / stats['data_requests']
            else:
                stats['cache_hit_rate'] = 0
            
            # 计算平均处理时间
            if stats['data_requests'] > 0:
                stats['avg_processing_time'] = stats['processing_time'] / stats['data_requests']
                stats['avg_total_time'] = stats['total_time'] / stats['data_requests']
            else:
                stats['avg_processing_time'] = 0
                stats['avg_total_time'] = 0
            
            return stats
    
    def invalidate_cache(self, symbol: Optional[str] = None, timeframe: Optional[TimeFrame] = None) -> int:
        """
        使缓存失效
        
        Args:
            symbol: 交易品种代码，如果为None则使所有品种缓存失效
            timeframe: 时间周期，如果为None则使所有时间周期缓存失效
            
        Returns:
            失效的缓存项数量
        """
        return self.cache.invalidate(symbol, timeframe)
    
    def clear_cache(self) -> None:
        """清空所有缓存"""
        self.cache.clear()
    
    def get_pipeline(self, name: str) -> Optional[ProcessingPipeline]:
        """
        获取指定名称的处理流水线
        
        Args:
            name: 流水线名称
            
        Returns:
            处理流水线，如果不存在则返回None
        """
        return self.pipelines.get(name)
    
    def get_pipelines(self) -> List[str]:
        """
        获取所有处理流水线的名称
        
        Returns:
            流水线名称列表
        """
        return list(self.pipelines.keys())
    
    def remove_pipeline(self, name: str) -> bool:
        """
        移除处理流水线
        
        Args:
            name: 要移除的流水线名称
            
        Returns:
            是否成功移除
        """
        if name in self.pipelines:
            del self.pipelines[name]
            logger.info(f"移除流水线: {name}")
            return True
        return False
    
    def _init_data_sources(self) -> None:
        """初始化所有配置的数据源"""
        # 添加Yahoo Finance数据源（如果配置了）
        if self.config.get('enable_yahoo', False) and YahooFinanceDataSource is not None:
            yahoo_config = self.config.get('yahoo_config', {})
            try:
                yahoo_source = YahooFinanceDataSource("yahoo", yahoo_config)
                if yahoo_source.connect():
                    self.data_sources['yahoo'] = yahoo_source
                    self.logger.info("成功初始化Yahoo Finance数据源")
                else:
                    self.logger.warning("无法连接到Yahoo Finance数据源")
            except Exception as e:
                self.logger.error(f"初始化Yahoo Finance数据源失败: {e}")
    
    def get_available_sources(self) -> List[str]:
        """获取所有可用的数据源名称列表"""
        return list(self.data_sources.keys())
        
    def get_international_symbols(self) -> Dict[str, List[str]]:
        """获取国际市场交易品种列表
        
        Returns:
            按市场分类的交易品种字典
        """
        result = {}
        
        # 优先使用Yahoo Finance数据源
        if 'yahoo' in self.data_sources:
            yahoo_source = self.data_sources['yahoo']
            
            # 从Yahoo数据源获取热门股票
            if hasattr(yahoo_source, 'popular_symbols'):
                return yahoo_source.popular_symbols
                
            # 或者获取所有支持的股票
            symbols = yahoo_source.get_symbols()
            result['international'] = symbols
            
        return result
        
    def get_historical_international_data(self, 
                                        symbol: str, 
                                        timeframe: TimeFrame, 
                                        start_time: datetime, 
                                        end_time: Optional[datetime] = None, 
                                        source: str = 'yahoo',
                                        limit: Optional[int] = None) -> pd.DataFrame:
        """获取国际市场历史数据
        
        Args:
            symbol: 交易品种代码
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间，默认为当前时间
            source: 数据源名称，默认为yahoo
            limit: 返回的最大数据条数
            
        Returns:
            包含历史数据的DataFrame
        """
        if source not in self.data_sources:
            self.logger.error(f"数据源 {source} 不可用")
            return pd.DataFrame()
            
        data_source = self.data_sources[source]
        
        return data_source.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
    def get_latest_international_data(self, 
                                    symbol: str, 
                                    timeframe: TimeFrame,
                                    source: str = 'yahoo') -> pd.DataFrame:
        """获取国际市场最新数据
        
        Args:
            symbol: 交易品种代码
            timeframe: 时间周期
            source: 数据源名称，默认为yahoo
            
        Returns:
            包含最新数据的DataFrame
        """
        if source not in self.data_sources:
            self.logger.error(f"数据源 {source} 不可用")
            return pd.DataFrame()
            
        data_source = self.data_sources[source]
        
        return data_source.get_latest_data(
            symbol=symbol,
            timeframe=timeframe
        ) 