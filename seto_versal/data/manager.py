#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据管理模块的核心实现。

该模块提供:
- 时间周期枚举（TimeFrame）
- 数据源抽象类（DataSource）
- 数据管理器（DataManager）
"""

import os
import enum
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Type, Set
from abc import ABC, abstractmethod
from seto_versal.data.quality import DataQualityChecker
from seto_versal.data.quality_monitor import DataQualityMonitor


class TimeFrame(enum.Enum):
    """
    时间周期枚举，定义支持的数据频率
    """
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"
    
    def __str__(self):
        return self.value
    
    @classmethod
    def from_string(cls, value: str) -> 'TimeFrame':
        """
        从字符串创建时间周期枚举
        
        Args:
            value: 时间周期字符串
            
        Returns:
            对应的TimeFrame枚举值
            
        Raises:
            ValueError: 如果字符串不匹配任何时间周期
        """
        for tf in cls:
            if tf.value == value:
                return tf
        raise ValueError(f"无效的时间周期: {value}")
    
    def to_minutes(self) -> int:
        """
        将时间周期转换为分钟数
        
        Returns:
            时间周期对应的分钟数
        """
        if self == TimeFrame.MINUTE_1:
            return 1
        elif self == TimeFrame.MINUTE_5:
            return 5
        elif self == TimeFrame.MINUTE_15:
            return 15
        elif self == TimeFrame.MINUTE_30:
            return 30
        elif self == TimeFrame.HOUR_1:
            return 60
        elif self == TimeFrame.HOUR_4:
            return 240
        elif self == TimeFrame.DAY_1:
            return 1440
        elif self == TimeFrame.WEEK_1:
            return 10080
        elif self == TimeFrame.MONTH_1:
            return 43200
        return 0


class DataSource(ABC):
    """
    数据源抽象基类，定义所有数据源必须实现的接口
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化数据源
        
        Args:
            name: 数据源名称
            config: 数据源配置参数
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"data.source.{name}")
        self.logger.info(f"初始化数据源: {name}")
        
        # 数据源支持的时间周期
        self.supported_timeframes: Set[TimeFrame] = set()
        
        # 数据源支持的交易品种
        self.available_symbols: List[str] = []
    
    @abstractmethod
    def connect(self) -> bool:
        """
        连接到数据源
        
        Returns:
            连接是否成功
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开与数据源的连接
        
        Returns:
            断开连接是否成功
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        检查是否已连接到数据源
        
        Returns:
            是否已连接
        """
        pass
    
    @abstractmethod
    def get_symbols(self) -> List[str]:
        """
        获取数据源支持的所有交易品种
        
        Returns:
            交易品种列表
        """
        pass
    
    @abstractmethod
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
            包含历史数据的DataFrame，包含时间、开盘价、最高价、最低价、收盘价、成交量等列
        """
        pass
    
    @abstractmethod
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
        pass
    
    def validate_timeframe(self, timeframe: TimeFrame) -> bool:
        """
        验证时间周期是否被当前数据源支持
        
        Args:
            timeframe: 要验证的时间周期
            
        Returns:
            时间周期是否受支持
        """
        return timeframe in self.supported_timeframes
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        验证交易品种是否被当前数据源支持
        
        Args:
            symbol: 要验证的交易品种代码
            
        Returns:
            交易品种是否受支持
        """
        if not self.available_symbols:
            # 如果还没加载可用品种，尝试加载
            self.available_symbols = self.get_symbols()
        
        return symbol in self.available_symbols


class CSVDataSource(DataSource):
    """
    CSV文件数据源，从本地CSV文件读取历史数据
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化CSV数据源
        
        Args:
            name: 数据源名称
            config: 数据源配置参数，必须包含data_path
        """
        super().__init__(name, config)
        
        # 数据文件目录
        self.data_path = self.config.get('data_path', 'data/market')
        
        # 支持的时间周期
        self.supported_timeframes = {
            TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15, 
            TimeFrame.MINUTE_30, TimeFrame.HOUR_1, TimeFrame.HOUR_4, 
            TimeFrame.DAY_1
        }
        
        # 缓存已加载的数据
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        # 初始化完成
        self.logger.info(f"CSV数据源初始化完成，数据路径: {self.data_path}")
    
    def connect(self) -> bool:
        """
        连接到数据源（对于CSV数据源，检查数据目录是否存在）
        
        Returns:
            连接是否成功
        """
        if not os.path.exists(self.data_path):
            self.logger.error(f"数据目录不存在: {self.data_path}")
            return False
        
        return True
    
    def disconnect(self) -> bool:
        """
        断开与数据源的连接（对于CSV数据源，清空缓存）
        
        Returns:
            断开连接是否成功
        """
        self.data_cache.clear()
        return True
    
    def is_connected(self) -> bool:
        """
        检查是否已连接到数据源
        
        Returns:
            是否已连接
        """
        return os.path.exists(self.data_path)
    
    def get_symbols(self) -> List[str]:
        """
        获取数据源支持的所有交易品种
        
        Returns:
            交易品种列表
        """
        if not self.is_connected():
            self.logger.error("数据源未连接")
            return []
        
        symbols = set()
        
        # 搜索数据目录下的所有CSV文件
        for filename in os.listdir(self.data_path):
            if filename.endswith('.csv'):
                # 从文件名提取符号和时间周期
                parts = filename.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    symbols.add(symbol)
        
        self.available_symbols = list(symbols)
        return self.available_symbols
    
    def _get_file_path(self, symbol: str, timeframe: TimeFrame) -> str:
        """
        获取指定交易品种和时间周期的数据文件路径
        
        Args:
            symbol: 交易品种代码
            timeframe: 时间周期
            
        Returns:
            数据文件路径
        """
        return os.path.join(self.data_path, f"{symbol}_{timeframe.value}.csv")
    
    def _load_data(self, symbol: str, timeframe: TimeFrame) -> pd.DataFrame:
        """
        加载指定交易品种和时间周期的数据
        
        Args:
            symbol: 交易品种代码
            timeframe: 时间周期
            
        Returns:
            包含历史数据的DataFrame
        """
        file_path = self._get_file_path(symbol, timeframe)
        
        if not os.path.exists(file_path):
            self.logger.error(f"数据文件不存在: {file_path}")
            return pd.DataFrame()
        
        try:
            # 加载CSV文件
            df = pd.read_csv(file_path)
            
            # 确保时间列是datetime类型
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
                df.drop('time', axis=1, inplace=True)
            else:
                self.logger.warning(f"数据文件中没有时间列: {file_path}")
            
            # 确保列名标准化
            rename_map = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
            
            df.rename(columns={col: rename_map.get(col.lower(), col) for col in df.columns}, inplace=True)
            
            # 按时间排序
            if 'timestamp' in df.columns:
                df.sort_values('timestamp', inplace=True)
            
            # 缓存数据
            cache_key = f"{symbol}_{timeframe.value}"
            self.data_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            self.logger.error(f"加载数据文件失败: {file_path}, 错误: {str(e)}")
            return pd.DataFrame()
    
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
        if not self.validate_timeframe(timeframe):
            self.logger.error(f"不支持的时间周期: {timeframe}")
            return pd.DataFrame()
        
        if not self.validate_symbol(symbol):
            self.logger.error(f"不支持的交易品种: {symbol}")
            return pd.DataFrame()
        
        # 确定结束时间
        if end_time is None:
            end_time = datetime.now()
        
        # 尝试从缓存获取数据
        cache_key = f"{symbol}_{timeframe.value}"
        
        if cache_key in self.data_cache:
            df = self.data_cache[cache_key]
        else:
            df = self._load_data(symbol, timeframe)
        
        if df.empty:
            return df
        
        # 筛选日期范围
        if 'timestamp' in df.columns:
            mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
            filtered_df = df[mask].copy()
        else:
            self.logger.warning(f"数据中没有时间列，无法按时间筛选")
            filtered_df = df.copy()
        
        # 限制返回的数据条数
        if limit is not None and len(filtered_df) > limit:
            filtered_df = filtered_df.tail(limit)
        
        return filtered_df
    
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
        if not self.validate_timeframe(timeframe):
            self.logger.error(f"不支持的时间周期: {timeframe}")
            return pd.DataFrame()
        
        if not self.validate_symbol(symbol):
            self.logger.error(f"不支持的交易品种: {symbol}")
            return pd.DataFrame()
        
        # 尝试从缓存获取数据
        cache_key = f"{symbol}_{timeframe.value}"
        
        if cache_key in self.data_cache:
            df = self.data_cache[cache_key]
        else:
            df = self._load_data(symbol, timeframe)
        
        if df.empty:
            return df
        
        # 返回最后一条数据
        return df.tail(1)


class DataManager:
    """
    数据管理器，负责协调多个数据源，提供统一的数据访问接口
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化数据管理器
        
        Args:
            config: 数据管理器配置参数
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化数据管理器")
        
        # 配置
        self.config = config or {}
        
        # 数据源映射表
        self.data_sources: Dict[str, DataSource] = {}
        
        # 默认数据源
        self.default_source: Optional[str] = None
        
        # 数据缓存
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        # 初始化完成
        self.logger.info("数据管理器初始化完成")
    
    def add_data_source(self, source: DataSource, is_default: bool = False) -> bool:
        """
        添加数据源
        
        Args:
            source: 数据源实例
            is_default: 是否设为默认数据源
            
        Returns:
            添加是否成功
        """
        if source.name in self.data_sources:
            self.logger.warning(f"数据源已存在: {source.name}")
            return False
        
        self.data_sources[source.name] = source
        
        if is_default or self.default_source is None:
            self.default_source = source.name
            self.logger.info(f"设置默认数据源: {source.name}")
        
        self.logger.info(f"添加数据源: {source.name}")
        return True
    
    def remove_data_source(self, source_name: str) -> bool:
        """
        移除数据源
        
        Args:
            source_name: 数据源名称
            
        Returns:
            移除是否成功
        """
        if source_name not in self.data_sources:
            self.logger.warning(f"数据源不存在: {source_name}")
            return False
        
        # 如果移除的是默认数据源，需要重新设置默认数据源
        if source_name == self.default_source:
            if len(self.data_sources) > 1:
                # 选择另一个数据源作为默认数据源
                for name in self.data_sources.keys():
                    if name != source_name:
                        self.default_source = name
                        self.logger.info(f"设置新的默认数据源: {name}")
                        break
            else:
                self.default_source = None
        
        # 断开连接并移除数据源
        try:
            self.data_sources[source_name].disconnect()
        except Exception as e:
            self.logger.warning(f"断开数据源连接时出错: {source_name}, 错误: {str(e)}")
        
        del self.data_sources[source_name]
        self.logger.info(f"移除数据源: {source_name}")
        
        return True
    
    def get_data_source(self, source_name: Optional[str] = None) -> Optional[DataSource]:
        """
        获取数据源
        
        Args:
            source_name: 数据源名称，如果为None则返回默认数据源
            
        Returns:
            数据源实例，如果不存在则返回None
        """
        if source_name is None:
            if self.default_source is None:
                self.logger.error("没有设置默认数据源")
                return None
            source_name = self.default_source
        
        if source_name not in self.data_sources:
            self.logger.error(f"数据源不存在: {source_name}")
            return None
        
        return self.data_sources[source_name]
    
    def get_historical_data(self, 
                          symbol: str, 
                          timeframe: TimeFrame, 
                          start_time: datetime, 
                          end_time: Optional[datetime] = None, 
                          limit: Optional[int] = None,
                          source_name: Optional[str] = None) -> pd.DataFrame:
        """
        获取历史市场数据
        
        Args:
            symbol: 交易品种代码
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间，默认为当前时间
            limit: 返回的最大数据条数
            source_name: 数据源名称，如果为None则使用默认数据源
            
        Returns:
            包含历史数据的DataFrame
        """
        source = self.get_data_source(source_name)
        if source is None:
            return pd.DataFrame()
        
        # 检查数据源是否已连接
        if not source.is_connected():
            if not source.connect():
                self.logger.error(f"无法连接到数据源: {source.name}")
                return pd.DataFrame()
        
        try:
            return source.get_historical_data(symbol, timeframe, start_time, end_time, limit)
        except Exception as e:
            self.logger.error(f"获取历史数据失败: {symbol} {timeframe}, 错误: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_data(self, 
                      symbol: str, 
                      timeframe: TimeFrame,
                      source_name: Optional[str] = None) -> pd.DataFrame:
        """
        获取最新的市场数据
        
        Args:
            symbol: 交易品种代码
            timeframe: 时间周期
            source_name: 数据源名称，如果为None则使用默认数据源
            
        Returns:
            包含最新数据的DataFrame
        """
        source = self.get_data_source(source_name)
        if source is None:
            return pd.DataFrame()
        
        # 检查数据源是否已连接
        if not source.is_connected():
            if not source.connect():
                self.logger.error(f"无法连接到数据源: {source.name}")
                return pd.DataFrame()
        
        try:
            return source.get_latest_data(symbol, timeframe)
        except Exception as e:
            self.logger.error(f"获取最新数据失败: {symbol} {timeframe}, 错误: {str(e)}")
            return pd.DataFrame()
    
    def get_available_symbols(self, source_name: Optional[str] = None) -> List[str]:
        """
        获取可用的交易品种列表
        
        Args:
            source_name: 数据源名称，如果为None则使用默认数据源
            
        Returns:
            交易品种列表
        """
        source = self.get_data_source(source_name)
        if source is None:
            return []
        
        # 检查数据源是否已连接
        if not source.is_connected():
            if not source.connect():
                self.logger.error(f"无法连接到数据源: {source.name}")
                return []
        
        try:
            return source.get_symbols()
        except Exception as e:
            self.logger.error(f"获取交易品种列表失败，错误: {str(e)}")
            return []
    
    def save_data(self, 
                data: pd.DataFrame, 
                symbol: str, 
                timeframe: TimeFrame, 
                output_dir: Optional[str] = None) -> bool:
        """
        保存数据到CSV文件
        
        Args:
            data: 要保存的数据
            symbol: 交易品种代码
            timeframe: 时间周期
            output_dir: 输出目录，如果为None则使用默认目录
            
        Returns:
            保存是否成功
        """
        if data.empty:
            self.logger.warning(f"没有要保存的数据: {symbol} {timeframe}")
            return False
        
        if output_dir is None:
            output_dir = self.config.get('data_path', 'data/market')
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建文件路径
        file_path = os.path.join(output_dir, f"{symbol}_{timeframe.value}.csv")
        
        try:
            # 确保数据按时间排序
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp')
            
            # 保存到CSV文件
            data.to_csv(file_path, index=False)
            self.logger.info(f"数据已保存到文件: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存数据失败: {file_path}, 错误: {str(e)}")
            return False
    
    def create_csv_data_source(self, 
                             name: str, 
                             data_path: str, 
                             is_default: bool = False) -> Optional[DataSource]:
        """
        创建CSV数据源
        
        Args:
            name: 数据源名称
            data_path: 数据文件目录
            is_default: 是否设为默认数据源
            
        Returns:
            创建的数据源，如果创建失败则返回None
        """
        config = {'data_path': data_path}
        source = CSVDataSource(name, config)
        
        # 尝试连接
        if not source.connect():
            self.logger.error(f"无法连接到CSV数据源: {data_path}")
            return None
        
        # 添加到数据源列表
        if self.add_data_source(source, is_default):
            return source
        
        return None
    
    def get_data_sources(self) -> Dict[str, str]:
        """
        获取所有已注册的数据源
        
        Returns:
            数据源名称到类型的映射
        """
        return {name: type(source).__name__ for name, source in self.data_sources.items()}
    
    def resample_data(self, 
                    data: pd.DataFrame, 
                    source_timeframe: TimeFrame, 
                    target_timeframe: TimeFrame) -> pd.DataFrame:
        """
        将数据重采样到不同的时间周期
        
        Args:
            data: 源数据
            source_timeframe: 源数据的时间周期
            target_timeframe: 目标时间周期
            
        Returns:
            重采样后的数据
        """
        if data.empty:
            return pd.DataFrame()
        
        if source_timeframe == target_timeframe:
            return data.copy()
        
        # 检查时间周期大小关系
        source_minutes = source_timeframe.to_minutes()
        target_minutes = target_timeframe.to_minutes()
        
        if source_minutes > target_minutes:
            self.logger.error("不支持降采样到更小的时间周期")
            return pd.DataFrame()
        
        if target_minutes % source_minutes != 0:
            self.logger.warning(f"目标时间周期 {target_timeframe} 不是源时间周期 {source_timeframe} 的整数倍")
        
        try:
            # 确保时间列是索引
            if 'timestamp' in data.columns:
                df = data.set_index('timestamp')
            else:
                df = data.copy()
                self.logger.warning("数据中没有timestamp列，无法进行重采样")
                return pd.DataFrame()
            
            # 计算重采样规则
            rule = None
            if target_timeframe in [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15, TimeFrame.MINUTE_30]:
                rule = f"{target_minutes}min"
            elif target_timeframe in [TimeFrame.HOUR_1, TimeFrame.HOUR_4]:
                rule = f"{target_minutes // 60}H"
            elif target_timeframe == TimeFrame.DAY_1:
                rule = "1D"
            elif target_timeframe == TimeFrame.WEEK_1:
                rule = "1W"
            elif target_timeframe == TimeFrame.MONTH_1:
                rule = "1M"
            
            if rule is None:
                self.logger.error(f"无法确定重采样规则: {target_timeframe}")
                return pd.DataFrame()
            
            # 执行重采样
            resampled = df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # 重置索引
            resampled = resampled.reset_index()
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"重采样数据失败，错误: {str(e)}")
            return pd.DataFrame() 