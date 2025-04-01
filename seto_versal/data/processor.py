#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal 数据处理模块

提供高效的数据处理功能，包括数据清洗、转换和特征工程。
支持基于管道的数据处理流程，可配置的特征计算，以及向量化的数据操作。
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
from collections import OrderedDict

from seto_versal.data.manager import TimeFrame

logger = logging.getLogger(__name__)

class ProcessorType(Enum):
    """处理器类型枚举"""
    CLEANER = "cleaner"       # 数据清洗
    TRANSFORMER = "transformer"  # 数据转换
    FEATURE = "feature"       # 特征计算
    ANOMALY = "anomaly"       # 异常检测
    FILTER = "filter"         # 数据过滤

class DataProcessor:
    """
    数据处理器基类
    """
    def __init__(self, name: str, processor_type: ProcessorType, config: Dict[str, Any] = None):
        """
        初始化数据处理器
        
        Args:
            name: 处理器名称
            processor_type: 处理器类型
            config: 配置参数
        """
        self.name = name
        self.type = processor_type
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        logger.debug(f"初始化数据处理器: {name} ({processor_type.value})")
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理数据
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        if not self.enabled or data is None or data.empty:
            return data
        
        try:
            return self._process_impl(data)
        except Exception as e:
            logger.error(f"处理器 {self.name} 处理失败: {e}")
            return data
    
    def _process_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        实际数据处理实现，由子类重写
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        return data
    
    def is_enabled(self) -> bool:
        """返回处理器是否启用"""
        return self.enabled
    
    def enable(self) -> None:
        """启用处理器"""
        self.enabled = True
    
    def disable(self) -> None:
        """禁用处理器"""
        self.enabled = False


class DataCleaner(DataProcessor):
    """
    数据清洗处理器
    """
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化数据清洗处理器
        
        Args:
            name: 处理器名称
            config: 配置参数，包括要清洗的列、填充方法、异常值处理等
        """
        super().__init__(name, ProcessorType.CLEANER, config)
        
        # 清洗配置
        self.fill_method = self.config.get('fill_method', 'ffill')
        self.drop_na = self.config.get('drop_na', False)
        self.remove_outliers = self.config.get('remove_outliers', False)
        self.outlier_std_threshold = self.config.get('outlier_std_threshold', 3.0)
        self.columns = self.config.get('columns', None)  # 如果为None，处理所有列
    
    def _process_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        实现数据清洗
        
        Args:
            data: 输入数据
            
        Returns:
            清洗后的数据
        """
        if data is None or data.empty:
            return data
        
        # 创建数据副本以避免修改原始数据
        df = data.copy()
        
        # 获取要处理的列
        cols = self.columns if self.columns else df.columns
        
        # 处理缺失值
        if self.fill_method == 'ffill':
            df[cols] = df[cols].ffill()
        elif self.fill_method == 'bfill':
            df[cols] = df[cols].bfill()
        elif self.fill_method == 'mean':
            for col in cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
        elif self.fill_method == 'median':
            for col in cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
        
        # 如果仍有缺失值且需要删除
        if self.drop_na:
            df = df.dropna(subset=cols)
        
        # 处理异常值
        if self.remove_outliers:
            for col in cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    mean = df[col].mean()
                    std = df[col].std()
                    threshold = self.outlier_std_threshold * std
                    df[col] = df[col].mask(
                        (df[col] > mean + threshold) | (df[col] < mean - threshold),
                        np.nan
                    )
                    # 填充处理后的NaN值
                    if self.fill_method == 'ffill':
                        df[col] = df[col].ffill().bfill()
                    elif self.fill_method == 'bfill':
                        df[col] = df[col].bfill().ffill()
                    elif self.fill_method == 'mean':
                        df[col] = df[col].fillna(df[col].mean())
                    elif self.fill_method == 'median':
                        df[col] = df[col].fillna(df[col].median())
        
        return df


class FeatureGenerator(DataProcessor):
    """
    特征生成处理器
    """
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化特征生成处理器
        
        Args:
            name: 处理器名称
            config: 配置参数，包括要生成的特征类型、参数等
        """
        super().__init__(name, ProcessorType.FEATURE, config)
        
        # 特征配置
        self.features = self.config.get('features', [])
        self.price_col = self.config.get('price_col', 'close')
        self.volume_col = self.config.get('volume_col', 'volume')
        self.inplace = self.config.get('inplace', True)
    
    def _process_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        实现特征生成
        
        Args:
            data: 输入数据
            
        Returns:
            添加特征后的数据
        """
        if data is None or data.empty:
            return data
        
        # 创建数据副本或使用原始数据
        df = data if self.inplace else data.copy()
        
        # 生成各种特征
        for feature in self.features:
            feature_type = feature.get('type')
            
            if feature_type == 'sma':
                # 简单移动平均线
                window = feature.get('window', 20)
                col_name = f"sma_{window}"
                df[col_name] = df[self.price_col].rolling(window=window).mean()
                
            elif feature_type == 'ema':
                # 指数移动平均线
                window = feature.get('window', 20)
                col_name = f"ema_{window}"
                df[col_name] = df[self.price_col].ewm(span=window, adjust=False).mean()
                
            elif feature_type == 'rsi':
                # 相对强弱指数
                window = feature.get('window', 14)
                col_name = f"rsi_{window}"
                delta = df[self.price_col].diff()
                gain = delta.mask(delta < 0, 0)
                loss = -delta.mask(delta > 0, 0)
                avg_gain = gain.rolling(window=window).mean()
                avg_loss = loss.rolling(window=window).mean()
                rs = avg_gain / avg_loss
                df[col_name] = 100 - (100 / (1 + rs))
                
            elif feature_type == 'macd':
                # MACD指标
                fast = feature.get('fast', 12)
                slow = feature.get('slow', 26)
                signal = feature.get('signal', 9)
                
                ema_fast = df[self.price_col].ewm(span=fast, adjust=False).mean()
                ema_slow = df[self.price_col].ewm(span=slow, adjust=False).mean()
                
                df[f"macd_line"] = ema_fast - ema_slow
                df[f"macd_signal"] = df[f"macd_line"].ewm(span=signal, adjust=False).mean()
                df[f"macd_hist"] = df[f"macd_line"] - df[f"macd_signal"]
                
            elif feature_type == 'bollinger':
                # 布林带
                window = feature.get('window', 20)
                std_dev = feature.get('std_dev', 2)
                
                df[f"bb_middle_{window}"] = df[self.price_col].rolling(window=window).mean()
                df[f"bb_std_{window}"] = df[self.price_col].rolling(window=window).std()
                df[f"bb_upper_{window}"] = df[f"bb_middle_{window}"] + std_dev * df[f"bb_std_{window}"]
                df[f"bb_lower_{window}"] = df[f"bb_middle_{window}"] - std_dev * df[f"bb_std_{window}"]
                
            elif feature_type == 'atr':
                # 平均真实范围
                window = feature.get('window', 14)
                col_name = f"atr_{window}"
                
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df[self.price_col].shift())
                low_close = np.abs(df['low'] - df[self.price_col].shift())
                
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                df[col_name] = true_range.rolling(window=window).mean()
            
            elif feature_type == 'volume_sma':
                # 成交量移动平均
                window = feature.get('window', 20)
                col_name = f"volume_sma_{window}"
                df[col_name] = df[self.volume_col].rolling(window=window).mean()
            
            elif feature_type == 'percentage_change':
                # 价格变化百分比
                window = feature.get('window', 1)
                col_name = f"pct_change_{window}"
                df[col_name] = df[self.price_col].pct_change(periods=window) * 100
        
        return df


class ProcessingPipeline:
    """
    数据处理流水线，将多个处理器组合在一起顺序处理数据
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化处理流水线
        
        Args:
            name: 流水线名称
            config: 配置参数
        """
        self.name = name
        self.config = config or {}
        self.processors: List[DataProcessor] = []
        
        logger.info(f"初始化数据处理流水线: {name}")
    
    def add_processor(self, processor: DataProcessor, position: Optional[int] = None) -> None:
        """
        添加处理器到流水线
        
        Args:
            processor: 要添加的处理器
            position: 添加位置，如果为None则添加到末尾
        """
        if position is None:
            self.processors.append(processor)
        else:
            self.processors.insert(position, processor)
        
        logger.debug(f"添加处理器到流水线 {self.name}: {processor.name}")
    
    def remove_processor(self, processor_name: str) -> bool:
        """
        从流水线中移除处理器
        
        Args:
            processor_name: 要移除的处理器名称
            
        Returns:
            是否成功移除
        """
        for i, processor in enumerate(self.processors):
            if processor.name == processor_name:
                del self.processors[i]
                logger.debug(f"从流水线 {self.name} 中移除处理器: {processor_name}")
                return True
        
        logger.warning(f"处理器 {processor_name} 不在流水线 {self.name} 中")
        return False
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        通过流水线处理数据
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        if data is None or data.empty:
            return data
        
        current_data = data.copy()
        
        for processor in self.processors:
            if processor.is_enabled():
                start_time = datetime.now()
                current_data = processor.process(current_data)
                elapsed = (datetime.now() - start_time).total_seconds()
                
                logger.debug(f"流水线 {self.name} 处理器 {processor.name} 耗时: {elapsed:.4f}s")
        
        return current_data
    
    def get_processors(self) -> List[Dict[str, Any]]:
        """
        获取流水线中的所有处理器信息
        
        Returns:
            处理器信息列表
        """
        return [
            {
                'name': processor.name,
                'type': processor.type.value,
                'enabled': processor.is_enabled()
            }
            for processor in self.processors
        ] 