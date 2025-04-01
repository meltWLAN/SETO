#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略基类模块

定义所有交易策略的基础接口和公共功能
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

@dataclass
class TradeSignal:
    """交易信号数据类"""
    symbol: str  # 股票代码
    signal_type: str  # 信号类型: 'buy', 'sell'
    price: float  # 交易价格
    quantity: int  # 交易数量
    timestamp: datetime = None  # 信号产生时间
    reason: str = ""  # 产生信号的原因
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class BaseStrategy:
    """策略基类"""
    
    def __init__(self, name: str = "基础策略"):
        """
        初始化策略
        
        Args:
            name: 策略名称
        """
        self.name = name
        self.parameters = {}
        self.performance_metrics = {}
    
    def set_parameters(self, **kwargs):
        """
        设置策略参数
        
        Args:
            **kwargs: 参数名称和值
        """
        for key, value in kwargs.items():
            self.parameters[key] = value
    
    def generate_signals(self, 
                        market_data: Dict[str, Any], 
                        positions: Dict[str, Any] = None,
                        market_state: Dict[str, Any] = None) -> List[TradeSignal]:
        """
        生成交易信号
        
        Args:
            market_data: 市场数据，包含价格、成交量等
            positions: 当前持仓信息
            market_state: 市场状态，如日期、交易时段等
            
        Returns:
            交易信号列表
        """
        # 在子类中实现具体的信号生成逻辑
        return []
    
    def update_performance(self, metrics: Dict[str, Any]):
        """
        更新策略性能指标
        
        Args:
            metrics: 性能指标
        """
        self.performance_metrics.update(metrics)
    
    def get_performance(self) -> Dict[str, Any]:
        """
        获取策略性能指标
        
        Returns:
            性能指标
        """
        return self.performance_metrics
    
    def reset(self):
        """
        重置策略状态
        """
        self.performance_metrics = {}
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.name} Strategy" 