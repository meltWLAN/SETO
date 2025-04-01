#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal strategies package

This package contains trading strategies that analyze market data
and generate trading signals for agents to act upon.
"""

from seto_versal.strategies.base import BaseStrategy
from seto_versal.strategies.breakout import BreakoutStrategy
from seto_versal.strategies.moving_average import MovingAverageStrategy
from seto_versal.strategies.momentum import MomentumStrategy

# 添加我们实现的测试策略
from seto_versal.strategies.test_strategy import TestStrategy

# 添加更多策略导入
from seto_versal.strategies.breakout_volume import BreakoutVolumeStrategy
from seto_versal.strategies.momentum_short import MomentumShortStrategy
from seto_versal.strategies.ema_golden_cross import EmaGoldenCrossStrategy
from seto_versal.strategies.adx_trend import AdxTrendStrategy
from seto_versal.strategies.rsi_oversold import RsiOversoldStrategy
from seto_versal.strategies.macd_divergence import MacdDivergenceStrategy
from seto_versal.strategies.sector_momentum import SectorMomentumStrategy
from seto_versal.strategies.industry_leader import IndustryLeaderStrategy
from seto_versal.strategies.market_hedge import MarketHedgeStrategy
from seto_versal.strategies.volatility_filter import VolatilityFilterStrategy

__all__ = [
    'BaseStrategy',
    'BreakoutStrategy',
    'MovingAverageStrategy',
    'MomentumStrategy',
    'TestStrategy',  # 添加测试策略到导出列表
    'BreakoutVolumeStrategy',
    'MomentumShortStrategy',
    'EmaGoldenCrossStrategy',
    'AdxTrendStrategy',
    'RsiOversoldStrategy',
    'MacdDivergenceStrategy',
    'SectorMomentumStrategy',
    'IndustryLeaderStrategy',
    'MarketHedgeStrategy',
    'VolatilityFilterStrategy',
]
