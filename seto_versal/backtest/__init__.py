#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测模块用于评估交易策略历史表现。
提供灵活的回测配置和详细的结果分析。
"""

from seto_versal.backtest.engine import Backtest, BacktestResult

__all__ = ['Backtest', 'BacktestResult'] 