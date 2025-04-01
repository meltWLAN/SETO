#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal 数据源模块

包含各种市场数据源的实现
"""

try:
    from seto_versal.data.sources.yahoo_finance import YahooFinanceDataSource
    __all__ = ["YahooFinanceDataSource"]
except ImportError:
    import logging
    logging.warning("无法导入Yahoo Finance数据源，可能是缺少依赖")
    __all__ = [] 