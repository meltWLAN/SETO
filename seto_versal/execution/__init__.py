#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
执行模块提供了交易执行接口和各种交易所适配器的实现。
"""

from seto_versal.execution.client import (
    ExecutionClient, OrderType, OrderStatus, Order, OrderManager
)
from seto_versal.execution.simulator import SimulatedExecutionClient
from seto_versal.execution.exchange import BinanceExecutionClient

__all__ = [
    'ExecutionClient', 'OrderType', 'OrderStatus', 'Order', 'OrderManager',
    'SimulatedExecutionClient', 'BinanceExecutionClient'
] 