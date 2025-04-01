#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common constants for SETO-Versal
"""

from enum import Enum, auto

class SignalType(Enum):
    """Trade signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit" 