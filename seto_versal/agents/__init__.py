#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agents module for SETO-Versal
Provides trading agent implementations with different strategies and personalities
"""

from seto_versal.agents.base import Agent, AgentType, AgentDecision, ConfidenceLevel
from seto_versal.agents.trend import TrendAgent
from seto_versal.agents.rapid_agent import RapidAgent
from seto_versal.agents.reversal import ReversalAgent
from seto_versal.agents.sector_rotation import SectorRotationAgent
from seto_versal.agents.defensive import DefensiveAgent

__all__ = [
    'Agent',
    'AgentType',
    'AgentDecision',
    'ConfidenceLevel',
    'TrendAgent',
    'RapidAgent',
    'ReversalAgent',
    'SectorRotationAgent',
    'DefensiveAgent'
]
