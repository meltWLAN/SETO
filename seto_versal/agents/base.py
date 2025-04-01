#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base agent module for SETO-Versal
Defines base agent class and supporting types for all trading agents
"""

import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import json
import pandas as pd
import numpy as np
import random

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of trading agents"""
    TREND = "trend"                  # Follows established market trends
    RAPID = "rapid"                  # Fast profit on breakouts (T+1 focused)
    REVERSAL = "reversal"            # Identifies trend reversals
    SECTOR_ROTATION = "sector"       # Tracks sector rotation and hot themes
    DEFENSIVE = "defensive"          # Risk management and capital preservation
    CUSTOM = "custom"                # Custom agent type

class ConfidenceLevel(Enum):
    """Confidence levels for agent decisions"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5

class AgentDecision:
    """
    Represents a trading decision made by an agent
    """
    
    def __init__(self, 
                 agent_id: str,
                 symbol: str,
                 decision_type: str,  # "buy", "sell", "hold", "reduce", "increase"
                 confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
                 target_price: Optional[float] = None,
                 stop_loss: Optional[float] = None,
                 quantity: Optional[int] = None,
                 timeframe: Optional[str] = None,
                 reasoning: Optional[str] = None,
                 metrics: Optional[Dict[str, Any]] = None,
                 entry_window: Optional[Tuple[str, str]] = None):
        """
        Initialize an agent decision
        
        Args:
            agent_id (str): ID of the agent making the decision
            symbol (str): Trading symbol
            decision_type (str): Type of decision - buy, sell, hold, reduce, increase
            confidence (ConfidenceLevel): Confidence level of the decision
            target_price (float, optional): Target price for the trade
            stop_loss (float, optional): Stop loss price for the trade
            quantity (int, optional): Recommended quantity to trade
            timeframe (str, optional): Expected holding timeframe
            reasoning (str, optional): Reasoning behind the decision
            metrics (dict, optional): Metrics supporting the decision
            entry_window (tuple, optional): Valid time window for entry (start, end)
        """
        self.id = str(uuid.uuid4())
        self.agent_id = agent_id
        self.symbol = symbol
        self.decision_type = decision_type.lower()
        self.confidence = confidence
        self.target_price = target_price
        self.stop_loss = stop_loss
        self.quantity = quantity
        self.timeframe = timeframe
        self.reasoning = reasoning
        self.metrics = metrics or {}
        self.timestamp = datetime.now()
        self.entry_window = entry_window
        self.executed = False
        self.execution_info = {}
        
        # Validate decision type
        valid_decisions = ['buy', 'sell', 'hold', 'reduce', 'increase']
        if self.decision_type not in valid_decisions:
            raise ValueError(f"Invalid decision type: {decision_type}. Must be one of {valid_decisions}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert decision to dictionary
        
        Returns:
            dict: Dictionary representation of the decision
        """
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'symbol': self.symbol,
            'decision_type': self.decision_type,
            'confidence': self.confidence.value,
            'confidence_name': self.confidence.name,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'quantity': self.quantity,
            'timeframe': self.timeframe,
            'reasoning': self.reasoning,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat(),
            'entry_window': self.entry_window,
            'executed': self.executed,
            'execution_info': self.execution_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentDecision':
        """
        Create decision from dictionary
        
        Args:
            data (dict): Dictionary representation of the decision
            
        Returns:
            AgentDecision: Reconstructed decision object
        """
        decision = cls(
            agent_id=data['agent_id'],
            symbol=data['symbol'],
            decision_type=data['decision_type'],
            confidence=ConfidenceLevel(data['confidence']),
            target_price=data.get('target_price'),
            stop_loss=data.get('stop_loss'),
            quantity=data.get('quantity'),
            timeframe=data.get('timeframe'),
            reasoning=data.get('reasoning'),
            metrics=data.get('metrics', {}),
            entry_window=data.get('entry_window')
        )
        
        # Restore additional fields
        decision.id = data['id']
        decision.timestamp = datetime.fromisoformat(data['timestamp'])
        decision.executed = data.get('executed', False)
        decision.execution_info = data.get('execution_info', {})
        
        return decision
    
    def __str__(self) -> str:
        """String representation of the decision"""
        return (f"Decision {self.id[:8]} - {self.decision_type.upper()} {self.symbol} "
                f"(Confidence: {self.confidence.name})")

class Agent:
    """Base agent class"""
    
    def __init__(self, name, confidence_threshold=0.7, max_positions=5, weight=1.0, strategies=None):
        """
        Initialize the agent
        
        Args:
            name (str): Agent name
            confidence_threshold (float): Confidence threshold for trading
            max_positions (int): Maximum number of positions
            weight (float): Agent weight in ensemble
            strategies (list): List of strategies
        """
        self.name = name
        self.confidence_threshold = confidence_threshold
        self.max_positions = max_positions
        self.weight = weight
        self.strategies = strategies or []
        self.type = self.__class__.__name__.lower().replace('agent', '')
        
        # Initialize state
        self.positions = {}
        self.orders = []
        self.performance_metrics = {
            'total_return': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0
        }
        
        logger.info(f"Initialized {self.type} agent: {name}")
        
    def add_strategy(self, strategy):
        """Add a strategy to the agent"""
        self.strategies.append(strategy)
        
    def get_strategies(self):
        """Get list of strategies"""
        return self.strategies
        
    def get_performance_metrics(self):
        """Get agent performance metrics"""
        return self.performance_metrics
        
    def reset(self):
        """Reset agent state"""
        self.positions = {}
        self.orders = []
        self.performance_metrics = {
            'total_return': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0
        }
        
    def pause(self):
        """Pause agent trading"""
        pass
        
    def resume(self):
        """Resume agent trading"""
        pass

    def analyze(self, market_state, symbols=None, **kwargs) -> List[AgentDecision]:
        """
        Analyze market data and generate trading decisions
        
        Args:
            market_state: Current market state
            symbols (list, optional): Specific symbols to analyze
            **kwargs: Additional analysis parameters
            
        Returns:
            list: List of AgentDecision objects
        """
        # This is an abstract method that should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the analyze method")
    
    def calculate_confidence(self, signal_strength, trend_alignment=None, 
                            historical_accuracy=None, volume_confirmation=None) -> ConfidenceLevel:
        """
        Calculate confidence level for a decision
        
        Args:
            signal_strength (float): Strength of the signal (0.0-1.0)
            trend_alignment (float, optional): Alignment with overall trend (0.0-1.0)
            historical_accuracy (float, optional): Historical accuracy in similar conditions (0.0-1.0)
            volume_confirmation (float, optional): Volume confirmation (0.0-1.0)
            
        Returns:
            ConfidenceLevel: Calculated confidence level
        """
        # Default values for missing inputs
        trend_alignment = trend_alignment if trend_alignment is not None else 0.5
        historical_accuracy = historical_accuracy if historical_accuracy is not None else 0.5
        volume_confirmation = volume_confirmation if volume_confirmation is not None else 0.5
        
        # Get weights for each factor
        weights = self.confidence_factors
        
        # Calculate weighted confidence score
        confidence_score = (
            signal_strength * weights['signal_strength'] +
            trend_alignment * weights['trend_alignment'] +
            historical_accuracy * weights['historical_accuracy'] +
            volume_confirmation * weights['volume_confirmation']
        )
        
        # Map score to confidence level
        if confidence_score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def record_decision(self, decision: AgentDecision) -> None:
        """
        Record a decision made by this agent
        
        Args:
            decision (AgentDecision): The decision to record
        """
        self.decisions.append(decision)
        
        # Limit history size
        if len(self.decisions) > self.max_decisions_history:
            self.decisions = self.decisions[-self.max_decisions_history:]
    
    def update_performance(self, trade_results: List[Dict[str, Any]]) -> None:
        """
        Update agent performance based on trade results
        
        Args:
            trade_results (list): List of trade result dictionaries
        """
        if not trade_results:
            return
        
        # Existing totals
        total_trades = self.performance['total_trades']
        win_count = self.performance['win_count']
        loss_count = self.performance['loss_count']
        total_profit = self.performance['total_profit']
        total_loss = self.performance['total_loss']
        
        # Update with new trades
        for result in trade_results:
            total_trades += 1
            pnl = result.get('pnl', 0)
            
            if pnl > 0:
                win_count += 1
                total_profit += pnl
            elif pnl < 0:
                loss_count += 1
                total_loss += abs(pnl)
        
        # Calculate derived metrics
        win_rate = win_count / total_trades if total_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        avg_win = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        
        # Update performance metrics
        self.performance.update({
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'last_updated': datetime.now().isoformat()
        })
        
        logger.info(f"Updated performance for agent {self.name}: Win rate={win_rate:.2f}, PF={profit_factor:.2f}")
    
    def adapt(self, changes: Dict[str, Any], reason: str = None) -> Dict[str, Any]:
        """
        Adapt agent behavior based on feedback
        
        Args:
            changes (dict): Parameter changes to apply
            reason (str, optional): Reason for adaptation
            
        Returns:
            dict: Results of adaptation
        """
        # Process activation/deactivation
        if 'active' in changes:
            self.is_active = changes['active']
        
        # Update parameters if provided
        if 'parameter_adjustments' in changes and isinstance(changes['parameter_adjustments'], dict):
            for param, modifier in changes['parameter_adjustments'].items():
                if param in self.parameters:
                    # Apply multiplier to parameter
                    self.parameters[param] *= modifier
            
        # Refresh parameters from default if requested
        if changes.get('refresh_parameters', False):
            default_params = self.config.get('default_parameters', {})
            for param, value in default_params.items():
                self.parameters[param] = value
        
        # Log the adaptation
        adaptation_result = {
            'agent_id': self.id,
            'agent_name': self.name,
            'agent_type': self.type.value,
            'changes_applied': changes,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
        
        logger.info(f"Agent {self.name} adapted: {reason or 'no reason provided'}")
        return adaptation_result
    
    def save_state(self, filename: str = None) -> bool:
        """
        Save agent state to file
        
        Args:
            filename (str, optional): Filename to save to
            
        Returns:
            bool: True if successful, False otherwise
        """
        import os
        if filename is None:
            os.makedirs(self.data_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_path}/agent_{self.id}_{timestamp}.json"
        
        try:
            # Prepare state data
            decision_data = [d.to_dict() for d in self.decisions]
            
            state = {
                'id': self.id,
                'name': self.name,
                'type': self.type.value,
                'is_active': self.is_active,
                'performance': self.performance,
                'parameters': self.parameters,
                'confidence_factors': self.confidence_factors,
                'decisions': decision_data,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Agent state saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving agent state: {str(e)}")
            return False
    
    def load_state(self, filename: str) -> bool:
        """
        Load agent state from file
        
        Args:
            filename (str): Filename to load from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            # Validate that the ID matches
            if state['id'] != self.id:
                logger.warning(f"State ID mismatch: {state['id']} != {self.id}")
                return False
            
            # Restore state
            self.name = state['name']
            self.type = AgentType(state['type'])
            self.is_active = state['is_active']
            self.performance = state['performance']
            self.parameters = state['parameters']
            self.confidence_factors = state['confidence_factors']
            
            # Restore decisions
            self.decisions = [AgentDecision.from_dict(d) for d in state['decisions']]
            
            logger.info(f"Agent state loaded from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading agent state: {str(e)}")
            return False

    def get_recommendations(self):
        """
        获取代理推荐的交易行动
        
        Returns:
            list: 交易建议列表，每个建议包含symbol、action、price、confidence和reason字段
        """
        # 基类实现返回随机建议，实际应由子类根据自身策略重写
        recommendations = []
        
        # 示例的上市公司列表
        stocks = [
            '600519.SH', '000651.SZ', '600036.SH', '601318.SH', '000858.SZ',
            '601166.SH', '000333.SZ', '600887.SH', '600276.SH', '601888.SH',
            '600030.SH', '601398.SH', '600000.SH', '601288.SH', '000568.SZ'
        ]
        
        # 随机选择几只股票生成建议
        sample_size = random.randint(2, 5)
        selected_stocks = random.sample(stocks, sample_size)
        
        for stock in selected_stocks:
            # 随机决定是买入还是卖出
            action = random.choice(['buy', 'sell'])
            # 随机生成价格
            price = round(random.uniform(20, 100), 2)
            # 随机生成置信度
            confidence = round(random.uniform(0.6, 0.95), 2)
            
            # 根据行动类型选择推荐原因
            if action == 'buy':
                reasons = [
                    "技术形态突破，看好上涨空间",
                    "K线形成多头排列，趋势向好",
                    "突破前期高点，有望继续上涨",
                    "基本面改善，业绩超预期",
                    "行业景气度提升，估值具吸引力"
                ]
            else:
                reasons = [
                    "上涨动能减弱，建议获利了结",
                    "技术指标顶背离，存在回调风险",
                    "估值已处高位，继续上行空间有限",
                    "K线形成顶部结构，短期有回落风险",
                    "行业竞争加剧，盈利能力承压"
                ]
            
            # 随机选择一个原因
            reason = random.choice(reasons)
            
            recommendations.append({
                'symbol': stock,
                'action': action,
                'price': price,
                'confidence': confidence,
                'reason': reason
            })
        
        return recommendations
    
    def update(self, market_state):
        """
        更新代理状态
        
        Args:
            market_state: 市场状态
            
        Returns:
            bool: 更新是否成功
        """
        # 基类实现，子类应重写
        try:
            # 更新基础指标
            self._update_performance_metrics(market_state)
            return True
        except Exception as e:
            logger.error(f"Error updating agent {self.name}: {e}")
            return False
    
    def _update_performance_metrics(self, market_state):
        """
        更新绩效指标
        
        Args:
            market_state: 市场状态
        """
        # 示例实现，生成随机绩效数据
        self.performance_metrics = {
            'total_return': round(random.uniform(-10, 20), 2),
            'win_rate': round(random.uniform(0.4, 0.8), 2),
            'sharpe_ratio': round(random.uniform(0.5, 2.5), 2),
            'max_drawdown': round(random.uniform(0.02, 0.1), 2)
        }
    
    def generate_signals(self, market_state):
        """
        生成交易信号
        
        Args:
            market_state: 市场状态
            
        Returns:
            list: 交易信号列表
        """
        # 基类实现，子类应重写
        return [] 