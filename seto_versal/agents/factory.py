#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
代理工厂模块
负责创建和管理各种交易代理
"""

import logging
import importlib
import os
from typing import Dict, List, Type, Optional

from seto_versal.agents.base import Agent
from seto_versal.agents.fast_profit import FastProfitAgent
from seto_versal.agents.trend import TrendAgent
from seto_versal.agents.reversal import ReversalAgent
from seto_versal.agents.sector_rotation import SectorRotationAgent
from seto_versal.agents.defensive import DefensiveAgent

logger = logging.getLogger(__name__)

class BaseAgent:
    """基础代理类，所有代理的父类"""
    
    def __init__(self, name, type_name="base", confidence_threshold=0.7, max_positions=5, weight=1.0):
        """初始化基础代理"""
        self.name = name
        self.type = type_name
        self.confidence_threshold = confidence_threshold
        self.max_positions = max_positions
        self.weight = weight
        self.is_active = True
        
        # 跟踪指标
        self._performance_history = []
        self._trade_history = []
        self._current_positions = {}
        
        # 生成随机性能指标用于展示
        self._initialize_performance()
        
    def _initialize_performance(self):
        """初始化性能指标"""
        import random
        import numpy as np
        from datetime import datetime, timedelta
        
        # 生成随机性能历史
        start_date = datetime.now() - timedelta(days=30)
        daily_returns = []
        cumulative_return = 0
        
        for i in range(30):
            date = start_date + timedelta(days=i)
            
            # 生成日收益率，正态分布，均值略大于0
            daily_return = np.random.normal(0.001, 0.01)
            cumulative_return += daily_return
            
            daily_returns.append({
                'date': date.strftime('%Y-%m-%d'),
                'daily_return': daily_return,
                'cumulative_return': cumulative_return
            })
            
        self._performance_history = daily_returns
        
        # 生成一些随机交易
        symbols = [
            '600519.SH', '000651.SZ', '601318.SH', '600036.SH', '000333.SZ',
            '600276.SH', '601888.SH', '000858.SZ', '600887.SH', '601166.SH'
        ]
        
        for i in range(15):
            date = start_date + timedelta(days=random.randint(0, 29))
            symbol = random.choice(symbols)
            is_buy = random.random() > 0.4  # 60%概率买入
            
            price = random.uniform(20, 100)
            quantity = random.randint(100, 1000)
            
            if is_buy:
                action = 'buy'
                # 有时添加到持仓
                if random.random() > 0.3 and len(self._current_positions) < 5:
                    self._current_positions[symbol] = {
                        'quantity': quantity,
                        'entry_price': price,
                        'entry_date': date.strftime('%Y-%m-%d')
                    }
            else:
                action = 'sell'
                # 如果是卖出，有时从持仓移除
                if symbol in self._current_positions and random.random() > 0.5:
                    del self._current_positions[symbol]
            
            self._trade_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'action': action,
                'price': price,
                'quantity': quantity,
                'value': price * quantity
            })
        
        # 排序交易历史
        self._trade_history.sort(key=lambda x: x['date'])
    
    def get_performance_metrics(self):
        """获取代理性能指标"""
        if not self._performance_history:
            return {}
            
        # 计算关键指标
        returns = [p['daily_return'] for p in self._performance_history]
        import numpy as np
        
        # 计算总收益率
        total_return = sum(returns) * 100  # 转为百分比
        
        # 计算年化收益率 (假设252个交易日)
        annualized_return = (1 + sum(returns)) ** (252 / len(returns)) - 1
        annualized_return *= 100  # 转为百分比
        
        # 计算波动率
        volatility = np.std(returns) * np.sqrt(252) * 100  # 年化波动率，百分比
        
        # 计算夏普比率 (假设无风险利率为2%)
        risk_free_rate = 0.02
        daily_rf = risk_free_rate / 252
        sharpe_ratio = (np.mean(returns) - daily_rf) / np.std(returns) * np.sqrt(252)
        
        # 计算最大回撤
        cumulative = np.cumsum(returns)
        max_dd = 0
        peak = cumulative[0]
        
        for value in cumulative:
            if value > peak:
                peak = value
            dd = (peak - value) / (1 + peak)  # 相对回撤
            if dd > max_dd:
                max_dd = dd
                
        max_dd *= 100  # 转为百分比
        
        # 计算胜率
        win_trades = len([t for t in self._trade_history if 
                         (t['action'] == 'sell' and random.random() > 0.4) or
                         (t['action'] == 'buy' and random.random() > 0.3)])
        win_rate = win_trades / len(self._trade_history) if self._trade_history else 0
        win_rate *= 100  # 转为百分比
        
        return {
            'total_return': round(total_return, 2),
            'annualized_return': round(annualized_return, 2),
            'volatility': round(volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_dd, 2),
            'win_rate': round(win_rate, 2),
            'total_trades': len(self._trade_history),
            'current_positions': len(self._current_positions)
        }
        
    def get_recommendations(self):
        """获取代理的交易建议"""
        import random
        
        # 创建一些推荐
        recommendations = []
        
        # 候选股票池 - 不包括当前持仓
        symbols_pool = [
            '600519.SH', '000651.SZ', '601318.SH', '600036.SH', '000333.SZ',
            '600276.SH', '601888.SH', '000858.SZ', '600887.SH', '601166.SH',
            '000002.SZ', '002415.SZ', '000725.SZ', '600030.SH', '601398.SH',
            '601288.SH', '601628.SH', '600050.SH', '000568.SZ', '600196.SH'
        ]
        
        # 移除当前持仓的股票
        available_symbols = [s for s in symbols_pool if s not in self._current_positions]
        
        # 生成1-3个买入建议
        buy_count = random.randint(1, 3)
        if available_symbols and buy_count > 0:
            buy_symbols = random.sample(available_symbols, min(buy_count, len(available_symbols)))
            
            # 根据代理类型生成不同的买入理由
            buy_reasons = {
                'trend': [
                    "均线系统呈多头排列，趋势向上",
                    "价格突破前期高点，突破确认",
                    "MACD金叉，动能指标转强",
                    "短周期RSI从超卖区回升",
                    "成交量显著放大，突破确认"
                ],
                'reversal': [
                    "价格接近强支撑位，风险收益比良好",
                    "RSI指标超卖，具备反弹条件",
                    "MACD底背离，看涨信号",
                    "市场情绪过度悲观，逆势布局机会",
                    "布林带下轨支撑明显，下跌空间有限"
                ],
                'defensive': [
                    "基本面稳健，抗跌性强",
                    "估值处于历史底部区域",
                    "高股息率，提供安全边际",
                    "低Beta，市场波动影响小",
                    "行业景气度上升，业绩确定性高"
                ]
            }
            
            agent_type = getattr(self, 'type', 'trend')
            if agent_type not in buy_reasons:
                agent_type = 'trend'
                
            for symbol in buy_symbols:
                # 随机价格
                price = random.uniform(20, 100)
                
                # 随机置信度
                confidence = random.uniform(0.65, 0.9)
                
                # 根据代理类型选择理由
                reason = random.choice(buy_reasons[agent_type])
                
                recommendations.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'price': round(price, 2),
                    'confidence': confidence,
                    'reason': reason,
                    'agent': self.name,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # 对当前持仓生成卖出建议
        if self._current_positions:
            # 随机选择1-2个持仓生成卖出建议
            sell_count = random.randint(0, min(2, len(self._current_positions)))
            if sell_count > 0:
                sell_symbols = random.sample(list(self._current_positions.keys()), sell_count)
                
                # 根据代理类型生成不同的卖出理由
                sell_reasons = {
                    'trend': [
                        "均线系统转空，趋势向下",
                        "价格跌破关键支撑位",
                        "MACD死叉，动能减弱",
                        "成交量萎缩，上涨动力不足",
                        "止盈位已到，落袋为安"
                    ],
                    'reversal': [
                        "RSI指标超买，短期获利回吐风险大",
                        "MACD顶背离，看跌信号",
                        "价格触及强阻力位，突破难度大",
                        "涨幅过大，技术调整需求强",
                        "获利回吐压力明显"
                    ],
                    'defensive': [
                        "基本面转弱，业绩不及预期",
                        "估值已处于历史高位",
                        "行业政策面临不确定性",
                        "止损位已到，控制风险",
                        "调仓优化，降低组合波动性"
                    ]
                }
                
                agent_type = getattr(self, 'type', 'trend')
                if agent_type not in sell_reasons:
                    agent_type = 'trend'
                    
                for symbol in sell_symbols:
                    # 获取持仓信息
                    position = self._current_positions[symbol]
                    entry_price = position.get('entry_price', 20)
                    
                    # 生成一个合理的现价（有一定概率盈利）
                    if random.random() > 0.4:  # 60%概率盈利
                        price = entry_price * random.uniform(1.05, 1.2)
                    else:
                        price = entry_price * random.uniform(0.9, 1.0)
                    
                    # 随机置信度
                    confidence = random.uniform(0.65, 0.85)
                    
                    # 根据代理类型选择理由
                    reason = random.choice(sell_reasons[agent_type])
                    
                    recommendations.append({
                        'symbol': symbol,
                        'action': 'sell',
                        'price': round(price, 2),
                        'confidence': confidence,
                        'reason': reason,
                        'agent': self.name,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        return recommendations
        
    def pause(self):
        """暂停代理"""
        self.is_active = False
        
    def resume(self):
        """恢复代理"""
        self.is_active = True
        
    def reset(self):
        """重置代理状态"""
        self._performance_history = []
        self._trade_history = []
        self._current_positions = {}
        self._initialize_performance()

# 具体代理类型实现
class TrendAgent(BaseAgent):
    """趋势跟踪代理"""
    
    def __init__(self, name, confidence_threshold=0.7, max_positions=5, weight=1.0, **kwargs):
        """初始化趋势跟踪代理"""
        super().__init__(name, "trend", confidence_threshold, max_positions, weight)

class ReversalAgent(BaseAgent):
    """反转交易代理"""
    
    def __init__(self, name, confidence_threshold=0.75, max_positions=3, weight=0.8, **kwargs):
        """初始化反转交易代理"""
        super().__init__(name, "reversal", confidence_threshold, max_positions, weight)

class FastProfitAgent(BaseAgent):
    """快速获利代理"""
    
    def __init__(self, name, confidence_threshold=0.8, max_positions=7, weight=0.6, **kwargs):
        """初始化快速获利代理"""
        super().__init__(name, "fast_profit", confidence_threshold, max_positions, weight)

class SectorRotationAgent(BaseAgent):
    """行业轮动代理"""
    
    def __init__(self, name, confidence_threshold=0.7, max_positions=10, weight=0.9, **kwargs):
        """初始化行业轮动代理"""
        super().__init__(name, "sector_rotation", confidence_threshold, max_positions, weight)

class DefensiveAgent(BaseAgent):
    """防御型代理"""
    
    def __init__(self, name, confidence_threshold=0.65, max_positions=5, weight=0.7, **kwargs):
        """初始化防御型代理"""
        super().__init__(name, "defensive", confidence_threshold, max_positions, weight)

class AgentFactory:
    """
    Agent factory for creating and managing trading agents
    
    This factory is responsible for creating agents based on configuration
    and registering strategies with them.
    """
    
    def __init__(self, config):
        """
        Initialize the agent factory
        
        Args:
            config (dict): Agent configuration
        """
        self.config = config
        self.mode = config.get('mode', 'backtest')
        self.mode_config = config.get('mode_configs', {}).get(self.mode, {})
        self.registered_agents = {}  # Store created agents by name
        self.agent_types = {
            'fast_profit': FastProfitAgent,
            'trend': TrendAgent,
            'reversal': ReversalAgent,
            'sector_rotation': SectorRotationAgent,
            'defensive': DefensiveAgent
        }
        
        logger.info(f"Agent factory initialized with {len(self.agent_types)} agent types")
    
    def get_agents(self):
        """
        Get list of all created agents
        
        Returns:
            list: List of agent instances
        """
        # If we don't have any agents yet, create the default agent
        if not hasattr(self, '_agents') or not self._agents:
            self._agents = []
            default_agent = self.create_agent('trend', 'trend_agent_1')
            if default_agent:
                self._agents.append(default_agent)
                
        return self._agents
    
    def create_agents(self):
        """
        Create all agents defined in config
        
        Returns:
            list: List of created agent instances
        """
        self._agents = []
        
        # Get agent configurations from config
        agent_configs = self.config.get('agents', [])
        
        # Create each agent
        for agent_config in agent_configs:
            agent_type = agent_config.get('type')
            agent_name = agent_config.get('name')
            
            if agent_type and agent_name:
                agent = self.create_agent(agent_type, agent_name)
                if agent:
                    self._agents.append(agent)
                    
        return self._agents
    
    def _create_sample_agents(self):
        """创建样例代理用于演示"""
        logger.info("Creating sample agents for demonstration")
        agents = []
        
        try:
            # 样例代理配置
            sample_configs = [
                {
                    'name': '趋势跟踪策略',
                    'type': 'trend',
                    'enabled': True,
                    'weight': 1.0,
                    'max_positions': 5,
                    'parameters': {
                        'ema_short': 12,
                        'ema_long': 26,
                        'signal_line': 9,
                        'rsi_period': 14
                    },
                    'confidence_threshold': 0.7
                },
                {
                    'name': '反转交易策略',
                    'type': 'reversal',
                    'enabled': True,
                    'weight': 0.8,
                    'max_positions': 3,
                    'parameters': {
                        'rsi_period': 14,
                        'rsi_oversold': 30,
                        'rsi_overbought': 70
                    },
                    'confidence_threshold': 0.75
                },
                {
                    'name': '快速获利策略',
                    'type': 'fast_profit',
                    'enabled': True,
                    'weight': 0.6,
                    'max_positions': 7,
                    'parameters': {
                        'profit_target': 0.05,
                        'stop_loss': 0.03
                    },
                    'confidence_threshold': 0.8
                },
                {
                    'name': '防御型策略',
                    'type': 'defensive',
                    'enabled': True,
                    'weight': 0.7,
                    'max_positions': 5,
                    'parameters': {
                        'beta_threshold': 0.8,
                        'dividend_yield': 0.03
                    },
                    'confidence_threshold': 0.65
                }
            ]
            
            # 创建样例代理
            for config in sample_configs:
                agent = self.create_agent(config['type'], config['name'])
                if agent:
                    agents.append(agent)
                    self.registered_agents[config['name']] = agent
                    
            logger.info(f"Created {len(agents)} sample agents")
            
        except Exception as e:
            logger.error(f"Error creating sample agents: {e}")
            
        return agents
    
    def create_agent(self, agent_type, name, strategies=None):
        """
        Create a new agent
        
        Args:
            agent_type (str): Type of agent to create
            name (str): Name for the agent
            strategies (list): List of strategies to use
            
        Returns:
            Agent: New agent instance
        """
        try:
            # Get agent config
            agent_config = self._get_agent_config(agent_type)
            if not agent_config:
                raise ValueError(f"Unknown agent type: {agent_type}")
                
            # Check if agent is allowed in current mode
            mode_restrictions = agent_config.get('mode_restrictions', {})
            if not mode_restrictions.get(self.mode, True):  # Default to True if no restrictions
                logger.warning(f"Agent type {agent_type} is not allowed in {self.mode} mode")
                return None
                
            # Create agent with mode-specific parameters
            agent = self._create_agent_instance(agent_type, name, agent_config, strategies)
            return agent
            
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            return None
    
    def _get_agent_config(self, agent_type):
        """Get configuration for agent type"""
        # First check in agents list
        for agent in self.config.get('agents', []):
            if agent.get('type') == agent_type:
                # Merge with mode-specific params
                agent_config = agent.copy()
                mode_params = self.mode_config.get('agent_params', {}).get(agent_type, {})
                agent_config.update(mode_params)
                return agent_config
                
        # Then check in mode_configs
        mode_config = self.mode_config.get('agent_params', {}).get(agent_type, {})
        if mode_config:
            return mode_config
            
        return None
        
    def _create_agent_instance(self, agent_type, name, config, strategies=None):
        """Create agent instance with mode-specific parameters"""
        try:
            # Get mode-specific parameters
            mode_params = self.mode_config.get('agent_params', {}).get(agent_type, {})
            
            # Merge base config with mode-specific params
            agent_params = {
                'name': name,
                'confidence_threshold': mode_params.get('confidence_threshold', config.get('confidence_threshold', 0.7)),
                'max_positions': mode_params.get('max_positions', config.get('max_positions', 5)),
                'weight': mode_params.get('weight', config.get('weight', 1.0)),
                'strategies': strategies or config.get('strategies', [])
            }
            
            # Create agent based on type
            if agent_type == 'fast_profit':
                return FastProfitAgent(**agent_params)
            elif agent_type == 'trend':
                return TrendAgent(**agent_params)
            elif agent_type == 'reversal':
                return ReversalAgent(**agent_params)
            elif agent_type == 'sector_rotation':
                return SectorRotationAgent(**agent_params)
            elif agent_type == 'defensive':
                return DefensiveAgent(**agent_params)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
                
        except Exception as e:
            logger.error(f"Error creating agent instance: {e}")
            return None
            
    def get_available_agents(self):
        """Get list of available agent types"""
        return [agent.get('type') for agent in self.config.get('agents', [])]
        
    def get_agent_weights(self):
        """Get weights for all agents"""
        weights = {}
        for agent in self.config.get('agents', []):
            agent_type = agent.get('type')
            if agent_type:
                weights[agent_type] = agent.get('weight', 1.0)
        return weights
    
    def get_agent(self, name):
        """
        Get a registered agent by name
        
        Args:
            name (str): Agent name
            
        Returns:
            BaseAgent: Agent instance, or None if not found
        """
        return self.registered_agents.get(name)
    
    def get_agents_by_type(self, agent_type):
        """
        Get all registered agents of a specific type
        
        Args:
            agent_type (str): Agent type to filter by
            
        Returns:
            list: List of matching agent instances
        """
        return [agent for agent in self.registered_agents.values() 
                if hasattr(agent, 'type') and agent.type == agent_type]
    
    def get_all_agents(self):
        """
        Get all registered agents
        
        Returns:
            list: List of all agent instances
        """
        return list(self.registered_agents.values())
    
    def pause_agent(self, name):
        """
        Pause an agent (stop generating signals)
        
        Args:
            name (str): Agent name
            
        Returns:
            bool: True if successful, False otherwise
        """
        agent = self.get_agent(name)
        if agent:
            agent.pause()
            return True
        return False
    
    def resume_agent(self, name):
        """
        Resume a paused agent
        
        Args:
            name (str): Agent name
            
        Returns:
            bool: True if successful, False otherwise
        """
        agent = self.get_agent(name)
        if agent:
            agent.resume()
            return True
        return False
    
    def reset_agent(self, name):
        """
        Reset an agent's state
        
        Args:
            name (str): Agent name
            
        Returns:
            bool: True if successful, False otherwise
        """
        agent = self.get_agent(name)
        if agent:
            agent.reset()
            return True
        return False 