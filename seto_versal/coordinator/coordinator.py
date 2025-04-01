#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Coordinator module for SETO-Versal
Provides decision coordination and integration across multiple agents
"""

import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import json
from collections import defaultdict
import numpy as np

from seto_versal.agents.base import AgentDecision, ConfidenceLevel, AgentType

logger = logging.getLogger(__name__)

class CoordinationMethod(Enum):
    """Methods of coordinating decisions across agents"""
    WEIGHTED_VOTE = "weighted_vote"        # Weight by confidence & agent performance
    HIGHEST_CONFIDENCE = "highest_confidence"  # Select decision with highest confidence
    CONSENSUS = "consensus"                # Require consensus among agents
    DOMINANT_AGENT = "dominant_agent"      # Give priority to specific agent types
    MARKET_ADAPTIVE = "market_adaptive"    # Adapt based on market conditions

class CoordinatedDecision:
    """
    Represents a coordinated decision across multiple agents
    """
    
    def __init__(self, 
                agent_decisions: List[AgentDecision],
                symbol: str,
                decision_type: str,
                confidence: float,
                primary_agent_id: str,
                primary_agent_type: str,
                target_price: Optional[float] = None,
                stop_loss: Optional[float] = None,
                quantity: Optional[int] = None,
                timeframe: Optional[str] = None,
                reasoning: Optional[str] = None,
                supporting_agents: Optional[List[str]] = None):
        """
        Initialize a coordinated decision
        
        Args:
            agent_decisions (list): Original agent decisions that form this decision
            symbol (str): Trading symbol
            decision_type (str): Type of decision (buy, sell, etc.)
            confidence (float): Aggregated confidence (0.0-1.0)
            primary_agent_id (str): ID of the primary agent driving this decision
            primary_agent_type (str): Type of the primary agent
            target_price (float, optional): Target price
            stop_loss (float, optional): Stop loss price
            quantity (int, optional): Quantity to trade
            timeframe (str, optional): Expected holding timeframe
            reasoning (str, optional): Reasoning behind the decision
            supporting_agents (list, optional): List of supporting agent IDs
        """
        self.id = str(uuid.uuid4())
        self.agent_decisions = agent_decisions
        self.symbol = symbol
        self.decision_type = decision_type.lower()
        self.confidence = confidence
        self.primary_agent_id = primary_agent_id
        self.primary_agent_type = primary_agent_type
        self.target_price = target_price
        self.stop_loss = stop_loss
        self.quantity = quantity
        self.timeframe = timeframe
        self.reasoning = reasoning
        self.supporting_agents = supporting_agents or []
        self.timestamp = datetime.now()
        self.executed = False
        self.execution_info = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert decision to dictionary
        
        Returns:
            dict: Dictionary representation of the decision
        """
        return {
            'id': self.id,
            'symbol': self.symbol,
            'decision_type': self.decision_type,
            'confidence': self.confidence,
            'primary_agent_id': self.primary_agent_id,
            'primary_agent_type': self.primary_agent_type,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'quantity': self.quantity,
            'timeframe': self.timeframe,
            'reasoning': self.reasoning,
            'supporting_agents': self.supporting_agents,
            'timestamp': self.timestamp.isoformat(),
            'executed': self.executed,
            'execution_info': self.execution_info,
            'original_decisions': [d.id for d in self.agent_decisions]
        }
    
    def __str__(self) -> str:
        """String representation of the coordinated decision"""
        return (f"CoordinatedDecision {self.id[:8]} - {self.decision_type.upper()} {self.symbol} "
                f"(Confidence: {self.confidence:.2f}, Primary: {self.primary_agent_type})")

class TradeCoordinator:
    """
    Coordinates and integrates trade decisions from multiple agents
    Resolves conflicts and prioritizes trades based on confidence and context
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trade coordinator with configuration
        
        Args:
            config (dict): Coordinator configuration
        """
        self.config = config
        self.name = config.get('name', 'trade_coordinator')
        
        # Coordination settings
        self.method = CoordinationMethod(config.get('method', 'weighted_vote'))
        self.min_confidence = config.get('min_confidence', 0.5)
        self.min_consensus = config.get('min_consensus', 0.5)  # Percentage of agents needed for consensus
        self.max_trades_per_cycle = config.get('max_trades_per_cycle', 5)
        self.agent_weights = config.get('agent_weights', {})  # Agent type to weight mapping
        
        # Default agent weights if not specified
        if not self.agent_weights:
            self.agent_weights = {
                AgentType.TREND.value: 1.0,
                AgentType.RAPID.value: 1.0,
                AgentType.REVERSAL.value: 1.0,
                AgentType.SECTOR_ROTATION.value: 1.0,
                AgentType.DEFENSIVE.value: 1.0,
                AgentType.CUSTOM.value: 0.8
            }
        
        # Agent type priorities by market regime
        self.regime_priorities = config.get('regime_priorities', {
            'bull': [AgentType.TREND.value, AgentType.RAPID.value, 
                    AgentType.SECTOR_ROTATION.value, AgentType.REVERSAL.value, 
                    AgentType.DEFENSIVE.value],
            'bear': [AgentType.DEFENSIVE.value, AgentType.REVERSAL.value, 
                   AgentType.TREND.value, AgentType.SECTOR_ROTATION.value, 
                   AgentType.RAPID.value],
            'range': [AgentType.SECTOR_ROTATION.value, AgentType.REVERSAL.value, 
                    AgentType.RAPID.value, AgentType.DEFENSIVE.value, 
                    AgentType.TREND.value],
            'volatile': [AgentType.DEFENSIVE.value, AgentType.RAPID.value, 
                       AgentType.REVERSAL.value, AgentType.SECTOR_ROTATION.value, 
                       AgentType.TREND.value],
            'recovery': [AgentType.TREND.value, AgentType.REVERSAL.value, 
                       AgentType.RAPID.value, AgentType.SECTOR_ROTATION.value, 
                       AgentType.DEFENSIVE.value],
            'topping': [AgentType.DEFENSIVE.value, AgentType.SECTOR_ROTATION.value, 
                      AgentType.REVERSAL.value, AgentType.RAPID.value, 
                      AgentType.TREND.value],
            'unknown': [AgentType.DEFENSIVE.value, AgentType.TREND.value, 
                      AgentType.RAPID.value, AgentType.REVERSAL.value, 
                      AgentType.SECTOR_ROTATION.value]
        })
        
        # Portfolio constraints
        self.max_exposure_per_symbol = config.get('max_exposure_per_symbol', 0.2)  # Max 20% in one symbol
        self.max_exposure_per_sector = config.get('max_exposure_per_sector', 0.3)  # Max 30% in one sector
        
        # Tracking data
        self.coordinated_decisions = []
        self.max_history = config.get('max_history', 1000)
        self.conflicts_resolved = 0
        self.last_coordination_time = None
        
        # Data path for saving state
        self.data_path = config.get('data_path', 'seto_versal/data/coordinator')
        
        # Save recent intentions for recommendation
        self.last_intentions = []
        
        logger.info(f"Trade coordinator '{self.name}' initialized with {self.method.value} method")
    
    def coordinate(self, intentions, market_state):
        """
        协调多个Agent的交易意图，生成最终交易决策
        
        Args:
            intentions (list): Agent产生的交易意图列表
            market_state (MarketState): 当前市场状态
            
        Returns:
            list: 交易决策列表
        """
        logger.info(f"Coordinating {len(intentions)} trading intentions")
        
        # Save reference to market state
        self.market_state = market_state
        
        # Save intentions for recommendation
        self.last_intentions = intentions
        
        # If no intentions, return empty list
        if not intentions:
            return []
        
        # Record coordination time
        self.last_coordination_time = datetime.now()
        
        # Group decisions by symbol and decision type
        grouped_decisions = self._group_decisions(intentions)
        
        # Update agent weights based on market regime if using market adaptive method
        if self.method == CoordinationMethod.MARKET_ADAPTIVE:
            self._update_agent_weights(market_state)
        
        # Coordinate each group
        coordinated_decisions = []
        
        for (symbol, decision_type), decisions in grouped_decisions.items():
            # Skip if only one decision in group
            if len(decisions) == 1:
                # Convert single decision to coordinated decision
                coord_decision = self._create_coordinated_decision_from_single(decisions[0])
                coordinated_decisions.append(coord_decision)
                continue
            
            # Coordinate based on selected method
            if self.method == CoordinationMethod.WEIGHTED_VOTE:
                coord_decision = self._coordinate_weighted_vote(decisions, market_state)
            elif self.method == CoordinationMethod.HIGHEST_CONFIDENCE:
                coord_decision = self._coordinate_highest_confidence(decisions)
            elif self.method == CoordinationMethod.CONSENSUS:
                coord_decision = self._coordinate_consensus(decisions)
            elif self.method == CoordinationMethod.DOMINANT_AGENT:
                coord_decision = self._coordinate_dominant_agent(decisions, market_state)
            elif self.method == CoordinationMethod.MARKET_ADAPTIVE:
                coord_decision = self._coordinate_market_adaptive(decisions, market_state)
            else:
                # Default to weighted vote
                coord_decision = self._coordinate_weighted_vote(decisions, market_state)
            
            if coord_decision:
                coordinated_decisions.append(coord_decision)
        
        # Apply portfolio constraints and prioritize trades
        final_decisions = self._apply_constraints(coordinated_decisions, market_state)
        
        # Record coordinated decisions
        self.coordinated_decisions.extend(final_decisions)
        
        # Limit history size
        if len(self.coordinated_decisions) > self.max_history:
            self.coordinated_decisions = self.coordinated_decisions[-self.max_history:]
        
        logger.info(f"Coordinated {len(intentions)} intentions into {len(final_decisions)} trades")
        return final_decisions
    
    def _group_decisions(self, decisions: List[AgentDecision]) -> Dict[Tuple[str, str], List[AgentDecision]]:
        """
        Group decisions by symbol and decision type
        
        Args:
            decisions (list): List of agent decisions
            
        Returns:
            dict: Dictionary grouped by (symbol, decision_type)
        """
        grouped = defaultdict(list)
        
        for decision in decisions:
            key = (decision.symbol, decision.decision_type)
            grouped[key].append(decision)
        
        return grouped
    
    def _update_agent_weights(self, market_state) -> None:
        """
        Update agent weights based on market regime
        
        Args:
            market_state: Current market state
        """
        # Get current market regime
        regime = market_state.get_market_regime()
        
        # Get agent priorities for this regime
        priorities = self.regime_priorities.get(regime, self.regime_priorities['unknown'])
        
        # Adjust weights based on priorities
        # First agent type gets highest weight, decreasing for others
        base_weight = 1.0
        decrement = 0.15
        
        # Reset weights
        for agent_type in self.agent_weights:
            self.agent_weights[agent_type] = base_weight
        
        # Apply priority adjustments
        for i, agent_type in enumerate(priorities):
            self.agent_weights[agent_type] = base_weight - (i * decrement)
            # Ensure minimum weight is 0.4
            self.agent_weights[agent_type] = max(0.4, self.agent_weights[agent_type])
        
        logger.debug(f"Updated agent weights for {regime} regime: {self.agent_weights}")
    
    def _create_coordinated_decision_from_single(self, decision: AgentDecision) -> CoordinatedDecision:
        """
        Create a coordinated decision from a single agent decision
        
        Args:
            decision (AgentDecision): The agent decision
            
        Returns:
            CoordinatedDecision: Coordinated decision
        """
        # Convert confidence level to float
        confidence_float = decision.confidence.value / ConfidenceLevel.VERY_HIGH.value
        
        return CoordinatedDecision(
            agent_decisions=[decision],
            symbol=decision.symbol,
            decision_type=decision.decision_type,
            confidence=confidence_float,
            primary_agent_id=decision.agent_id,
            primary_agent_type=self._get_agent_type_from_id(decision.agent_id),
            target_price=decision.target_price,
            stop_loss=decision.stop_loss,
            quantity=decision.quantity,
            timeframe=decision.timeframe,
            reasoning=decision.reasoning,
            supporting_agents=[]
        )
    
    def _get_agent_type_from_id(self, agent_id: str) -> str:
        """
        Extract agent type from agent ID
        
        Args:
            agent_id (str): Agent ID
            
        Returns:
            str: Agent type
        """
        # In a real implementation, we would look up the agent
        # For now, just extract from ID if available or return 'unknown'
        for agent_type in AgentType:
            if agent_type.value in agent_id.lower():
                return agent_type.value
        
        return "unknown"
    
    def _coordinate_weighted_vote(self, decisions: List[AgentDecision], 
                                 market_state) -> Optional[CoordinatedDecision]:
        """
        Coordinate decisions using weighted voting
        
        Args:
            decisions (list): List of agent decisions
            market_state: Current market state
            
        Returns:
            CoordinatedDecision: Coordinated decision or None
        """
        if not decisions:
            return None
        
        # Calculate weighted votes
        total_weight = 0.0
        weighted_confidence = 0.0
        supporting_agents = []
        
        # Track parameters for final decision
        target_prices = []
        stop_losses = []
        quantities = []
        timeframes = []
        reasonings = []
        
        for decision in decisions:
            # Get agent type
            agent_type = self._get_agent_type_from_id(decision.agent_id)
            
            # Get agent weight
            agent_weight = self.agent_weights.get(agent_type, 1.0)
            
            # Calculate decision confidence as float
            decision_confidence = decision.confidence.value / ConfidenceLevel.VERY_HIGH.value
            
            # Add to weighted sum
            weighted_confidence += decision_confidence * agent_weight
            total_weight += agent_weight
            
            # Add to supporting agents
            supporting_agents.append(decision.agent_id)
            
            # Collect parameters if available
            if decision.target_price is not None:
                target_prices.append(decision.target_price)
            
            if decision.stop_loss is not None:
                stop_losses.append(decision.stop_loss)
            
            if decision.quantity is not None:
                quantities.append(decision.quantity)
            
            if decision.timeframe is not None:
                timeframes.append(decision.timeframe)
            
            if decision.reasoning is not None:
                reasonings.append(f"{agent_type}: {decision.reasoning}")
        
        # Calculate aggregate confidence
        if total_weight > 0:
            aggregate_confidence = weighted_confidence / total_weight
        else:
            aggregate_confidence = 0.0
        
        # Check minimum confidence
        if aggregate_confidence < self.min_confidence:
            logger.debug(f"Insufficient confidence ({aggregate_confidence:.2f}) for {decisions[0].symbol}")
            return None
        
        # Find decision with highest confidence * weight to use as primary
        primary_decision = max(
            decisions, 
            key=lambda d: (d.confidence.value / ConfidenceLevel.VERY_HIGH.value) * 
                          self.agent_weights.get(self._get_agent_type_from_id(d.agent_id), 1.0)
        )
        
        # Calculate aggregate parameters
        target_price = np.median(target_prices) if target_prices else primary_decision.target_price
        stop_loss = np.median(stop_losses) if stop_losses else primary_decision.stop_loss
        quantity = int(np.median(quantities)) if quantities else primary_decision.quantity
        timeframe = max(set(timeframes), key=timeframes.count) if timeframes else primary_decision.timeframe
        reasoning = " + ".join(reasonings) if reasonings else primary_decision.reasoning
        
        # Create coordinated decision
        return CoordinatedDecision(
            agent_decisions=decisions,
            symbol=primary_decision.symbol,
            decision_type=primary_decision.decision_type,
            confidence=aggregate_confidence,
            primary_agent_id=primary_decision.agent_id,
            primary_agent_type=self._get_agent_type_from_id(primary_decision.agent_id),
            target_price=target_price,
            stop_loss=stop_loss,
            quantity=quantity,
            timeframe=timeframe,
            reasoning=reasoning,
            supporting_agents=supporting_agents
        )
    
    def _coordinate_highest_confidence(self, decisions: List[AgentDecision]) -> Optional[CoordinatedDecision]:
        """
        Coordinate decisions by choosing the one with highest confidence
        
        Args:
            decisions (list): List of agent decisions
            
        Returns:
            CoordinatedDecision: Coordinated decision or None
        """
        if not decisions:
            return None
        
        # Find decision with highest confidence
        primary_decision = max(decisions, key=lambda d: d.confidence.value)
        
        # Check if confidence meets minimum requirement
        confidence_float = primary_decision.confidence.value / ConfidenceLevel.VERY_HIGH.value
        if confidence_float < self.min_confidence:
            logger.debug(f"Highest confidence ({confidence_float:.2f}) below threshold for {primary_decision.symbol}")
            return None
        
        # List other agents as supporting
        supporting_agents = [d.agent_id for d in decisions if d != primary_decision]
        
        # Create coordinated decision
        return CoordinatedDecision(
            agent_decisions=decisions,
            symbol=primary_decision.symbol,
            decision_type=primary_decision.decision_type,
            confidence=confidence_float,
            primary_agent_id=primary_decision.agent_id,
            primary_agent_type=self._get_agent_type_from_id(primary_decision.agent_id),
            target_price=primary_decision.target_price,
            stop_loss=primary_decision.stop_loss,
            quantity=primary_decision.quantity,
            timeframe=primary_decision.timeframe,
            reasoning=primary_decision.reasoning,
            supporting_agents=supporting_agents
        )
    
    def _coordinate_consensus(self, decisions: List[AgentDecision]) -> Optional[CoordinatedDecision]:
        """
        Coordinate decisions requiring consensus among agents
        
        Args:
            decisions (list): List of agent decisions
            
        Returns:
            CoordinatedDecision: Coordinated decision or None
        """
        if not decisions:
            return None
        
        # Check if enough agents agree (based on min_consensus percentage)
        required_count = max(2, int(len(decisions) * self.min_consensus))
        
        if len(decisions) < required_count:
            logger.debug(f"Not enough agents ({len(decisions)}) for consensus on {decisions[0].symbol}")
            return None
        
        # Sort decisions by confidence (descending)
        sorted_decisions = sorted(decisions, key=lambda d: d.confidence.value, reverse=True)
        primary_decision = sorted_decisions[0]
        
        # Calculate average confidence
        avg_confidence = sum(d.confidence.value for d in decisions) / len(decisions)
        avg_confidence_float = avg_confidence / ConfidenceLevel.VERY_HIGH.value
        
        # Collect supporting agents
        supporting_agents = [d.agent_id for d in decisions if d != primary_decision]
        
        # Create coordinated decision
        return CoordinatedDecision(
            agent_decisions=decisions,
            symbol=primary_decision.symbol,
            decision_type=primary_decision.decision_type,
            confidence=avg_confidence_float,
            primary_agent_id=primary_decision.agent_id,
            primary_agent_type=self._get_agent_type_from_id(primary_decision.agent_id),
            target_price=primary_decision.target_price,
            stop_loss=primary_decision.stop_loss,
            quantity=primary_decision.quantity,
            timeframe=primary_decision.timeframe,
            reasoning=f"Consensus decision with {len(decisions)} agents",
            supporting_agents=supporting_agents
        )
    
    def _coordinate_dominant_agent(self, decisions: List[AgentDecision], 
                                  market_state) -> Optional[CoordinatedDecision]:
        """
        Coordinate decisions giving priority to specific agent types
        
        Args:
            decisions (list): List of agent decisions
            market_state: Current market state
            
        Returns:
            CoordinatedDecision: Coordinated decision or None
        """
        if not decisions:
            return None
        
        # Get current market regime
        regime = market_state.get_market_regime()
        
        # Get agent priorities for this regime
        priorities = self.regime_priorities.get(regime, self.regime_priorities['unknown'])
        
        # Classify decisions by agent type
        decisions_by_type = defaultdict(list)
        for decision in decisions:
            agent_type = self._get_agent_type_from_id(decision.agent_id)
            decisions_by_type[agent_type].append(decision)
        
        # Select primary decision from highest priority agent type
        primary_decision = None
        for agent_type in priorities:
            if agent_type in decisions_by_type:
                # Get highest confidence decision for this agent type
                type_decisions = decisions_by_type[agent_type]
                type_primary = max(type_decisions, key=lambda d: d.confidence.value)
                
                # Check if confidence is sufficient
                confidence_float = type_primary.confidence.value / ConfidenceLevel.VERY_HIGH.value
                if confidence_float >= self.min_confidence:
                    primary_decision = type_primary
                    break
        
        # If no suitable decision found, return None
        if primary_decision is None:
            logger.debug(f"No suitable dominant agent decision for {decisions[0].symbol}")
            return None
        
        # List other agents as supporting
        supporting_agents = [d.agent_id for d in decisions if d != primary_decision]
        
        # Create coordinated decision
        return CoordinatedDecision(
            agent_decisions=decisions,
            symbol=primary_decision.symbol,
            decision_type=primary_decision.decision_type,
            confidence=primary_decision.confidence.value / ConfidenceLevel.VERY_HIGH.value,
            primary_agent_id=primary_decision.agent_id,
            primary_agent_type=self._get_agent_type_from_id(primary_decision.agent_id),
            target_price=primary_decision.target_price,
            stop_loss=primary_decision.stop_loss,
            quantity=primary_decision.quantity,
            timeframe=primary_decision.timeframe,
            reasoning=f"Priority decision from {self._get_agent_type_from_id(primary_decision.agent_id)} agent",
            supporting_agents=supporting_agents
        )
    
    def _coordinate_market_adaptive(self, decisions: List[AgentDecision], 
                                   market_state) -> Optional[CoordinatedDecision]:
        """
        Coordinate decisions adapting to market conditions
        
        Args:
            decisions (list): List of agent decisions
            market_state: Current market state
            
        Returns:
            CoordinatedDecision: Coordinated decision or None
        """
        # This method selects different coordination strategies based on market conditions
        regime = market_state.get_market_regime()
        volatility = market_state.get_market_volatility()
        
        # Choose coordination method based on market conditions
        if regime in ['volatile', 'bear'] or (volatility is not None and volatility > 0.2):
            # In volatile or bear markets, be more conservative - use consensus
            return self._coordinate_consensus(decisions)
        elif regime in ['bull', 'recovery']:
            # In bull markets, be more aggressive - use highest confidence
            return self._coordinate_highest_confidence(decisions)
        else:
            # In other markets, use weighted vote as balanced approach
            return self._coordinate_weighted_vote(decisions, market_state)
    
    def _apply_constraints(self, decisions: List[CoordinatedDecision], 
                          market_state) -> List[CoordinatedDecision]:
        """
        Apply constraints and prioritize trade decisions
        
        Args:
            decisions (list): List of coordinated decisions
            market_state: Current market state
            
        Returns:
            list: Filtered and prioritized list of coordinated decisions
        """
        # If no decisions, return empty list
        if not decisions:
            return []
        
        # Sort decisions by confidence (descending)
        sorted_decisions = sorted(decisions, key=lambda d: d.confidence, reverse=True)
        
        # Apply maximum trades per cycle constraint
        max_trades = self.max_trades_per_cycle
        
        # Adjust max trades based on market regime
        regime = market_state.get_market_regime()
        if regime == 'bear':
            # Be more conservative in bear markets
            max_trades = max(1, int(max_trades * 0.7))
        elif regime == 'volatile':
            # Be more selective in volatile markets
            max_trades = max(1, int(max_trades * 0.8))
        elif regime in ['bull', 'recovery']:
            # Can be more active in bull or recovery markets
            max_trades = min(int(max_trades * 1.2), len(sorted_decisions))
        
        # Final list of decisions after applying constraints
        final_decisions = []
        
        # Apply constraints
        for decision in sorted_decisions:
            # Check if adding this position would exceed exposure limits
            # (This would require implementing exposure calculation based on portfolio_state)
            # For now, skip this check
            
            # Add decision to final list
            final_decisions.append(decision)
            
            # Stop if max trades reached
            if len(final_decisions) >= max_trades:
                break
        
        return final_decisions
    
    def save_state(self, filename: str = None) -> bool:
        """
        Save coordinator state to file
        
        Args:
            filename (str, optional): Filename to save to
            
        Returns:
            bool: True if successful, False otherwise
        """
        import os
        if filename is None:
            os.makedirs(self.data_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_path}/coordinator_{timestamp}.json"
        
        try:
            # Prepare state data
            decision_data = [d.to_dict() for d in self.coordinated_decisions]
            
            state = {
                'name': self.name,
                'method': self.method.value,
                'min_confidence': self.min_confidence,
                'min_consensus': self.min_consensus,
                'max_trades_per_cycle': self.max_trades_per_cycle,
                'agent_weights': self.agent_weights,
                'regime_priorities': self.regime_priorities,
                'max_exposure_per_symbol': self.max_exposure_per_symbol,
                'max_exposure_per_sector': self.max_exposure_per_sector,
                'coordinated_decisions': decision_data,
                'conflicts_resolved': self.conflicts_resolved,
                'last_coordination_time': self.last_coordination_time.isoformat() if self.last_coordination_time else None,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Coordinator state saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving coordinator state: {str(e)}")
            return False

    def _generate_sample_data(self, num_samples=5, recommendation_type="buy"):
        """
        生成样本交易推荐数据用于测试和展示
        
        Args:
            num_samples (int): 要生成的样本数量
            recommendation_type (str): 推荐类型，"buy"或"sell"
            
        Returns:
            list: 样本推荐数据列表
        """
        import random
        
        # 股票样本数据
        sample_stocks = [
            {"symbol": "000001.SZ", "name": "平安银行", "price": 12.45},
            {"symbol": "600000.SH", "name": "浦发银行", "price": 8.76},
            {"symbol": "601318.SH", "name": "中国平安", "price": 45.60},
            {"symbol": "601398.SH", "name": "工商银行", "price": 5.23},
            {"symbol": "000651.SZ", "name": "格力电器", "price": 38.52},
            {"symbol": "000858.SZ", "name": "五粮液", "price": 168.20},
            {"symbol": "600519.SH", "name": "贵州茅台", "price": 1820.50},
            {"symbol": "601988.SH", "name": "中国银行", "price": 3.82},
            {"symbol": "600036.SH", "name": "招商银行", "price": 34.95},
            {"symbol": "002415.SZ", "name": "海康威视", "price": 29.76}
        ]
        
        # 理由样本
        buy_reasons = [
            "突破20日均线，成交量放大",
            "RSI超卖反弹信号",
            "底部形态确认，有望反弹",
            "基本面改善，业绩超预期",
            "行业龙头优势明显",
            "市盈率低于行业平均水平",
            "机构持仓明显增加",
            "技术形态良好，多头排列",
            "突破重要阻力位",
            "量价配合，上涨趋势确认"
        ]
        
        sell_reasons = [
            "MACD死叉，量能萎缩",
            "突破下行通道，趋势转弱",
            "头肩顶形态确认，看空信号",
            "业绩低于预期，估值过高",
            "行业竞争加剧，优势减弱",
            "技术指标超买，短期调整可能性大",
            "大股东减持，市场信心不足",
            "突破重要支撑位，看空信号",
            "均线空头排列，趋势向下",
            "市场情绪转弱，资金流出"
        ]
        
        # 代理名称样本
        agents = ["趋势跟踪", "反转交易", "快速获利", "防御型", "板块轮动"]
        
        # 生成样本数据
        samples = []
        for _ in range(min(num_samples, len(sample_stocks))):
            stock = random.choice(sample_stocks)
            # 避免重复
            sample_stocks.remove(stock)
            
            if recommendation_type == "buy":
                sample = {
                    "symbol": stock["symbol"],
                    "name": stock["name"],
                    "price": stock["price"],
                    "reason": random.choice(buy_reasons),
                    "target_price": round(stock["price"] * (1 + random.uniform(0.05, 0.15)), 2),
                    "confidence": random.uniform(0.65, 0.95),
                    "agent": random.choice(agents)
                }
            else:  # sell
                sample = {
                    "symbol": stock["symbol"],
                    "name": stock["name"],
                    "price": stock["price"],
                    "reason": random.choice(sell_reasons),
                    "profit": f"+{random.uniform(5, 15):.1f}%" if random.random() > 0.3 else f"-{random.uniform(1, 5):.1f}%",
                    "confidence": random.uniform(0.65, 0.95),
                    "agent": random.choice(agents)
                }
            
            samples.append(sample)
            
        return samples

    def get_buy_recommendations(self, max_recommendations=5):
        """
        获取买入推荐
        
        根据当前交易意图，过滤出信心较高的买入推荐
        
        Args:
            max_recommendations (int): 最大推荐数量
            
        Returns:
            list: 买入推荐列表
        """
        buy_recommendations = []
        
        # 尝试从交易意图中提取买入推荐
        if hasattr(self, 'last_intentions') and self.last_intentions:
            # 筛选信心等级较高的买入交易意图
            buy_intents = [intent for intent in self.last_intentions 
                          if intent.decision_type == 'buy' and intent.confidence.value >= 3]
            
            # 提取买入推荐信息
            for intent in buy_intents[:max_recommendations]:
                stock_info = self.market_state.get_stock_info(intent.symbol)
                if not stock_info:
                    continue
                    
                buy_recommendations.append({
                    "symbol": intent.symbol,
                    "name": stock_info.get('name', intent.symbol),
                    "price": stock_info.get('price', 0.0),
                    "reason": intent.reasoning or "技术指标触发买入信号",
                    "target_price": intent.target_price or round(stock_info.get('price', 0.0) * 1.1, 2),
                    "confidence": intent.confidence.value / 5.0,  # 转换为0-1范围
                    "agent": intent.agent_id
                })
        
        # 如果没有足够的推荐，生成一些样本数据
        if len(buy_recommendations) < max_recommendations:
            sample_count = max_recommendations - len(buy_recommendations)
            buy_recommendations.extend(self._generate_sample_data(sample_count, "buy"))
            
        return buy_recommendations
            
    def get_sell_recommendations(self, max_recommendations=5):
        """
        获取卖出推荐
        
        根据当前交易意图，过滤出信心较高的卖出推荐
        
        Args:
            max_recommendations (int): 最大推荐数量
            
        Returns:
            list: 卖出推荐列表
        """
        sell_recommendations = []
        
        # 尝试从交易意图中提取卖出推荐
        if hasattr(self, 'last_intentions') and self.last_intentions:
            # 筛选信心等级较高的卖出交易意图
            sell_intents = [intent for intent in self.last_intentions 
                           if intent.decision_type == 'sell' and intent.confidence.value >= 3]
            
            # 提取卖出推荐信息
            for intent in sell_intents[:max_recommendations]:
                stock_info = self.market_state.get_stock_info(intent.symbol)
                if not stock_info:
                    continue
                    
                # 计算盈亏百分比（如果有仓位）
                profit = "+0.0%"
                if hasattr(self, 'portfolio_state') and self.portfolio_state:
                    position = self.portfolio_state.get_position(intent.symbol)
                    if position and position.average_cost > 0:
                        profit_pct = (stock_info.get('price', 0.0) - position.average_cost) / position.average_cost * 100
                        profit = f"+{profit_pct:.1f}%" if profit_pct >= 0 else f"{profit_pct:.1f}%"
                
                sell_recommendations.append({
                    "symbol": intent.symbol,
                    "name": stock_info.get('name', intent.symbol),
                    "price": stock_info.get('price', 0.0),
                    "reason": intent.reasoning or "技术指标触发卖出信号",
                    "profit": profit,
                    "confidence": intent.confidence.value / 5.0,  # 转换为0-1范围
                    "agent": intent.agent_id
                })
        
        # 如果没有足够的推荐，生成一些样本数据
        if len(sell_recommendations) < max_recommendations:
            sample_count = max_recommendations - len(sell_recommendations)
            sell_recommendations.extend(self._generate_sample_data(sample_count, "sell"))
            
        return sell_recommendations 