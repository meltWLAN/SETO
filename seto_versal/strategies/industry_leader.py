#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Industry Leader Strategy

This strategy focuses on identifying and investing in the leading companies
within specific industries, based on market share, growth, and relative performance.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

from seto_versal.strategies.base import BaseStrategy
from seto_versal.common.constants import SignalType, OrderType
from seto_versal.common.models import Signal

logger = logging.getLogger(__name__)

class IndustryLeaderStrategy(BaseStrategy):
    """
    Industry Leader Strategy - Identifies and invests in industry-leading companies
    
    This strategy:
    1. Identifies industries with positive growth trends
    2. Locates the leading companies in these industries
    3. Generates buy signals for industry leaders showing positive technicals
    4. Exits positions when industry trends weaken or company fundamentals deteriorate
    """
    
    def __init__(
        self,
        market_share_threshold: float = 0.2,
        min_relative_performance: float = 0.1,
        min_growth_rate: float = 0.08,
        max_pe_ratio: float = 30.0,
        lookback_period: int = 60,
        profit_take: float = 0.2,
        stop_loss: float = 0.1,
        **kwargs
    ):
        """
        Initialize Industry Leader Strategy
        
        Args:
            market_share_threshold: Minimum market share to qualify as a leader
            min_relative_performance: Minimum outperformance vs industry average
            min_growth_rate: Minimum annual growth rate
            max_pe_ratio: Maximum price-to-earnings ratio
            lookback_period: Period (in days) to calculate performance metrics
            profit_take: Profit-taking threshold
            stop_loss: Stop-loss threshold
        """
        super().__init__(**kwargs)
        self.market_share_threshold = market_share_threshold
        self.min_relative_performance = min_relative_performance
        self.min_growth_rate = min_growth_rate
        self.max_pe_ratio = max_pe_ratio
        self.lookback_period = lookback_period
        self.profit_take = profit_take
        self.stop_loss = stop_loss
        
        logger.info(
            f"Initialized IndustryLeaderStrategy: market_share={market_share_threshold}, "
            f"min_growth={min_growth_rate}, lookback={lookback_period}"
        )
    
    def generate_signals(
        self,
        market_data: Dict[str, Dict[datetime, Dict[str, float]]],
        positions: Dict[str, Dict[str, Any]] = None,
        market_state: Dict[str, Any] = None,
        **kwargs
    ) -> List[Signal]:
        """
        Generate trading signals based on industry leaders analysis
        
        Args:
            market_data: Dictionary of market data by symbol
            positions: Current positions
            market_state: Current market state information
            
        Returns:
            List of trading signals
        """
        signals = []
        
        try:
            # Validate we have enough data
            if not market_data:
                logger.warning("No market data provided to IndustryLeaderStrategy")
                return signals
                
            # Extract industry information from market_state
            if not market_state or 'industry_data' not in market_state:
                logger.warning("No industry data in market_state")
                return signals
                
            industry_data = market_state.get('industry_data', {})
            fundamental_data = market_state.get('fundamental_data', {})
            
            # 1. Identify growing industries
            growing_industries = self._identify_growing_industries(industry_data)
            if not growing_industries:
                logger.info("No growing industries identified")
                return signals
                
            logger.debug(f"Growing industries identified: {', '.join(growing_industries)}")
            
            # 2. For each growing industry, identify leaders
            for industry in growing_industries:
                # Get companies in this industry
                companies = industry_data.get(industry, {}).get('companies', [])
                if not companies:
                    continue
                    
                # Identify leaders in this industry
                leaders = self._identify_industry_leaders(
                    companies, 
                    industry_data.get(industry, {}),
                    fundamental_data
                )
                
                # 3. Generate signals for leaders with positive technicals
                for symbol in leaders:
                    # Skip if not in market data
                    if symbol not in market_data:
                        continue
                        
                    # Check technical indicators
                    if not self._has_positive_technicals(symbol, market_data):
                        continue
                        
                    # Create buy signal
                    signal = self._create_buy_signal(
                        symbol, 
                        market_data, 
                        industry, 
                        fundamental_data.get(symbol, {})
                    )
                    if signal:
                        signals.append(signal)
            
            # 4. Process exit signals for current positions
            if positions:
                exit_signals = self._process_exits(
                    positions, 
                    market_data, 
                    industry_data,
                    fundamental_data
                )
                signals.extend(exit_signals)
                
            logger.info(f"IndustryLeaderStrategy generated {len(signals)} signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error in IndustryLeaderStrategy: {str(e)}", exc_info=True)
            return []
    
    def _identify_growing_industries(self, industry_data: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Identify industries with positive growth trends
        
        Args:
            industry_data: Dictionary of industry metrics
            
        Returns:
            List of growing industry names
        """
        growing_industries = []
        
        for industry, data in industry_data.items():
            # Get industry growth rate
            growth_rate = data.get('growth_rate', 0)
            
            # Check if growth rate meets minimum threshold
            if growth_rate >= self.min_growth_rate:
                growing_industries.append(industry)
                
        return growing_industries
    
    def _identify_industry_leaders(
        self,
        companies: List[str],
        industry_data: Dict[str, Any],
        fundamental_data: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Identify leading companies in an industry
        
        Args:
            companies: List of companies in the industry
            industry_data: Industry-specific metrics
            fundamental_data: Fundamental data for all companies
            
        Returns:
            List of industry leader symbols
        """
        leaders = []
        industry_avg_performance = industry_data.get('avg_performance', 0)
        
        for symbol in companies:
            # Skip if no fundamental data
            if symbol not in fundamental_data:
                continue
                
            company_data = fundamental_data[symbol]
            
            # Check market share
            market_share = company_data.get('market_share', 0)
            if market_share < self.market_share_threshold:
                continue
                
            # Check relative performance vs industry
            performance = company_data.get('performance', 0)
            if performance < industry_avg_performance * (1 + self.min_relative_performance):
                continue
                
            # Check PE ratio if available
            pe_ratio = company_data.get('pe_ratio', 0)
            if pe_ratio > self.max_pe_ratio and pe_ratio > 0:
                continue
                
            # This company qualifies as a leader
            leaders.append(symbol)
            
        return leaders
    
    def _has_positive_technicals(
        self,
        symbol: str,
        market_data: Dict[str, Dict[datetime, Dict[str, float]]]
    ) -> bool:
        """
        Check if a stock has positive technical indicators
        
        Args:
            symbol: Stock symbol
            market_data: Market data dictionary
            
        Returns:
            True if technicals are positive, False otherwise
        """
        data = market_data[symbol]
        dates = sorted(data.keys())
        
        # Skip if not enough data
        if len(dates) < self.lookback_period:
            return False
            
        # Calculate short-term trend (20-day)
        current_price = data[dates[-1]].get('close', 0)
        price_20_days_ago = data[dates[-20]].get('close', 0) if len(dates) >= 20 else 0
        
        if price_20_days_ago <= 0:
            return False
            
        short_term_trend = (current_price - price_20_days_ago) / price_20_days_ago
        
        # Check if price is above 50-day moving average
        ma_50 = sum(data[dates[-i]].get('close', 0) for i in range(1, min(51, len(dates)))) / min(50, len(dates) - 1)
        
        # Calculate recent volume trend
        avg_volume_recent = sum(data[dates[-i]].get('volume', 0) for i in range(1, 11)) / 10
        avg_volume_prior = sum(data[dates[-i]].get('volume', 0) for i in range(11, 31)) / 20
        
        volume_trend = avg_volume_recent / avg_volume_prior if avg_volume_prior > 0 else 0
        
        # Define technical criteria
        criteria = [
            short_term_trend > 0,  # Positive short-term trend
            current_price > ma_50,  # Price above 50-day MA
            volume_trend >= 0.9,  # Volume not significantly decreasing
        ]
        
        # Return True if most criteria are met
        return sum(criteria) >= 2
    
    def _create_buy_signal(
        self,
        symbol: str,
        market_data: Dict[str, Dict[datetime, Dict[str, float]]],
        industry: str,
        fundamental_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Create a buy signal for an industry leader
        
        Args:
            symbol: Stock symbol
            market_data: Market data dictionary
            industry: Industry name
            fundamental_data: Fundamental data for the company
            
        Returns:
            Signal object or None
        """
        try:
            data = market_data[symbol]
            dates = sorted(data.keys())
            
            if not dates:
                return None
                
            current_data = data[dates[-1]]
            current_price = current_data.get('close', 0)
            
            if current_price <= 0:
                return None
            
            # Calculate target and stop prices
            target_price = current_price * (1 + self.profit_take)
            stop_loss_price = current_price * (1 - self.stop_loss)
            
            # Calculate confidence based on leadership metrics
            market_share = fundamental_data.get('market_share', 0)
            growth_rate = fundamental_data.get('growth_rate', 0)
            
            confidence = min(0.9, 0.5 + (market_share / self.market_share_threshold) * 0.2 + 
                           (growth_rate / self.min_growth_rate) * 0.2)
            
            # Create reason text
            reason = f"Industry leader in {industry}"
            if growth_rate > 0:
                reason += f" with {growth_rate:.1%} growth rate"
            if market_share > 0:
                reason += f" and {market_share:.1%} market share"
            
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                quantity=1,  # Quantity will be determined by position sizing
                order_type=OrderType.MARKET,
                confidence=confidence,
                target_price=target_price,
                stop_loss_price=stop_loss_price,
                reason=reason,
                metadata={
                    "strategy": "industry_leader",
                    "industry": industry,
                    "market_share": market_share,
                    "growth_rate": growth_rate
                }
            )
        except Exception as e:
            logger.error(f"Error creating buy signal for {symbol}: {str(e)}")
            return None
    
    def _process_exits(
        self,
        positions: Dict[str, Dict[str, Any]],
        market_data: Dict[str, Dict[datetime, Dict[str, float]]],
        industry_data: Dict[str, Dict[str, Any]],
        fundamental_data: Dict[str, Dict[str, Any]]
    ) -> List[Signal]:
        """
        Process exit signals for current positions
        
        Args:
            positions: Current positions
            market_data: Market data dictionary
            industry_data: Industry metrics
            fundamental_data: Fundamental data for companies
            
        Returns:
            List of exit signals
        """
        exit_signals = []
        
        for symbol, position in positions.items():
            # Skip if not in market data
            if symbol not in market_data:
                continue
                
            # Extract position data
            entry_price = position.get('entry_price', 0)
            industry = position.get('metadata', {}).get('industry', '')
            
            # Skip if missing data
            if entry_price <= 0 or not industry:
                continue
                
            data = market_data[symbol]
            dates = sorted(data.keys())
            
            if not dates:
                continue
                
            current_price = data[dates[-1]].get('close', 0)
            
            # Exit conditions:
            # 1. Industry trend has turned negative
            # 2. Company has lost market leadership
            # 3. Stock has hit target profit
            # 4. Stock has hit stop loss
            
            sell_reason = None
            
            # Check industry trend
            if industry in industry_data:
                industry_growth = industry_data[industry].get('growth_rate', 0)
                if industry_growth < 0:
                    sell_reason = f"Negative growth trend in {industry} industry"
            
            # Check leadership status
            if symbol in fundamental_data:
                market_share = fundamental_data[symbol].get('market_share', 0)
                if market_share < self.market_share_threshold * 0.8:
                    sell_reason = "Declining market leadership position"
            
            # Check profit/loss targets
            if current_price >= entry_price * (1 + self.profit_take):
                sell_reason = f"Target profit reached (+{self.profit_take:.1%})"
            elif current_price <= entry_price * (1 - self.stop_loss):
                sell_reason = f"Stop loss triggered (-{self.stop_loss:.1%})"
                
            if sell_reason:
                exit_signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    quantity=position.get('quantity', 0),
                    order_type=OrderType.MARKET,
                    confidence=0.7,
                    reason=sell_reason,
                    metadata={
                        "strategy": "industry_leader",
                        "industry": industry
                    }
                ))
                
        return exit_signals 