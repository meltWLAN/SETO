#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Market Hedge Strategy

This strategy focuses on providing portfolio protection during market downturns
by identifying and investing in defensive assets when market conditions deteriorate.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

from seto_versal.strategies.base import BaseStrategy
from seto_versal.common.constants import SignalType, OrderType
from seto_versal.common.models import Signal

logger = logging.getLogger(__name__)

class MarketHedgeStrategy(BaseStrategy):
    """
    Market Hedge Strategy - Provides portfolio protection during market downturns
    
    This strategy:
    1. Monitors market conditions and risk indicators
    2. Identifies potential market downturns
    3. Allocates to defensive assets (inverse ETFs, VIX products, bonds, etc.)
    4. Reduces hedge positions when market conditions improve
    """
    
    def __init__(
        self,
        risk_threshold: float = 0.6,
        vix_threshold: float = 25.0,
        max_allocation: float = 0.2,
        hedge_symbols: List[str] = None,
        lookback_period: int = 20,
        exit_threshold: float = 0.4,
        rebalance_interval: int = 5,
        profit_take: float = 0.15,
        **kwargs
    ):
        """
        Initialize Market Hedge Strategy
        
        Args:
            risk_threshold: Risk indicator threshold to trigger hedging
            vix_threshold: VIX level to trigger hedging
            max_allocation: Maximum portfolio allocation to hedges
            hedge_symbols: List of symbols to use as hedges
            lookback_period: Period (in days) to calculate market metrics
            exit_threshold: Risk threshold to exit hedge positions
            rebalance_interval: Days between hedge position rebalancing
            profit_take: Profit-taking threshold for hedge positions
        """
        super().__init__(**kwargs)
        self.risk_threshold = risk_threshold
        self.vix_threshold = vix_threshold
        self.max_allocation = max_allocation
        self.lookback_period = lookback_period
        self.exit_threshold = exit_threshold
        self.rebalance_interval = rebalance_interval
        self.profit_take = profit_take
        
        # Default hedge symbols if none provided
        self.hedge_symbols = hedge_symbols or [
            "SH",   # Inverse S&P 500
            "VXX",  # VIX Short-Term Futures
            "TLT",  # 20+ Year Treasury Bond
            "GLD",  # Gold
            "HDGE"  # Active Bear ETF
        ]
        
        self.last_rebalance_date = None
        
        logger.info(
            f"Initialized MarketHedgeStrategy: risk_threshold={risk_threshold}, "
            f"vix_threshold={vix_threshold}, max_allocation={max_allocation}"
        )
    
    def generate_signals(
        self,
        market_data: Dict[str, Dict[datetime, Dict[str, float]]],
        positions: Dict[str, Dict[str, Any]] = None,
        market_state: Dict[str, Any] = None,
        **kwargs
    ) -> List[Signal]:
        """
        Generate trading signals based on market risk assessment
        
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
                logger.warning("No market data provided to MarketHedgeStrategy")
                return signals
                
            # Get current date from market data
            current_date = self._get_latest_date(market_data)
            if not current_date:
                logger.warning("Could not determine current date")
                return signals
                
            # Check if rebalance is needed based on interval
            if self.last_rebalance_date and (current_date - self.last_rebalance_date).days < self.rebalance_interval:
                logger.debug(f"Skipping hedge rebalance - next rebalance in {self.rebalance_interval - (current_date - self.last_rebalance_date).days} days")
                return signals
                
            # Calculate market risk score
            risk_score = self._calculate_risk_score(market_data, market_state)
            logger.info(f"Current market risk score: {risk_score:.2f}")
            
            # Determine if hedging is needed
            hedge_needed = self._is_hedge_needed(risk_score, market_state)
            
            # Get current hedge allocations
            current_hedges = self._get_current_hedges(positions)
            current_allocation = sum(position.get('allocation', 0) for position in current_hedges.values())
            
            if hedge_needed:
                # Market risk is high - add or increase hedge positions
                logger.info(f"Market risk is elevated ({risk_score:.2f}) - adding hedge positions")
                
                # Calculate target allocation based on risk score
                target_allocation = min(self.max_allocation, self.max_allocation * (risk_score - self.exit_threshold) / 
                                     (self.risk_threshold - self.exit_threshold))
                
                # Determine hedge quality for each symbol
                hedge_quality = self._evaluate_hedge_quality(market_data, market_state)
                
                # Generate buy signals to reach target allocation
                if target_allocation > current_allocation:
                    buy_signals = self._generate_buy_signals(
                        market_data, 
                        hedge_quality, 
                        target_allocation - current_allocation,
                        current_hedges
                    )
                    signals.extend(buy_signals)
            else:
                # Market risk is low - reduce or exit hedge positions
                if current_hedges:
                    logger.info(f"Market risk is reduced ({risk_score:.2f}) - reducing hedge positions")
                    sell_signals = self._generate_sell_signals(
                        market_data, 
                        current_hedges, 
                        risk_score
                    )
                    signals.extend(sell_signals)
            
            # Update last rebalance date
            if signals:
                self.last_rebalance_date = current_date
                
            logger.info(f"MarketHedgeStrategy generated {len(signals)} signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error in MarketHedgeStrategy: {str(e)}", exc_info=True)
            return []
    
    def _get_latest_date(self, market_data: Dict[str, Dict[datetime, Dict[str, float]]]) -> Optional[datetime]:
        """
        Get the latest date from market data
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Latest date or None
        """
        latest_date = None
        
        # Look for SPY or any market index first
        for symbol in ["SPY", "^GSPC", "^DJI"]:
            if symbol in market_data:
                dates = sorted(market_data[symbol].keys())
                if dates:
                    latest_date = dates[-1]
                    break
        
        # If not found, use any symbol's data
        if not latest_date and market_data:
            symbol = next(iter(market_data))
            dates = sorted(market_data[symbol].keys())
            if dates:
                latest_date = dates[-1]
                
        return latest_date
    
    def _calculate_risk_score(
        self, 
        market_data: Dict[str, Dict[datetime, Dict[str, float]]],
        market_state: Dict[str, Any]
    ) -> float:
        """
        Calculate market risk score based on various indicators
        
        Args:
            market_data: Market data dictionary
            market_state: Market state information
            
        Returns:
            Risk score between 0 and 1
        """
        risk_factors = []
        
        # 1. Market trend (price vs moving averages)
        market_trend = self._calculate_market_trend(market_data)
        if market_trend is not None:
            # Negative trend increases risk score
            risk_factors.append(0.7 if market_trend < 0 else 0.3)
        
        # 2. Volatility (VIX level)
        vix_level = self._get_vix_level(market_data, market_state)
        if vix_level is not None:
            # Higher VIX increases risk score
            vix_factor = min(1.0, vix_level / self.vix_threshold)
            risk_factors.append(vix_factor)
        
        # 3. Breadth indicators (advance/decline, new highs/lows)
        market_breadth = market_state.get('market_breadth', {}).get('breadth_ratio', 0.5)
        # Lower breadth increases risk score
        risk_factors.append(1.0 - market_breadth)
        
        # 4. Correlation spike (correlation among sectors)
        correlation = market_state.get('correlation', 0.5)
        # Higher correlation increases risk score
        risk_factors.append(correlation)
        
        # Calculate combined risk score
        if risk_factors:
            return sum(risk_factors) / len(risk_factors)
        else:
            return 0.5  # Default to moderate risk if no factors available
    
    def _calculate_market_trend(self, market_data: Dict[str, Dict[datetime, Dict[str, float]]]) -> Optional[float]:
        """
        Calculate market trend based on price vs moving averages
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Trend indicator or None
        """
        for symbol in ["SPY", "^GSPC", "^DJI"]:
            if symbol not in market_data:
                continue
                
            data = market_data[symbol]
            dates = sorted(data.keys())
            
            # Skip if not enough data
            if len(dates) < self.lookback_period:
                continue
                
            # Calculate 50-day and 200-day moving averages
            if len(dates) >= 50:
                current_price = data[dates[-1]].get('close', 0)
                ma50 = sum(data[dates[-i]].get('close', 0) for i in range(1, 51)) / 50
                
                # Use 200-day MA if available, otherwise use available data
                ma_days = min(200, len(dates) - 1)
                ma200 = sum(data[dates[-i]].get('close', 0) for i in range(1, ma_days + 1)) / ma_days
                
                # Calculate trend indicator
                if ma50 > 0 and ma200 > 0:
                    # Positive values indicate bullish trend, negative values indicate bearish trend
                    return ((current_price / ma50) - 1) + ((current_price / ma200) - 1)
                    
        return None
    
    def _get_vix_level(
        self, 
        market_data: Dict[str, Dict[datetime, Dict[str, float]]],
        market_state: Dict[str, Any]
    ) -> Optional[float]:
        """
        Get current VIX level from data
        
        Args:
            market_data: Market data dictionary
            market_state: Market state information
            
        Returns:
            VIX level or None
        """
        # Try to get VIX from market data
        for symbol in ["^VIX", "VIX"]:
            if symbol in market_data:
                dates = sorted(market_data[symbol].keys())
                if dates:
                    return market_data[symbol][dates[-1]].get('close', 0)
        
        # If not in market data, try market state
        vix = market_state.get('volatility', {}).get('vix', None)
        if vix is not None:
            return vix
            
        return None
    
    def _is_hedge_needed(self, risk_score: float, market_state: Dict[str, Any]) -> bool:
        """
        Determine if hedging is needed based on risk score and market conditions
        
        Args:
            risk_score: Calculated risk score
            market_state: Market state information
            
        Returns:
            True if hedging is needed, False otherwise
        """
        # Primary criterion is risk threshold
        if risk_score >= self.risk_threshold:
            return True
            
        # Check VIX level
        vix = market_state.get('volatility', {}).get('vix', 0)
        if vix >= self.vix_threshold:
            return True
            
        # Check additional risk signals
        if (market_state.get('volatility', {}).get('high', False) and 
            market_state.get('trend', {}).get('bearish', False)):
            return True
            
        return False
    
    def _get_current_hedges(self, positions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Extract current hedge positions
        
        Args:
            positions: Current positions
            
        Returns:
            Dictionary of current hedge positions
        """
        if not positions:
            return {}
            
        hedges = {}
        for symbol, position in positions.items():
            # Check if position is from this strategy
            metadata = position.get('metadata', {})
            if metadata.get('strategy') == 'market_hedge':
                hedges[symbol] = position
                
        return hedges
    
    def _evaluate_hedge_quality(
        self, 
        market_data: Dict[str, Dict[datetime, Dict[str, float]]],
        market_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluate quality of potential hedge instruments
        
        Args:
            market_data: Market data dictionary
            market_state: Market state information
            
        Returns:
            Dictionary of hedge symbols and their quality scores
        """
        hedge_quality = {}
        
        for symbol in self.hedge_symbols:
            if symbol not in market_data:
                continue
                
            data = market_data[symbol]
            dates = sorted(data.keys())
            
            # Skip if not enough data
            if len(dates) < self.lookback_period:
                continue
                
            # Calculate hedge quality metrics
            quality_score = 0.5  # Default moderate quality
            
            # 1. Negative correlation with market
            market_correlation = market_state.get('correlations', {}).get(symbol, 0)
            if market_correlation < 0:
                # Negative correlation is good for hedges
                quality_score += 0.2 * min(1.0, abs(market_correlation))
            
            # 2. Recent performance during market stress
            # Positive performance during stress is good
            stress_performance = market_state.get('stress_performance', {}).get(symbol, 0)
            if stress_performance > 0:
                quality_score += 0.2 * min(1.0, stress_performance / 0.05)
            
            # 3. Liquidity (higher is better)
            avg_volume = sum(data[d].get('volume', 0) for d in dates[-10:]) / 10
            if avg_volume > 100000:
                quality_score += 0.1
            
            # 4. Volatility (lower is better for some hedges, higher for others)
            # Default preference for lower volatility
            if symbol in ["TLT", "GLD"]:
                volatility = self._calculate_volatility(data, dates)
                quality_score += 0.1 * (1.0 - min(1.0, volatility / 0.02))
                
            hedge_quality[symbol] = quality_score
            
        return hedge_quality
    
    def _calculate_volatility(
        self, 
        data: Dict[datetime, Dict[str, float]], 
        dates: List[datetime]
    ) -> float:
        """
        Calculate price volatility
        
        Args:
            data: Price data
            dates: Date list
            
        Returns:
            Volatility measure
        """
        # Calculate returns
        returns = []
        for i in range(1, min(21, len(dates))):
            prev_close = data[dates[-i-1]].get('close', 0)
            curr_close = data[dates[-i]].get('close', 0)
            if prev_close > 0:
                returns.append((curr_close - prev_close) / prev_close)
                
        # Calculate standard deviation
        if returns:
            return np.std(returns)
        else:
            return 0.02  # Default moderate volatility
    
    def _generate_buy_signals(
        self,
        market_data: Dict[str, Dict[datetime, Dict[str, float]]],
        hedge_quality: Dict[str, float],
        target_allocation: float,
        current_hedges: Dict[str, Dict[str, Any]]
    ) -> List[Signal]:
        """
        Generate buy signals for hedge positions
        
        Args:
            market_data: Market data dictionary
            hedge_quality: Dictionary of hedge quality scores
            target_allocation: Target allocation to achieve
            current_hedges: Current hedge positions
            
        Returns:
            List of buy signals
        """
        buy_signals = []
        
        # Sort hedge instruments by quality
        sorted_hedges = sorted(
            hedge_quality.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Determine number of instruments to use
        num_instruments = min(3, len(sorted_hedges))
        allocation_per_instrument = target_allocation / num_instruments
        
        # Generate buy signals
        for symbol, quality in sorted_hedges[:num_instruments]:
            # Skip if not in market data
            if symbol not in market_data:
                continue
                
            data = market_data[symbol]
            dates = sorted(data.keys())
            
            if not dates:
                continue
                
            current_price = data[dates[-1]].get('close', 0)
            
            if current_price <= 0:
                continue
                
            # Adjust allocation if already holding
            current_allocation = 0
            if symbol in current_hedges:
                current_allocation = current_hedges[symbol].get('allocation', 0)
                
            if allocation_per_instrument <= current_allocation:
                continue
                
            # Calculate additional allocation needed
            additional_allocation = allocation_per_instrument - current_allocation
            
            # Create buy signal
            confidence = min(0.9, 0.6 + quality * 0.3)
            
            # Determine reason based on market conditions
            reason = "Market hedge - elevated risk"
            if 'VIX' in symbol or 'VXX' in symbol:
                reason = "Market hedge - rising volatility"
            elif 'TLT' in symbol or 'GLD' in symbol:
                reason = "Market hedge - safe haven allocation"
            elif 'SH' in symbol or 'HDGE' in symbol:
                reason = "Market hedge - short position"
                
            buy_signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                quantity=1,  # Quantity will be determined by position sizing
                order_type=OrderType.MARKET,
                confidence=confidence,
                target_price=current_price * (1 + self.profit_take),
                reason=reason,
                metadata={
                    "strategy": "market_hedge",
                    "hedge_type": self._determine_hedge_type(symbol),
                    "allocation": additional_allocation,
                    "quality_score": quality
                }
            ))
                
        return buy_signals
    
    def _generate_sell_signals(
        self,
        market_data: Dict[str, Dict[datetime, Dict[str, float]]],
        current_hedges: Dict[str, Dict[str, Any]],
        risk_score: float
    ) -> List[Signal]:
        """
        Generate sell signals for hedge positions
        
        Args:
            market_data: Market data dictionary
            current_hedges: Current hedge positions
            risk_score: Current risk score
            
        Returns:
            List of sell signals
        """
        sell_signals = []
        
        # Calculate exit percentage based on risk score
        exit_pct = 1.0 - (risk_score / self.exit_threshold) if risk_score < self.exit_threshold else 0.0
        
        # Generate sell signals for each hedge position
        for symbol, position in current_hedges.items():
            # Skip if not in market data
            if symbol not in market_data:
                continue
                
            data = market_data[symbol]
            dates = sorted(data.keys())
            
            if not dates:
                continue
                
            current_price = data[dates[-1]].get('close', 0)
            entry_price = position.get('entry_price', 0)
            
            if current_price <= 0 or entry_price <= 0:
                continue
                
            # Determine if full exit or partial reduction
            full_exit = risk_score < self.exit_threshold * 0.8
            
            # Check for profit target
            profit_pct = (current_price - entry_price) / entry_price
            profit_target_hit = profit_pct >= self.profit_take
            
            # Determine sell quantity
            quantity = position.get('quantity', 0)
            if not full_exit and not profit_target_hit:
                quantity = int(quantity * exit_pct)
                
            if quantity <= 0:
                continue
                
            # Determine sell reason
            reason = "Reducing hedge - market risk decreased"
            if full_exit:
                reason = "Exiting hedge - market risk subsided"
            elif profit_target_hit:
                reason = f"Taking profit on hedge (+{profit_pct:.1%})"
                
            sell_signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=current_price,
                quantity=quantity,
                order_type=OrderType.MARKET,
                confidence=0.7,
                reason=reason,
                metadata={
                    "strategy": "market_hedge",
                    "hedge_type": position.get('metadata', {}).get('hedge_type', ''),
                    "profit_pct": profit_pct
                }
            ))
                
        return sell_signals
    
    def _determine_hedge_type(self, symbol: str) -> str:
        """
        Determine the type of hedge based on symbol
        
        Args:
            symbol: Security symbol
            
        Returns:
            Hedge type description
        """
        if symbol in ['SH', 'SDS', 'HDGE']:
            return 'short_equity'
        elif symbol in ['VXX', 'UVXY', '^VIX']:
            return 'volatility'
        elif symbol in ['TLT', 'IEF']:
            return 'treasury'
        elif symbol in ['GLD', 'IAU']:
            return 'gold'
        else:
            return 'other' 