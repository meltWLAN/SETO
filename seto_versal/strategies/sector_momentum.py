#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sector Momentum Strategy

This strategy focuses on identifying strong momentum in specific market sectors
and investing in the top performing stocks within those sectors.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

from seto_versal.strategies.base import BaseStrategy
from seto_versal.common.constants import SignalType, OrderType
from seto_versal.common.models import Signal

logger = logging.getLogger(__name__)

class SectorMomentumStrategy(BaseStrategy):
    """
    Sector Momentum Strategy - Identifies strong performing sectors and invests in top stocks
    
    This strategy:
    1. Identifies sectors with strong positive momentum 
    2. Selects top performing stocks within those sectors
    3. Generates buy signals for these stocks
    4. Exits positions when sector momentum weakens or stock performance deteriorates
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        sector_count: int = 3,
        stocks_per_sector: int = 3,
        min_sector_momentum: float = 0.05,
        min_relative_strength: float = 1.2,
        profit_take: float = 0.15,
        stop_loss: float = 0.07,
        **kwargs
    ):
        """
        Initialize Sector Momentum Strategy
        
        Args:
            lookback_period: Period (in days) to calculate momentum
            sector_count: Number of top sectors to select
            stocks_per_sector: Number of top stocks to select per sector
            min_sector_momentum: Minimum momentum required for a sector to qualify
            min_relative_strength: Minimum relative strength vs market for a stock to qualify
            profit_take: Profit-taking threshold
            stop_loss: Stop-loss threshold
        """
        super().__init__(**kwargs)
        self.lookback_period = lookback_period
        self.sector_count = sector_count
        self.stocks_per_sector = stocks_per_sector
        self.min_sector_momentum = min_sector_momentum
        self.min_relative_strength = min_relative_strength
        self.profit_take = profit_take
        self.stop_loss = stop_loss
        
        logger.info(
            f"Initialized SectorMomentumStrategy: lookback={lookback_period}, "
            f"sectors={sector_count}, stocks_per_sector={stocks_per_sector}"
        )
    
    def generate_signals(
        self, 
        market_data: Dict[str, Dict[datetime, Dict[str, float]]], 
        positions: Dict[str, Dict[str, Any]] = None,
        market_state: Dict[str, Any] = None,
        **kwargs
    ) -> List[Signal]:
        """
        Generate trading signals based on sector momentum
        
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
                logger.warning("No market data provided to SectorMomentumStrategy")
                return signals
                
            # Extract sector information from market_state
            if not market_state or 'sector_performance' not in market_state:
                logger.warning("No sector performance data in market_state")
                return signals
                
            sector_performance = market_state.get('sector_performance', {})
            sector_stocks = market_state.get('sector_stocks', {})
            
            # 1. Identify top performing sectors
            top_sectors = self._get_top_sectors(sector_performance)
            if not top_sectors:
                logger.info("No sectors meeting momentum criteria found")
                return signals
                
            logger.debug(f"Top sectors identified: {', '.join(top_sectors)}")
            
            # 2. For each top sector, identify top performing stocks
            for sector in top_sectors:
                # Skip if no stocks for this sector
                if sector not in sector_stocks:
                    continue
                    
                # Get stocks in this sector
                stocks_in_sector = sector_stocks[sector]
                if not stocks_in_sector:
                    continue
                    
                # Calculate stock performance
                stock_performance = self._calculate_stock_performance(
                    stocks_in_sector, market_data
                )
                
                # Select top performing stocks
                top_stocks = self._get_top_stocks(stock_performance)
                
                # Generate signals for top stocks
                for symbol in top_stocks:
                    if symbol not in market_data:
                        continue
                        
                    signal = self._create_buy_signal(symbol, market_data, sector)
                    if signal:
                        signals.append(signal)
            
            # 3. Process exit signals for current positions
            if positions:
                exit_signals = self._process_exits(
                    positions, market_data, sector_performance
                )
                signals.extend(exit_signals)
                
            logger.info(f"SectorMomentumStrategy generated {len(signals)} signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error in SectorMomentumStrategy: {str(e)}", exc_info=True)
            return []
    
    def _get_top_sectors(self, sector_performance: Dict[str, float]) -> List[str]:
        """
        Identify top performing sectors based on momentum
        
        Args:
            sector_performance: Dictionary of sector performance metrics
            
        Returns:
            List of top sector names
        """
        # Filter sectors by minimum momentum
        qualified_sectors = {
            sector: perf for sector, perf in sector_performance.items()
            if perf >= self.min_sector_momentum
        }
        
        # Sort by performance (descending)
        sorted_sectors = sorted(
            qualified_sectors.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Take top N sectors
        top_sectors = [
            sector for sector, _ in sorted_sectors[:self.sector_count]
        ]
        
        return top_sectors
    
    def _calculate_stock_performance(
        self,
        symbols: List[str],
        market_data: Dict[str, Dict[datetime, Dict[str, float]]]
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for a list of stocks
        
        Args:
            symbols: List of stock symbols
            market_data: Market data dictionary
            
        Returns:
            Dictionary of stock performance metrics
        """
        stock_performance = {}
        
        for symbol in symbols:
            if symbol not in market_data:
                continue
                
            data = market_data[symbol]
            dates = sorted(data.keys())
            
            # Skip if not enough data
            if len(dates) < self.lookback_period:
                continue
                
            # Calculate performance over lookback period
            current_price = data[dates[-1]].get('close', 0)
            past_price = data[dates[-self.lookback_period]].get('close', 0)
            
            if past_price <= 0:
                continue
                
            # Calculate performance as percentage change
            performance = (current_price - past_price) / past_price
            stock_performance[symbol] = performance
            
        return stock_performance
    
    def _get_top_stocks(self, stock_performance: Dict[str, float]) -> List[str]:
        """
        Select top performing stocks from performance dictionary
        
        Args:
            stock_performance: Dictionary of stock performance metrics
            
        Returns:
            List of top stock symbols
        """
        # Filter stocks by minimum relative strength
        market_avg = np.mean(list(stock_performance.values())) if stock_performance else 0
        
        qualified_stocks = {
            symbol: perf for symbol, perf in stock_performance.items()
            if perf >= market_avg * self.min_relative_strength
        }
        
        # Sort by performance (descending)
        sorted_stocks = sorted(
            qualified_stocks.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take top N stocks
        top_stocks = [
            symbol for symbol, _ in sorted_stocks[:self.stocks_per_sector]
        ]
        
        return top_stocks
    
    def _create_buy_signal(
        self, 
        symbol: str, 
        market_data: Dict[str, Dict[datetime, Dict[str, float]]],
        sector: str
    ) -> Optional[Signal]:
        """
        Create a buy signal for a stock
        
        Args:
            symbol: Stock symbol
            market_data: Market data dictionary
            sector: Sector name
            
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
            
            # Calculate confidence based on sector momentum and stock performance
            confidence = min(0.8, 0.5 + (current_data.get('volume', 0) / 
                             (sum(data[d].get('volume', 0) for d in dates[-5:]) / 5)))
            
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                quantity=1,  # Quantity will be determined by position sizing
                order_type=OrderType.MARKET,
                confidence=confidence,
                target_price=target_price,
                stop_loss_price=stop_loss_price,
                reason=f"Strong sector momentum in {sector}",
                metadata={
                    "strategy": "sector_momentum",
                    "sector": sector
                }
            )
        except Exception as e:
            logger.error(f"Error creating buy signal for {symbol}: {str(e)}")
            return None
    
    def _process_exits(
        self,
        positions: Dict[str, Dict[str, Any]],
        market_data: Dict[str, Dict[datetime, Dict[str, float]]],
        sector_performance: Dict[str, float]
    ) -> List[Signal]:
        """
        Process exit signals for current positions
        
        Args:
            positions: Current positions
            market_data: Market data dictionary
            sector_performance: Sector performance metrics
            
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
            sector = position.get('metadata', {}).get('sector', '')
            
            # Skip if missing data
            if entry_price <= 0 or not sector:
                continue
                
            data = market_data[symbol]
            dates = sorted(data.keys())
            
            if not dates:
                continue
                
            current_price = data[dates[-1]].get('close', 0)
            
            # Exit conditions:
            # 1. Sector momentum has weakened
            # 2. Stock has hit target profit
            # 3. Stock has hit stop loss
            
            # Check sector momentum
            sector_momentum = sector_performance.get(sector, 0)
            sell_reason = None
            
            if sector_momentum < self.min_sector_momentum / 2:
                sell_reason = f"Weakening momentum in {sector} sector"
            elif current_price >= entry_price * (1 + self.profit_take):
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
                        "strategy": "sector_momentum",
                        "sector": sector
                    }
                ))
                
        return exit_signals 