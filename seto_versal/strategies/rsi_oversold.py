#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RSI Oversold Strategy

This strategy identifies oversold conditions using the Relative Strength Index (RSI)
and generates buy signals when a security becomes oversold and starts to recover.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

from seto_versal.strategies.base import BaseStrategy
from seto_versal.common.constants import SignalType, OrderType
from seto_versal.common.models import Signal

logger = logging.getLogger(__name__)

class RsiOversoldStrategy(BaseStrategy):
    """
    RSI Oversold Strategy - Identifies oversold conditions using RSI
    
    This strategy:
    1. Calculates RSI for securities
    2. Identifies when securities become oversold (RSI below threshold)
    3. Generates buy signals when RSI rises from oversold levels
    4. Provides exit signals when RSI reaches overbought levels
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30.0,
        lookback_period: int = 5,
        exit_threshold: float = 70.0,
        min_price: float = 5.0,
        volume_increase: float = 1.2,
        prior_downtrend: bool = True,
        downtrend_days: int = 5,
        **kwargs
    ):
        """
        Initialize RSI Oversold Strategy
        
        Args:
            rsi_period: Period for RSI calculation
            oversold_threshold: RSI level considered oversold
            lookback_period: Days to look back for oversold conditions
            exit_threshold: RSI level for exit (overbought)
            min_price: Minimum price for consideration
            volume_increase: Required volume increase factor
            prior_downtrend: Whether a prior downtrend is required
            downtrend_days: Days to check for prior downtrend
        """
        super().__init__(**kwargs)
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.lookback_period = lookback_period
        self.exit_threshold = exit_threshold
        self.min_price = min_price
        self.volume_increase = volume_increase
        self.prior_downtrend = prior_downtrend
        self.downtrend_days = downtrend_days
        
        logger.info(
            f"Initialized RsiOversoldStrategy: period={rsi_period}, "
            f"oversold={oversold_threshold}, exit={exit_threshold}"
        )
    
    def generate_signals(
        self, 
        market_data: Dict[str, Dict[datetime, Dict[str, float]]], 
        positions: Dict[str, Dict[str, Any]] = None,
        market_state: Dict[str, Any] = None,
        **kwargs
    ) -> List[Signal]:
        """
        Generate trading signals based on RSI oversold conditions
        
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
                logger.warning("No market data provided to RsiOversoldStrategy")
                return signals
            
            # Process each symbol
            for symbol, data in market_data.items():
                # Skip if not enough data
                if len(data) < self.rsi_period + self.lookback_period:
                    continue
                
                # Check if we already have a position
                has_position = False
                if positions:
                    has_position = symbol in positions
                
                # Skip if we already have a position
                if has_position:
                    # Process exit signal instead
                    exit_signal = self._check_exit_signal(symbol, data, positions[symbol])
                    if exit_signal:
                        signals.append(exit_signal)
                    continue
                
                # Calculate RSI
                dates = sorted(data.keys())
                
                # Get current price
                current_date = dates[-1]
                current_price = data[current_date].get('close', 0)
                
                # Skip if price too low
                if current_price < self.min_price:
                    continue
                
                # Calculate RSI for the lookback period
                rsi_values = self._calculate_rsi(data, dates)
                
                # Check for oversold condition and recovery
                if self._is_oversold_recovery(rsi_values):
                    # Check additional conditions
                    if self._check_additional_conditions(data, dates):
                        # Create buy signal
                        signal = self._create_buy_signal(symbol, data, dates, rsi_values)
                        if signal:
                            signals.append(signal)
            
            logger.info(f"RsiOversoldStrategy generated {len(signals)} signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error in RsiOversoldStrategy: {str(e)}", exc_info=True)
            return []
    
    def _calculate_rsi(self, data: Dict[datetime, Dict[str, float]], dates: List[datetime]) -> List[float]:
        """
        Calculate RSI values
        
        Args:
            data: Price data dictionary
            dates: Sorted list of dates
            
        Returns:
            List of RSI values
        """
        # Calculate price changes
        changes = []
        for i in range(1, len(dates)):
            prev_close = data[dates[i-1]].get('close', 0)
            curr_close = data[dates[i]].get('close', 0)
            if prev_close > 0:
                changes.append(curr_close - prev_close)
        
        # Not enough data for RSI calculation
        if len(changes) < self.rsi_period:
            return []
        
        # Calculate RSI values
        rsi_values = []
        for i in range(len(changes) - self.rsi_period + 1):
            period_changes = changes[i:i+self.rsi_period]
            gains = [max(0, change) for change in period_changes]
            losses = [abs(min(0, change)) for change in period_changes]
            
            avg_gain = sum(gains) / self.rsi_period
            avg_loss = sum(losses) / self.rsi_period
            
            if avg_loss == 0:
                rsi = 100  # No losses means RSI = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    
    def _is_oversold_recovery(self, rsi_values: List[float]) -> bool:
        """
        Check if there's an oversold recovery pattern
        
        Args:
            rsi_values: List of RSI values
            
        Returns:
            True if oversold recovery is detected, False otherwise
        """
        if len(rsi_values) < 3:
            return False
        
        # Check if RSI was below oversold threshold and is now rising
        recent_values = rsi_values[-3:]
        
        # Was oversold in last 3 readings
        was_oversold = any(rsi <= self.oversold_threshold for rsi in recent_values)
        
        # Is trending upward now
        is_rising = recent_values[-1] > recent_values[-2]
        
        return was_oversold and is_rising
    
    def _check_additional_conditions(self, data: Dict[datetime, Dict[str, float]], dates: List[datetime]) -> bool:
        """
        Check additional confirmation conditions
        
        Args:
            data: Price data dictionary
            dates: Sorted list of dates
            
        Returns:
            True if additional conditions are met, False otherwise
        """
        # Check volume increase
        if self.volume_increase > 1.0:
            recent_volume = data[dates[-1]].get('volume', 0)
            avg_volume = sum(data[dates[-i]].get('volume', 0) for i in range(2, 6)) / 4
            
            if avg_volume > 0 and recent_volume < avg_volume * self.volume_increase:
                return False
        
        # Check for prior downtrend if required
        if self.prior_downtrend:
            if len(dates) < self.downtrend_days + 1:
                return False
                
            closes = [data[dates[-i]].get('close', 0) for i in range(1, self.downtrend_days + 2)]
            closes.reverse()  # Now in chronological order
            
            # Check if price was generally declining
            downtrend_count = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
            if downtrend_count < self.downtrend_days * 0.6:  # At least 60% of days showed decline
                return False
        
        return True
    
    def _create_buy_signal(
        self, 
        symbol: str, 
        data: Dict[datetime, Dict[str, float]], 
        dates: List[datetime],
        rsi_values: List[float]
    ) -> Optional[Signal]:
        """
        Create buy signal for oversold condition
        
        Args:
            symbol: Stock symbol
            data: Price data dictionary
            dates: Sorted list of dates
            rsi_values: RSI values
            
        Returns:
            Signal object or None
        """
        try:
            current_date = dates[-1]
            current_price = data[current_date].get('close', 0)
            current_rsi = rsi_values[-1]
            
            # Calculate target and stop prices
            # Target based on historical resistance or 10% gain
            recent_high = max(data[dates[-i]].get('high', 0) for i in range(1, min(21, len(dates))))
            target_price = max(current_price * 1.1, recent_high)
            
            # Stop loss based on recent low
            recent_low = min(data[dates[-i]].get('low', 0) for i in range(1, min(10, len(dates))))
            stop_loss_price = max(current_price * 0.92, recent_low * 0.98)
            
            # Calculate confidence level based on RSI depth and recovery
            lowest_rsi = min(rsi_values[-3:])
            rsi_depth = self.oversold_threshold - lowest_rsi if lowest_rsi < self.oversold_threshold else 0
            rsi_recovery = current_rsi - lowest_rsi
            
            # Normalize to 0-1 range
            normalized_depth = min(1.0, rsi_depth / 10)  # 10 points below oversold is max
            normalized_recovery = min(1.0, rsi_recovery / 15)  # 15 points recovery is max
            
            confidence = 0.5 + normalized_depth * 0.25 + normalized_recovery * 0.25
            
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                quantity=1,  # Quantity will be determined by position sizing
                order_type=OrderType.MARKET,
                confidence=confidence,
                target_price=target_price,
                stop_loss_price=stop_loss_price,
                reason=f"RSI oversold recovery (RSI: {current_rsi:.1f}, from {lowest_rsi:.1f})",
                metadata={
                    "strategy": "rsi_oversold",
                    "rsi_current": current_rsi,
                    "rsi_lowest": lowest_rsi,
                    "rsi_depth": rsi_depth,
                    "rsi_recovery": rsi_recovery
                }
            )
        except Exception as e:
            logger.error(f"Error creating buy signal for {symbol}: {str(e)}")
            return None
    
    def _check_exit_signal(
        self, 
        symbol: str, 
        data: Dict[datetime, Dict[str, float]], 
        position: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Check if we should exit an existing position
        
        Args:
            symbol: Stock symbol
            data: Price data dictionary
            position: Current position data
            
        Returns:
            Signal object or None
        """
        try:
            dates = sorted(data.keys())
            if len(dates) < self.rsi_period:
                return None
                
            current_date = dates[-1]
            current_price = data[current_date].get('close', 0)
            
            # Calculate RSI
            rsi_values = self._calculate_rsi(data, dates)
            if not rsi_values:
                return None
                
            current_rsi = rsi_values[-1]
            
            # Check if RSI is now overbought
            if current_rsi > self.exit_threshold:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    quantity=position.get('quantity', 0),
                    order_type=OrderType.MARKET,
                    confidence=0.7,
                    reason=f"RSI overbought (RSI: {current_rsi:.1f})",
                    metadata={
                        "strategy": "rsi_oversold",
                        "rsi_current": current_rsi,
                        "exit_type": "overbought"
                    }
                )
                
            # Check if we've hit target or stop loss
            entry_price = position.get('entry_price', 0)
            target_price = position.get('target_price', 0)
            stop_loss_price = position.get('stop_loss_price', 0)
            
            if entry_price > 0:
                # Check target
                if target_price > 0 and current_price >= target_price:
                    profit_pct = (current_price - entry_price) / entry_price
                    return Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        price=current_price,
                        quantity=position.get('quantity', 0),
                        order_type=OrderType.MARKET,
                        confidence=0.8,
                        reason=f"Target price reached (+{profit_pct:.1%})",
                        metadata={
                            "strategy": "rsi_oversold",
                            "rsi_current": current_rsi,
                            "exit_type": "target",
                            "profit_pct": profit_pct
                        }
                    )
                
                # Check stop loss
                if stop_loss_price > 0 and current_price <= stop_loss_price:
                    loss_pct = (entry_price - current_price) / entry_price
                    return Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        price=current_price,
                        quantity=position.get('quantity', 0),
                        order_type=OrderType.MARKET,
                        confidence=0.9,
                        reason=f"Stop loss triggered (-{loss_pct:.1%})",
                        metadata={
                            "strategy": "rsi_oversold",
                            "rsi_current": current_rsi,
                            "exit_type": "stop_loss",
                            "loss_pct": loss_pct
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking exit signal for {symbol}: {str(e)}")
            return None 