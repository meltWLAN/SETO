#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Volatility Filter Strategy

This strategy acts as a filter for other strategies, reducing position sizes
or avoiding trades during periods of high market volatility.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

from seto_versal.strategies.base import BaseStrategy
from seto_versal.common.constants import SignalType, OrderType
from seto_versal.common.models import Signal

logger = logging.getLogger(__name__)

class VolatilityFilterStrategy(BaseStrategy):
    """
    Volatility Filter Strategy - Adjusts positions based on market volatility
    
    This strategy:
    1. Monitors market volatility using VIX and/or historical volatility
    2. Acts as a filter for signals from other strategies
    3. Reduces position sizes during high volatility periods
    4. May generate exit signals for existing positions during extreme volatility
    
    This is typically used in conjunction with other strategies, not on its own.
    """
    
    def __init__(
        self,
        high_volatility_threshold: float = 25.0,
        extreme_volatility_threshold: float = 35.0,
        position_reduction_pct: float = 0.5,
        lookback_period: int = 20,
        min_position_pct: float = 0.25,
        atr_multiplier: float = 2.0,
        vix_weight: float = 0.7,
        hist_vol_weight: float = 0.3,
        **kwargs
    ):
        """
        Initialize Volatility Filter Strategy
        
        Args:
            high_volatility_threshold: VIX/volatility level considered high
            extreme_volatility_threshold: VIX/volatility level considered extreme
            position_reduction_pct: Percentage to reduce positions by during high vol
            lookback_period: Period (in days) to calculate historical volatility
            min_position_pct: Minimum position size as percentage of normal
            atr_multiplier: Multiplier for ATR in stop calculation
            vix_weight: Weight to give VIX in combined volatility score
            hist_vol_weight: Weight to give historical volatility in combined score
        """
        super().__init__(**kwargs)
        self.high_volatility_threshold = high_volatility_threshold
        self.extreme_volatility_threshold = extreme_volatility_threshold
        self.position_reduction_pct = position_reduction_pct
        self.lookback_period = lookback_period
        self.min_position_pct = min_position_pct
        self.atr_multiplier = atr_multiplier
        self.vix_weight = vix_weight
        self.hist_vol_weight = hist_vol_weight
        
        logger.info(
            f"Initialized VolatilityFilterStrategy: high_threshold={high_volatility_threshold}, "
            f"extreme_threshold={extreme_volatility_threshold}, reduction={position_reduction_pct}"
        )
    
    def generate_signals(
        self,
        market_data: Dict[str, Dict[datetime, Dict[str, float]]],
        positions: Dict[str, Dict[str, Any]] = None,
        market_state: Dict[str, Any] = None,
        input_signals: List[Signal] = None,
        **kwargs
    ) -> List[Signal]:
        """
        Generate trading signals based on volatility filtering
        
        Args:
            market_data: Dictionary of market data by symbol
            positions: Current positions
            market_state: Current market state information
            input_signals: Signals from other strategies to filter
            
        Returns:
            List of filtered trading signals and possible exit signals
        """
        filtered_signals = []
        
        try:
            # Validate we have enough data
            if not market_data:
                logger.warning("No market data provided to VolatilityFilterStrategy")
                return filtered_signals
                
            # Calculate current volatility level
            volatility_level = self._calculate_volatility(market_data, market_state)
            logger.info(f"Current volatility level: {volatility_level:.2f}")
            
            # Determine volatility regime
            vol_regime = self._determine_volatility_regime(volatility_level)
            logger.debug(f"Current volatility regime: {vol_regime}")
            
            # Process input signals if provided
            if input_signals:
                filtered_signals = self._filter_signals(input_signals, vol_regime, volatility_level)
                
            # Generate exit signals for existing positions during extreme volatility
            if vol_regime == "extreme" and positions:
                exit_signals = self._generate_exit_signals(
                    positions, 
                    market_data, 
                    volatility_level
                )
                filtered_signals.extend(exit_signals)
                
            logger.info(f"VolatilityFilterStrategy processed {len(input_signals or [])} signals, "
                       f"output {len(filtered_signals)} signals")
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error in VolatilityFilterStrategy: {str(e)}", exc_info=True)
            return []
    
    def _calculate_volatility(
        self, 
        market_data: Dict[str, Dict[datetime, Dict[str, float]]],
        market_state: Dict[str, Any]
    ) -> float:
        """
        Calculate current volatility level using VIX and historical volatility
        
        Args:
            market_data: Market data dictionary
            market_state: Market state information
            
        Returns:
            Combined volatility measure
        """
        # Get VIX level if available
        vix_value = None
        for symbol in ["^VIX", "VIX"]:
            if symbol in market_data:
                dates = sorted(market_data[symbol].keys())
                if dates:
                    vix_value = market_data[symbol][dates[-1]].get('close', 0)
                    break
        
        # If VIX not in market data, try market state
        if vix_value is None:
            vix_value = market_state.get('volatility', {}).get('vix', None)
            
        # Calculate historical volatility using SPY or S&P 500
        hist_vol = None
        for symbol in ["SPY", "^GSPC"]:
            if symbol in market_data:
                data = market_data[symbol]
                dates = sorted(data.keys())
                
                if len(dates) >= self.lookback_period:
                    # Calculate daily returns
                    returns = []
                    for i in range(1, self.lookback_period + 1):
                        if i < len(dates):
                            prev_close = data[dates[-i-1]].get('close', 0)
                            curr_close = data[dates[-i]].get('close', 0)
                            if prev_close > 0:
                                returns.append((curr_close - prev_close) / prev_close)
                    
                    # Calculate annualized volatility
                    if returns:
                        daily_std = np.std(returns)
                        hist_vol = daily_std * np.sqrt(252)  # Annualize
                        break
        
        # Use market state volatility if historical calculation failed
        if hist_vol is None:
            hist_vol = market_state.get('volatility', {}).get('historical', 0.2)  # Default 20%
            
        # Combine VIX and historical volatility
        if vix_value is not None:
            # Normalize VIX to same scale as hist_vol (percentage)
            vix_pct = vix_value / 100
            combined_vol = (self.vix_weight * vix_pct) + (self.hist_vol_weight * hist_vol)
        else:
            combined_vol = hist_vol
            
        return combined_vol * 100  # Convert to same scale as VIX
    
    def _determine_volatility_regime(self, volatility_level: float) -> str:
        """
        Determine volatility regime based on thresholds
        
        Args:
            volatility_level: Current volatility level
            
        Returns:
            Volatility regime ("normal", "high", or "extreme")
        """
        if volatility_level >= self.extreme_volatility_threshold:
            return "extreme"
        elif volatility_level >= self.high_volatility_threshold:
            return "high"
        else:
            return "normal"
    
    def _filter_signals(
        self, 
        signals: List[Signal], 
        vol_regime: str,
        volatility_level: float
    ) -> List[Signal]:
        """
        Filter input signals based on volatility regime
        
        Args:
            signals: List of signals to filter
            vol_regime: Current volatility regime
            volatility_level: Current volatility level
            
        Returns:
            List of filtered signals
        """
        filtered_signals = []
        
        for signal in signals:
            # Skip non-buy signals (don't filter sell signals)
            if signal.signal_type != SignalType.BUY:
                filtered_signals.append(signal)
                continue
                
            # Determine position size adjustment
            adjustment = self._calculate_position_adjustment(vol_regime, volatility_level)
            
            if adjustment <= 0:
                # Skip signal entirely
                logger.debug(f"Filtering out {signal.symbol} signal due to extreme volatility")
                continue
                
            # Create adjusted signal
            adjusted_signal = self._adjust_signal(signal, adjustment, vol_regime)
            filtered_signals.append(adjusted_signal)
            
        return filtered_signals
    
    def _calculate_position_adjustment(self, vol_regime: str, volatility_level: float) -> float:
        """
        Calculate position size adjustment factor based on volatility
        
        Args:
            vol_regime: Current volatility regime
            volatility_level: Current volatility level
            
        Returns:
            Adjustment factor (0.0 to 1.0)
        """
        if vol_regime == "normal":
            return 1.0
        elif vol_regime == "high":
            # Linear reduction based on volatility level
            excess_vol = volatility_level - self.high_volatility_threshold
            vol_range = self.extreme_volatility_threshold - self.high_volatility_threshold
            
            if vol_range <= 0:
                return 1.0 - self.position_reduction_pct
                
            reduction = self.position_reduction_pct * (excess_vol / vol_range)
            return max(self.min_position_pct, 1.0 - reduction)
        else:  # extreme
            # Minimum position size or zero
            if volatility_level > self.extreme_volatility_threshold * 1.5:
                return 0.0  # No positions in severe volatility
            else:
                return self.min_position_pct
    
    def _adjust_signal(self, signal: Signal, adjustment: float, vol_regime: str) -> Signal:
        """
        Adjust signal based on volatility regime
        
        Args:
            signal: Original signal
            adjustment: Position size adjustment factor
            vol_regime: Current volatility regime
            
        Returns:
            Adjusted signal
        """
        # Create a copy of the signal
        adjusted_signal = Signal(
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            price=signal.price,
            quantity=signal.quantity,
            order_type=signal.order_type,
            confidence=signal.confidence,
            target_price=signal.target_price,
            stop_loss_price=signal.stop_loss_price,
            reason=signal.reason,
            metadata=signal.metadata.copy() if signal.metadata else {}
        )
        
        # Adjust position size
        adjusted_signal.quantity = max(1, int(signal.quantity * adjustment))
        
        # Adjust confidence level
        adjusted_signal.confidence = signal.confidence * adjustment
        
        # Adjust stop loss to be tighter in high volatility
        if vol_regime != "normal" and signal.stop_loss_price and signal.price:
            # Calculate original stop distance
            orig_stop_distance = abs(signal.price - signal.stop_loss_price)
            
            # Tighten stop based on volatility
            if signal.signal_type == SignalType.BUY:
                new_stop = signal.price - (orig_stop_distance * adjustment)
                adjusted_signal.stop_loss_price = new_stop
            else:
                new_stop = signal.price + (orig_stop_distance * adjustment)
                adjusted_signal.stop_loss_price = new_stop
        
        # Add volatility information to metadata
        adjusted_signal.metadata.update({
            "volatility_adjusted": True,
            "adjustment_factor": adjustment,
            "volatility_regime": vol_regime
        })
        
        # Update reason
        if vol_regime != "normal":
            adjusted_signal.reason = f"{signal.reason} (position reduced due to {vol_regime} volatility)"
            
        return adjusted_signal
    
    def _generate_exit_signals(
        self,
        positions: Dict[str, Dict[str, Any]],
        market_data: Dict[str, Dict[datetime, Dict[str, float]]],
        volatility_level: float
    ) -> List[Signal]:
        """
        Generate exit signals for existing positions during extreme volatility
        
        Args:
            positions: Current positions
            market_data: Market data dictionary
            volatility_level: Current volatility level
            
        Returns:
            List of exit signals
        """
        exit_signals = []
        
        # Determine exit percentage based on volatility level
        excess_vol = volatility_level - self.extreme_volatility_threshold
        vol_severity = min(1.0, excess_vol / (self.extreme_volatility_threshold * 0.5))
        
        # Exit more positions as volatility increases
        exit_pct = vol_severity
        
        # Sort positions by risk (using stop distance as proxy)
        position_risks = []
        for symbol, position in positions.items():
            # Skip if not in market data
            if symbol not in market_data:
                continue
                
            data = market_data[symbol]
            dates = sorted(data.keys())
            
            if not dates:
                continue
                
            current_price = data[dates[-1]].get('close', 0)
            entry_price = position.get('entry_price', 0)
            stop_price = position.get('stop_loss_price', 0)
            
            if current_price <= 0 or entry_price <= 0:
                continue
                
            # Calculate stop distance as percentage
            if stop_price > 0:
                stop_distance = abs(current_price - stop_price) / current_price
            else:
                # Use ATR to estimate risk if no stop loss defined
                atr = self._calculate_atr(data, dates)
                stop_distance = (atr * self.atr_multiplier) / current_price
                
            # Check if position is held by a risk-sensitive strategy
            metadata = position.get('metadata', {})
            strategy = metadata.get('strategy', '')
            
            # Add extra risk factor for certain strategies
            strategy_risk = 1.0
            if strategy in ['momentum', 'breakout']:
                strategy_risk = 1.5  # Higher risk for momentum/breakout strategies
                
            # Calculate total risk score
            risk_score = stop_distance * strategy_risk
            
            position_risks.append((symbol, position, risk_score))
            
        # Sort by risk (highest first)
        position_risks.sort(key=lambda x: x[2], reverse=True)
        
        # Determine how many positions to exit
        num_to_exit = max(1, int(len(position_risks) * exit_pct))
        
        # Generate exit signals for highest risk positions
        for symbol, position, risk_score in position_risks[:num_to_exit]:
            data = market_data[symbol]
            dates = sorted(data.keys())
            current_price = data[dates[-1]].get('close', 0)
            
            exit_signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=current_price,
                quantity=position.get('quantity', 0),
                order_type=OrderType.MARKET,
                confidence=0.8,
                reason=f"Exiting position due to extreme market volatility ({volatility_level:.1f})",
                metadata={
                    "strategy": "volatility_filter",
                    "original_strategy": position.get('metadata', {}).get('strategy', ''),
                    "risk_score": risk_score,
                    "volatility_level": volatility_level
                }
            ))
            
        return exit_signals
    
    def _calculate_atr(self, data: Dict[datetime, Dict[str, float]], dates: List[datetime]) -> float:
        """
        Calculate Average True Range (ATR)
        
        Args:
            data: Price data
            dates: Date list
            
        Returns:
            ATR value
        """
        period = min(14, len(dates) - 1)
        true_ranges = []
        
        for i in range(1, period + 1):
            if i >= len(dates):
                break
                
            high = data[dates[-i]].get('high', 0)
            low = data[dates[-i]].get('low', 0)
            prev_close = data[dates[-i-1]].get('close', 0)
            
            if high <= 0 or low <= 0 or prev_close <= 0:
                continue
                
            # Calculate true range
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
            
        if true_ranges:
            return sum(true_ranges) / len(true_ranges)
        else:
            # Default ATR if calculation fails
            return data[dates[-1]].get('close', 100) * 0.02  # 2% of price 