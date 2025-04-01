#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ADX Trend strategy module for SETO-Versal
Identifies strong trends using the Average Directional Index (ADX)
"""

import logging
import numpy as np
from datetime import datetime

from seto_versal.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class AdxTrendStrategy(BaseStrategy):
    """
    ADX Trend strategy for detecting strong trends using the Average Directional Index
    
    This strategy identifies stocks with strong trends using ADX, which measures
    trend strength regardless of direction, and +DI/-DI to determine trend direction.
    """
    
    def __init__(self, name="adx_trend", **kwargs):
        """
        Initialize the ADX trend strategy
        
        Args:
            name (str): Strategy name
            **kwargs: Strategy parameters
        """
        kwargs['name'] = name
        kwargs['category'] = 'trend'
        super().__init__(**kwargs)
        
        # Initialize parameters with defaults
        self.parameters = {
            'adx_period': kwargs.get('adx_period', 14),
            'di_period': kwargs.get('di_period', 14),
            'min_adx': kwargs.get('min_adx', 25.0),
            'lookback_period': kwargs.get('lookback_period', 20),
            'entry_threshold': kwargs.get('entry_threshold', 5.0),
            'exit_threshold': kwargs.get('exit_threshold', 5.0),
            'adx_rising': kwargs.get('adx_rising', True)
        }
        
        logger.info(f"Initialized AdxTrendStrategy: period={self.parameters['adx_period']}, min_adx={self.parameters['min_adx']}")
    
    def generate_signals(self, market_data, **kwargs):
        """
        Generate trend signals based on ADX and directional indicators
        
        Args:
            market_data (dict): Market data including price and volume history
            **kwargs: Additional parameters
            
        Returns:
            list: List of ADX trend signals
        """
        signals = []
        
        # Validate inputs
        if not market_data or 'history' not in market_data:
            logger.warning(f"Strategy '{self.name}': Invalid market data format")
            return signals
        
        # Extract parameters
        adx_period = self.parameters['adx_period']
        di_period = self.parameters['di_period']
        min_adx = self.parameters['min_adx']
        lookback_period = self.parameters['lookback_period']
        entry_threshold = self.parameters['entry_threshold']
        exit_threshold = self.parameters['exit_threshold']
        adx_rising = self.parameters['adx_rising']
        
        # Need enough data for calculations
        min_data_points = lookback_period + adx_period + 10  # Add buffer for calculations
        
        # Process each symbol in market data
        for symbol, data in market_data.get('symbols', {}).items():
            try:
                # Get price and volume data
                if 'ohlcv' not in data:
                    continue
                
                ohlcv = data['ohlcv']
                if len(ohlcv) < min_data_points:
                    continue
                
                # Extract price data
                high_prices = np.array([bar['high'] for bar in ohlcv[-lookback_period:]])
                low_prices = np.array([bar['low'] for bar in ohlcv[-lookback_period:]])
                close_prices = np.array([bar['close'] for bar in ohlcv[-lookback_period:]])
                
                # Calculate ADX and directional indicators
                adx, plus_di, minus_di = self._calculate_adx(high_prices, low_prices, close_prices, adx_period, di_period)
                
                if len(adx) < 5:
                    continue  # Not enough ADX data points
                
                # Get current values
                current_adx = adx[-1]
                current_plus_di = plus_di[-1]
                current_minus_di = minus_di[-1]
                current_price = close_prices[-1]
                
                # Check for minimum ADX (strong trend)
                if current_adx < min_adx:
                    continue  # Trend not strong enough
                
                # Check if ADX is rising (strengthening trend) if required
                if adx_rising and adx[-1] <= adx[-2]:
                    continue  # ADX not rising
                
                # Determine trend direction based on +DI and -DI
                bull_trend = current_plus_di > current_minus_di
                bear_trend = current_minus_di > current_plus_di
                
                # Check for sufficient separation between +DI and -DI
                di_separation = abs(current_plus_di - current_minus_di)
                if di_separation < entry_threshold:
                    continue  # Not enough separation between indicators
                
                # Determine signal type based on trend direction
                signal_type = "buy" if bull_trend else "sell" if bear_trend else None
                
                if not signal_type:
                    continue  # No clear trend direction
                
                # Check for recent crossover (entry signal)
                crossover_detected = False
                if bull_trend:
                    # Look for recent bullish crossover (+DI crosses above -DI)
                    for i in range(2, min(6, len(plus_di))):
                        if plus_di[-i] <= minus_di[-i] and plus_di[-i+1] > minus_di[-i+1]:
                            crossover_detected = True
                            break
                elif bear_trend:
                    # Look for recent bearish crossover (-DI crosses above +DI)
                    for i in range(2, min(6, len(minus_di))):
                        if minus_di[-i] <= plus_di[-i] and minus_di[-i+1] > plus_di[-i+1]:
                            crossover_detected = True
                            break
                
                # If no recent crossover detected, this may be an established trend
                # We'll still generate a signal but with lower confidence
                
                # Calculate confidence based on ADX strength and DI separation
                adx_confidence = min(1.0, current_adx / 40)  # Normalize ADX (25-40 range typical)
                di_confidence = min(1.0, di_separation / 20)  # Normalize DI separation
                
                # Add crossover factor if recent crossover detected
                crossover_confidence = 0.2 if crossover_detected else 0
                
                # Combined confidence with weights
                confidence = (adx_confidence * 0.5) + (di_confidence * 0.3) + crossover_confidence
                if bull_trend and adx_rising:
                    confidence += 0.1  # Bonus for bullish trend with rising ADX
                
                confidence = min(0.95, confidence)  # Cap at 0.95
                
                # Calculate target and stop loss based on ATR
                atr = self._calculate_atr(high_prices, low_prices, close_prices, 14)
                if atr is None:
                    atr = current_price * 0.02  # Fallback to 2% of price
                
                # Different targets for buy vs sell
                if signal_type == "buy":
                    target_price = current_price + (atr * 3)  # 3x ATR for target
                    stop_loss = current_price - (atr * 1.5)   # 1.5x ATR for stop loss
                else:  # sell
                    target_price = current_price - (atr * 3)  # 3x ATR for target (downside)
                    stop_loss = current_price + (atr * 1.5)   # 1.5x ATR for stop loss (upside)
                
                # Create signal
                reason = f"Strong {'bullish' if bull_trend else 'bearish'} trend: ADX={current_adx:.1f}, "
                reason += f"+DI={current_plus_di:.1f}, -DI={current_minus_di:.1f}"
                
                signal = {
                    'symbol': symbol,
                    'type': signal_type,
                    'timestamp': datetime.now(),
                    'price': current_price,
                    'confidence': confidence,
                    'parameters': {k: v for k, v in self.parameters.items()},
                    'reason': reason,
                    'target_price': target_price,
                    'stop_loss': stop_loss,
                    'expected_holding_period': '2-6 weeks'  # Medium-term trend following
                }
                
                signals.append(signal)
                logger.debug(f"ADX Trend signal for {symbol}: {signal_type}, confidence={confidence:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol} for ADX trend: {e}")
        
        # Update strategy metrics
        self.signals_generated += len(signals)
        
        return signals
    
    def _calculate_tr(self, high, low, close):
        """
        Calculate True Range
        
        Args:
            high (float): Current high price
            low (float): Current low price
            prev_close (float): Previous close price
            
        Returns:
            float: True Range value
        """
        return max(high - low, abs(high - close), abs(low - close))
    
    def _calculate_atr(self, high_prices, low_prices, close_prices, period=14):
        """
        Calculate Average True Range (ATR)
        
        Args:
            high_prices (np.array): Array of high prices
            low_prices (np.array): Array of low prices
            close_prices (np.array): Array of close prices
            period (int): ATR period
            
        Returns:
            float: ATR value
        """
        if len(high_prices) < period + 1:
            return None
            
        # Calculate true ranges
        tr_values = []
        for i in range(1, len(high_prices)):
            tr = self._calculate_tr(high_prices[i], low_prices[i], close_prices[i-1])
            tr_values.append(tr)
        
        # Calculate ATR as simple moving average of true ranges
        atr = np.mean(tr_values[-period:])
        return atr
    
    def _calculate_adx(self, high_prices, low_prices, close_prices, adx_period=14, di_period=14):
        """
        Calculate Average Directional Index (ADX) and Directional Indicators (+DI, -DI)
        
        Args:
            high_prices (np.array): Array of high prices
            low_prices (np.array): Array of low prices
            close_prices (np.array): Array of close prices
            adx_period (int): ADX smoothing period
            di_period (int): DI calculation period
            
        Returns:
            tuple: (ADX, +DI, -DI) as numpy arrays
        """
        if len(high_prices) < adx_period + di_period + 1:
            return np.array([]), np.array([]), np.array([])
        
        # Calculate True Range
        tr_values = []
        for i in range(1, len(high_prices)):
            tr = self._calculate_tr(high_prices[i], low_prices[i], close_prices[i-1])
            tr_values.append(tr)
        
        tr_values = np.array(tr_values)
        
        # Calculate Directional Movement
        plus_dm = []
        minus_dm = []
        
        for i in range(1, len(high_prices)):
            up_move = high_prices[i] - high_prices[i-1]
            down_move = low_prices[i-1] - low_prices[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)
                
            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)
        
        plus_dm = np.array(plus_dm)
        minus_dm = np.array(minus_dm)
        
        # Smooth TR and DMs with EMA
        smoothed_tr = self._smooth_data(tr_values, di_period)
        smoothed_plus_dm = self._smooth_data(plus_dm, di_period)
        smoothed_minus_dm = self._smooth_data(minus_dm, di_period)
        
        # Calculate Directional Indicators
        plus_di = (smoothed_plus_dm / smoothed_tr) * 100
        minus_di = (smoothed_minus_dm / smoothed_tr) * 100
        
        # Calculate Directional Index (DX)
        dx_values = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX by smoothing DX
        adx = self._smooth_data(dx_values, adx_period)
        
        return adx, plus_di, minus_di
    
    def _smooth_data(self, data, period):
        """
        Apply Welles Wilder's smoothing method (similar to EMA but with Wilder's alpha)
        
        Args:
            data (np.array): Data to smooth
            period (int): Smoothing period
            
        Returns:
            np.array: Smoothed data
        """
        # Initialize smoothed array with first value
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        
        # Apply smoothing formula
        for i in range(1, len(data)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + data[i]
            
        return smoothed 