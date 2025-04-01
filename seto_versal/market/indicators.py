#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Technical indicators module for SETO-Versal
Provides functions to calculate various technical indicators for market analysis
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def calculate_indicators(df):
    """
    Calculate technical indicators for a given dataframe
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pandas.DataFrame: DataFrame with added technical indicators
    """
    # Check if required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'vol']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"DataFrame missing required columns for indicator calculation")
        return df
    
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    try:
        # Moving Averages
        result['sma5'] = calculate_sma(result['close'], 5)
        result['sma10'] = calculate_sma(result['close'], 10)
        result['sma20'] = calculate_sma(result['close'], 20)
        result['sma50'] = calculate_sma(result['close'], 50)
        result['ema5'] = calculate_ema(result['close'], 5)
        result['ema10'] = calculate_ema(result['close'], 10)
        result['ema20'] = calculate_ema(result['close'], 20)
        
        # Bollinger Bands (20, 2)
        result['bb_middle'] = result['sma20']
        result['bb_std'] = calculate_rolling_std(result['close'], 20)
        result['bb_upper'] = result['bb_middle'] + 2 * result['bb_std']
        result['bb_lower'] = result['bb_middle'] - 2 * result['bb_std']
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        
        # RSI (14)
        result['rsi'] = calculate_rsi(result['close'], 14)
        
        # MACD (12, 26, 9)
        result['macd'], result['macd_signal'], result['macd_hist'] = calculate_macd(
            result['close'], 12, 26, 9)
        
        # Stochastic Oscillator (14, 3)
        result['stoch_k'], result['stoch_d'] = calculate_stochastic(
            result['high'], result['low'], result['close'], 14, 3)
        
        # Average Directional Index (14)
        result['adx'] = calculate_adx(result['high'], result['low'], result['close'], 14)
        
        # On-Balance Volume
        result['obv'] = calculate_obv(result['close'], result['vol'])
        
        # Volume indicators
        result['volume_ma5'] = calculate_sma(result['vol'], 5)
        result['volume_ma20'] = calculate_sma(result['vol'], 20)
        result['volume_ratio'] = result['vol'] / result['volume_ma20']
        
        # Momentum indicators
        result['momentum'] = calculate_momentum(result['close'], 10)
        result['rate_of_change'] = calculate_roc(result['close'], 10)
        
        # Price channels
        result['highest_high_20'] = calculate_rolling_max(result['high'], 20)
        result['lowest_low_20'] = calculate_rolling_min(result['low'], 20)
        
        # Ichimoku Cloud
        tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span = calculate_ichimoku(
            result['high'], result['low'], result['close'])
        result['ichimoku_tenkan'] = tenkan_sen
        result['ichimoku_kijun'] = kijun_sen
        result['ichimoku_senkou_a'] = senkou_span_a
        result['ichimoku_senkou_b'] = senkou_span_b
        result['ichimoku_chikou'] = chikou_span
        
        # Additional indicators
        # Chaikin Money Flow
        result['cmf'] = calculate_cmf(result['high'], result['low'], result['close'], result['vol'], 20)
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
    
    return result


def calculate_sma(series, period):
    """Calculate Simple Moving Average"""
    return series.rolling(window=period).mean()


def calculate_ema(series, period, smoothing=2):
    """Calculate Exponential Moving Average"""
    # span = 2 / (1 + period) * price + (1 - (2 / (1 + period))) * previous_ema
    return series.ewm(span=period, adjust=False).mean()


def calculate_rolling_std(series, period):
    """Calculate rolling standard deviation"""
    return series.rolling(window=period).std()


def calculate_rolling_max(series, period):
    """Calculate rolling maximum"""
    return series.rolling(window=period).max()


def calculate_rolling_min(series, period):
    """Calculate rolling minimum"""
    return series.rolling(window=period).min()


def calculate_rsi(series, period):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # For values after the initial period
    for i in range(period, len(gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
    
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(series, fast_period, slow_period, signal_period):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    fast_ema = calculate_ema(series, fast_period)
    slow_ema = calculate_ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_stochastic(high, low, close, k_period, d_period):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    # Fast %K
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    # Fast %D (3-period SMA of %K)
    d = k.rolling(window=d_period).mean()
    
    return k, d


def calculate_adx(high, low, close, period):
    """Calculate Average Directional Index"""
    # True Range
    tr1 = abs(high - low)
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Plus Directional Movement (+DM)
    plus_dm = high - high.shift(1)
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > (low.shift(1) - low)), 0)
    
    # Minus Directional Movement (-DM)
    minus_dm = low.shift(1) - low
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > (high - high.shift(1))), 0)
    
    # Smoothed +DM and -DM
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Directional Movement Index (DX)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.finfo(float).eps)
    
    # Average Directional Index (ADX)
    adx = dx.rolling(window=period).mean()
    
    return adx


def calculate_obv(close, volume):
    """Calculate On-Balance Volume"""
    obv = pd.Series(index=close.index, dtype='float64')
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def calculate_momentum(series, period):
    """Calculate Momentum"""
    return series / series.shift(period) - 1


def calculate_roc(series, period):
    """Calculate Rate of Change"""
    return 100 * (series / series.shift(period) - 1)


def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 (shifted forward 26 periods)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 (shifted forward 26 periods)
    senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
    
    # Chikou Span (Lagging Span): Close shifted back 26 periods
    chikou_span = close.shift(-26)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span


def calculate_cmf(high, low, close, volume, period):
    """Calculate Chaikin Money Flow"""
    # Money Flow Multiplier: ((Close - Low) - (High - Close)) / (High - Low)
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.finfo(float).eps)
    
    # Money Flow Volume: Money Flow Multiplier * Volume
    mfv = mfm * volume
    
    # Chaikin Money Flow: Sum of MFV over period / Sum of Volume over period
    cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    
    return cmf 