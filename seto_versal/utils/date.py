#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Date utilities for SETO-Versal
Provides functions for handling trading calendar and dates
"""

import logging
from datetime import datetime, date, timedelta
import pandas as pd
import tushare as ts
import os
import pickle

logger = logging.getLogger(__name__)

# Cache for trading calendar
_TRADING_DAYS_CACHE = None
_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'trading_days_cache.pkl')

def _ensure_cache_dir():
    """Ensure that the cache directory exists"""
    cache_dir = os.path.dirname(_CACHE_FILE)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

def _load_trading_days_cache():
    """Load trading days from cache if available"""
    global _TRADING_DAYS_CACHE
    try:
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE, 'rb') as f:
                _TRADING_DAYS_CACHE = pickle.load(f)
                logger.debug(f"Loaded trading days cache with {len(_TRADING_DAYS_CACHE)} days")
    except Exception as e:
        logger.warning(f"Failed to load trading days cache: {e}")
        _TRADING_DAYS_CACHE = None

def _save_trading_days_cache():
    """Save trading days to cache"""
    if _TRADING_DAYS_CACHE is not None:
        try:
            _ensure_cache_dir()
            with open(_CACHE_FILE, 'wb') as f:
                pickle.dump(_TRADING_DAYS_CACHE, f)
                logger.debug("Saved trading days cache")
        except Exception as e:
            logger.warning(f"Failed to save trading days cache: {e}")

def get_trading_days(start_date=None, end_date=None, api_key=None):
    """
    Get a list of trading days for the given period
    
    Args:
        start_date (str or datetime.date): Start date in 'YYYYMMDD' format or as date object
        end_date (str or datetime.date): End date in 'YYYYMMDD' format or as date object
        api_key (str): Tushare API key
        
    Returns:
        list: List of trading days as datetime.date objects
    """
    global _TRADING_DAYS_CACHE
    
    # Initialize cache if not already done
    if _TRADING_DAYS_CACHE is None:
        _load_trading_days_cache()
        
    if _TRADING_DAYS_CACHE is None:
        _TRADING_DAYS_CACHE = []
    
    # Convert date objects to strings if needed
    if isinstance(start_date, date):
        start_date = start_date.strftime('%Y%m%d')
    if isinstance(end_date, date):
        end_date = end_date.strftime('%Y%m%d')
    
    # Use default dates if not specified
    if start_date is None:
        start_date = '20100101'  # Default to start of 2010
    if end_date is None:
        end_date = (datetime.now() + timedelta(days=30)).strftime('%Y%m%d')  # Default to 30 days in future
    
    # Check if we already have the data in cache
    cached_dates = sorted(_TRADING_DAYS_CACHE)
    cached_start = min(cached_dates).strftime('%Y%m%d') if cached_dates else '99999999'
    cached_end = max(cached_dates).strftime('%Y%m%d') if cached_dates else '00000000'
    
    # If the requested dates are within the cache, use cached data
    if start_date >= cached_start and end_date <= cached_end:
        filtered_days = [d for d in cached_dates if start_date <= d.strftime('%Y%m%d') <= end_date]
        return filtered_days
    
    # Otherwise, need to fetch from API
    try:
        # Setup Tushare
        if api_key:
            ts.set_token(api_key)
        pro = ts.pro_api()
        
        # Get trading calendar
        df = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open='1')
        
        if df.empty:
            logger.warning(f"No trading days found from {start_date} to {end_date}")
            return []
        
        # Parse dates
        trading_days = [datetime.strptime(d, '%Y%m%d').date() for d in df['cal_date'].tolist()]
        
        # Update cache
        _TRADING_DAYS_CACHE.extend([d for d in trading_days if d not in _TRADING_DAYS_CACHE])
        _save_trading_days_cache()
        
        return sorted(trading_days)
        
    except Exception as e:
        logger.error(f"Error fetching trading calendar: {e}")
        # Fallback to simple weekday check if API fails
        logger.warning("Falling back to weekday-based trading day estimation")
        
        # Generate dates in the range
        start = datetime.strptime(start_date, '%Y%m%d').date()
        end = datetime.strptime(end_date, '%Y%m%d').date()
        all_days = [start + timedelta(days=x) for x in range((end - start).days + 1)]
        
        # Filter to weekdays (0=Monday, 4=Friday)
        trading_days = [d for d in all_days if d.weekday() < 5]
        
        return trading_days

def is_trading_day(day=None, api_key=None):
    """
    Check if a given day is a trading day
    
    Args:
        day (datetime.date): The day to check, defaults to today
        api_key (str): Tushare API key
        
    Returns:
        bool: True if the day is a trading day, False otherwise
    """
    if day is None:
        day = datetime.now().date()
    
    # Quick check for weekends
    if day.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Get trading days for the month
    start_date = (day.replace(day=1) - timedelta(days=1)).strftime('%Y%m%d')
    end_date = (day.replace(day=28) + timedelta(days=7)).strftime('%Y%m%d')
    
    trading_days = get_trading_days(start_date, end_date, api_key)
    
    return day in trading_days

def get_next_trading_day(from_date=None, api_key=None):
    """
    Get the next trading day after the given date
    
    Args:
        from_date (datetime.date): Starting date, defaults to today
        api_key (str): Tushare API key
        
    Returns:
        datetime.date: The next trading day
    """
    if from_date is None:
        from_date = datetime.now().date()
    
    # Get trading days for the next month
    start_date = from_date.strftime('%Y%m%d')
    end_date = (from_date + timedelta(days=31)).strftime('%Y%m%d')
    
    trading_days = get_trading_days(start_date, end_date, api_key)
    
    # Filter to days after from_date
    future_trading_days = [d for d in trading_days if d > from_date]
    
    if future_trading_days:
        return min(future_trading_days)
    else:
        # If no trading days found, estimate by finding the next weekday
        next_day = from_date + timedelta(days=1)
        while next_day.weekday() >= 5:  # Skip weekends
            next_day += timedelta(days=1)
        return next_day

def get_previous_trading_day(from_date=None, api_key=None):
    """
    Get the previous trading day before the given date
    
    Args:
        from_date (datetime.date): Starting date, defaults to today
        api_key (str): Tushare API key
        
    Returns:
        datetime.date: The previous trading day
    """
    if from_date is None:
        from_date = datetime.now().date()
    
    # Get trading days for the previous month
    end_date = from_date.strftime('%Y%m%d')
    start_date = (from_date - timedelta(days=31)).strftime('%Y%m%d')
    
    trading_days = get_trading_days(start_date, end_date, api_key)
    
    # Filter to days before from_date
    past_trading_days = [d for d in trading_days if d < from_date]
    
    if past_trading_days:
        return max(past_trading_days)
    else:
        # If no trading days found, estimate by finding the previous weekday
        prev_day = from_date - timedelta(days=1)
        while prev_day.weekday() >= 5:  # Skip weekends
            prev_day -= timedelta(days=1)
        return prev_day 