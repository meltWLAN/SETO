#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data source module for SETO-Versal
Provides interfaces and implementations for market data acquisition
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
import os
import json
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)

class DataProvider(Enum):
    """Available data providers"""
    TUSHARE = "tushare"
    AKSHARE = "akshare"
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO = "yahoo"
    CSV = "csv"
    MOCK = "mock"
    DATABASE = "database"

class DataSource(ABC):
    """
    Abstract base class for data sources
    Defines interfaces that all data source implementations must support
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data source with configuration
        
        Args:
            config (dict): Configuration for the data source
        """
        self.config = config
        self.name = config.get('name', 'default_data_source')
        self.provider = DataProvider(config.get('provider', 'mock'))
        self.cache_dir = config.get('cache_dir', 'seto_versal/data/cache')
        self.use_cache = config.get('use_cache', True)
        self.cache_expiry = config.get('cache_expiry', 86400)  # 1 day in seconds
        
        # Create cache directory if it doesn't exist
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Track API rate limits
        self.rate_limit = config.get('rate_limit', {})
        self.last_request_time = {}
        
        logger.info(f"Initialized {self.provider.value} data source")
    
    @abstractmethod
    def get_price_data(self, symbol: str, start_date: str, end_date: str, 
                    interval: str = 'daily') -> pd.DataFrame:
        """
        Get historical price data for a symbol
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval ('daily', 'weekly', '1m', '5m', etc.)
            
        Returns:
            pd.DataFrame: DataFrame with price data
        """
        pass
    
    @abstractmethod
    def get_market_index(self, index_symbol: str, start_date: str, end_date: str,
                       interval: str = 'daily') -> pd.DataFrame:
        """
        Get market index data
        
        Args:
            index_symbol (str): Index symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval ('daily', 'weekly', '1m', '5m', etc.)
            
        Returns:
            pd.DataFrame: DataFrame with index data
        """
        pass
    
    @abstractmethod
    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Get fundamental data for a symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Dictionary of fundamental data
        """
        pass
    
    @abstractmethod
    def get_symbol_list(self, market: str = None) -> List[str]:
        """
        Get list of available symbols
        
        Args:
            market (str, optional): Market filter
            
        Returns:
            list: List of symbols
        """
        pass
    
    def _get_cache_path(self, key: str) -> str:
        """
        Get path for a cache file
        
        Args:
            key (str): Cache key
            
        Returns:
            str: Path to cache file
        """
        # Create a safe filename from the key
        safe_key = key.replace('/', '_').replace('\\', '_').replace(':', '_')
        return os.path.join(self.cache_dir, f"{safe_key}.json")
    
    def _save_to_cache(self, key: str, data: Any) -> None:
        """
        Save data to cache
        
        Args:
            key (str): Cache key
            data (any): Data to cache (must be JSON serializable)
        """
        if not self.use_cache:
            return
        
        cache_path = self._get_cache_path(key)
        
        try:
            # Convert DataFrame to dict for JSON serialization
            if isinstance(data, pd.DataFrame):
                serializable_data = {
                    'data': data.to_dict(orient='records'),
                    'timestamp': datetime.now().timestamp(),
                    'columns': data.columns.tolist(),
                    'index': data.index.tolist() if not data.index.equals(pd.RangeIndex(len(data))) else None
                }
            else:
                serializable_data = {
                    'data': data,
                    'timestamp': datetime.now().timestamp()
                }
            
            with open(cache_path, 'w') as f:
                json.dump(serializable_data, f)
            
            logger.debug(f"Saved data to cache: {key}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {str(e)}")
    
    def _load_from_cache(self, key: str) -> Tuple[bool, Any]:
        """
        Load data from cache if available and not expired
        
        Args:
            key (str): Cache key
            
        Returns:
            tuple: (cache_hit, data)
                - cache_hit (bool): Whether data was found in cache
                - data (any): Cached data or None
        """
        if not self.use_cache:
            return False, None
        
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return False, None
        
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            
            # Check if cache is expired
            timestamp = cached.get('timestamp', 0)
            if (datetime.now().timestamp() - timestamp) > self.cache_expiry:
                logger.debug(f"Cache expired for: {key}")
                return False, None
            
            # Reconstruct DataFrame if needed
            if 'columns' in cached:
                df = pd.DataFrame(cached['data'])
                if cached.get('index') is not None:
                    df.index = cached['index']
                return True, df
            else:
                return True, cached['data']
                
        except Exception as e:
            logger.warning(f"Failed to load from cache: {str(e)}")
            return False, None
    
    def _respect_rate_limit(self, endpoint: str = 'default') -> None:
        """
        Ensure rate limits are respected by adding delays between requests
        
        Args:
            endpoint (str): API endpoint being accessed
        """
        # Get rate limit for this endpoint
        rate_limit = self.rate_limit.get(endpoint, self.rate_limit.get('default', 0))
        
        if rate_limit > 0:
            # Get last request time
            last_time = self.last_request_time.get(endpoint, 0)
            
            # Calculate minimum time between requests (in seconds)
            min_interval = 1.0 / rate_limit
            
            # Calculate how long to wait
            current_time = time.time()
            elapsed = current_time - last_time
            
            if elapsed < min_interval:
                delay = min_interval - elapsed
                logger.debug(f"Rate limiting: waiting {delay:.2f}s for {endpoint}")
                time.sleep(delay)
            
            # Update last request time
            self.last_request_time[endpoint] = time.time()

class MockDataSource(DataSource):
    """
    Mock data source for testing
    Generates synthetic data that resembles real market data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize mock data source
        
        Args:
            config (dict): Configuration for the data source
        """
        config['provider'] = 'mock'
        super().__init__(config)
        
        # Mock data generation parameters
        self.volatility = config.get('volatility', 0.01)
        self.trend = config.get('trend', 0.0001)
        self.symbols = config.get('symbols', ['MOCK1', 'MOCK2', 'MOCK3', 'MOCK4', 'MOCK5'])
        self.index_symbols = config.get('index_symbols', ['MOCKINDEX'])
        
        # Seed for reproducibility
        self.seed = config.get('seed', 42)
        np.random.seed(self.seed)
    
    def get_price_data(self, symbol: str, start_date: str, end_date: str, 
                      interval: str = 'daily') -> pd.DataFrame:
        """
        Generate mock price data
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval ('daily', 'weekly', '1m', '5m', etc.)
            
        Returns:
            pd.DataFrame: DataFrame with price data
        """
        # Check cache first
        cache_key = f"price_{symbol}_{start_date}_{end_date}_{interval}"
        cache_hit, cached_data = self._load_from_cache(cache_key)
        if cache_hit:
            return cached_data
        
        # Convert dates to datetime
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Determine number of periods based on interval
        if interval == 'daily':
            # Number of days between dates
            delta = (end - start).days + 1
            date_range = [start + timedelta(days=i) for i in range(delta)]
            
        elif interval == 'weekly':
            # Calculate number of weeks
            delta = (end - start).days // 7 + 1
            date_range = [start + timedelta(weeks=i) for i in range(delta)]
            
        elif interval.endswith('m'):
            # Intraday data (minutes)
            minutes = int(interval[:-1])
            trading_hours = 6.5  # 6.5 hours of trading per day
            periods_per_day = int(trading_hours * 60 / minutes)
            
            delta = (end - start).days + 1
            date_range = []
            
            for i in range(delta):
                day = start + timedelta(days=i)
                # Skip weekends
                if day.weekday() >= 5:  # 5=Saturday, 6=Sunday
                    continue
                
                # Add periods for this day
                day_open = datetime(day.year, day.month, day.day, 9, 30)  # 9:30 AM
                for j in range(periods_per_day):
                    date_range.append(day_open + timedelta(minutes=j*minutes))
        else:
            # Default to daily
            delta = (end - start).days + 1
            date_range = [start + timedelta(days=i) for i in range(delta)]
        
        # Generate random walk prices
        price = 100.0  # Starting price
        prices = []
        volumes = []
        
        # Use symbol as additional seed for variation
        symbol_seed = sum(ord(c) for c in symbol)
        np.random.seed(self.seed + symbol_seed)
        
        for i, date in enumerate(date_range):
            # Skip weekends for daily data
            if interval == 'daily' and date.weekday() >= 5:
                continue
                
            # Random price change with trend
            change = np.random.normal(self.trend, self.volatility)
            price *= (1 + change)
            
            # Generate OHLC prices
            daily_volatility = self.volatility * price
            open_price = price
            high_price = price + abs(np.random.normal(0, daily_volatility))
            low_price = price - abs(np.random.normal(0, daily_volatility))
            close_price = price
            
            # Generate volume
            volume = np.random.randint(100000, 10000000)
            
            prices.append([date, open_price, high_price, low_price, close_price])
            volumes.append(volume)
        
        # Create DataFrame
        df = pd.DataFrame(prices, columns=['date', 'open', 'high', 'low', 'close'])
        df['volume'] = volumes
        df['symbol'] = symbol
        df.set_index('date', inplace=True)
        
        # Cache the results
        self._save_to_cache(cache_key, df)
        
        return df
    
    def get_market_index(self, index_symbol: str, start_date: str, end_date: str,
                       interval: str = 'daily') -> pd.DataFrame:
        """
        Generate mock market index data
        
        Args:
            index_symbol (str): Index symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval ('daily', 'weekly', '1m', '5m', etc.)
            
        Returns:
            pd.DataFrame: DataFrame with index data
        """
        # Similar to get_price_data but with lower volatility
        old_volatility = self.volatility
        self.volatility = self.volatility * 0.6  # Lower volatility for indices
        
        df = self.get_price_data(index_symbol, start_date, end_date, interval)
        df['is_index'] = True
        
        # Restore volatility
        self.volatility = old_volatility
        
        return df
    
    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Generate mock fundamental data
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Dictionary of fundamental data
        """
        # Check cache first
        cache_key = f"fundamentals_{symbol}"
        cache_hit, cached_data = self._load_from_cache(cache_key)
        if cache_hit:
            return cached_data
        
        # Use symbol as seed for variation
        symbol_seed = sum(ord(c) for c in symbol)
        np.random.seed(self.seed + symbol_seed)
        
        # Generate mock fundamentals
        market_cap = np.random.randint(1000000, 10000000000)
        pe_ratio = np.random.uniform(5, 50)
        dividend_yield = np.random.uniform(0, 0.05)
        eps = np.random.uniform(0.1, 10)
        revenue = np.random.randint(10000, 10000000000)
        profit_margin = np.random.uniform(0.01, 0.3)
        debt_to_equity = np.random.uniform(0, 2)
        
        fundamentals = {
            'symbol': symbol,
            'company_name': f"Mock Company {symbol}",
            'market_cap': market_cap,
            'pe_ratio': pe_ratio,
            'dividend_yield': dividend_yield,
            'eps': eps,
            'revenue': revenue,
            'profit_margin': profit_margin,
            'debt_to_equity': debt_to_equity,
            'sector': np.random.choice(['Technology', 'Finance', 'Healthcare', 'Consumer', 'Energy']),
            'industry': np.random.choice(['Software', 'Banking', 'Pharmaceuticals', 'Retail', 'Oil & Gas']),
            'employees': np.random.randint(50, 100000),
            'founded_year': np.random.randint(1900, 2020),
            'ceo': f"John Doe {symbol}",
            'headquarters': np.random.choice(['New York', 'San Francisco', 'Chicago', 'Boston', 'Seattle']),
            'last_updated': datetime.now().strftime("%Y-%m-%d")
        }
        
        # Cache the results
        self._save_to_cache(cache_key, fundamentals)
        
        return fundamentals
    
    def get_symbol_list(self, market: str = None) -> List[str]:
        """
        Get list of available mock symbols
        
        Args:
            market (str, optional): Market filter
            
        Returns:
            list: List of symbols
        """
        return self.symbols

class TushareDataSource(DataSource):
    """
    Tushare data source implementation
    Uses Tushare API to get real market data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Tushare data source
        
        Args:
            config (dict): Configuration for the data source
        """
        super().__init__(config)
        self.provider = DataProvider.TUSHARE
        
        # Tushare API setup
        self.api_key = config.get('api_key')
        if not self.api_key:
            logger.warning("No Tushare API key provided. Data access may be limited.")
        
        # 初始化Tushare API
        try:
            import tushare as ts
            ts.set_token(self.api_key)
            self.api = ts.pro_api()
            logger.info("Tushare API initialized successfully")
        except ImportError:
            logger.error("Failed to import tushare. Please install it using: pip install tushare")
            self.api = None
        except Exception as e:
            logger.error(f"Failed to initialize Tushare API: {str(e)}")
            self.api = None
    
    def get_price_data(self, symbol: str, start_date: str, end_date: str, 
                    interval: str = 'daily') -> pd.DataFrame:
        """
        Get historical price data for a symbol using Tushare
        
        Args:
            symbol (str): Stock symbol (格式：000001.SZ)
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval ('daily', 'weekly', 'monthly')
            
        Returns:
            pd.DataFrame: DataFrame with price data
        """
        if self.api is None:
            logger.error("Tushare API is not initialized")
            return self._generate_mock_price_data(symbol, start_date, end_date, interval)
        
        # 缓存键
        cache_key = f"tushare_{symbol}_{start_date}_{end_date}_{interval}"
        
        # 尝试从缓存加载
        cache_hit, cached_data = self._load_from_cache(cache_key)
        if cache_hit:
            logger.debug(f"Using cached data for {symbol}")
            if isinstance(cached_data.get('data'), list) and 'columns' in cached_data:
                df = pd.DataFrame(cached_data['data'], columns=cached_data['columns'])
                # 恢复索引如果存在
                if cached_data.get('index'):
                    df.index = cached_data['index']
                return df
            return pd.DataFrame()
        
        try:
            # 遵守API速率限制
            self._respect_rate_limit('daily')
            
            # 转换日期格式 (YYYY-MM-DD -> YYYYMMDD)
            start_date_fmt = start_date.replace('-', '')
            end_date_fmt = end_date.replace('-', '')
            
            # 根据interval参数确定要调用的API
            if interval == 'daily':
                df = self.api.daily(ts_code=symbol, start_date=start_date_fmt, end_date=end_date_fmt)
            elif interval == 'weekly':
                df = self.api.weekly(ts_code=symbol, start_date=start_date_fmt, end_date=end_date_fmt)
            elif interval == 'monthly':
                df = self.api.monthly(ts_code=symbol, start_date=start_date_fmt, end_date=end_date_fmt)
            else:
                logger.warning(f"Unsupported interval: {interval}, using daily")
                df = self.api.daily(ts_code=symbol, start_date=start_date_fmt, end_date=end_date_fmt)
            
            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # 按交易日期排序
            df = df.sort_values('trade_date')
            
            # 重命名列以符合标准格式
            df = df.rename(columns={
                'trade_date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume',
                'amount': 'amount',
                'change': 'change',
                'pct_chg': 'pct_change'
            })
            
            # 缓存数据
            self._save_to_cache(cache_key, df)
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting price data for {symbol}: {str(e)}")
            # 发生错误时生成模拟数据
            return self._generate_mock_price_data(symbol, start_date, end_date, interval)
    
    def get_market_index(self, index_symbol: str, start_date: str, end_date: str,
                       interval: str = 'daily') -> pd.DataFrame:
        """
        Get market index data using Tushare
        
        Args:
            index_symbol (str): Index symbol (e.g. 000001.SH for Shanghai Composite)
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval ('daily', 'weekly', 'monthly')
            
        Returns:
            pd.DataFrame: DataFrame with index data
        """
        # 指数数据使用相同的API接口，只是使用不同的代码
        return self.get_price_data(index_symbol, start_date, end_date, interval)
    
    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Get fundamental data for a symbol using Tushare
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Dictionary of fundamental data
        """
        if self.api is None:
            logger.error("Tushare API is not initialized")
            return {}
        
        # 缓存键
        cache_key = f"tushare_fundamentals_{symbol}"
        
        # 尝试从缓存加载
        cache_hit, cached_data = self._load_from_cache(cache_key)
        if cache_hit and isinstance(cached_data.get('data'), dict):
            logger.debug(f"Using cached fundamental data for {symbol}")
            return cached_data['data']
        
        fundamentals = {}
        
        try:
            # 遵守API速率限制
            self._respect_rate_limit('fundamentals')
            
            # 获取公司基本信息
            basic_info = self.api.stock_company(ts_code=symbol)
            if not basic_info.empty:
                fundamentals['basic_info'] = basic_info.iloc[0].to_dict()
            
            # 获取最新财务指标
            self._respect_rate_limit('fundamentals')
            financial = self.api.fina_indicator(ts_code=symbol, period='')
            if not financial.empty:
                # 只取最新的财报数据
                fundamentals['financial'] = financial.iloc[0].to_dict()
            
            # 获取最新估值指标
            self._respect_rate_limit('fundamentals')
            valuation = self.api.daily_basic(ts_code=symbol)
            if not valuation.empty:
                fundamentals['valuation'] = valuation.iloc[0].to_dict()
            
            # 缓存数据
            self._save_to_cache(cache_key, fundamentals)
            
            return fundamentals
        
        except Exception as e:
            logger.error(f"Error getting fundamental data for {symbol}: {str(e)}")
            return {}
    
    def get_symbol_list(self, market: str = None) -> List[str]:
        """
        Get list of available symbols from Tushare
        
        Args:
            market (str, optional): Market filter, e.g. 'SSE' for Shanghai, 'SZSE' for Shenzhen
            
        Returns:
            list: List of symbols
        """
        if self.api is None:
            logger.error("Tushare API is not initialized")
            return []
        
        # 缓存键
        cache_key = f"tushare_symbols_{market or 'all'}"
        
        # 尝试从缓存加载
        cache_hit, cached_data = self._load_from_cache(cache_key)
        if cache_hit and isinstance(cached_data.get('data'), list):
            logger.debug(f"Using cached symbol list for {market or 'all'}")
            return cached_data['data']
        
        try:
            # 遵守API速率限制
            self._respect_rate_limit('stock_list')
            
            # 获取股票列表
            if market:
                df = self.api.stock_basic(exchange=market)
            else:
                df = self.api.stock_basic()
            
            if df is None or df.empty:
                logger.warning("No symbols returned")
                return []
            
            symbols = df['ts_code'].tolist()
            
            # 缓存数据
            self._save_to_cache(cache_key, symbols)
            
            return symbols
        
        except Exception as e:
            logger.error(f"Error getting symbol list: {str(e)}")
            return []
    
    def _generate_mock_price_data(self, symbol: str, start_date: str, end_date: str, 
                               interval: str = 'daily') -> pd.DataFrame:
        """
        生成模拟价格数据（当真实数据获取失败时使用）
        
        Args:
            symbol (str): 股票代码
            start_date (str): 开始日期
            end_date (str): 结束日期
            interval (str): 数据间隔
            
        Returns:
            pd.DataFrame: 模拟价格数据
        """
        logger.warning(f"Generating mock data for {symbol}")
        
        # 解析日期
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 计算交易日数量（简化计算，忽略周末和节假日）
        if interval == 'daily':
            days = (end - start).days + 1
        elif interval == 'weekly':
            days = ((end - start).days // 7) + 1
        elif interval == 'monthly':
            # 计算月份差异
            months = (end.year - start.year) * 12 + end.month - start.month + 1
            days = months
        else:
            days = (end - start).days + 1
        
        # 生成随机价格
        base_price = 50.0  # 初始价格
        volatility = 0.02  # 日波动率
        
        dates = []
        current = start
        
        # 生成交易日期
        if interval == 'daily':
            while current <= end:
                if current.weekday() < 5:  # 0-4表示周一至周五
                    dates.append(current)
                current += timedelta(days=1)
        elif interval == 'weekly':
            # 从星期一开始
            while current.weekday() != 0:
                current += timedelta(days=1)
            while current <= end:
                dates.append(current)
                current += timedelta(days=7)
        elif interval == 'monthly':
            while current <= end:
                dates.append(current)
                # 增加一个月
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
        
        # 生成价格数据
        data = []
        price = base_price
        
        for i, date in enumerate(dates):
            # 价格变动
            daily_return = np.random.normal(0, volatility)
            price *= (1 + daily_return)
            
            # 日内价格范围
            high = price * (1 + abs(np.random.normal(0, volatility/2)))
            low = price * (1 - abs(np.random.normal(0, volatility/2)))
            open_price = low + (high - low) * np.random.random()
            
            # 成交量 (模拟成交量，与价格相关)
            volume = int(np.random.normal(1000000, 500000) * (1 + abs(daily_return) * 10))
            volume = max(100, volume)  # 确保成交量为正
            
            # 添加到数据列表
            data.append({
                'date': date.strftime('%Y%m%d'),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(price, 2),
                'volume': volume,
                'amount': round(price * volume, 2),
                'change': round(price - open_price, 2),
                'pct_change': round((price / open_price - 1) * 100, 2)
            })
        
        return pd.DataFrame(data)

class AkshareDataSource(DataSource):
    """
    Akshare data source implementation
    Uses Akshare API to get real market data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Akshare data source
        
        Args:
            config (dict): Configuration for the data source
        """
        super().__init__(config)
        self.provider = DataProvider.AKSHARE
        
        # 初始化Akshare API
        try:
            import akshare as ak
            self.ak = ak
            logger.info("Akshare API initialized successfully")
        except ImportError:
            logger.error("Failed to import akshare. Please install it using: pip install akshare")
            self.ak = None
        except Exception as e:
            logger.error(f"Failed to initialize Akshare API: {str(e)}")
            self.ak = None
    
    def get_price_data(self, symbol: str, start_date: str, end_date: str, 
                    interval: str = 'daily') -> pd.DataFrame:
        """
        Get historical price data for a symbol using Akshare
        
        Args:
            symbol (str): Stock symbol (格式：000001 或 000001.SZ)
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval ('daily', 'weekly', 'monthly')
            
        Returns:
            pd.DataFrame: DataFrame with price data
        """
        if self.ak is None:
            logger.error("Akshare API is not initialized")
            return pd.DataFrame()
        
        # 提取股票代码（去掉市场后缀）
        if '.' in symbol:
            stock_code = symbol.split('.')[0]
        else:
            stock_code = symbol
        
        # 缓存键
        cache_key = f"akshare_{symbol}_{start_date}_{end_date}_{interval}"
        
        # 尝试从缓存加载
        cache_hit, cached_data = self._load_from_cache(cache_key)
        if cache_hit:
            logger.debug(f"Using cached data for {symbol}")
            if isinstance(cached_data.get('data'), list) and 'columns' in cached_data:
                df = pd.DataFrame(cached_data['data'], columns=cached_data['columns'])
                # 恢复索引如果存在
                if cached_data.get('index'):
                    df.index = cached_data['index']
                return df
            return pd.DataFrame()
        
        try:
            # 遵守API速率限制
            self._respect_rate_limit('stock_hist')
            
            # 根据interval参数确定要调用的API
            if interval == 'daily':
                df = self.ak.stock_zh_a_hist(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="qfq")
            elif interval == 'weekly':
                df = self.ak.stock_zh_a_hist_weekly(symbol=stock_code, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''), adjust="qfq")
            elif interval == 'monthly':
                df = self.ak.stock_zh_a_hist_monthly(symbol=stock_code, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''), adjust="qfq")
            else:
                logger.warning(f"Unsupported interval: {interval}, using daily")
                df = self.ak.stock_zh_a_hist(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="qfq")
            
            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # 重命名列以符合标准格式
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change',
                '换手率': 'turnover'
            })
            
            # 确保日期格式标准化
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
            
            # 排序
            df = df.sort_values('date')
            
            # 缓存数据
            self._save_to_cache(cache_key, df)
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting price data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_market_index(self, index_symbol: str, start_date: str, end_date: str,
                       interval: str = 'daily') -> pd.DataFrame:
        """
        Get market index data using Akshare
        
        Args:
            index_symbol (str): Index symbol (e.g. 000001 for SSE Composite)
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval ('daily', 'weekly', 'monthly')
            
        Returns:
            pd.DataFrame: DataFrame with index data
        """
        if self.ak is None:
            logger.error("Akshare API is not initialized")
            return pd.DataFrame()
        
        # 提取指数代码（去掉市场后缀）
        if '.' in index_symbol:
            index_code = index_symbol.split('.')[0]
        else:
            index_code = index_symbol
        
        # 缓存键
        cache_key = f"akshare_index_{index_symbol}_{start_date}_{end_date}_{interval}"
        
        # 尝试从缓存加载
        cache_hit, cached_data = self._load_from_cache(cache_key)
        if cache_hit:
            logger.debug(f"Using cached data for {index_symbol}")
            if isinstance(cached_data.get('data'), list) and 'columns' in cached_data:
                df = pd.DataFrame(cached_data['data'], columns=cached_data['columns'])
                # 恢复索引如果存在
                if cached_data.get('index'):
                    df.index = cached_data['index']
                return df
            return pd.DataFrame()
        
        try:
            # 遵守API速率限制
            self._respect_rate_limit('index_hist')
            
            # 确定指数所属市场
            if index_code.startswith('0'):
                # 上证指数
                df = self.ak.stock_zh_index_daily(symbol=f"sh{index_code}")
            elif index_code.startswith('3'):
                # 深证指数
                df = self.ak.stock_zh_index_daily(symbol=f"sz{index_code}")
            else:
                logger.warning(f"Unknown index market for {index_symbol}, assuming Shanghai")
                df = self.ak.stock_zh_index_daily(symbol=f"sh{index_code}")
            
            # 按日期筛选
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            if df is None or df.empty:
                logger.warning(f"No data returned for index {index_symbol}")
                return pd.DataFrame()
            
            # 重命名列以符合标准格式
            df = df.rename(columns={
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'amount': 'amount',
                'change': 'change',
            })
            
            # 确保日期格式标准化
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
            
            # 排序
            df = df.sort_values('date')
            
            # 缓存数据
            self._save_to_cache(cache_key, df)
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting index data for {index_symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Get fundamental data for a symbol using Akshare
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Dictionary of fundamental data
        """
        if self.ak is None:
            logger.error("Akshare API is not initialized")
            return {}
        
        # 提取股票代码（去掉市场后缀）
        if '.' in symbol:
            stock_code = symbol.split('.')[0]
        else:
            stock_code = symbol
        
        # 缓存键
        cache_key = f"akshare_fundamentals_{symbol}"
        
        # 尝试从缓存加载
        cache_hit, cached_data = self._load_from_cache(cache_key)
        if cache_hit and isinstance(cached_data.get('data'), dict):
            logger.debug(f"Using cached fundamental data for {symbol}")
            return cached_data['data']
        
        fundamentals = {}
        
        try:
            # 遵守API速率限制
            self._respect_rate_limit('fundamentals')
            
            # 获取公司基本信息
            try:
                stock_info = self.ak.stock_individual_info_em(symbol=stock_code)
                if not stock_info.empty:
                    info_dict = {}
                    for _, row in stock_info.iterrows():
                        info_dict[row['item']] = row['value']
                    fundamentals['basic_info'] = info_dict
            except Exception as e:
                logger.warning(f"Error getting basic info for {symbol}: {str(e)}")
            
            # 获取最新财务指标
            try:
                self._respect_rate_limit('fundamentals')
                financial = self.ak.stock_financial_analysis_indicator(symbol=stock_code)
                if not financial.empty:
                    # 只取最新的财报数据
                    fundamentals['financial'] = financial.iloc[0].to_dict()
            except Exception as e:
                logger.warning(f"Error getting financial data for {symbol}: {str(e)}")
            
            # 获取最新估值指标
            try:
                self._respect_rate_limit('fundamentals')
                valuation = self.ak.stock_zh_valuation_baidu(symbol=stock_code)
                if not valuation.empty:
                    fundamentals['valuation'] = valuation.iloc[0].to_dict()
            except Exception as e:
                logger.warning(f"Error getting valuation data for {symbol}: {str(e)}")
            
            # 缓存数据
            self._save_to_cache(cache_key, fundamentals)
            
            return fundamentals
        
        except Exception as e:
            logger.error(f"Error getting fundamental data for {symbol}: {str(e)}")
            return {}
    
    def get_symbol_list(self, market: str = None) -> List[str]:
        """
        Get list of available symbols from Akshare
        
        Args:
            market (str, optional): Market filter, e.g. 'sh' for Shanghai, 'sz' for Shenzhen
            
        Returns:
            list: List of symbols
        """
        if self.ak is None:
            logger.error("Akshare API is not initialized")
            return []
        
        # 缓存键
        cache_key = f"akshare_symbols_{market or 'all'}"
        
        # 尝试从缓存加载
        cache_hit, cached_data = self._load_from_cache(cache_key)
        if cache_hit and isinstance(cached_data.get('data'), list):
            logger.debug(f"Using cached symbol list for {market or 'all'}")
            return cached_data['data']
        
        try:
            # 遵守API速率限制
            self._respect_rate_limit('stock_list')
            
            # 获取A股列表
            df = self.ak.stock_zh_a_spot_em()
            
            if df is None or df.empty:
                logger.warning("No symbols returned")
                return []
            
            # 应用市场过滤
            if market:
                if market.lower() == 'sh' or market.lower() == 'sse':
                    df = df[df['代码'].str.startswith(('6', '9'))]
                elif market.lower() == 'sz' or market.lower() == 'szse':
                    df = df[df['代码'].str.startswith(('0', '3'))]
            
            # 根据代码确定市场后缀并生成标准格式的股票代码
            symbols = []
            for code in df['代码']:
                if code.startswith(('6', '9')):
                    symbols.append(f"{code}.SH")
                elif code.startswith(('0', '3')):
                    symbols.append(f"{code}.SZ")
                else:
                    # 未知市场，默认为上海
                    symbols.append(f"{code}.SH")
            
            # 缓存数据
            self._save_to_cache(cache_key, symbols)
            
            return symbols
        
        except Exception as e:
            logger.error(f"Error getting symbol list: {str(e)}")
            return []

def create_data_source(config: Dict[str, Any]) -> DataSource:
    """
    Factory function to create a data source
    
    Args:
        config (dict): Configuration for the data source
        
    Returns:
        DataSource: Data source instance
    """
    provider = config.get('provider', 'mock')
    
    if provider == 'tushare' or provider == DataProvider.TUSHARE.value:
        return TushareDataSource(config)
    elif provider == 'akshare' or provider == DataProvider.AKSHARE.value:
        return AkshareDataSource(config)
    elif provider == 'mock' or provider == DataProvider.MOCK.value:
        return MockDataSource(config)
    # 添加对其他数据源的支持
    else:
        logger.warning(f"Unknown data provider: {provider}, falling back to mock data")
    return MockDataSource(config) 