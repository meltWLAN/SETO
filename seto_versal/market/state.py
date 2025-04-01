#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Market state module for SETO-Versal
Tracks market conditions and provides data to agents
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
import json
import os
import random

logger = logging.getLogger(__name__)

class MarketState:
    """
    Market state class for tracking market conditions
    
    Provides data, indicators, and market state information to agents
    """
    
    def __init__(self, mode='paper', data_dir='data/market', universe='hs300'):
        """
        Initialize the market state
        
        Args:
            mode (str): Trading mode ('paper', 'backtest', or 'real')
            data_dir (str): Directory containing market data
            universe (str): Name of the universe to load
        """
        self.mode = mode
        self.data_dir = data_dir
        self.symbols = self._load_universe(universe)
        self.benchmark = '000300.SH'  # Default to CSI 300
        self.market_hours = {
            'open': '09:30:00',
            'close': '15:00:00'
        }
        self.trading_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        self.cache = {}
        self.current_data = {}
        self.history = []
        self.fundamentals = {}
        self.should_evolve_flag = False
        
        # Initialize data and fundamentals
        self.initialize_data()
        self.initialize_fundamentals()
        
        logger.info(f"Initialized MarketState with {len(self.symbols)} symbols in {mode} mode")
        
    def initialize_data(self):
        """Initialize market data"""
        try:
            if self.mode == 'paper':
                # Use simulated data for paper trading
                self._initialize_paper_data()
            elif self.mode == 'backtest':
                # Load historical data for backtesting
                self._initialize_backtest_data()
            else:
                # Use real market data
                self._initialize_real_data()
                
        except Exception as e:
            logger.error(f"Error initializing data: {e}")
            
    def _initialize_paper_data(self):
        """Initialize paper trading data"""
        try:
            # 定义几种常见的股票价格范围和行业分类
            price_ranges = [
                (5, 15),     # 小盘股
                (15, 30),    # 中盘股
                (30, 60),    # 大盘股
                (60, 120),   # 蓝筹股
                (120, 500)   # 白马股
            ]
            
            sectors = [
                "金融",
                "科技",
                "消费",
                "医药",
                "能源",
                "制造",
                "房地产"
            ]
            
            # 为每个股票生成真实的初始数据
            for i, symbol in enumerate(self.symbols):
                # 确定这个股票属于哪个价格范围
                price_range_idx = i % len(price_ranges)
                min_price, max_price = price_ranges[price_range_idx]
                
                # 生成基础价格
                base_price = round(random.uniform(min_price, max_price), 2)
                
                # 根据股票代码判断交易所
                if '.SH' in symbol:
                    # 上证股票偏向传统行业
                    sector_idx = i % 4  # 主要分布在金融、能源、制造、房地产
                else:
                    # 深证股票偏向成长行业
                    sector_idx = (i % 3) + 1  # 主要分布在科技、消费、医药
                
                sector_name = sectors[sector_idx]
                
                # 根据行业特性调整成交量
                if sector_name in ["科技", "医药"]:
                    volume_factor = random.uniform(1.2, 2.0)  # 成长股成交活跃
                elif sector_name in ["金融", "能源"]:
                    volume_factor = random.uniform(0.8, 1.5)  # 价值股相对稳定
                else:
                    volume_factor = random.uniform(0.6, 1.8)
                
                # 生成昨日收盘价，有一定概率与基础价格不同
                if random.random() < 0.7:  # 70%概率有变动
                    prev_close = round(base_price * random.uniform(0.97, 1.03), 2)
                else:
                    prev_close = base_price
                
                # 生成今日开盘价
                open_price = round(prev_close * random.uniform(0.99, 1.01), 2)
                
                # 生成最高价和最低价
                price_range = prev_close * random.uniform(0.02, 0.04)  # 2%-4%的日内波动
                high_price = round(max(base_price, open_price) + price_range * random.uniform(0.3, 1.0), 2)
                low_price = round(min(base_price, open_price) - price_range * random.uniform(0.3, 1.0), 2)
                low_price = max(low_price, 0.1)  # 确保最低价大于0
                
                # 生成成交量，与价格成正比
                base_volume = int(base_price * 100000 * volume_factor)
                volume = int(base_volume * random.uniform(0.8, 1.2))
                
                # 创建数据
                self.current_data[symbol] = {
                    'symbol': symbol,
                    'price': base_price,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'prev_close': prev_close,
                    'volume': volume,
                    'sector': sector_name,  # 添加行业信息
                    'timestamp': datetime.now()
                }
                
            logger.info("Initialized paper trading data with realistic price distributions")
        
        except Exception as e:
            logger.error(f"Error initializing paper data: {e}")
            
    def _initialize_real_data(self):
        """Initialize real market data"""
        try:
            # Load latest market data for each symbol
            for symbol in self.symbols:
                try:
                    # Try to load cached data first
                    data_file = os.path.join(self.data_dir, f"{symbol}_latest.json")
                    if os.path.exists(data_file):
                        with open(data_file, 'r') as f:
                            self.current_data[symbol] = json.load(f)
                    else:
                        # Use default values if no data available
                        self.current_data[symbol] = {
                            'symbol': symbol,
                            'price': 0.0,
                            'open': 0.0,
                            'high': 0.0,
                            'low': 0.0,
                            'volume': 0,
                            'timestamp': datetime.now()
                        }
                
                except Exception as e:
                    logger.error(f"Error loading data for {symbol}: {e}")
                    
            logger.info("Initialized real market data")
            
        except Exception as e:
            logger.error(f"Error initializing real data: {e}")
            
    def get_state(self):
        """Get current market state"""
        return self
        
    def get_market_data(self):
        """Get current market data"""
        return self.current_data
        
    def get_trade_history(self):
        """Get trade history"""
        # This would be implemented based on your trade tracking system
        return []
        
    def get_performance_metrics(self):
        """Get performance metrics"""
        # This would be implemented based on your performance tracking system
        return {
            'pnl': 0.0,
            'win_rate': 0.0,
            'total_trades': 0
        }
        
    def update(self):
        """Update market state"""
        try:
            if self.mode == 'backtest':
                success = self._update_backtest_data()
            elif self.mode == 'paper':
                success = self._update_paper_data()
            else:
                success = self._update_real_data()
                
            if success:
                self.last_update_time = datetime.now()
                
            return success
            
        except Exception as e:
            logger.error(f"Error updating market state: {e}")
            return False
            
    def _update_paper_data(self):
        """Update paper trading data"""
        try:
            for symbol in self.symbols:
                current = self.current_data[symbol]
                base_price = current['price']
                
                # 添加一些行业特性和个股特性
                symbol_index = self.symbols.index(symbol) % 5
                sector_bias = [-0.005, 0.01, 0.0, -0.01, 0.015][symbol_index]  # 不同行业的偏向性
                
                # 基于当前时间的市场整体趋势
                hour = datetime.now().hour
                minute = datetime.now().minute
                market_trend = 0.0
                
                # 早盘通常波动较大
                if hour < 11:
                    market_trend = 0.005 * np.sin(minute / 10)
                # 午盘相对平稳
                elif 11 <= hour < 14:
                    market_trend = 0.002 * np.sin(minute / 15)
                # 尾盘可能出现方向性
                else:
                    market_trend = 0.008 * np.sin(minute / 8)
                
                # 个股波动性
                vol_factor = 1.0 + (symbol_index - 2) * 0.2  # 不同股票波动率不同
                
                # 生成一个更真实的价格变动
                # 基于随机波动 + 市场趋势 + 行业特性 + 个股特性
                change = (
                    random.normalvariate(-0.001, 0.01) * vol_factor +  # 随机波动
                    market_trend +                                     # 市场趋势
                    sector_bias                                       # 行业特性
                )
                
                # 价格变动限制，模拟涨跌停限制
                change = max(min(change, 0.1), -0.1)  # 限制在±10%
                
                new_price = base_price * (1 + change)
                
                # 确保价格不会变为负数或过小
                new_price = max(new_price, 0.1)
                
                # 保留两位小数，模拟实际价格
                new_price = round(new_price, 2)
                
                # 成交量与价格变动正相关
                volume_change = 1 + abs(change) * 5  # 价格变动越大，成交量变化越大
                new_volume = int(current['volume'] * volume_change * random.uniform(0.8, 1.2))
                
                # 确保成交量在合理范围内
                new_volume = max(new_volume, 100000)  # 最小成交量
                new_volume = min(new_volume, 100000000)  # 最大成交量
                
                # 更新数据
                self.current_data[symbol] = {
                    'symbol': symbol,
                    'price': new_price,
                    'open': current.get('open', base_price),
                    'high': max(current.get('high', base_price), new_price),
                    'low': min(current.get('low', base_price), new_price),
                    'volume': new_volume,
                    'prev_close': current.get('prev_close', base_price * 0.99),  # 添加昨收价
                    'timestamp': datetime.now()
                }
            
            return True
        
        except Exception as e:
            logger.error(f"Error updating paper data: {e}")
            return False
            
    def _update_real_data(self):
        """Update real market data"""
        try:
            # This would be implemented to fetch real market data
            # For now, just return True
            return True
            
        except Exception as e:
            logger.error(f"Error updating real data: {e}")
            return False
            
    def is_evolution_time(self):
        """Check if it's time for evolution"""
        # For paper trading, use random evolution
        if self.mode in ['backtest', 'test', 'paper']:
            return random.random() < 0.1
            
        # For live trading, check market hours
        if not self.trading_day:
            return False
        
        try:
            now = datetime.now().time()
            
            # Get close time
            close_time_str = self.market_hours.get('close', '15:00')
            try:
                close_time = datetime.strptime(close_time_str, '%H:%M').time()
            except ValueError:
                close_time = datetime.strptime('15:00', '%H:%M').time()
                
            # Evolution window is 30 minutes after close
            evolution_time = (datetime.combine(datetime.today(), close_time) + timedelta(minutes=30)).time()
            
            return close_time <= now <= evolution_time
            
        except Exception as e:
            logger.error(f"Error checking evolution time: {e}")
            return False
    
    def is_market_open(self):
        """
        检查当前市场是否开放交易
        
        Returns:
            bool: 市场是否开放
        """
        try:
            # 获取当前时间
            current_time = datetime.now().time()
            current_day = datetime.now().strftime('%A')
            
            # 如果是周末，市场关闭
            if current_day not in self.trading_days:
                return False
                
            # 初始化交易日标志（如果尚未设置）
            if not hasattr(self, 'trading_day'):
                self.trading_day = current_day
            
            # 如果是测试模式，默认市场开放
            if self.mode == 'test':
                return True
                
            # 获取市场开盘和收盘时间
            open_time = datetime.strptime(self.market_hours.get('open', '09:30:00'), '%H:%M:%S').time()
            close_time = datetime.strptime(self.market_hours.get('close', '15:00:00'), '%H:%M:%S').time()
            
            # 判断当前时间是否在交易时段
            if open_time <= current_time <= close_time:
                return True
                
            # 不在交易时段
            return False
        except Exception as e:
            logger.error(f"Error checking market open status: {e}")
            # 默认返回市场关闭
            return False
        
    def _load_universe(self, universe):
        """
        Load symbols for a predefined universe
        
        Args:
            universe (str): Name of the universe to load
            
        Returns:
            list: List of symbols in the universe
        """
        logger.info(f"Loading universe: {universe}")
        
        try:
            # Check if universe file exists
            universe_file = os.path.join(self.data_dir, f"{universe}.json")
            if os.path.exists(universe_file):
                try:
                    with open(universe_file, 'r') as f:
                        symbols = json.load(f)
                    logger.info(f"Loaded {len(symbols)} symbols from universe file: {universe_file}")
                    return symbols
                except Exception as e:
                    logger.error(f"Error loading universe file: {e}")
            
            # Check if stock_list.json exists in data directory
            stock_list_file = os.path.join(self.data_dir, "stock_list.json")
            if os.path.exists(stock_list_file):
                try:
                    with open(stock_list_file, 'r') as f:
                        symbols = json.load(f)
                    logger.info(f"Loaded {len(symbols)} symbols from stock list file")
                    return symbols
                except Exception as e:
                    logger.error(f"Error loading stock list file: {e}")
            
            # Use any available stocks from data/market/price directory
            try:
                price_dir = os.path.join(self.data_dir, 'price')
                if os.path.exists(price_dir):
                    symbols = [d for d in os.listdir(price_dir) if os.path.isdir(os.path.join(price_dir, d))]
                    if symbols:
                        logger.info(f"Found {len(symbols)} symbols in price directory")
                        return symbols
            except Exception as e:
                logger.error(f"Error listing price directory: {e}")
            
            # If no files found, provide default universes based on the universe name
            logger.warning(f"No existing symbol files found, using built-in {universe} universe")
            
            if universe == 'hs300':
                # Return top 60 stocks from HS300 (just a sample for the system)
                return [
                    # 金融
                    '600036.SH', '601318.SH', '601398.SH', '601288.SH', '600030.SH',
                    '601166.SH', '601328.SH', '601601.SH', '601988.SH', '600016.SH',
                    '601169.SH', '601818.SH', '601628.SH', '601336.SH', '601688.SH',
                    # 消费
                    '600519.SH', '000858.SZ', '002304.SZ', '600887.SH', '600276.SH',
                    '000333.SZ', '000651.SZ', '601888.SH', '603288.SH', '000568.SZ',
                    # 科技
                    '000725.SZ', '002415.SZ', '002594.SZ', '600703.SH', '002230.SZ',
                    '600050.SH', '002475.SZ', '002241.SZ', '000063.SZ', '002027.SZ',
                    # 工业
                    '601766.SH', '600031.SH', '600009.SH', '601857.SH', '600019.SH',
                    '601668.SH', '601800.SH', '601390.SH', '601186.SH', '600028.SH',
                    # 医药
                    '600196.SH', '000538.SZ', '002252.SZ', '600085.SH', '600518.SH',
                    # 地产
                    '000002.SZ', '600048.SH', '001979.SZ', '600606.SH', '000069.SZ',
                    # 交通运输
                    '601111.SH', '600029.SH', '601021.SH', '601333.SH', '600115.SH',
                    # 农业
                    '000998.SZ', '600598.SH', '002714.SZ', '600108.SH', '000876.SZ'
                ]
            elif universe == 'test' or universe == 'demo':
                # Return a small set for testing
                return ['000001.SZ', '600000.SH', '600036.SH', '601318.SH', '600519.SH', 
                        '000858.SZ', '600887.SH', '601888.SH', '000651.SZ', '002415.SZ']
            elif universe == 'minimal':
                # Return a minimal set of 5 stocks for quick testing
                return ['000001.SZ', '600519.SH', '601318.SH', '000858.SZ', '600276.SH']
            elif universe == 'any' or universe == 'default':
                # Return a default test set
                logger.info("Using default stock set")
                return ['000001.SZ', '600519.SH', '601318.SH', '000858.SZ', '600276.SH']
            else:
                # Unknown universe, return minimal set
                logger.warning(f"Unknown universe '{universe}', using minimal set")
                return ['000001.SZ', '600519.SH', '601318.SH', '000858.SZ', '600276.SH']
            
        except Exception as e:
            logger.error(f"Error loading universe: {e}")
            # Return minimal set in case of error
            return ['000001.SZ', '600519.SH', '601318.SH', '000858.SZ', '600276.SH'] 
    
    def get_price_previous(self, symbol):
        """
        Get previous day's price for a symbol
        
        Args:
            symbol (str): Symbol code
            
        Returns:
            float: Previous day's price
        """
        # Default implementation returns 99% of current price as a fallback
        current_price = self.get_price(symbol)
        if current_price is None:
            return None
        return current_price * 0.99
        
    def get_price_history(self, symbol, lookback_period=20):
        """
        Get historical prices for a symbol
        
        Args:
            symbol (str): Symbol code
            lookback_period (int): Number of periods to look back
            
        Returns:
            list: List of historical prices
        """
        # 对于回测模式，使用回测专用方法
        if self.mode == 'backtest' and hasattr(self, 'historical_data'):
            return self.get_price_history_backtest(symbol, lookback_period)
        
        # 默认实现生成随机样本数据
        import numpy as np
        current_price = self.get_price(symbol)
        if current_price is None:
            return None
            
        # Generate random price history with a slight upward trend
        prices = [current_price]
        for i in range(lookback_period - 1):
            # Random change between -2% and +3%
            change = np.random.normal(0.001, 0.015)
            prev_price = prices[-1] * (1 - change)  # Going backward in time
            prices.append(prev_price)
            
        # Reverse to get chronological order
        return prices[::-1]

    def get_price(self, symbol):
        """
        获取股票当前价格
        
        Args:
            symbol (str): 股票代码
            
        Returns:
            float: 当前价格
        """
        try:
            # 从数据字典中获取价格
            if symbol in self.current_data:
                return self.current_data[symbol].get('price', 0.0)
            return None
        except Exception as e:
            logger.error(f"获取股票{symbol}价格失败: {e}")
            return None

    def get_volume(self, symbol):
        """
        获取股票当前成交量
        
        Args:
            symbol (str): 股票代码
            
        Returns:
            int: 成交量
        """
        try:
            # 从数据字典中获取成交量
            if symbol in self.current_data:
                return self.current_data[symbol].get('volume', 0)
            return 0
        except Exception as e:
            logger.error(f"获取股票{symbol}成交量失败: {e}")
            return 0
            
    def get_tradable_symbols(self):
        """
        Get list of currently tradable symbols
        
        Returns:
            list: List of symbol strings that are currently tradable
        """
        if not hasattr(self, 'symbols'):
            logger.warning("MarketState has no symbols attribute")
            return []
        
        # 在测试模式下返回所有股票
        if self.mode == 'test':
            return list(self.symbols)
        
        # 在实盘模式下，检查市场是否开放
        if not self.is_market_open():
            logger.info("Market is closed, no tradable symbols")
            return []
        
        # 检查每个股票是否有有效数据
        tradable = []
        for symbol in self.symbols:
            if symbol in self.current_data and self.current_data[symbol].get('price', 0) > 0:
                tradable.append(symbol)
            
        return tradable
    
    def get_market_summary(self):
        """
        获取市场概况摘要
        
        Returns:
            dict: 市场概况数据
        """
        try:
            # 计算市场整体指标
            total_symbols = len(self.current_data)
            if total_symbols == 0:
                return {
                    'up_count': 0,
                    'down_count': 0,
                    'flat_count': 0,
                    'avg_change': 0.0,
                    'total_volume': 0,
                    'market_status': '休市' if not self.is_market_open() else '开盘'
                }
                
            # 统计涨跌家数和平均涨跌幅
            up_count = 0
            down_count = 0
            flat_count = 0
            total_change = 0.0
            total_volume = 0
            
            for symbol, data in self.current_data.items():
                price = data.get('price', 0.0)
                prev_price = data.get('prev_close', self.get_price_previous(symbol))
                
                if price > prev_price:
                    up_count += 1
                elif price < prev_price:
                    down_count += 1
                else:
                    flat_count += 1
                    
                if prev_price and prev_price > 0:
                    change_pct = (price - prev_price) / prev_price * 100
                    total_change += change_pct
                    
                total_volume += data.get('volume', 0)
                
            avg_change = total_change / total_symbols if total_symbols > 0 else 0.0
            
            return {
                'up_count': up_count,
                'down_count': down_count,
                'flat_count': flat_count,
                'avg_change': avg_change,
                'total_volume': total_volume,
                'market_status': '休市' if not self.is_market_open() else '开盘'
            }
        except Exception as e:
            logger.error(f"获取市场概况失败: {e}")
            return {
                'up_count': 0,
                'down_count': 0,
                'flat_count': 0,
                'avg_change': 0.0,
                'total_volume': 0,
                'market_status': '未知'
            } 

    def get_trading_signals(self):
        """
        获取当前的交易信号
        
        Returns:
            dict: 包含买入和卖出建议的字典
        """
        try:
            # 初始化结果字典
            signals = {
                'buy': [],
                'sell': []
            }
            
            # 如果是回测或模拟模式，生成更有现实意义的信号
            if self.mode in ['backtest', 'paper']:
                # 计算一些简单指标
                signal_candidates = {}
                
                for symbol in self.symbols:
                    if symbol not in self.current_data:
                        continue
                        
                    data = self.current_data[symbol]
                    price = data.get('price', 0)
                    prev_close = data.get('prev_close', 0)
                    
                    if not price or not prev_close:
                        continue
                    
                    # 计算涨跌幅
                    change_pct = (price - prev_close) / prev_close * 100
                    
                    # 获取价格历史
                    price_history = self.get_price_history(symbol, 10)
                    if not price_history or len(price_history) < 10:
                        continue
                    
                    # 计算简单移动平均线
                    ma5 = sum(price_history[-5:]) / 5
                    ma10 = sum(price_history) / len(price_history)
                    
                    # 判断趋势方向
                    trend = "up" if ma5 > ma10 else "down"
                    
                    # 计算波动率
                    prices = np.array(price_history)
                    volatility = np.std(prices) / np.mean(prices)
                    
                    # 计算价格突破
                    highest = max(price_history[:-1])
                    lowest = min(price_history[:-1])
                    
                    signal_candidates[symbol] = {
                        'price': price,
                        'change_pct': change_pct,
                        'ma5': ma5,
                        'ma10': ma10,
                        'trend': trend,
                        'volatility': volatility,
                        'highest': highest,
                        'lowest': lowest
                    }
                
                # 生成买入信号
                buy_candidates = []
                for symbol, metrics in signal_candidates.items():
                    # 金叉或突破前期高点
                    if (metrics['trend'] == "up" and metrics['ma5'] > metrics['ma10'] and 
                        metrics['price'] > metrics['highest'] and metrics['volatility'] < 0.1):
                        buy_candidates.append((symbol, 0.85, "突破阻力位，伴随均线金叉"))
                        
                    # 超跌反弹
                    elif (metrics['change_pct'] < -5 and metrics['price'] < metrics['lowest'] * 1.05 and
                         metrics['volatility'] > 0.1):
                        buy_candidates.append((symbol, 0.75, "超跌反弹，风险收益比良好"))
                        
                    # 强势上涨
                    elif (metrics['change_pct'] > 3 and metrics['trend'] == "up" and 
                          metrics['price'] > metrics['ma5'] > metrics['ma10']):
                        buy_candidates.append((symbol, 0.7, "强势上涨，动能持续增强"))
                
                # 生成卖出信号
                sell_candidates = []
                for symbol, metrics in signal_candidates.items():
                    # 死叉或跌破支撑
                    if (metrics['trend'] == "down" and metrics['ma5'] < metrics['ma10'] and 
                        metrics['price'] < metrics['lowest']):
                        sell_candidates.append((symbol, 0.8, "跌破支撑位，伴随均线死叉"))
                        
                    # 超涨回落
                    elif (metrics['change_pct'] > 7 and metrics['price'] > metrics['highest'] * 1.05 and
                         metrics['volatility'] > 0.15):
                        sell_candidates.append((symbol, 0.78, "超涨风险，建议适当获利了结"))
                        
                    # 弱势下跌
                    elif (metrics['change_pct'] < -2 and metrics['trend'] == "down" and 
                          metrics['price'] < metrics['ma5'] < metrics['ma10']):
                        sell_candidates.append((symbol, 0.72, "弱势下跌，避免深度调整"))
                
                # 随机选择一些买入和卖出信号
                import random
                
                # 确保信号数量合理
                buy_count = min(len(buy_candidates), 5)
                sell_count = min(len(sell_candidates), 5)
                
                if buy_candidates:
                    selected_buys = random.sample(buy_candidates, buy_count) if buy_count > 0 else []
                    for symbol, confidence, reason in selected_buys:
                        signals['buy'].append({
                            'symbol': symbol,
                            'action': 'buy',
                            'price': self.current_data[symbol]['price'],
                            'confidence': confidence,
                            'reason': reason,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                
                if sell_candidates:
                    selected_sells = random.sample(sell_candidates, sell_count) if sell_count > 0 else []
                    for symbol, confidence, reason in selected_sells:
                        signals['sell'].append({
                            'symbol': symbol,
                            'action': 'sell',
                            'price': self.current_data[symbol]['price'],
                            'confidence': confidence,
                            'reason': reason,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                
                # 如果没有足够的信号，添加一些随机信号
                if len(signals['buy']) < 3 or len(signals['sell']) < 3:
                    # 选择一些随机股票
                    available_symbols = [s for s in self.symbols if s in self.current_data]
                    if len(available_symbols) > 5:
                        random_symbols = random.sample(available_symbols, 5)
                        
                        for symbol in random_symbols:
                            # 随机决定是买入还是卖出信号
                            signal_type = random.choice(['buy', 'sell'])
                            
                            # 如果该类型信号不足3个，则添加
                            if len(signals[signal_type]) < 3:
                                # 获取当前价格
                                price = self.current_data[symbol].get('price', 0.0)
                                
                                # 生成随机置信度
                                confidence = random.uniform(0.6, 0.7)  # 较低的置信度表示随机性
                                
                                # 生成随机理由
                                if signal_type == 'buy':
                                    reasons = [
                                        "技术形态有改善迹象",
                                        "市场情绪转为积极",
                                        "短期超跌，具备反弹机会",
                                        "行业政策面有利好预期",
                                        "基本面存在低估可能"
                                    ]
                                else:
                                    reasons = [
                                        "短期技术面转弱",
                                        "压力位阻力明显",
                                        "获利回吐压力增大",
                                        "市场风险偏好下降",
                                        "行业竞争压力加剧"
                                    ]
                                
                                reason = random.choice(reasons)
                                
                                # 创建信号
                                signals[signal_type].append({
                                    'symbol': symbol,
                                    'action': signal_type,
                                    'price': price,
                                    'confidence': confidence,
                                    'reason': reason,
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                })
            else:
                # 实盘模式：如果有连接到外部数据源，从数据源获取信号
                # 这里可以添加与实际数据源连接的代码
                pass
                
            return signals
            
        except Exception as e:
            logger.error(f"获取交易信号失败: {e}")
            return {'buy': [], 'sell': []} 

    def _initialize_backtest_data(self):
        """初始化回测数据"""
        try:
            # 获取回测配置
            backtest_config = self.mode_config.get('backtest', {})
            self.start_date = backtest_config.get('start_date')
            self.end_date = backtest_config.get('end_date')
            
            # 如果没有设置日期，使用默认值
            if not self.start_date:
                # 默认回测最近1年的数据
                self.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if not self.end_date:
                self.end_date = datetime.now().strftime('%Y-%m-%d')
                
            # 将字符串转换为日期对象
            if isinstance(self.start_date, str):
                self.start_date = datetime.strptime(self.start_date, '%Y-%m-%d').date()
            if isinstance(self.end_date, str):
                self.end_date = datetime.strptime(self.end_date, '%Y-%m-%d').date()
                
            # 设置当前日期为开始日期
            self.current_date = self.start_date
            
            # 初始化历史数据结构
            self.historical_data = {}
            
            # 尝试加载历史数据
            self._load_historical_data()
            
            # 初始化当前数据
            self._update_backtest_data(set_initial=True)
            
            logger.info(f"初始化回测数据: 从{self.start_date}到{self.end_date}")
            
        except Exception as e:
            logger.error(f"初始化回测数据失败: {e}")
            # 如果失败，创建一些模拟数据
            self._generate_simulated_historical_data()
    
    def _load_historical_data(self):
        """加载历史数据"""
        try:
            # 检查历史数据目录
            historical_data_dir = os.path.join(self.data_dir, 'historical')
            if not os.path.exists(historical_data_dir):
                logger.warning(f"历史数据目录不存在: {historical_data_dir}")
                self._generate_simulated_historical_data()
                return
                
            # 为每个股票加载历史数据
            for symbol in self.symbols:
                # 构建数据文件路径
                symbol_data_file = os.path.join(historical_data_dir, f"{symbol}.csv")
                
                if os.path.exists(symbol_data_file):
                    try:
                        # 使用pandas读取CSV文件
                        df = pd.read_csv(symbol_data_file, parse_dates=['date'])
                        
                        # 过滤指定日期范围的数据
                        mask = (df['date'].dt.date >= self.start_date) & (df['date'].dt.date <= self.end_date)
                        filtered_df = df.loc[mask].copy()
                        
                        # 如果数据为空，生成模拟数据
                        if filtered_df.empty:
                            logger.warning(f"股票{symbol}在指定日期范围内没有数据")
                            continue
                            
                        # 转换为字典列表存储
                        data_records = []
                        for _, row in filtered_df.iterrows():
                            record = {
                                'date': row['date'].date(),
                                'price': row['close'],
                                'open': row['open'],
                                'high': row['high'],
                                'low': row['low'],
                                'volume': row['volume'],
                                'prev_close': row['prev_close'] if 'prev_close' in row else 0.0
                            }
                            data_records.append(record)
                            
                        self.historical_data[symbol] = data_records
                        logger.info(f"加载{symbol}历史数据: {len(data_records)}条记录")
                        
                    except Exception as e:
                        logger.error(f"加载{symbol}历史数据失败: {e}")
                        
            # 如果没有加载到任何数据，生成模拟数据
            if not self.historical_data:
                logger.warning("未能加载任何历史数据，使用模拟数据")
                self._generate_simulated_historical_data()
                
        except Exception as e:
            logger.error(f"加载历史数据失败: {e}")
            self._generate_simulated_historical_data()
    
    def _generate_simulated_historical_data(self):
        """生成模拟历史数据"""
        try:
            # 计算总天数
            days = (self.end_date - self.start_date).days + 1
            dates = [self.start_date + timedelta(days=i) for i in range(days)]
            
            # 过滤出交易日(周一到周五)
            trading_dates = [date for date in dates if date.weekday() < 5]
            
            # 定义基本价格范围
            price_ranges = {
                'small': (5, 15),      # 小盘股
                'medium': (15, 30),    # 中盘股
                'large': (30, 60),     # 大盘股
                'blue_chip': (60, 120), # 蓝筹股
                'premium': (120, 500)  # 白马股
            }
            
            # 为每个股票生成历史数据
            for i, symbol in enumerate(self.symbols):
                # 决定股票类型
                stock_type = list(price_ranges.keys())[i % len(price_ranges)]
                min_price, max_price = price_ranges[stock_type]
                
                # 生成起始价格
                base_price = random.uniform(min_price, max_price)
                
                # 生成股票特性
                volatility = random.uniform(0.01, 0.03)  # 每日波动率
                trend = random.uniform(-0.0005, 0.001)   # 长期趋势
                
                # 可以给不同板块添加不同特性
                if '.SH' in symbol:
                    sector_volatility = 1.0  # 上证波动率基准
                    sector_trend = 0.0002    # 上证长期趋势
                else:
                    sector_volatility = 1.2  # 深证波动率稍高
                    sector_trend = 0.0001    # 深证长期趋势基准
                
                # 应用板块特性
                volatility *= sector_volatility
                trend += sector_trend
                
                # 生成每日价格
                data_records = []
                current_price = base_price
                
                # 添加一些市场事件、季节性和周期性
                # 定义一些重要的市场事件
                market_events = {}
                for _ in range(3):  # 随机添加3个市场事件
                    event_date = random.choice(trading_dates[10:-10])  # 避开头尾
                    market_events[event_date] = {
                        'impact': random.choice([-0.08, -0.05, 0.05, 0.08]),  # 事件影响
                        'duration': random.randint(1, 5)  # 事件影响持续天数
                    }
                
                prev_close = current_price
                
                for date in trading_dates:
                    # 考虑市场事件的影响
                    event_impact = 0
                    for event_date, event in market_events.items():
                        days_from_event = (date - event_date).days
                        if 0 <= days_from_event < event['duration']:
                            decay = 1 - (days_from_event / event['duration'])  # 影响随时间衰减
                            event_impact += event['impact'] * decay
                    
                    # 计算日变动
                    daily_change = np.random.normal(trend, volatility) + event_impact
                    
                    # 应用变动
                    current_price *= (1 + daily_change)
                    current_price = max(current_price, 0.1)  # 确保价格为正
                    
                    # 生成开盘价、最高价、最低价
                    daily_volatility = volatility * random.uniform(0.5, 1.5)
                    open_price = prev_close * (1 + np.random.normal(0, daily_volatility * 0.5))
                    open_price = max(open_price, 0.1)
                    
                    price_range = current_price * daily_volatility
                    high_price = max(current_price, open_price) + abs(np.random.normal(0, price_range * 0.5))
                    low_price = min(current_price, open_price) * (1 - abs(np.random.normal(0, daily_volatility * 0.5)))
                    low_price = max(low_price, 0.1)
                    
                    # 生成成交量 - 与价格变动和市场事件相关
                    base_volume = int(current_price * 100000)
                    volume_multiplier = 1 + abs(daily_change) * 10 + abs(event_impact) * 20
                    volume = int(base_volume * volume_multiplier * random.uniform(0.8, 1.2))
                    
                    # 添加记录
                    record = {
                        'date': date,
                        'price': round(current_price, 2),
                        'open': round(open_price, 2),
                        'high': round(high_price, 2),
                        'low': round(low_price, 2),
                        'volume': volume,
                        'prev_close': round(prev_close, 2)
                    }
                    data_records.append(record)
                    
                    # 更新前一日收盘价
                    prev_close = current_price
                
                # 保存历史数据
                self.historical_data[symbol] = data_records
                
            logger.info(f"生成模拟历史数据: {len(self.symbols)}支股票，每支{len(trading_dates)}天的数据")
            
        except Exception as e:
            logger.error(f"生成模拟历史数据失败: {e}")
    
    def _update_backtest_data(self, set_initial=False):
        """更新回测数据
        
        Args:
            set_initial (bool): 是否设置为初始状态
        """
        try:
            # 如果是初始设置，或者需要前进到下一个交易日
            if set_initial or self.current_date < self.end_date:
                # 如果不是初始设置，前进到下一个交易日
                if not set_initial:
                    # 找到下一个交易日
                    next_date = self.current_date + timedelta(days=1)
                    while next_date.weekday() >= 5:  # 跳过周末
                        next_date += timedelta(days=1)
                    self.current_date = next_date
                
                # 更新所有股票的当前数据
                for symbol, historical_data in self.historical_data.items():
                    # 找到当前日期的数据
                    current_data = None
                    for data in historical_data:
                        if data['date'] == self.current_date:
                            current_data = data
                            break
                    
                    # 如果没有当前日期的数据，使用最近的数据
                    if not current_data and historical_data:
                        # 找到小于等于当前日期的最近记录
                        valid_records = [d for d in historical_data if d['date'] <= self.current_date]
                        if valid_records:
                            current_data = max(valid_records, key=lambda x: x['date'])
                    
                    # 更新当前数据
                    if current_data:
                        self.current_data[symbol] = {
                            'symbol': symbol,
                            'price': current_data['price'],
                            'open': current_data['open'],
                            'high': current_data['high'],
                            'low': current_data['low'],
                            'volume': current_data['volume'],
                            'prev_close': current_data['prev_close'],
                            'timestamp': datetime.combine(self.current_date, time(15, 0))
                        }
                
                return True
            else:
                # 回测结束
                logger.info("回测数据已到达结束日期")
                return False
                
        except Exception as e:
            logger.error(f"更新回测数据失败: {e}")
            return False
            
    def get_price_history_backtest(self, symbol, lookback_period=20):
        """获取回测模式下的历史价格
        
        Args:
            symbol (str): 股票代码
            lookback_period (int): 回溯天数
            
        Returns:
            list: 历史价格列表
        """
        try:
            if symbol not in self.historical_data:
                return None
                
            # 获取当前日期索引
            historical_data = self.historical_data[symbol]
            current_index = None
            
            for i, data in enumerate(historical_data):
                if data['date'] == self.current_date:
                    current_index = i
                    break
                    
            if current_index is None:
                # 找到小于等于当前日期的最近记录
                valid_records = [(i, d) for i, d in enumerate(historical_data) if d['date'] <= self.current_date]
                if valid_records:
                    current_index = max(valid_records, key=lambda x: x[1]['date'])[0]
                else:
                    return None
            
            # 获取历史价格
            start_index = max(0, current_index - lookback_period + 1)
            price_history = [data['price'] for data in historical_data[start_index:current_index+1]]
            
            return price_history
            
        except Exception as e:
            logger.error(f"获取回测历史价格失败: {e}")
            return None

    def get_history(self, symbol, start_date=None, end_date=None):
        """
        获取股票的历史数据
        """
        if symbol not in self.history:
            # 如果历史数据不存在，生成一些模拟数据
            self._generate_history_data(symbol)
        
        data = self.history.get(symbol, [])
        if start_date and end_date:
            # 过滤日期范围内的数据
            filtered_data = [d for d in data if start_date <= d.get('date') <= end_date]
            return filtered_data
        return data
    
    def _generate_history_data(self, symbol):
        """
        生成模拟的历史数据
        """
        import random
        import datetime
        
        # 生成过去30天的数据
        end_date = datetime.datetime.now()
        dates = [(end_date - datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
        dates.reverse()  # 按日期升序排列
        
        price = random.uniform(10, 100)
        history_data = []
        
        for date in dates:
            # 生成当天的OHLCV数据
            change_percent = random.uniform(-0.05, 0.05)
            open_price = price * (1 + random.uniform(-0.02, 0.02))
            high_price = price * (1 + random.uniform(0, 0.05))
            low_price = price * (1 - random.uniform(0, 0.05))
            close_price = price * (1 + change_percent)
            volume = random.randint(10000, 1000000)
            
            data_point = {
                'date': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
            
            history_data.append(data_point)
            price = close_price  # 更新价格为收盘价
        
        self.history[symbol] = history_data
    
    def initialize_fundamentals(self):
        """
        初始化股票的基本面数据
        """
        for symbol in self.symbols:
            if symbol not in self.fundamentals:
                self.fundamentals[symbol] = self._generate_fundamental_data(symbol)
    
    def _generate_fundamental_data(self, symbol):
        """
        生成模拟的基本面数据
        """
        import random
        
        return {
            'pe_ratio': random.uniform(5, 30),
            'pb_ratio': random.uniform(0.5, 5),
            'dividend_yield': random.uniform(0, 0.05),
            'market_cap': random.uniform(1e9, 1e12),
            'revenue': random.uniform(1e8, 1e11),
            'profit_margin': random.uniform(0.05, 0.3),
            'debt_to_equity': random.uniform(0.1, 2.0),
            'current_ratio': random.uniform(0.8, 3.0),
            'roe': random.uniform(0.05, 0.25),
            'roa': random.uniform(0.02, 0.15)
        }
    
    def get_ohlcv(self, symbol, period='1d'):
        """
        获取指定股票的OHLCV数据
        """
        if symbol in self.cache and period in self.cache[symbol]:
            return self.cache[symbol][period]
        
        # 如果缓存中没有，尝试从历史数据中获取
        history_data = self.get_history(symbol)
        if history_data:
            latest_data = history_data[-1]
            return {
                'open': latest_data.get('open', 0),
                'high': latest_data.get('high', 0),
                'low': latest_data.get('low', 0),
                'close': latest_data.get('close', 0),
                'volume': latest_data.get('volume', 0)
            }
        
        # 如果没有历史数据，返回默认值
        return {'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0}
    
    def get_market_regime(self):
        """
        获取市场状态
        """
        if self.market_regime is None:
            import random
            regimes = ['bullish', 'bearish', 'neutral', 'volatile']
            self.market_regime = random.choice(regimes)
        
        return self.market_regime
    
    def should_evolve(self):
        """
        判断是否应该执行进化操作
        """
        return self.should_evolve_flag
    
    def _update_backtest_data(self):
        # ... existing code ...
        # 更新基本面数据
        if not self.fundamentals or len(self.fundamentals) == 0:
            self.initialize_fundamentals()
    
    def _update_live_data(self):
        # ... existing code ...
        # 更新基本面数据
        if not self.fundamentals or len(self.fundamentals) == 0:
            self.initialize_fundamentals() 