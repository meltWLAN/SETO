#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal GUI Dashboard
Provides visualization and monitoring interface for the trading system
Uses real data from the data management system
"""

import os
import sys
import threading
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import json

# 导入SETO-Versal系统组件
from seto_versal.market.state import MarketState
from seto_versal.data.manager import DataManager, TimeFrame
from seto_versal.utils.config import load_config

logger = logging.getLogger(__name__)

class GUIDashboard:
    """
    GUI Dashboard for visualizing trading data and system state
    Uses real data from the SETO-Versal data management system
    """
    
    def __init__(self, config=None):
        """
        Initialize the dashboard with configuration
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        self.running = False
        self.thread = None
        
        # 初始化数据管理器
        data_config = self.config.get('data', {})
        self.data_manager = DataManager(data_config)
        
        # 加载数据源
        self._init_data_sources()
        
        # 数据存储
        self.market_data = {}
        self.trading_recommendations = {
            'buy': [],
            'sell': []
        }
        self.transactions = []
        self.system_state = {
            'market_status': 'initializing',
            'active_agents': 0,
            'account_balance': 100000.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'daily_pnl': 0.0,
            'timestamp': datetime.now()
        }
        
        # 加载初始数据
        self._load_market_data()
        logger.info("GUI Dashboard initialized with real data")
    
    def _init_data_sources(self):
        """
        初始化所有数据源
        """
        try:
            # 配置的数据源
            data_sources = self.config.get('data_sources', [])
            
            if not data_sources:
                # 默认CSV数据源
                data_path = self.config.get('data_path', 'data/market')
                self.data_manager.create_csv_data_source("csv", {"data_path": data_path}, is_default=True)
                logger.info(f"Created default CSV data source from {data_path}")
            else:
                # 从配置加载数据源
                for i, source_config in enumerate(data_sources):
                    source_type = source_config.get('type', 'csv')
                    source_name = source_config.get('name', f"source_{i}")
                    is_default = source_config.get('is_default', i == 0)
                    
                    if source_type == 'csv':
                        self.data_manager.create_csv_data_source(
                            source_name, 
                            source_config, 
                            is_default=is_default
                        )
                    elif source_type == 'api':
                        self.data_manager.create_api_data_source(
                            source_name, 
                            source_config, 
                            is_default=is_default
                        )
                    elif source_type == 'database':
                        self.data_manager.create_database_data_source(
                            source_name, 
                            source_config, 
                            is_default=is_default
                        )
                    logger.info(f"Created {source_type} data source: {source_name}")
            
            # 连接数据源
            self.data_manager.connect_all()
        except Exception as e:
            logger.error(f"Error initializing data sources: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def start(self):
        """
        Start the dashboard update thread
        """
        if self.running:
            logger.warning("Dashboard already running")
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._update_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Dashboard update thread started")
        return True
    
    def stop(self):
        """
        Stop the dashboard update thread
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        
        # 断开数据源连接
        self.data_manager.disconnect_all()
        logger.info("Dashboard stopped")
        return True
    
    def _update_loop(self):
        """
        Main dashboard update loop
        """
        try:
            while self.running:
                # 更新市场数据
                self._load_market_data()
                
                # 生成交易推荐
                self._generate_recommendations()
                
                # 更新其他状态
                self._update_system_state()
                
                # 等待更新间隔
                update_interval = self.config.get('update_interval', 5)
                time.sleep(update_interval)
                
        except Exception as e:
            logger.error(f"Error in dashboard update loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.running = False
    
    def _load_market_data(self):
        """
        从数据管理系统加载市场数据
        """
        try:
            # 获取可用股票列表
            symbols = self.data_manager.get_available_symbols()
            if not symbols:
                logger.warning("没有可用股票数据")
                return False
            
            # 获取关注的股票列表
            watch_list = self.config.get('watch_list', symbols[:20])  # 默认关注前20个
            
            # 获取最近的市场数据
            end_time = datetime.now()
            start_time = end_time - timedelta(days=5)  # 获取最近5天数据
            
            # 更新市场数据
            market_data = {}
            for symbol in watch_list:
                try:
                    # 获取日线数据
                    df = self.data_manager.get_historical_data(
                        symbol, 
                        TimeFrame.DAY_1, 
                        start_time, 
                        end_time
                    )
                    
                    if df is not None and not df.empty:
                        # 获取最新价格数据
                        latest = df.iloc[-1]
                        prev_close = df.iloc[-2]['close'] if len(df) > 1 else latest['open']
                        
                        # 计算价格变动
                        change_pct = (latest['close'] - prev_close) / prev_close
                        
                        # 获取股票名称
                        name = self._get_stock_name(symbol)
                        
                        # 构建市场数据
                        market_data[symbol] = {
                            'symbol': symbol,
                            'name': name,
                            'price': latest['close'],
                            'change': change_pct,
                            'open': latest['open'],
                            'high': latest['high'],
                            'low': latest['low'],
                            'volume': latest['volume'],
                            'timestamp': latest.name if isinstance(latest.name, datetime) else datetime.now()
                        }
                except Exception as e:
                    logger.error(f"获取{symbol}数据失败: {e}")
            
            # 更新内部存储
            self.market_data = market_data
            logger.debug(f"已加载{len(market_data)}只股票的市场数据")
            return True
                
        except Exception as e:
            logger.error(f"加载市场数据失败: {e}")
            return False
    
    def _get_stock_name(self, symbol):
        """获取股票名称"""
        # 首先从数据管理器获取
        name = self.data_manager.get_symbol_info(symbol, 'name')
        if name:
            return name
            
        # 或者从本地映射文件获取
        try:
            stock_list_path = os.path.join(
                os.path.dirname(__file__), 
                '../../data/market/stock_list.json'
            )
            if os.path.exists(stock_list_path):
                with open(stock_list_path, 'r', encoding='utf-8') as f:
                    stock_list = json.load(f)
                    if symbol in stock_list:
                        return stock_list[symbol]['name']
        except Exception as e:
            logger.debug(f"从本地文件获取股票名称失败: {e}")
        
        # 如果无法获取名称，返回代码作为名称
        return symbol
    
    def _generate_recommendations(self):
        """
        基于市场数据生成交易推荐
        """
        try:
            # 检查市场数据是否可用
            if not self.market_data:
                logger.warning("市场数据不可用，无法生成推荐")
                return
            
            # 从交易系统获取推荐
            # 这里需要集成实际的交易策略和智能体
            # 当前为示例实现
            
            from seto_versal.trading.analyzer import MarketAnalyzer
            from seto_versal.trading.strategy import StrategyManager
            
            # 初始化分析器和策略
            analyzer = MarketAnalyzer()
            strategy_manager = StrategyManager()
            
            # 分析市场状态
            market_state = analyzer.analyze_market(self.market_data)
            
            # 获取活跃策略
            active_strategies = strategy_manager.get_active_strategies(market_state)
            
            # 生成推荐
            buy_recommendations = []
            sell_recommendations = []
            
            for strategy in active_strategies:
                # 获取策略产生的信号
                signals = strategy.generate_signals(self.market_data)
                
                for signal in signals:
                    if signal['action'] == 'buy':
                        buy_recommendations.append({
                            'symbol': signal['symbol'],
                            'name': self._get_stock_name(signal['symbol']),
                            'price': self.market_data[signal['symbol']]['price'],
                            'reason': signal['reason'],
                            'confidence': signal['confidence'],
                            'target_price': signal['target_price'],
                            'strategy': strategy.name,
                            'timestamp': datetime.now()
                        })
                    elif signal['action'] == 'sell':
                        sell_recommendations.append({
                            'symbol': signal['symbol'],
                            'name': self._get_stock_name(signal['symbol']),
                            'price': self.market_data[signal['symbol']]['price'],
                            'reason': signal['reason'],
                            'confidence': signal['confidence'],
                            'profit': signal['profit'],
                            'strategy': strategy.name,
                            'timestamp': datetime.now()
                        })
            
            # 更新推荐列表
            self.trading_recommendations = {
                'buy': buy_recommendations,
                'sell': sell_recommendations
            }
            
            logger.debug(f"已生成 {len(buy_recommendations)} 个买入建议和 {len(sell_recommendations)} 个卖出建议")
            
        except Exception as e:
            logger.error(f"生成交易推荐失败: {e}")
    
    def _update_system_state(self):
        """
        更新系统状态信息
        """
        try:
            current_time = datetime.now()
            
            # 获取市场状态
            from seto_versal.market.calendar import MarketCalendar
            calendar = MarketCalendar()
            
            market_state = 'closed'  # 默认关闭
            
            # 检查是否为交易日
            if calendar.is_trading_day(current_time.date()):
                # 检查交易时间
                if calendar.is_trading_time(current_time):
                    market_state = 'open'
                elif calendar.is_pre_market(current_time):
                    market_state = 'pre-market'
                elif calendar.is_post_market(current_time):
                    market_state = 'post-market'
            
            # 获取账户信息
            from seto_versal.account.manager import AccountManager
            account_manager = AccountManager()
            account_info = account_manager.get_account_info()
            
            # 更新系统状态
            self.system_state.update({
                'market_status': market_state,
                'account_balance': account_info.get('balance', 0.0),
                'total_trades': account_info.get('total_trades', 0),
                'win_rate': account_info.get('win_rate', 0.0),
                'daily_pnl': account_info.get('daily_pnl', 0.0),
                'timestamp': current_time
            })
            
        except Exception as e:
            logger.error(f"更新系统状态失败: {e}")
    
    def get_dashboard_data(self):
        """
        获取当前仪表盘数据
        
        Returns:
            dict: 仪表盘数据结构
        """
        return {
            'market_data': self.market_data,
            'recommendations': self.trading_recommendations,
            'transactions': self.transactions,
            'system_state': self.system_state,
            'timestamp': datetime.now()
        }
    
    def export_data(self, file_path=None):
        """
        导出仪表盘数据到文件
        
        Args:
            file_path (str, optional): 输出文件路径
            
        Returns:
            str: 导出文件路径
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(os.path.dirname(__file__), f"dashboard_data_{timestamp}.json")
        
        try:
            # 创建可序列化的表示
            data = self.get_dashboard_data()
            
            # 转换日期时间对象为字符串
            def convert_datetime(obj):
                if isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                elif isinstance(obj, datetime):
                    return obj.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    return obj
            
            serializable_data = convert_datetime(data)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2)
            
            logger.info(f"导出仪表盘数据到 {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"导出仪表盘数据失败: {e}")
            return None


def create_dashboard(config=None):
    """
    工厂函数，创建仪表盘实例
    
    Args:
        config (dict, optional): 仪表盘配置
        
    Returns:
        GUIDashboard: 配置好的仪表盘实例
    """
    # 加载配置
    if config is None:
        config_path = os.getenv('SETO_CONFIG', 'config.yaml')
        try:
            config = load_config(config_path)
            dashboard_config = config.get('visualization', {}).get('dashboard', {})
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，使用默认配置")
            dashboard_config = {}
    else:
        dashboard_config = config
        
    return GUIDashboard(dashboard_config)


if __name__ == "__main__":
    # 设置基本日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建并启动仪表盘
    dashboard = create_dashboard()
    dashboard.start()
    
    try:
        # 运行演示
        print("仪表盘运行中。按Ctrl+C退出。")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        dashboard.stop()
        print("仪表盘已停止。")
