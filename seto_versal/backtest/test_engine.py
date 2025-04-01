#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测引擎模块的测试文件
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import shutil
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from seto_versal.data.manager import DataManager, TimeFrame, CSVDataSource
from seto_versal.strategy.manager import Strategy, StrategyManager
from seto_versal.backtest.engine import Backtest, BacktestResult, Trade, Position, Portfolio


class TestTrade(unittest.TestCase):
    """Trade类的测试用例"""
    
    def test_trade_creation(self):
        """测试创建交易对象"""
        timestamp = datetime(2023, 1, 1, 10, 0)
        trade = Trade(
            symbol="AAPL",
            timestamp=timestamp,
            direction="BUY",
            quantity=100,
            price=150.0,
            strategy_id="test_strategy"
        )
        
        self.assertEqual(trade.symbol, "AAPL")
        self.assertEqual(trade.timestamp, timestamp)
        self.assertEqual(trade.direction, "BUY")
        self.assertEqual(trade.quantity, 100)
        self.assertEqual(trade.price, 150.0)
        self.assertEqual(trade.amount, 15000.0)
        self.assertEqual(trade.strategy_id, "test_strategy")
    
    def test_trade_serialization(self):
        """测试交易序列化和反序列化"""
        timestamp = datetime(2023, 1, 1, 10, 0)
        trade = Trade(
            symbol="AAPL",
            timestamp=timestamp,
            direction="BUY",
            quantity=100,
            price=150.0
        )
        
        trade_dict = trade.to_dict()
        reconstructed_trade = Trade.from_dict(trade_dict)
        
        self.assertEqual(reconstructed_trade.symbol, trade.symbol)
        self.assertEqual(reconstructed_trade.direction, trade.direction)
        self.assertEqual(reconstructed_trade.quantity, trade.quantity)
        self.assertEqual(reconstructed_trade.price, trade.price)
        self.assertEqual(reconstructed_trade.amount, trade.amount)


class TestPosition(unittest.TestCase):
    """Position类的测试用例"""
    
    def test_position_creation(self):
        """测试创建持仓对象"""
        position = Position("AAPL")
        self.assertEqual(position.symbol, "AAPL")
        self.assertEqual(position.quantity, 0.0)
        self.assertEqual(position.average_price, 0.0)
        self.assertEqual(position.cost_basis, 0.0)
        self.assertEqual(position.realized_pnl, 0.0)
        self.assertEqual(position.unrealized_pnl, 0.0)
    
    def test_update_from_trade_buy(self):
        """测试买入交易更新持仓"""
        position = Position("AAPL")
        
        # 第一笔交易 - 买入开仓
        trade1 = Trade(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 10, 0),
            direction="BUY",
            quantity=100,
            price=150.0
        )
        position.update_from_trade(trade1)
        
        self.assertEqual(position.quantity, 100)
        self.assertEqual(position.average_price, 150.0)
        self.assertEqual(position.cost_basis, 15000.0)
        
        # 第二笔交易 - 继续买入
        trade2 = Trade(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 11, 0),
            direction="BUY",
            quantity=50,
            price=155.0
        )
        position.update_from_trade(trade2)
        
        expected_avg_price = (15000.0 + 7750.0) / 150.0
        self.assertEqual(position.quantity, 150)
        self.assertEqual(position.average_price, expected_avg_price)
        self.assertEqual(position.cost_basis, 15000.0 + 7750.0)
    
    def test_update_from_trade_sell(self):
        """测试卖出交易更新持仓"""
        position = Position("AAPL")
        
        # 先买入
        trade1 = Trade(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 10, 0),
            direction="BUY",
            quantity=100,
            price=150.0
        )
        position.update_from_trade(trade1)
        
        # 然后卖出平仓
        trade2 = Trade(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 11, 0),
            direction="SELL",
            quantity=50,
            price=160.0
        )
        position.update_from_trade(trade2, is_opening=False)
        
        # 预期实现盈亏 = (160 - 150) * 50 = 500
        self.assertEqual(position.quantity, 50)
        self.assertEqual(position.realized_pnl, 500.0)
    
    def test_update_market_price(self):
        """测试更新市场价格"""
        position = Position("AAPL")
        
        # 买入建仓
        trade = Trade(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 10, 0),
            direction="BUY",
            quantity=100,
            price=150.0
        )
        position.update_from_trade(trade)
        
        # 更新市场价格
        position.update_market_price(160.0)
        
        # 预期未实现盈亏 = (160 - 150) * 100 = 1000
        self.assertEqual(position.unrealized_pnl, 1000.0)


class TestPortfolio(unittest.TestCase):
    """Portfolio类的测试用例"""
    
    def setUp(self):
        """设置测试环境"""
        self.initial_capital = 100000.0
        self.portfolio = Portfolio(self.initial_capital)
    
    def test_portfolio_creation(self):
        """测试创建投资组合"""
        self.assertEqual(self.portfolio.initial_capital, self.initial_capital)
        self.assertEqual(self.portfolio.cash, self.initial_capital)
        self.assertEqual(self.portfolio.equity, self.initial_capital)
        self.assertEqual(len(self.portfolio.positions), 0)
    
    def test_process_trade(self):
        """测试处理交易"""
        trade = Trade(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 10, 0),
            direction="BUY",
            quantity=100,
            price=150.0
        )
        self.portfolio.process_trade(trade)
        
        # 验证持仓和现金变化
        self.assertEqual(len(self.portfolio.positions), 1)
        self.assertIn("AAPL", self.portfolio.positions)
        self.assertEqual(self.portfolio.positions["AAPL"].quantity, 100)
        self.assertEqual(self.portfolio.cash, self.initial_capital - 15000.0)
        
        # 验证组合价值
        self.assertEqual(self.portfolio.equity, self.initial_capital)
    
    def test_update_market_prices(self):
        """测试更新市场价格"""
        # 买入AAPL
        trade1 = Trade(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 10, 0),
            direction="BUY",
            quantity=100,
            price=150.0
        )
        self.portfolio.process_trade(trade1)
        
        # 买入MSFT
        trade2 = Trade(
            symbol="MSFT",
            timestamp=datetime(2023, 1, 1, 10, 0),
            direction="BUY",
            quantity=50,
            price=250.0
        )
        self.portfolio.process_trade(trade2)
        
        # 更新市场价格
        prices = {"AAPL": 160.0, "MSFT": 260.0}
        self.portfolio.update_market_prices(prices)
        
        # 预期未实现盈亏 = (160-150)*100 + (260-250)*50 = 1000 + 500 = 1500
        expected_equity = self.initial_capital + 1500.0
        self.assertEqual(self.portfolio.equity, expected_equity)
        
        # 验证高水位和回撤
        self.assertEqual(self.portfolio.high_watermark, expected_equity)
        self.assertEqual(self.portfolio.max_drawdown, 0.0)


class TestBacktestResult(unittest.TestCase):
    """BacktestResult类的测试用例"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建一个包含历史数据的投资组合
        self.portfolio = Portfolio(100000.0)
        
        # 添加一些交易和历史记录点
        trade1 = Trade(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1, 10, 0),
            direction="BUY",
            quantity=100,
            price=150.0
        )
        self.portfolio.process_trade(trade1)
        
        # 更新市场价格
        self.portfolio.update_market_prices({"AAPL": 160.0})
        
        # 再添加一笔交易
        trade2 = Trade(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 2, 10, 0),
            direction="SELL",
            quantity=50,
            price=165.0
        )
        self.portfolio.process_trade(trade2, is_opening=False)
        
        # 更新市场价格
        self.portfolio.update_market_prices({"AAPL": 170.0})
        
        # 创建回测结果
        self.config = {
            "initial_capital": 100000.0,
            "commission_rate": 0.0003,
            "slippage": 0.0001,
            "risk_free_rate": 0.02
        }
        self.start_time = datetime(2023, 1, 1)
        self.end_time = datetime(2023, 1, 10)
        self.result = BacktestResult(self.portfolio, self.config, self.start_time, self.end_time)
    
    def test_calculate_performance_metrics(self):
        """测试计算绩效指标"""
        # 验证基本指标
        self.assertIn("total_return", self.result.metrics)
        self.assertIn("annualized_return", self.result.metrics)
        self.assertIn("max_drawdown", self.result.metrics)
        self.assertIn("sharpe_ratio", self.result.metrics)
        
        # 验证总收益率计算
        expected_return = (self.portfolio.equity / self.portfolio.initial_capital - 1) * 100
        self.assertAlmostEqual(self.result.metrics["total_return"], expected_return, delta=0.01)
    
    def test_get_history_dataframe(self):
        """测试获取历史数据DataFrame"""
        df = self.result.get_history_dataframe()
        
        # 验证DataFrame结构
        self.assertFalse(df.empty)
        self.assertIn("cash", df.columns)
        self.assertIn("equity", df.columns)
        self.assertIn("drawdown_percentage", df.columns)
        
        # 验证数据点数量
        self.assertEqual(len(df), len(self.portfolio.history))
    
    def test_to_dict(self):
        """测试转换为字典"""
        result_dict = self.result.to_dict()
        
        # 验证字典结构
        self.assertIn("config", result_dict)
        self.assertIn("metrics", result_dict)
        self.assertIn("portfolio", result_dict)
        self.assertIn("start_time", result_dict)
        self.assertIn("end_time", result_dict)


class SimpleStrategy(Strategy):
    """用于测试的简单策略"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.signals = []
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return data
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], timestamp: datetime) -> List[Dict[str, Any]]:
        # 模拟生成交易信号
        if "AAPL" in data and not data["AAPL"].empty:
            close_price = data["AAPL"].iloc[-1]["close"]
            
            # 简单策略：如果价格上涨超过前一天3%，则买入
            if len(data["AAPL"]) > 1:
                prev_close = data["AAPL"].iloc[-2]["close"]
                if close_price > prev_close * 1.03:
                    signal = {
                        "symbol": "AAPL",
                        "action": "BUY",
                        "quantity": 10,
                        "price": close_price,
                        "timestamp": timestamp,
                        "strategy_id": self.name
                    }
                    self.signals.append(signal)
                    return [signal]
            
            # 如果价格下跌超过前一天2%，则卖出
            if len(data["AAPL"]) > 1:
                prev_close = data["AAPL"].iloc[-2]["close"]
                if close_price < prev_close * 0.98:
                    signal = {
                        "symbol": "AAPL",
                        "action": "SELL",
                        "quantity": 10,
                        "price": close_price,
                        "timestamp": timestamp,
                        "strategy_id": self.name
                    }
                    self.signals.append(signal)
                    return [signal]
        
        return []
    
    def validate(self) -> Tuple[bool, str]:
        return True, ""
    
    def get_signals(self) -> List[Dict[str, Any]]:
        signals = self.signals.copy()
        self.signals = []
        return signals


class TestBacktest(unittest.TestCase):
    """Backtest类的测试用例"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试数据
        self.create_test_data()
        
        # 创建数据管理器
        config = {'data_path': self.temp_dir}
        data_source = CSVDataSource("test_csv", config)
        self.data_manager = DataManager()
        self.data_manager.add_data_source(data_source, is_default=True)
        
        # 创建策略管理器
        self.strategy_manager = StrategyManager()
        
        # 创建回测引擎
        self.backtest = Backtest(self.data_manager, self.strategy_manager)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_data(self):
        """创建测试数据文件"""
        # 创建AAPL日线数据
        aapl_daily = pd.DataFrame({
            'timestamp': [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
                datetime(2023, 1, 4),
                datetime(2023, 1, 5)
            ],
            'open': [150.0, 151.0, 152.0, 149.0, 147.0],
            'high': [155.0, 156.0, 157.0, 151.0, 150.0],
            'low': [149.0, 150.0, 148.0, 145.0, 144.0],
            'close': [153.0, 154.0, 149.0, 147.0, 145.0],
            'volume': [1000000, 1100000, 1200000, 1050000, 980000]
        })
        
        # 确保目录存在
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 保存数据
        aapl_daily.to_csv(os.path.join(self.temp_dir, "AAPL_1d.csv"), index=False)
    
    def test_backtest_creation(self):
        """测试创建回测引擎"""
        self.assertIsNotNone(self.backtest)
        self.assertEqual(self.backtest.data_manager, self.data_manager)
        self.assertEqual(self.backtest.strategy_manager, self.strategy_manager)
        self.assertIsNotNone(self.backtest.config)
        self.assertIsNotNone(self.backtest.portfolio)
    
    def test_set_config(self):
        """测试设置回测配置"""
        new_config = {
            "initial_capital": 200000.0,
            "commission_rate": 0.0005,
            "slippage": 0.0002
        }
        self.backtest.set_config(new_config)
        
        self.assertEqual(self.backtest.config["initial_capital"], 200000.0)
        self.assertEqual(self.backtest.config["commission_rate"], 0.0005)
        self.assertEqual(self.backtest.config["slippage"], 0.0002)
        self.assertEqual(self.backtest.portfolio.initial_capital, 200000.0)
    
    def test_add_strategy(self):
        """测试添加策略"""
        strategy = SimpleStrategy("test_strategy")
        self.backtest.add_strategy(strategy)
        
        strategies = self.strategy_manager.get_strategies()
        self.assertEqual(len(strategies), 1)
        self.assertEqual(strategies[0].name, "test_strategy")
    
    def test_run_backtest(self):
        """测试运行回测"""
        # 添加策略
        strategy = SimpleStrategy("test_strategy")
        self.backtest.add_strategy(strategy)
        
        # 运行回测
        result = self.backtest.run(
            symbols=["AAPL"],
            timeframe=TimeFrame.DAY_1,
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 5),
            initial_capital=100000.0
        )
        
        # 验证回测结果
        self.assertIsNotNone(result)
        self.assertIsInstance(result, BacktestResult)
        
        # 检查处理的数据
        self.assertGreater(self.backtest.stats["processed_bars"], 0)
        
        # 验证回测统计信息
        self.assertIn("processed_bars", self.backtest.stats)
        self.assertIn("generated_signals", self.backtest.stats)
        self.assertIn("executed_trades", self.backtest.stats)


if __name__ == "__main__":
    unittest.main() 