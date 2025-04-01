#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测引擎模块，用于策略的历史表现评估。
提供了完整的回测框架和结果分析功能。
"""

import os
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Type, Set
import matplotlib.pyplot as plt
from collections import defaultdict

from seto_versal.data.manager import DataManager, TimeFrame
from seto_versal.strategy.manager import Strategy, StrategyManager
from seto_versal.risk.controller import RiskController, RiskLevel


class Trade:
    """
    交易类，记录单笔交易的信息
    """
    
    def __init__(self, 
                 symbol: str, 
                 timestamp: datetime, 
                 direction: str, 
                 quantity: float, 
                 price: float, 
                 trade_id: Optional[str] = None,
                 strategy_id: Optional[str] = None,
                 tags: Optional[Dict[str, Any]] = None):
        """
        初始化交易
        
        Args:
            symbol: 交易品种
            timestamp: 交易时间
            direction: 交易方向，"BUY"或"SELL"
            quantity: 交易数量
            price: 交易价格
            trade_id: 交易ID
            strategy_id: 策略ID
            tags: 交易标签
        """
        self.symbol = symbol
        self.timestamp = timestamp
        self.direction = direction.upper()
        self.quantity = quantity
        self.price = price
        self.trade_id = trade_id or f"{symbol}_{int(timestamp.timestamp())}_{direction}"
        self.strategy_id = strategy_id
        self.tags = tags or {}
        
        # 计算交易金额
        self.amount = self.quantity * self.price
        
        # 交易费用
        self.commission = 0.0
        self.slippage = 0.0
        
        # 交易状态
        self.status = "EXECUTED"  # 回测中默认都是执行成功的
        
    def to_dict(self) -> Dict[str, Any]:
        """
        将交易转换为字典
        
        Returns:
            交易信息的字典表示
        """
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "quantity": self.quantity,
            "price": self.price,
            "amount": self.amount,
            "commission": self.commission,
            "slippage": self.slippage,
            "strategy_id": self.strategy_id,
            "status": self.status,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """
        从字典创建交易
        
        Args:
            data: 交易信息字典
            
        Returns:
            创建的交易对象
        """
        trade = cls(
            symbol=data["symbol"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            direction=data["direction"],
            quantity=data["quantity"],
            price=data["price"],
            trade_id=data.get("trade_id"),
            strategy_id=data.get("strategy_id"),
            tags=data.get("tags", {})
        )
        
        trade.commission = data.get("commission", 0.0)
        trade.slippage = data.get("slippage", 0.0)
        trade.status = data.get("status", "EXECUTED")
        
        return trade


class Position:
    """
    持仓类，记录单个交易品种的持仓信息
    """
    
    def __init__(self, symbol: str):
        """
        初始化持仓
        
        Args:
            symbol: 交易品种
        """
        self.symbol = symbol
        self.quantity = 0.0
        self.average_price = 0.0
        self.cost_basis = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.last_price = 0.0
        self.trades: List[Trade] = []
        
    def update_from_trade(self, trade: Trade, is_opening: bool = True) -> float:
        """
        根据交易更新持仓
        
        Args:
            trade: 交易对象
            is_opening: 是否为开仓交易
            
        Returns:
            实现盈亏
        """
        realized_pnl = 0.0
        
        if trade.symbol != self.symbol:
            logging.warning(f"交易品种 {trade.symbol} 与持仓品种 {self.symbol} 不匹配")
            return realized_pnl
        
        # 添加交易到历史
        self.trades.append(trade)
        
        # 根据交易方向更新持仓
        if trade.direction == "BUY":
            # 买入
            if is_opening:
                # 开仓
                if self.quantity == 0:
                    # 新建仓位
                    self.quantity = trade.quantity
                    self.average_price = trade.price
                    self.cost_basis = trade.amount
                else:
                    # 增加现有仓位
                    new_cost = self.cost_basis + trade.amount
                    new_quantity = self.quantity + trade.quantity
                    self.average_price = new_cost / new_quantity
                    self.cost_basis = new_cost
                    self.quantity = new_quantity
            else:
                # 平仓（买入平空）
                if self.quantity < 0:
                    # 计算平仓部分的实现盈亏
                    close_quantity = min(abs(self.quantity), trade.quantity)
                    realized_pnl = (self.average_price - trade.price) * close_quantity
                    self.realized_pnl += realized_pnl
                    
                    # 更新持仓
                    remaining_quantity = self.quantity + close_quantity
                    if remaining_quantity < 0:
                        # 仍有空头持仓
                        self.quantity = remaining_quantity
                    else:
                        # 平仓后变为多头或平仓
                        excess_quantity = remaining_quantity
                        if excess_quantity > 0:
                            # 多头
                            self.quantity = excess_quantity
                            self.average_price = trade.price
                            self.cost_basis = trade.price * excess_quantity
                        else:
                            # 完全平仓
                            self.reset()
                else:
                    # 已有多头持仓，应该使用is_opening=True
                    logging.warning("买入平空交易，但当前无空头持仓")
                    self.quantity += trade.quantity
                    self.cost_basis += trade.amount
                    self.average_price = self.cost_basis / self.quantity
        
        elif trade.direction == "SELL":
            # 卖出
            if is_opening:
                # 开仓（卖出开空）
                if self.quantity == 0:
                    # 新建空头仓位
                    self.quantity = -trade.quantity
                    self.average_price = trade.price
                    self.cost_basis = trade.amount
                else:
                    # 增加现有空头仓位或减少多头仓位
                    if self.quantity > 0:
                        # 有多头持仓，应该使用is_opening=False
                        logging.warning("卖出开空交易，但当前有多头持仓")
                    
                    # 增加空头仓位
                    new_quantity = self.quantity - trade.quantity
                    if self.quantity < 0:
                        # 已有空头，增加空头
                        new_cost = self.cost_basis + trade.amount
                        self.average_price = new_cost / abs(new_quantity)
                        self.cost_basis = new_cost
                    else:
                        # 从多头变为空头
                        new_cost = trade.amount - self.cost_basis
                        if new_quantity < 0:
                            self.average_price = new_cost / abs(new_quantity)
                            self.cost_basis = new_cost
                        else:
                            # 部分平仓
                            realized_pnl = (trade.price - self.average_price) * trade.quantity
                            self.realized_pnl += realized_pnl
                            if new_quantity > 0:
                                self.cost_basis = self.average_price * new_quantity
                            else:
                                self.cost_basis = 0
                                self.average_price = 0
                    
                    self.quantity = new_quantity
            else:
                # 平仓（卖出平多）
                if self.quantity > 0:
                    # 计算平仓部分的实现盈亏
                    close_quantity = min(self.quantity, trade.quantity)
                    realized_pnl = (trade.price - self.average_price) * close_quantity
                    self.realized_pnl += realized_pnl
                    
                    # 更新持仓
                    remaining_quantity = self.quantity - close_quantity
                    if remaining_quantity > 0:
                        # 仍有多头持仓
                        self.quantity = remaining_quantity
                        self.cost_basis = self.average_price * remaining_quantity
                    else:
                        # 平仓后变为空头或平仓
                        excess_quantity = trade.quantity - self.quantity
                        if excess_quantity > 0:
                            # 空头
                            self.quantity = -excess_quantity
                            self.average_price = trade.price
                            self.cost_basis = trade.price * excess_quantity
                        else:
                            # 完全平仓
                            self.reset()
                else:
                    # 已有空头持仓，应该使用is_opening=True
                    logging.warning("卖出平多交易，但当前无多头持仓")
                    self.quantity -= trade.quantity
                    if self.quantity < 0:
                        self.cost_basis += trade.amount
                        self.average_price = self.cost_basis / abs(self.quantity)
        
        # 更新最新价格
        self.last_price = trade.price
        
        return realized_pnl
    
    def update_market_price(self, price: float) -> None:
        """
        更新市场价格和未实现盈亏
        
        Args:
            price: 最新市场价格
        """
        self.last_price = price
        
        if self.quantity != 0:
            if self.quantity > 0:
                # 多头
                self.unrealized_pnl = (price - self.average_price) * self.quantity
            else:
                # 空头
                self.unrealized_pnl = (self.average_price - price) * abs(self.quantity)
        else:
            self.unrealized_pnl = 0.0
    
    def reset(self) -> None:
        """
        重置持仓
        """
        self.quantity = 0.0
        self.average_price = 0.0
        self.cost_basis = 0.0
        self.unrealized_pnl = 0.0
        # 不重置已实现盈亏和交易历史
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将持仓转换为字典
        
        Returns:
            持仓信息的字典表示
        """
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_price": self.average_price,
            "cost_basis": self.cost_basis,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "last_price": self.last_price,
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
            "trades": [trade.to_dict() for trade in self.trades]
        }


class Portfolio:
    """
    投资组合类，管理账户资金和持仓
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        初始化投资组合
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.history: List[Dict[str, Any]] = []
        
        # 绩效指标
        self.equity = initial_capital
        self.high_watermark = initial_capital
        self.max_drawdown = 0.0
        self.max_drawdown_percentage = 0.0
        
        # 当前时间戳
        self.current_timestamp = None
    
    def process_trade(self, trade: Trade, is_opening: bool = True) -> None:
        """
        处理交易
        
        Args:
            trade: 交易对象
            is_opening: 是否为开仓交易
        """
        # 检查是否有持仓
        if trade.symbol not in self.positions:
            self.positions[trade.symbol] = Position(trade.symbol)
        
        # 更新当前时间戳
        if self.current_timestamp is None or trade.timestamp > self.current_timestamp:
            self.current_timestamp = trade.timestamp
        
        # 计算交易成本（手续费+滑点）
        trade_cost = trade.commission + (trade.slippage * trade.amount)
        
        # 计算交易金额
        trade_amount = trade.amount + trade_cost
        
        # 更新持仓
        position = self.positions[trade.symbol]
        realized_pnl = position.update_from_trade(trade, is_opening)
        
        # 更新现金
        if trade.direction == "BUY":
            self.cash -= trade_amount
        else:  # SELL
            self.cash += trade_amount - trade_cost
        
        # 更新现金以反映实现盈亏（避免重复计算）
        if realized_pnl != 0:
            # 实现盈亏已经在position.update_from_trade中计算并添加到position.realized_pnl
            # 这里不需要再更新现金
            pass
        
        # 添加交易到历史
        self.trades.append(trade)
        
        # 更新组合价值
        self.update_portfolio_value()
        
        # 记录历史点
        self.record_history_point()
    
    def update_market_prices(self, symbol_prices: Dict[str, float]) -> None:
        """
        更新市场价格
        
        Args:
            symbol_prices: 品种价格映射
        """
        for symbol, price in symbol_prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_market_price(price)
        
        # 更新组合价值
        self.update_portfolio_value()
        
        # 记录历史点
        self.record_history_point()
    
    def update_portfolio_value(self) -> None:
        """
        更新投资组合价值
        """
        # 计算持仓总价值
        positions_value = sum(pos.last_price * abs(pos.quantity) for pos in self.positions.values() if pos.quantity != 0)
        
        # 计算总权益
        self.equity = self.cash + positions_value + sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # 更新高水位线
        if self.equity > self.high_watermark:
            self.high_watermark = self.equity
        
        # 计算回撤
        drawdown = self.high_watermark - self.equity
        drawdown_percentage = (drawdown / self.high_watermark) * 100 if self.high_watermark > 0 else 0
        
        # 更新最大回撤
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            self.max_drawdown_percentage = drawdown_percentage
    
    def record_history_point(self) -> None:
        """
        记录历史点
        """
        if self.current_timestamp is None:
            return
        
        # 创建历史记录点
        point = {
            "timestamp": self.current_timestamp,
            "cash": self.cash,
            "equity": self.equity,
            "positions_value": self.equity - self.cash,
            "drawdown": self.high_watermark - self.equity,
            "drawdown_percentage": (self.high_watermark - self.equity) / self.high_watermark * 100 if self.high_watermark > 0 else 0,
            "positions": {symbol: pos.to_dict() for symbol, pos in self.positions.items() if pos.quantity != 0}
        }
        
        self.history.append(point)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        获取持仓
        
        Args:
            symbol: 交易品种
            
        Returns:
            持仓对象
        """
        return self.positions.get(symbol)
    
    def get_total_value(self) -> float:
        """
        获取投资组合总价值
        
        Returns:
            总价值
        """
        return self.equity
    
    def get_positions_value(self) -> float:
        """
        获取持仓总价值
        
        Returns:
            持仓总价值
        """
        return sum(pos.last_price * abs(pos.quantity) for pos in self.positions.values() if pos.quantity != 0)
    
    def get_realized_pnl(self) -> float:
        """
        获取已实现盈亏
        
        Returns:
            已实现盈亏
        """
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    def get_unrealized_pnl(self) -> float:
        """
        获取未实现盈亏
        
        Returns:
            未实现盈亏
        """
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将投资组合转换为字典
        
        Returns:
            投资组合信息的字典表示
        """
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "equity": self.equity,
            "positions_value": self.get_positions_value(),
            "realized_pnl": self.get_realized_pnl(),
            "unrealized_pnl": self.get_unrealized_pnl(),
            "positions": {symbol: pos.to_dict() for symbol, pos in self.positions.items() if pos.quantity != 0},
            "max_drawdown": self.max_drawdown,
            "max_drawdown_percentage": self.max_drawdown_percentage,
            "current_timestamp": self.current_timestamp.isoformat() if self.current_timestamp else None,
            "high_watermark": self.high_watermark,
            "total_return": (self.equity / self.initial_capital - 1) * 100,
            "total_trades": len(self.trades)
        }
    
    def reset(self) -> None:
        """
        重置投资组合
        """
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.history = []
        self.equity = self.initial_capital
        self.high_watermark = self.initial_capital
        self.max_drawdown = 0.0
        self.max_drawdown_percentage = 0.0
        self.current_timestamp = None 


class BacktestResult:
    """
    回测结果类，用于存储和分析回测结果
    """
    
    def __init__(self, 
                portfolio: Portfolio, 
                backtest_config: Dict[str, Any], 
                start_time: datetime, 
                end_time: datetime):
        """
        初始化回测结果
        
        Args:
            portfolio: 回测完成后的投资组合
            backtest_config: 回测配置
            start_time: 回测开始时间
            end_time: 回测结束时间
        """
        self.portfolio = portfolio
        self.config = backtest_config
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        
        # 计算绩效指标
        self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self) -> None:
        """
        计算绩效指标
        """
        history_df = self.get_history_dataframe()
        
        if history_df.empty:
            self.metrics = {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_percentage": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "duration_days": 0,
                "risk_adjusted_return": 0.0
            }
            return
        
        # 总收益率
        initial_equity = self.portfolio.initial_capital
        final_equity = self.portfolio.equity
        total_return = (final_equity / initial_equity - 1) * 100
        
        # 年化收益率
        days = self.duration.days
        if days <= 0:
            days = 1
        annualized_return = (((final_equity / initial_equity) ** (365 / days)) - 1) * 100
        
        # 获取日收益率
        if len(history_df) > 1:
            history_df['daily_return'] = history_df['equity'].pct_change()
            # 过滤掉NaN值
            daily_returns = history_df['daily_return'].dropna()
        else:
            daily_returns = pd.Series([0.0])
        
        # 计算Sharpe比率和Sortino比率
        risk_free_rate = self.config.get('risk_free_rate', 0.0) / 365  # 日风险自由利率
        excess_returns = daily_returns - risk_free_rate
        
        if len(excess_returns) > 0 and excess_returns.std() > 0:
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)  # 年化
        else:
            sharpe_ratio = 0.0
        
        # Sortino比率
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = (daily_returns.mean() - risk_free_rate) / negative_returns.std() * np.sqrt(252)
        else:
            sortino_ratio = 0.0
        
        # 计算胜率和盈亏比
        trades = self.portfolio.trades
        if len(trades) > 0:
            # 计算每笔交易的盈亏
            trade_pnls = []
            open_trades = {}
            
            for trade in trades:
                symbol = trade.symbol
                direction = trade.direction
                quantity = trade.quantity
                price = trade.price
                
                # 非常简化的交易盈亏计算，实际应该考虑更复杂的持仓管理
                if symbol not in open_trades:
                    open_trades[symbol] = []
                
                open_trades[symbol].append(trade)
                
                # 这里的盈亏计算非常简化，实际应该通过配对交易计算
                # 由于交易记录本身不包含完整的盈亏信息，这里只是简单估算
            
            # 从已实现盈亏来计算
            all_positions = [pos for pos in self.portfolio.positions.values()]
            total_realized_pnl = sum(pos.realized_pnl for pos in all_positions)
            
            if total_realized_pnl > 0:
                winning_trades = sum(1 for pos in all_positions if pos.realized_pnl > 0)
                total_trades = len(trades)
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
            else:
                win_rate = 0.0
            
            # 盈亏比
            profit_factor = 0.0
            total_profit = sum(pos.realized_pnl for pos in all_positions if pos.realized_pnl > 0)
            total_loss = abs(sum(pos.realized_pnl for pos in all_positions if pos.realized_pnl < 0))
            
            if total_loss > 0:
                profit_factor = total_profit / total_loss
            elif total_profit > 0:
                profit_factor = float('inf')  # 没有亏损，盈亏比无限大
        else:
            win_rate = 0.0
            profit_factor = 0.0
        
        # 风险调整后的收益率
        risk_adjusted_return = annualized_return / (self.portfolio.max_drawdown_percentage + 0.1)
        
        # 存储计算的指标
        self.metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": self.portfolio.max_drawdown,
            "max_drawdown_percentage": self.portfolio.max_drawdown_percentage,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(trades),
            "duration_days": days,
            "risk_adjusted_return": risk_adjusted_return
        }
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """
        获取历史数据的DataFrame
        
        Returns:
            历史数据DataFrame
        """
        if not self.portfolio.history:
            return pd.DataFrame()
        
        # 转换历史记录为DataFrame
        history_data = []
        
        for point in self.portfolio.history:
            data = {
                "timestamp": point["timestamp"],
                "cash": point["cash"],
                "equity": point["equity"],
                "positions_value": point["positions_value"],
                "drawdown": point["drawdown"],
                "drawdown_percentage": point["drawdown_percentage"]
            }
            history_data.append(data)
        
        df = pd.DataFrame(history_data)
        df.set_index("timestamp", inplace=True)
        
        return df
    
    def plot_equity_curve(self, save_path: Optional[str] = None) -> None:
        """
        绘制权益曲线
        
        Args:
            save_path: 图表保存路径，如果为None则显示图表
        """
        history_df = self.get_history_dataframe()
        
        if history_df.empty:
            logging.warning("没有足够的历史数据来绘制权益曲线")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 绘制权益曲线
        plt.subplot(2, 1, 1)
        plt.plot(history_df.index, history_df['equity'], label='权益')
        plt.plot(history_df.index, history_df['cash'], label='现金', linestyle='--')
        plt.title('回测权益曲线')
        plt.legend()
        plt.grid(True)
        
        # 绘制回撤曲线
        plt.subplot(2, 1, 2)
        plt.fill_between(history_df.index, 0, history_df['drawdown_percentage'], alpha=0.3, color='red')
        plt.plot(history_df.index, history_df['drawdown_percentage'], color='red', label='回撤(%)')
        plt.title('回撤(%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_trade_analysis(self, save_path: Optional[str] = None) -> None:
        """
        绘制交易分析图表
        
        Args:
            save_path: 图表保存路径，如果为None则显示图表
        """
        if not self.portfolio.trades:
            logging.warning("没有交易数据来绘制交易分析图表")
            return
        
        # 准备交易数据
        trade_data = []
        for trade in self.portfolio.trades:
            data = {
                "timestamp": trade.timestamp,
                "symbol": trade.symbol,
                "direction": trade.direction,
                "quantity": trade.quantity,
                "price": trade.price,
                "amount": trade.amount
            }
            trade_data.append(data)
        
        trades_df = pd.DataFrame(trade_data)
        
        if trades_df.empty:
            logging.warning("没有足够的交易数据来绘制交易分析图表")
            return
        
        plt.figure(figsize=(12, 10))
        
        # 交易方向分布
        plt.subplot(2, 2, 1)
        direction_counts = trades_df['direction'].value_counts()
        plt.pie(direction_counts, labels=direction_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('交易方向分布')
        
        # 交易品种分布
        plt.subplot(2, 2, 2)
        symbol_counts = trades_df['symbol'].value_counts()
        plt.pie(symbol_counts.head(5), labels=symbol_counts.head(5).index, autopct='%1.1f%%', startangle=90)
        plt.title('交易品种分布 (Top 5)')
        
        # 交易金额分布
        plt.subplot(2, 2, 3)
        plt.hist(trades_df['amount'], bins=20, alpha=0.7)
        plt.title('交易金额分布')
        plt.xlabel('交易金额')
        plt.ylabel('频率')
        
        # 交易时间分布
        plt.subplot(2, 2, 4)
        trades_df['hour'] = trades_df['timestamp'].dt.hour
        hour_counts = trades_df['hour'].value_counts().sort_index()
        plt.bar(hour_counts.index, hour_counts.values)
        plt.title('交易时间分布')
        plt.xlabel('小时')
        plt.ylabel('交易次数')
        plt.xticks(range(0, 24, 2))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将回测结果转换为字典
        
        Returns:
            回测结果的字典表示
        """
        return {
            "config": self.config,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_days": self.duration.days,
            "metrics": self.metrics,
            "portfolio": self.portfolio.to_dict()
        }
    
    def save_to_file(self, file_path: str) -> bool:
        """
        将回测结果保存到文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            保存是否成功
        """
        try:
            result_dict = self.to_dict()
            
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logging.error(f"保存回测结果失败: {str(e)}")
            return False
    
    def print_summary(self) -> None:
        """
        打印回测结果摘要
        """
        print("\n====== 回测结果摘要 ======")
        print(f"回测期间: {self.start_time.strftime('%Y-%m-%d')} 至 {self.end_time.strftime('%Y-%m-%d')} ({self.duration.days}天)")
        print(f"初始资金: {self.portfolio.initial_capital:.2f}")
        print(f"最终权益: {self.portfolio.equity:.2f}")
        print(f"总收益率: {self.metrics['total_return']:.2f}%")
        print(f"年化收益率: {self.metrics['annualized_return']:.2f}%")
        print(f"最大回撤: {self.metrics['max_drawdown']:.2f} ({self.metrics['max_drawdown_percentage']:.2f}%)")
        print(f"Sharpe比率: {self.metrics['sharpe_ratio']:.4f}")
        print(f"Sortino比率: {self.metrics['sortino_ratio']:.4f}")
        print(f"胜率: {self.metrics['win_rate']:.2f}%")
        print(f"盈亏比: {self.metrics['profit_factor']:.4f}")
        print(f"总交易次数: {self.metrics['total_trades']}")
        print(f"风险调整后收益率: {self.metrics['risk_adjusted_return']:.4f}")
        print("==========================\n")


class Backtest:
    """
    回测类，用于策略的历史表现评估
    """
    
    def __init__(self, data_manager: DataManager, strategy_manager: Optional[StrategyManager] = None):
        """
        初始化回测
        
        Args:
            data_manager: 数据管理器
            strategy_manager: 策略管理器，如果为None则创建一个新的
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化回测引擎")
        
        self.data_manager = data_manager
        self.strategy_manager = strategy_manager or StrategyManager()
        
        # 回测配置
        self.config: Dict[str, Any] = {
            "initial_capital": 100000.0,
            "commission_rate": 0.0003,  # 手续费率
            "slippage": 0.0001,  # 滑点
            "risk_free_rate": 0.02,  # 年化无风险利率
            "enable_risk_control": True,  # 是否启用风险控制
            "position_size_limit": 0.2,  # 单一持仓上限（占总资金比例）
            "cash_reserve_ratio": 0.1,  # 现金保留比例
            "risk_rules_file": None,  # 风险规则文件路径
            "price_type": "close",  # 使用的价格类型，可以是"open", "close", "vwap"等
            "leverage": 1.0,  # 杠杆倍数
        }
        
        # 投资组合
        self.portfolio = Portfolio(self.config["initial_capital"])
        
        # 风险控制器
        self.risk_controller = None
        
        # 回测状态
        self.current_date = None
        self.is_running = False
        self.start_time = None
        self.end_time = None
        
        # 回测结果
        self.result = None
        
        # 回测统计信息
        self.stats = {
            "processed_bars": 0,
            "generated_signals": 0,
            "executed_trades": 0,
            "rejected_trades": 0,
            "processing_time": 0.0
        }
        
        self.logger.info("回测引擎初始化完成")
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        设置回测配置
        
        Args:
            config: 回测配置
        """
        self.config.update(config)
        
        # 更新投资组合的初始资金
        if "initial_capital" in config:
            self.portfolio = Portfolio(config["initial_capital"])
        
        # 设置风险控制器
        if self.config["enable_risk_control"]:
            if self.risk_controller is None:
                risk_config = {
                    "max_drawdown_percent": 20.0,
                    "max_position_percent": self.config["position_size_limit"] * 100,
                    "cash_reserve_percent": self.config["cash_reserve_ratio"] * 100,
                    "initial_risk_level": "MEDIUM"
                }
                
                if self.config["risk_rules_file"]:
                    risk_config["risk_rules_file"] = self.config["risk_rules_file"]
                
                self.risk_controller = RiskController("backtest", risk_config)
        else:
            self.risk_controller = None
    
    def add_strategy(self, strategy: Strategy, weight: float = 1.0) -> None:
        """
        添加策略
        
        Args:
            strategy: 策略对象
            weight: 策略权重
        """
        self.strategy_manager.add_strategy(strategy, weight)
    
    def run(self, 
          symbols: List[str], 
          timeframe: TimeFrame, 
          start_time: datetime, 
          end_time: Optional[datetime] = None, 
          initial_capital: float = 100000.0) -> BacktestResult:
        """
        运行回测
        
        Args:
            symbols: 交易品种列表
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间，如果为None则使用当前时间
            initial_capital: 初始资金
            
        Returns:
            回测结果
        """
        if end_time is None:
            end_time = datetime.now()
        
        self.logger.info(f"开始回测: {symbols}, {timeframe}, {start_time} 至 {end_time}")
        
        # 更新配置
        self.set_config({"initial_capital": initial_capital})
        
        # 重置投资组合和状态
        self.portfolio.reset()
        self.stats = {
            "processed_bars": 0,
            "generated_signals": 0,
            "executed_trades": 0,
            "rejected_trades": 0,
            "processing_time": 0.0
        }
        self.is_running = True
        self.start_time = start_time
        self.end_time = end_time
        
        # 获取历史数据
        all_data = {}
        for symbol in symbols:
            data = self.data_manager.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            if data.empty:
                self.logger.warning(f"无法获取{symbol}的历史数据")
                continue
            
            all_data[symbol] = data
        
        if not all_data:
            self.logger.error("没有可用的历史数据，回测终止")
            self.is_running = False
            return BacktestResult(self.portfolio, self.config, start_time, end_time)
        
        # 确定所有时间戳
        all_timestamps = set()
        for data in all_data.values():
            all_timestamps.update(data['timestamp'].tolist())
        
        all_timestamps = sorted(all_timestamps)
        
        # 运行回测循环
        start_processing_time = datetime.now()
        
        for timestamp in all_timestamps:
            self.current_date = timestamp
            
            # 准备当前时间戳的数据
            current_bars = {}
            for symbol, data in all_data.items():
                matching_data = data[data['timestamp'] == timestamp]
                if not matching_data.empty:
                    bar = matching_data.iloc[0].to_dict()
                    current_bars[symbol] = bar
            
            if not current_bars:
                continue
            
            # 为每个策略提供最新数据
            for strategy in self.strategy_manager.get_strategies():
                # 检查策略是否支持当前交易品种
                strategy_symbols = []
                for symbol in symbols:
                    if strategy.can_trade_symbol(symbol):
                        strategy_symbols.append(symbol)
                
                if not strategy_symbols:
                    continue
                
                # 提供数据并生成信号
                strategy_bars = {symbol: bar for symbol, bar in current_bars.items() if symbol in strategy_symbols}
                if strategy_bars:
                    strategy.on_data(strategy_bars, timestamp)
                    
                    # 获取策略生成的信号
                    signals = strategy.get_signals()
                    self.stats["generated_signals"] += len(signals)
                    
                    # 处理信号
                    self._process_signals(signals, current_bars)
            
            # 更新投资组合的市场价格
            symbol_prices = {}
            price_type = self.config.get('price_type', 'close')
            for symbol, bar in current_bars.items():
                if price_type in bar:
                    symbol_prices[symbol] = bar[price_type]
                elif 'close' in bar:
                    symbol_prices[symbol] = bar['close']
            
            self.portfolio.update_market_prices(symbol_prices)
            
            # 如果启用了风险控制，则更新风险级别
            if self.risk_controller is not None:
                portfolio_data = {
                    "equity": self.portfolio.equity,
                    "initial_capital": self.portfolio.initial_capital,
                    "max_drawdown_percent": self.portfolio.max_drawdown_percentage
                }
                self.risk_controller.update_risk_level(portfolio_data)
            
            self.stats["processed_bars"] += len(current_bars)
        
        end_processing_time = datetime.now()
        self.stats["processing_time"] = (end_processing_time - start_processing_time).total_seconds()
        
        self.is_running = False
        self.logger.info(f"回测完成: 处理了 {self.stats['processed_bars']} 个价格条，生成了 {self.stats['generated_signals']} 个信号，执行了 {self.stats['executed_trades']} 笔交易，拒绝了 {self.stats['rejected_trades']} 笔交易")
        
        # 创建回测结果
        self.result = BacktestResult(self.portfolio, self.config, start_time, end_time)
        return self.result
    
    def _process_signals(self, signals: List[Dict[str, Any]], current_bars: Dict[str, Dict[str, Any]]) -> None:
        """
        处理策略生成的信号
        
        Args:
            signals: 信号列表
            current_bars: 当前时间戳的价格条数据
        """
        for signal in signals:
            symbol = signal.get('symbol')
            action = signal.get('action')
            quantity = signal.get('quantity', 0)
            price = signal.get('price')
            strategy_id = signal.get('strategy_id')
            
            if not symbol or not action or quantity <= 0:
                continue
            
            # 如果没有指定价格，则使用当前价格条的价格
            if price is None and symbol in current_bars:
                price_type = self.config.get('price_type', 'close')
                if price_type in current_bars[symbol]:
                    price = current_bars[symbol][price_type]
                elif 'close' in current_bars[symbol]:
                    price = current_bars[symbol]['close']
                else:
                    self.logger.warning(f"无法确定{symbol}的交易价格")
                    continue
            
            # 创建交易对象
            trade = Trade(
                symbol=symbol,
                timestamp=self.current_date,
                direction=action.upper(),
                quantity=quantity,
                price=price,
                strategy_id=strategy_id,
                tags=signal.get('tags', {})
            )
            
            # 设置交易费用
            trade.commission = trade.amount * self.config['commission_rate']
            trade.slippage = trade.amount * self.config['slippage']
            
            # 如果启用了风险控制，则验证交易
            if self.risk_controller is not None:
                trade_data = {
                    "symbol": trade.symbol,
                    "direction": trade.direction,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "amount": trade.amount,
                    "portfolio_value": self.portfolio.get_total_value(),
                    "cash": self.portfolio.cash
                }
                
                is_valid, rejection_reason = self.risk_controller.validate_trade(trade_data)
                
                if not is_valid:
                    self.logger.warning(f"交易被风险控制拒绝: {rejection_reason}")
                    self.stats["rejected_trades"] += 1
                    continue
            
            # 处理交易
            is_opening = True
            if action.upper() == "BUY":
                position = self.portfolio.get_position(symbol)
                if position and position.quantity < 0:
                    is_opening = False  # 买入平空
            elif action.upper() == "SELL":
                position = self.portfolio.get_position(symbol)
                if position and position.quantity > 0:
                    is_opening = False  # 卖出平多
            
            self.portfolio.process_trade(trade, is_opening)
            self.stats["executed_trades"] += 1
    
    def get_result(self) -> Optional[BacktestResult]:
        """
        获取回测结果
        
        Returns:
            回测结果
        """
        return self.result 