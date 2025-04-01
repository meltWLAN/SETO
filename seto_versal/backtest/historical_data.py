#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
历史数据回测模块

使用真实的历史市场数据进行策略回测，实现更精确的策略评估
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union

try:
    from seto_versal.data.tushare_provider import TuShareProvider
except ImportError:
    TuShareProvider = None

from seto_versal.common.constants import TradingPeriod
from seto_versal.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class HistoricalDataBacktest:
    """使用历史数据进行策略回测的类"""
    
    def __init__(self, 
                data_dir: str = 'data/market',
                tushare_token: Optional[str] = None,
                use_cache: bool = True):
        """
        初始化历史数据回测器
        
        Args:
            data_dir: 数据目录
            tushare_token: TuShare API令牌，如不提供则尝试从环境变量获取
            use_cache: 是否使用缓存数据
        """
        self.data_dir = data_dir
        self.tushare_token = tushare_token or os.environ.get('TUSHARE_TOKEN')
        self.use_cache = use_cache
        
        # 初始化数据提供者
        if self.tushare_token and TuShareProvider:
            try:
                self.data_provider = TuShareProvider(
                    token=self.tushare_token,
                    cache_dir=os.path.join(data_dir, 'cache/tushare'),
                    cache_days=7 if use_cache else 0
                )
                logger.info("成功初始化TuShare数据提供者")
            except Exception as e:
                logger.error(f"初始化TuShare数据提供者失败: {e}")
                self.data_provider = None
        else:
            logger.warning("未提供TuShare令牌或未安装TuShare，将使用本地数据")
            self.data_provider = None
        
        # 存储历史数据
        self.historical_data = {}
        
        # 回测结果
        self.backtest_results = {}
        
        logger.info("历史数据回测模块初始化完成")
    
    def load_index_stocks(self, index_code: str) -> List[str]:
        """
        加载指数成分股
        
        Args:
            index_code: 指数代码，如'000300.SH'
            
        Returns:
            成分股代码列表
        """
        # 检查本地文件
        local_file = os.path.join(self.data_dir, f'indices/{index_code}.json')
        if os.path.exists(local_file):
            try:
                with open(local_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    stocks = data.get('stocks', [])
                    logger.info(f"从本地文件加载 {index_code} 成分股: {len(stocks)} 只")
                    return stocks
            except Exception as e:
                logger.warning(f"从本地文件加载 {index_code} 成分股失败: {e}")
        
        # 从TuShare加载
        if self.data_provider:
            try:
                stocks = self.data_provider.get_index_stocks(index_code)
                logger.info(f"从TuShare加载 {index_code} 成分股: {len(stocks)} 只")
                return stocks
            except Exception as e:
                logger.warning(f"从TuShare加载 {index_code} 成分股失败: {e}")
        
        logger.error(f"无法加载 {index_code} 成分股")
        return []
    
    def load_industry_stocks(self, industry: str) -> List[str]:
        """
        加载行业成分股
        
        Args:
            industry: 行业名称
            
        Returns:
            该行业的股票代码列表
        """
        # 检查本地文件
        local_file = os.path.join(self.data_dir, 'sectors/seto_format.json')
        if os.path.exists(local_file):
            try:
                with open(local_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if industry in data:
                        stocks = data[industry].get('stocks', [])
                        logger.info(f"从本地文件加载 {industry} 行业股票: {len(stocks)} 只")
                        return stocks
            except Exception as e:
                logger.warning(f"从本地文件加载 {industry} 行业股票失败: {e}")
        
        # 从TuShare加载
        if self.data_provider:
            try:
                seto_format = self.data_provider.convert_industry_format()
                if industry in seto_format:
                    stocks = seto_format[industry].get('stocks', [])
                    logger.info(f"从TuShare加载 {industry} 行业股票: {len(stocks)} 只")
                    return stocks
            except Exception as e:
                logger.warning(f"从TuShare加载 {industry} 行业股票失败: {e}")
        
        logger.error(f"无法加载 {industry} 行业股票")
        return []
    
    def load_stock_daily_data(self, 
                             symbols: List[str], 
                             start_date: Optional[str] = None, 
                             end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        加载股票日线数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            
        Returns:
            股票日线数据字典，key为股票代码，value为DataFrame
        """
        # 设置默认日期
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        # 初始化结果
        results = {}
        
        # 使用TuShare加载数据
        if self.data_provider:
            for symbol in symbols:
                try:
                    df = self.data_provider.get_stock_daily(
                        ts_code=symbol, start_date=start_date, end_date=end_date
                    )
                    if not df.empty:
                        results[symbol] = df
                except Exception as e:
                    logger.warning(f"加载股票 {symbol} 日线数据失败: {e}")
        else:
            # 从本地CSV文件加载数据
            for symbol in symbols:
                local_file = os.path.join(self.data_dir, f'price/{symbol}.csv')
                if os.path.exists(local_file):
                    try:
                        df = pd.read_csv(local_file, parse_dates=['trade_date'])
                        # 筛选日期范围
                        df = df[(df['trade_date'] >= start_date) & 
                               (df['trade_date'] <= end_date)]
                        if not df.empty:
                            results[symbol] = df
                    except Exception as e:
                        logger.warning(f"从本地文件加载股票 {symbol} 日线数据失败: {e}")
        
        logger.info(f"加载了 {len(results)} 只股票的日线数据")
        return results
    
    def prepare_backtest_data(self, 
                             symbols: List[str], 
                             start_date: str, 
                             end_date: str) -> bool:
        """
        准备回测数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            
        Returns:
            是否成功
        """
        try:
            # 加载股票日线数据
            daily_data = self.load_stock_daily_data(symbols, start_date, end_date)
            
            # 转换为回测格式
            for symbol, df in daily_data.items():
                # 初始化该股票的历史数据字典
                self.historical_data[symbol] = {}
                
                # 按日期组织数据
                for _, row in df.iterrows():
                    date_str = row['trade_date']
                    if isinstance(date_str, str):
                        date = datetime.strptime(date_str, '%Y%m%d')
                    else:
                        date = date_str
                    
                    # 存储OHLCV数据
                    self.historical_data[symbol][date] = {
                        'open': row.get('open', 0.0),
                        'high': row.get('high', 0.0),
                        'low': row.get('low', 0.0),
                        'close': row.get('close', 0.0),
                        'volume': row.get('vol', 0.0),
                        'amount': row.get('amount', 0.0),
                        'change': row.get('pct_chg', 0.0) / 100 if 'pct_chg' in row else 0.0
                    }
            
            logger.info(f"成功准备回测数据: {len(self.historical_data)} 只股票, "
                       f"从 {start_date} 到 {end_date}")
            return True
            
        except Exception as e:
            logger.error(f"准备回测数据失败: {e}")
            return False
    
    def run_backtest(self, strategy: BaseStrategy, 
                    initial_capital: float = 1000000.0,
                    commission_rate: float = 0.0003,
                    slippage: float = 0.0001) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            strategy: 策略对象
            initial_capital: 初始资金
            commission_rate: 佣金率
            slippage: 滑点
            
        Returns:
            回测结果
        """
        if not self.historical_data:
            logger.error("回测数据未准备，请先调用prepare_backtest_data")
            return {}
        
        try:
            # 提取所有交易日
            all_dates = set()
            for symbol, dates in self.historical_data.items():
                all_dates.update(dates.keys())
            
            # 按时间排序
            trading_days = sorted(all_dates)
            
            # 初始化模拟账户
            account = {
                'cash': initial_capital,
                'positions': {},
                'total_value': initial_capital,
                'total_commission': 0.0,
                'total_slippage': 0.0
            }
            
            # 初始化回测结果
            self.backtest_results = {
                'daily_returns': [],
                'positions': [],
                'trades': []
            }
            
            # 遍历每个交易日
            for day_idx, current_date in enumerate(trading_days):
                # 准备当日市场数据
                market_data = {}
                for symbol, dates in self.historical_data.items():
                    if current_date in dates:
                        market_data[symbol] = dates
                
                # 更新持仓价值
                positions_value = 0.0
                for symbol, position in account['positions'].items():
                    if symbol in market_data and current_date in market_data[symbol]:
                        current_price = market_data[symbol][current_date]['close']
                        position['current_price'] = current_price
                        position['current_value'] = position['shares'] * current_price
                        positions_value += position['current_value']
                
                prev_total_value = account['total_value']
                account['total_value'] = account['cash'] + positions_value
                
                # 计算每日收益率
                if day_idx > 0:
                    daily_return = (account['total_value'] - prev_total_value) / prev_total_value
                else:
                    daily_return = 0.0
                
                # 记录每日结果
                self.backtest_results['daily_returns'].append({
                    'date': current_date,
                    'total_value': account['total_value'],
                    'cash': account['cash'],
                    'positions_value': positions_value,
                    'return': daily_return
                })
                
                # 记录当日持仓
                positions = []
                for symbol, position in account['positions'].items():
                    positions.append({
                        'date': current_date,
                        'symbol': symbol,
                        'shares': position['shares'],
                        'current_price': position.get('current_price', 0.0),
                        'current_value': position.get('current_value', 0.0),
                        'cost_basis': position['cost_basis'],
                        'unrealized_pnl': position.get('current_value', 0.0) - position['cost_basis']
                    })
                
                self.backtest_results['positions'].append({
                    'date': current_date,
                    'positions': positions
                })
                
                # 生成交易信号
                signals = strategy.generate_signals(
                    market_data=market_data,
                    positions=account['positions'],
                    market_state={'date': current_date}
                )
                
                # 执行交易
                for signal in signals:
                    symbol = signal.symbol
                    signal_type = signal.signal_type
                    price = signal.price
                    quantity = signal.quantity
                    
                    # 添加滑点
                    if signal_type == 'buy':
                        execution_price = price * (1 + slippage)
                    else:
                        execution_price = price * (1 - slippage)
                    
                    # 计算佣金
                    commission = execution_price * quantity * commission_rate
                    
                    # 执行买入
                    if signal_type == 'buy':
                        # 检查资金是否足够
                        cost = execution_price * quantity + commission
                        if cost > account['cash']:
                            # 资金不足，按可用资金调整数量
                            adjusted_quantity = int((account['cash'] - commission) / execution_price)
                            if adjusted_quantity <= 0:
                                continue
                            
                            quantity = adjusted_quantity
                            cost = execution_price * quantity + commission
                        
                        # 更新持仓
                        if symbol not in account['positions']:
                            account['positions'][symbol] = {
                                'symbol': symbol,
                                'shares': quantity,
                                'cost_basis': execution_price * quantity,
                                'entry_price': execution_price
                            }
                        else:
                            position = account['positions'][symbol]
                            position['shares'] += quantity
                            position['cost_basis'] += execution_price * quantity
                        
                        # 更新资金
                        account['cash'] -= cost
                        account['total_commission'] += commission
                        account['total_slippage'] += price * quantity * slippage
                        
                        # 记录交易
                        self.backtest_results['trades'].append({
                            'date': current_date,
                            'symbol': symbol,
                            'type': 'buy',
                            'quantity': quantity,
                            'price': execution_price,
                            'commission': commission,
                            'slippage': price * quantity * slippage
                        })
                        
                    # 执行卖出
                    elif signal_type == 'sell':
                        # 检查持仓是否足够
                        if symbol not in account['positions'] or account['positions'][symbol]['shares'] < quantity:
                            # 持仓不足，跳过或调整数量
                            if symbol not in account['positions']:
                                continue
                            
                            quantity = account['positions'][symbol]['shares']
                        
                        # 计算收益
                        proceeds = execution_price * quantity - commission
                        
                        # 更新持仓
                        position = account['positions'][symbol]
                        position['shares'] -= quantity
                        position['cost_basis'] -= (position['cost_basis'] / (position['shares'] + quantity)) * quantity
                        
                        # 如果持仓为0，移除
                        if position['shares'] <= 0:
                            del account['positions'][symbol]
                        
                        # 更新资金
                        account['cash'] += proceeds
                        account['total_commission'] += commission
                        account['total_slippage'] += price * quantity * slippage
                        
                        # 记录交易
                        self.backtest_results['trades'].append({
                            'date': current_date,
                            'symbol': symbol,
                            'type': 'sell',
                            'quantity': quantity,
                            'price': execution_price,
                            'commission': commission,
                            'slippage': price * quantity * slippage
                        })
            
            # 计算回测统计数据
            stats = self._calculate_backtest_stats()
            
            logger.info(f"回测完成: 最终资产 {account['total_value']:.2f}, "
                       f"收益率 {stats['total_return']:.2%}")
            
            # 返回完整回测结果
            return {
                'account': account,
                'stats': stats,
                'daily_returns': self.backtest_results['daily_returns'],
                'positions': self.backtest_results['positions'],
                'trades': self.backtest_results['trades']
            }
            
        except Exception as e:
            logger.error(f"运行回测失败: {e}")
            return {}
    
    def _calculate_backtest_stats(self) -> Dict[str, Any]:
        """
        计算回测统计数据
        
        Returns:
            统计数据
        """
        if not self.backtest_results or not self.backtest_results.get('daily_returns'):
            return {}
        
        try:
            # 提取每日收益率
            daily_returns = [day['return'] for day in self.backtest_results['daily_returns']]
            
            # 计算总收益率
            total_return = (self.backtest_results['daily_returns'][-1]['total_value'] / 
                          self.backtest_results['daily_returns'][0]['total_value'] - 1)
            
            # 计算年化收益率
            days = len(daily_returns)
            annualized_return = (1 + total_return) ** (252 / days) - 1
            
            # 计算波动率
            volatility = np.std(daily_returns) * np.sqrt(252)
            
            # 计算夏普比率
            risk_free_rate = 0.02  # 假设无风险利率为2%
            if volatility > 0:
                sharpe_ratio = (annualized_return - risk_free_rate) / volatility
            else:
                sharpe_ratio = 0
            
            # 计算最大回撤
            total_values = [day['total_value'] for day in self.backtest_results['daily_returns']]
            peak = total_values[0]
            max_drawdown = 0
            
            for value in total_values:
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            
            # 计算交易统计
            trades = self.backtest_results['trades']
            total_trades = len(trades)
            
            # 计算胜率
            profitable_trades = sum(1 for trade in trades if trade['type'] == 'sell' and 
                                  trade['price'] > 0)  # 简化的胜率计算
            
            if total_trades > 0:
                win_rate = profitable_trades / total_trades
            else:
                win_rate = 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'win_rate': win_rate
            }
            
        except Exception as e:
            logger.error(f"计算回测统计数据失败: {e}")
            return {}
    
    def generate_backtest_report(self, output_file: Optional[str] = None) -> str:
        """
        生成回测报告
        
        Args:
            output_file: 输出文件路径，如果不提供则返回HTML字符串
            
        Returns:
            HTML报告字符串
        """
        if not self.backtest_results:
            logger.error("没有回测结果，无法生成报告")
            return ""
        
        try:
            # 计算统计数据
            stats = self._calculate_backtest_stats()
            
            # 生成HTML报告
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>SETO-Versal 回测报告</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>SETO-Versal 回测报告</h1>
                <h2>回测统计</h2>
                <table>
                    <tr><th>指标</th><th>值</th></tr>
                    <tr><td>总收益率</td><td>{stats.get('total_return', 0):.2%}</td></tr>
                    <tr><td>年化收益率</td><td>{stats.get('annualized_return', 0):.2%}</td></tr>
                    <tr><td>波动率</td><td>{stats.get('volatility', 0):.2%}</td></tr>
                    <tr><td>夏普比率</td><td>{stats.get('sharpe_ratio', 0):.2f}</td></tr>
                    <tr><td>最大回撤</td><td>{stats.get('max_drawdown', 0):.2%}</td></tr>
                    <tr><td>总交易次数</td><td>{stats.get('total_trades', 0)}</td></tr>
                    <tr><td>胜率</td><td>{stats.get('win_rate', 0):.2%}</td></tr>
                </table>
                
                <h2>资产曲线</h2>
                <div id="equity-curve" style="width:100%; height:400px;"></div>
                
                <h2>最近交易</h2>
                <table>
                    <tr>
                        <th>日期</th>
                        <th>股票</th>
                        <th>类型</th>
                        <th>数量</th>
                        <th>价格</th>
                        <th>佣金</th>
                    </tr>
            """
            
            # 添加最近的交易记录
            recent_trades = self.backtest_results['trades'][-10:] if len(self.backtest_results['trades']) > 10 else self.backtest_results['trades']
            for trade in recent_trades:
                html += f"""
                    <tr>
                        <td>{trade['date'].strftime('%Y-%m-%d')}</td>
                        <td>{trade['symbol']}</td>
                        <td>{trade['type']}</td>
                        <td>{trade['quantity']}</td>
                        <td>{trade['price']:.2f}</td>
                        <td>{trade['commission']:.2f}</td>
                    </tr>
                """
            
            html += """
                </table>
                
                <h2>当前持仓</h2>
                <table>
                    <tr>
                        <th>股票</th>
                        <th>数量</th>
                        <th>当前价格</th>
                        <th>当前价值</th>
                        <th>成本基础</th>
                        <th>未实现盈亏</th>
                    </tr>
            """
            
            # 添加当前持仓
            if self.backtest_results['positions']:
                latest_positions = self.backtest_results['positions'][-1]['positions']
                for position in latest_positions:
                    pnl_class = "positive" if position['unrealized_pnl'] >= 0 else "negative"
                    html += f"""
                        <tr>
                            <td>{position['symbol']}</td>
                            <td>{position['shares']}</td>
                            <td>{position['current_price']:.2f}</td>
                            <td>{position['current_value']:.2f}</td>
                            <td>{position['cost_basis']:.2f}</td>
                            <td class="{pnl_class}">{position['unrealized_pnl']:.2f}</td>
                        </tr>
                    """
            
            html += """
                </table>
                
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script>
                    // 绘制资产曲线
                    var dates = [
            """
            
            # 添加日期数据
            for day in self.backtest_results['daily_returns']:
                html += f"'{day['date'].strftime('%Y-%m-%d')}',"
            
            html += """
                    ];
                    var total_values = [
            """
            
            # 添加资产价值数据
            for day in self.backtest_results['daily_returns']:
                html += f"{day['total_value']},"
            
            html += """
                    ];
                    
                    var trace = {
                        x: dates,
                        y: total_values,
                        type: 'scatter',
                        mode: 'lines',
                        name: '资产价值',
                        line: {color: '#17BECF'}
                    };
                    
                    var layout = {
                        title: '资产曲线',
                        xaxis: {title: '日期'},
                        yaxis: {title: '价值'}
                    };
                    
                    Plotly.newPlot('equity-curve', [trace], layout);
                </script>
            </body>
            </html>
            """
            
            # 如果提供了输出文件路径，保存到文件
            if output_file:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html)
                logger.info(f"回测报告已保存到 {output_file}")
            
            return html
            
        except Exception as e:
            logger.error(f"生成回测报告失败: {e}")
            return ""


# 示例使用代码
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 检查示例策略
    try:
        from seto_versal.strategies.sector_template import SectorRotationStrategy
        
        # 初始化回测器
        backtest = HistoricalDataBacktest()
        
        # 加载沪深300成分股
        stocks = backtest.load_index_stocks('000300.SH')
        
        if stocks:
            # 准备回测数据
            start_date = '20220101'
            end_date = '20220630'
            success = backtest.prepare_backtest_data(stocks[:30], start_date, end_date)
            
            if success:
                # 初始化策略
                strategy = SectorRotationStrategy(
                    lookback_period=20,
                    sector_count=3,
                    stocks_per_sector=3
                )
                
                # 运行回测
                results = backtest.run_backtest(strategy)
                
                # 生成报告
                backtest.generate_backtest_report('reports/sector_rotation_backtest.html')
                
                print(f"回测完成: 总收益率 {results['stats']['total_return']:.2%}")
    except ImportError:
        print("无法导入示例策略，请确保已实现相关模块") 