import numpy as np
import logging
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class TrendAgent:
    """
    趋势交易代理，基于动量和趋势指标分析市场
    """
    
    def __init__(self, name=None, symbols=None, lookback_period=20, 
                 threshold=0.05, ema_periods=(5, 20), market_state=None,
                 risk_manager=None, logging_level=logging.INFO,
                 confidence_threshold=0.7, max_positions=5, **kwargs):
        """
        初始化趋势代理
        
        Args:
            name: 代理名称
            symbols: 交易符号列表
            lookback_period: 回溯周期
            threshold: 趋势识别阈值
            ema_periods: EMA周期元组 (短期, 长期)
            market_state: 市场状态实例
            risk_manager: 风险管理器实例
            logging_level: 日志级别
            confidence_threshold: 信号置信度阈值
            max_positions: 最大持仓数量
        """
        self.name = name or f"TrendAgent-{datetime.now().strftime('%m%d%H%M')}"
        self.symbols = symbols or []
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.short_period, self.long_period = ema_periods
        self.market_state = market_state
        self.risk_manager = risk_manager
        self.confidence_threshold = confidence_threshold
        self.max_positions = max_positions
        
        # 设置日志
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.logger.setLevel(logging_level)
        
        # 初始化性能指标
        self.performance_metrics = {
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'total_trades': 0,
            'profitable_trades': 0,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 交易历史
        self.trade_history = []
        
        # 当前持仓
        self.positions = {}
        
        # 初始化技术指标缓存
        self.indicators_cache = {}
        
        # 信号缓存
        self.signals_cache = {
            'last_update': None,
            'signals': {}
        }
        
        self.logger.info(f"Initialized {self.name} with {len(self.symbols)} symbols")
    
    def update(self):
        """
        更新代理状态和指标
        """
        try:
            self.logger.debug(f"Updating {self.name}")
            
            # 清除指标缓存
            self.indicators_cache = {}
            
            # 更新持仓盈亏
            self._update_positions()
            
            # 更新性能指标
            self._update_performance_metrics()
            
            # 生成新信号
            self._generate_signals()
            
            return True
        except Exception as e:
            self.logger.error(f"Error updating agent: {e}")
            return False
    
    def _update_positions(self):
        """更新当前持仓状态"""
        if not self.market_state:
            return
            
        updated_positions = {}
        
        for symbol, position in self.positions.items():
            current_price = self.market_state.get_price(symbol)
            if not current_price:
                continue
                
            entry_price = position.get('entry_price', current_price)
            quantity = position.get('quantity', 0)
            entry_date = position.get('entry_date', datetime.now().strftime('%Y-%m-%d'))
            
            # 计算持仓盈亏
            profit_loss = (current_price - entry_price) * quantity
            profit_loss_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
            
            # 更新持仓信息
            updated_positions[symbol] = {
                'symbol': symbol,
                'entry_price': entry_price,
                'current_price': current_price,
                'quantity': quantity,
                'profit_loss': round(profit_loss, 2),
                'profit_loss_pct': round(profit_loss_pct, 2),
                'entry_date': entry_date,
                'days_held': (datetime.now() - datetime.strptime(entry_date, '%Y-%m-%d')).days
            }
            
        self.positions = updated_positions
    
    def _update_performance_metrics(self):
        """更新性能指标"""
        if len(self.trade_history) == 0:
            return
            
        # 计算总交易数和盈利交易数
        total_trades = len(self.trade_history)
        profitable_trades = sum(1 for trade in self.trade_history if trade.get('profit_loss', 0) > 0)
        
        # 计算胜率
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # 计算平均盈利
        profits = [trade.get('profit_loss', 0) for trade in self.trade_history]
        avg_profit = sum(profits) / total_trades if total_trades > 0 else 0
        
        # 计算最大回撤
        # 简化计算，使用交易历史中的最大亏损
        losses = [p for p in profits if p < 0]
        max_drawdown = min(losses) if losses else 0
        
        # 计算夏普比率（简化版）
        profit_std = np.std(profits) if profits else 1
        sharpe_ratio = avg_profit / profit_std if profit_std > 0 else 0
        
        # 更新性能指标
        self.performance_metrics = {
            'win_rate': round(win_rate * 100, 2),
            'avg_profit': round(avg_profit, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_performance(self):
        """获取性能指标"""
        return self.performance_metrics
    
    def get_positions(self):
        """获取当前持仓"""
        return list(self.positions.values())
    
    def get_trade_history(self):
        """获取交易历史"""
        return self.trade_history
        
    def _calculate_indicators(self, symbol):
        """
        计算技术指标
        
        Args:
            symbol: 股票代码
            
        Returns:
            dict: 包含计算得到的指标
        """
        # 检查缓存
        if symbol in self.indicators_cache:
            return self.indicators_cache[symbol]
            
        if not self.market_state:
            return None
            
        # 获取价格历史
        prices = self.market_state.get_price_history(symbol, self.lookback_period)
        if not prices or len(prices) < self.long_period:
            return None
            
        # 将价格转换为numpy数组
        prices_array = np.array(prices)
        
        # 计算EMA
        ema_short = self._calculate_ema(prices_array, self.short_period)
        ema_long = self._calculate_ema(prices_array, self.long_period)
        
        # 计算MACD
        macd = ema_short - ema_long
        signal = self._calculate_ema(macd, 9)
        histogram = macd - signal
        
        # 计算RSI
        rsi = self._calculate_rsi(prices_array, 14)
        
        # 计算趋势强度
        trend_strength = self._calculate_trend_strength(prices_array, ema_short, ema_long)
        
        # 计算波动率
        volatility = self._calculate_volatility(prices_array)
        
        # 创建指标字典
        indicators = {
            'symbol': symbol,
            'current_price': prices_array[-1],
            'ema_short': ema_short[-1],
            'ema_long': ema_long[-1],
            'macd': macd[-1],
            'signal': signal[-1],
            'histogram': histogram[-1],
            'rsi': rsi[-1],
            'trend_strength': trend_strength,
            'volatility': volatility,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 缓存指标
        self.indicators_cache[symbol] = indicators
        
        return indicators
        
    def _calculate_ema(self, data, period):
        """计算指数移动平均线"""
        if len(data) < period:
            return np.array([data.mean()] * len(data))
            
        ema = np.zeros_like(data)
        ema[0:period] = data[0:period].mean()
        
        # EMA权重
        k = 2.0 / (period + 1)
        
        # 计算EMA
        for i in range(period, len(data)):
            ema[i] = data[i] * k + ema[i-1] * (1 - k)
            
        return ema
        
    def _calculate_rsi(self, data, period=14):
        """计算相对强弱指标(RSI)"""
        if len(data) <= period:
            return 50.0  # 默认中性值
            
        # 计算价格变化
        deltas = np.diff(data)
        seed = deltas[:period]
        
        # 分离上涨和下跌
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100.0
            
        rs = up / down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
        
    def _calculate_trend_strength(self, prices, ema_short, ema_long):
        """计算趋势强度"""
        # 计算EMA之间的差距百分比
        ema_diff = (ema_short[-1] - ema_long[-1]) / ema_long[-1] * 100
        
        # 计算价格斜率
        price_slope = 0
        if len(prices) > 5:
            price_slope = (prices[-1] - prices[-5]) / prices[-5] * 100
            
        # 权重综合
        trend_strength = 0.7 * ema_diff + 0.3 * price_slope
        
        return round(trend_strength, 2)
        
    def _calculate_volatility(self, prices, period=10):
        """计算波动率"""
        if len(prices) < period:
            return 0.0
            
        # 计算最近期间的标准差
        recent_prices = prices[-period:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns) * 100  # 百分比表示
        
        return round(volatility, 2)
    
    def _generate_signals(self):
        """生成交易信号"""
        if not self.market_state or not self.symbols:
            return
            
        now = datetime.now()
        
        # 检查信号缓存是否需要更新
        if (self.signals_cache['last_update'] and 
            (now - self.signals_cache['last_update']).total_seconds() < 300):  # 5分钟内不重新计算
            return
            
        buy_signals = []
        sell_signals = []
        
        for symbol in self.symbols:
            # 获取技术指标
            indicators = self._calculate_indicators(symbol)
            if not indicators:
                continue
                
            # 计算信号得分
            buy_score, sell_score = self._calculate_signal_scores(indicators)
            
            # 生成买入信号
            if buy_score >= self.confidence_threshold and len(buy_signals) < self.max_positions:
                # 检查风险控制
                if self.risk_manager and not self.risk_manager.check_limit(symbol, 'buy'):
                    self.logger.info(f"Risk limit exceeded for buying {symbol}")
                    continue
                    
                price = indicators['current_price']
                quantity = self._calculate_position_size(symbol, price)
                
                if quantity > 0:
                    buy_signals.append({
                        'symbol': symbol,
                        'price': price,
                        'quantity': quantity,
                        'confidence': round(buy_score, 2),
                        'type': 'trend_following',
                        'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                        'reason': self._generate_signal_reason(indicators, 'buy')
                    })
            
            # 生成卖出信号
            if symbol in self.positions and sell_score >= self.confidence_threshold:
                position = self.positions[symbol]
                
                sell_signals.append({
                    'symbol': symbol,
                    'price': indicators['current_price'],
                    'quantity': position.get('quantity', 0),
                    'confidence': round(sell_score, 2),
                    'type': 'trend_reversal',
                    'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                    'reason': self._generate_signal_reason(indicators, 'sell')
                })
        
        # 更新信号缓存
        self.signals_cache = {
            'last_update': now,
            'signals': {
                'buy': buy_signals,
                'sell': sell_signals
            }
        }
    
    def _calculate_signal_scores(self, indicators):
        """
        计算买入和卖出信号的置信度分数
        
        Args:
            indicators: 技术指标字典
            
        Returns:
            tuple: (买入分数, 卖出分数)
        """
        # 初始化分数
        buy_score = 0.0
        sell_score = 0.0
        
        # 1. EMA交叉信号 (权重: 30%)
        ema_short = indicators['ema_short']
        ema_long = indicators['ema_long']
        
        if ema_short > ema_long:
            buy_score += 0.3
        else:
            sell_score += 0.3
            
        # 2. MACD信号 (权重: 25%)
        macd = indicators['macd']
        signal = indicators['signal']
        histogram = indicators['histogram']
        
        if macd > signal and histogram > 0:
            buy_score += 0.25
        elif macd < signal and histogram < 0:
            sell_score += 0.25
            
        # 3. RSI指标 (权重: 20%)
        rsi = indicators['rsi']
        
        # RSI低于30为超卖, 高于70为超买
        if rsi < 30:
            buy_score += 0.2
        elif rsi > 70:
            sell_score += 0.2
            
        # 4. 趋势强度 (权重: 15%)
        trend_strength = indicators['trend_strength']
        
        if trend_strength > 0:
            buy_score += 0.15 * min(1.0, trend_strength / 5.0)
        else:
            sell_score += 0.15 * min(1.0, abs(trend_strength) / 5.0)
            
        # 5. 波动率调整 (权重: 10%)
        # 高波动率降低信号可信度
        volatility = indicators['volatility']
        volatility_factor = max(0, 1 - volatility / 10.0)
        
        buy_score *= (0.9 + 0.1 * volatility_factor)
        sell_score *= (0.9 + 0.1 * volatility_factor)
        
        return buy_score, sell_score
    
    def _calculate_position_size(self, symbol, price):
        """
        计算仓位大小
        
        Args:
            symbol: 股票代码
            price: 当前价格
            
        Returns:
            int: 建议的股票数量
        """
        if not self.risk_manager:
            return 100  # 默认购买100股
            
        # 获取可用资金
        available_cash = self.risk_manager.get_available_cash()
        
        # 计算最大允许的仓位大小
        max_position_value = available_cash * 0.95 / self.max_positions
        
        # 计算股票数量（向下取整到100股的倍数）
        quantity = int(max_position_value / price / 100) * 100
        
        # 确保至少购买100股
        quantity = max(100, quantity)
        
        return quantity
    
    def _generate_signal_reason(self, indicators, signal_type):
        """
        生成信号原因文本
        
        Args:
            indicators: 技术指标字典
            signal_type: 信号类型 ('buy' 或 'sell')
            
        Returns:
            str: 信号原因文本
        """
        reasons = []
        
        if signal_type == 'buy':
            if indicators['ema_short'] > indicators['ema_long']:
                reasons.append(f"短期EMA({self.short_period}日)上穿长期EMA({self.long_period}日)")
                
            if indicators['macd'] > indicators['signal']:
                reasons.append("MACD上穿信号线")
                
            if indicators['rsi'] < 30:
                reasons.append(f"RSI={indicators['rsi']:.1f}，处于超卖区")
                
            if indicators['trend_strength'] > 2:
                reasons.append(f"趋势强度={indicators['trend_strength']}，呈现明显上升趋势")
                
        else:  # sell signal
            if indicators['ema_short'] < indicators['ema_long']:
                reasons.append(f"短期EMA({self.short_period}日)下穿长期EMA({self.long_period}日)")
                
            if indicators['macd'] < indicators['signal']:
                reasons.append("MACD下穿信号线")
                
            if indicators['rsi'] > 70:
                reasons.append(f"RSI={indicators['rsi']:.1f}，处于超买区")
                
            if indicators['trend_strength'] < -2:
                reasons.append(f"趋势强度={indicators['trend_strength']}，呈现明显下降趋势")
                
            # 止损/止盈策略
            if signal_type == 'sell' and indicators['symbol'] in self.positions:
                position = self.positions[indicators['symbol']]
                profit_loss_pct = position.get('profit_loss_pct', 0)
                
                if profit_loss_pct <= -5:
                    reasons.append(f"触发止损，当前亏损{profit_loss_pct:.1f}%")
                elif profit_loss_pct >= 10:
                    reasons.append(f"触发止盈，当前盈利{profit_loss_pct:.1f}%")
        
        if not reasons:
            return "基于综合技术指标分析" if signal_type == 'buy' else "基于综合风险评估"
            
        return "；".join(reasons)
    
    def get_trading_signals(self):
        """
        获取当前交易信号
        
        Returns:
            dict: 包含买入和卖出信号的字典
        """
        # 如果缓存过期或不存在，重新生成信号
        if (not self.signals_cache['last_update'] or 
            (datetime.now() - self.signals_cache['last_update']).total_seconds() > 300):
            self._generate_signals()
            
        return self.signals_cache['signals']
    
    def execute_signals(self):
        """执行交易信号"""
        if not self.market_state:
            return False
            
        signals = self.get_trading_signals()
        
        # 处理卖出信号（优先执行）
        for sell_signal in signals.get('sell', []):
            symbol = sell_signal['symbol']
            price = sell_signal['price']
            quantity = sell_signal['quantity']
            
            if symbol in self.positions and quantity > 0:
                # 记录交易
                trade = {
                    'symbol': symbol,
                    'action': 'sell',
                    'price': price,
                    'quantity': quantity,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_price': self.positions[symbol].get('entry_price', price),
                    'profit_loss': (price - self.positions[symbol].get('entry_price', price)) * quantity,
                    'profit_loss_pct': (price - self.positions[symbol].get('entry_price', price)) / 
                                      self.positions[symbol].get('entry_price', price) * 100,
                    'reason': sell_signal.get('reason', 'Strategy signal')
                }
                
                self.trade_history.append(trade)
                
                # 从持仓中移除
                del self.positions[symbol]
                
                self.logger.info(f"Executed SELL: {symbol} x {quantity} @ {price}")
        
        # 处理买入信号
        for buy_signal in signals.get('buy', []):
            symbol = buy_signal['symbol']
            price = buy_signal['price']
            quantity = buy_signal['quantity']
            
            # 跳过已有持仓的股票
            if symbol in self.positions:
                continue
                
            # 检查持仓数量限制
            if len(self.positions) >= self.max_positions:
                self.logger.info(f"Maximum positions reached, skipping buy for {symbol}")
                continue
                
            # 添加到持仓
            self.positions[symbol] = {
                'symbol': symbol,
                'entry_price': price,
                'current_price': price,
                'quantity': quantity,
                'profit_loss': 0,
                'profit_loss_pct': 0,
                'entry_date': datetime.now().strftime('%Y-%m-%d'),
                'days_held': 0
            }
            
            # 记录交易
            trade = {
                'symbol': symbol,
                'action': 'buy',
                'price': price,
                'quantity': quantity,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'reason': buy_signal.get('reason', 'Strategy signal')
            }
            
            self.trade_history.append(trade)
            
            self.logger.info(f"Executed BUY: {symbol} x {quantity} @ {price}")
        
        # 更新性能指标
        self._update_performance_metrics()
        
        return True 