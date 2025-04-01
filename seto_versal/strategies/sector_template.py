#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
行业策略模板

这个模板提供了创建基于行业分析的策略的基础框架，
可以根据特定需求进行扩展和定制化。
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

from seto_versal.strategies.base import BaseStrategy
from seto_versal.common.constants import SignalType, OrderType
from seto_versal.common.models import Signal

logger = logging.getLogger(__name__)

class SectorBaseStrategy(BaseStrategy):
    """
    行业策略基类 - 为基于行业分析的策略提供基础框架
    
    这个基类提供了:
    1. 行业数据处理的标准接口
    2. 行业相对强度分析的基本方法
    3. 行业内股票筛选的通用功能
    4. 行业轮动识别的基础算法
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        sector_count: int = 3,
        stocks_per_sector: int = 3,
        min_sector_momentum: float = 0.05,
        min_relative_strength: float = 1.2,
        profit_take: float = 0.15,
        stop_loss: float = 0.07,
        enable_sector_timing: bool = True,
        enable_stock_timing: bool = True,
        **kwargs
    ):
        """
        初始化行业策略基类
        
        Args:
            lookback_period: 计算动量的历史区间(天)
            sector_count: 选择的行业数量
            stocks_per_sector: 每个行业选择的股票数量
            min_sector_momentum: 行业选择的最小动量阈值
            min_relative_strength: 股票选择的最小相对强度(相对行业)
            profit_take: 止盈阈值
            stop_loss: 止损阈值
            enable_sector_timing: 是否启用行业择时
            enable_stock_timing: 是否启用个股择时
        """
        super().__init__(**kwargs)
        self.lookback_period = lookback_period
        self.sector_count = sector_count
        self.stocks_per_sector = stocks_per_sector
        self.min_sector_momentum = min_sector_momentum
        self.min_relative_strength = min_relative_strength
        self.profit_take = profit_take
        self.stop_loss = stop_loss
        self.enable_sector_timing = enable_sector_timing
        self.enable_stock_timing = enable_stock_timing
        
        # 状态追踪
        self.current_sectors = []  # 当前关注的行业
        self.sector_entry_time = {}  # 行业入场时间
        self.sector_entry_price = {}  # 行业入场价格指数
        
        logger.info(
            f"初始化行业策略: lookback={lookback_period}, "
            f"sectors={sector_count}, stocks_per_sector={stocks_per_sector}"
        )
    
    def generate_signals(
        self, 
        market_data: Dict[str, Dict[datetime, Dict[str, float]]], 
        positions: Dict[str, Dict[str, Any]] = None,
        market_state: Dict[str, Any] = None,
        **kwargs
    ) -> List[Signal]:
        """
        根据行业分析生成交易信号
        
        Args:
            market_data: 按股票代码组织的市场数据
            positions: 当前持仓
            market_state: 当前市场状态
            
        Returns:
            交易信号列表
        """
        signals = []
        
        try:
            # 验证数据
            if not market_data:
                logger.warning("未提供市场数据")
                return signals
                
            # 从市场状态中提取行业信息
            if not market_state or 'sector_performance' not in market_state:
                logger.warning("市场状态中缺少行业表现数据")
                return signals
                
            sector_performance = market_state.get('sector_performance', {})
            sector_stocks = market_state.get('sector_stocks', {})
            
            # 1. 识别表现最好的行业
            top_sectors = self._get_top_sectors(sector_performance)
            if not top_sectors:
                logger.info("没有行业满足动量标准")
                return signals
                
            logger.debug(f"识别到的强势行业: {', '.join(top_sectors)}")
            
            # 更新当前关注的行业
            self.current_sectors = top_sectors
            
            # 2. 对每个强势行业，找出表现最好的股票
            for sector in top_sectors:
                # 如果该行业没有股票则跳过
                if sector not in sector_stocks:
                    continue
                    
                # 获取行业内的股票
                stocks_in_sector = sector_stocks[sector]
                if not stocks_in_sector:
                    continue
                    
                # 计算股票表现
                stock_performance = self._calculate_stock_performance(
                    stocks_in_sector, market_data
                )
                
                # 选择表现最好的股票
                top_stocks = self._get_top_stocks(stock_performance)
                
                # 对这些股票生成买入信号
                for symbol in top_stocks:
                    if symbol not in market_data:
                        continue
                        
                    # 如果启用个股择时，检查是否是买入时机
                    if self.enable_stock_timing and not self._is_good_entry(symbol, market_data):
                        logger.debug(f"股票 {symbol} 不是良好的入场时机")
                        continue
                    
                    signal = self._create_buy_signal(symbol, market_data, sector)
                    if signal:
                        signals.append(signal)
            
            # 3. 处理当前持仓的退出信号
            if positions:
                exit_signals = self._process_exits(
                    positions, market_data, sector_performance
                )
                signals.extend(exit_signals)
                
            logger.info(f"行业策略生成了 {len(signals)} 个交易信号")
            return signals
            
        except Exception as e:
            logger.error(f"行业策略出现错误: {str(e)}", exc_info=True)
            return []
    
    def _get_top_sectors(self, sector_performance: Dict[str, float]) -> List[str]:
        """
        根据动量识别表现最好的行业
        
        Args:
            sector_performance: 行业表现指标字典
            
        Returns:
            顶级行业名称列表
        """
        # 按最小动量过滤行业
        qualified_sectors = {
            sector: perf for sector, perf in sector_performance.items()
            if perf >= self.min_sector_momentum
        }
        
        # 按表现排序(降序)
        sorted_sectors = sorted(
            qualified_sectors.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 取前N个行业
        top_sectors = [
            sector for sector, _ in sorted_sectors[:self.sector_count]
        ]
        
        return top_sectors
    
    def _calculate_stock_performance(
        self,
        symbols: List[str],
        market_data: Dict[str, Dict[datetime, Dict[str, float]]]
    ) -> Dict[str, float]:
        """
        计算一组股票的表现指标
        
        Args:
            symbols: 股票代码列表
            market_data: 市场数据字典
            
        Returns:
            股票表现指标字典
        """
        stock_performance = {}
        
        for symbol in symbols:
            if symbol not in market_data:
                continue
                
            data = market_data[symbol]
            dates = sorted(data.keys())
            
            # 数据不足则跳过
            if len(dates) < self.lookback_period:
                continue
                
            # 计算回看期的表现
            current_price = data[dates[-1]].get('close', 0)
            past_price = data[dates[-self.lookback_period]].get('close', 0)
            
            if past_price <= 0:
                continue
                
            # 计算表现(百分比变化)
            performance = (current_price - past_price) / past_price
            stock_performance[symbol] = performance
            
        return stock_performance
    
    def _get_top_stocks(self, stock_performance: Dict[str, float]) -> List[str]:
        """
        从表现字典中选择表现最好的股票
        
        Args:
            stock_performance: 股票表现指标字典
            
        Returns:
            顶级股票代码列表
        """
        # 按最小相对强度过滤股票
        market_avg = np.mean(list(stock_performance.values())) if stock_performance else 0
        
        qualified_stocks = {
            symbol: perf for symbol, perf in stock_performance.items()
            if perf >= market_avg * self.min_relative_strength
        }
        
        # 按表现排序(降序)
        sorted_stocks = sorted(
            qualified_stocks.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 取前N个股票
        top_stocks = [
            symbol for symbol, _ in sorted_stocks[:self.stocks_per_sector]
        ]
        
        return top_stocks
    
    def _is_good_entry(
        self, 
        symbol: str, 
        market_data: Dict[str, Dict[datetime, Dict[str, float]]]
    ) -> bool:
        """
        检查股票是否是良好的入场点
        
        Args:
            symbol: 股票代码
            market_data: 市场数据
            
        Returns:
            如果是良好入场点则为True，否则为False
        """
        # 在实际应用中，这里应该包含更复杂的入场逻辑
        # 例如技术指标、突破形态等
        
        # 简单示例：5日均线上穿10日均线
        if symbol not in market_data:
            return False
            
        data = market_data[symbol]
        dates = sorted(data.keys())
        
        # 数据不足则跳过
        if len(dates) < 10:
            return False
            
        # 计算5日和10日均线
        prices = [data[date].get('close', 0) for date in dates[-10:]]
        ma5_prev = np.mean(prices[-6:-1])
        ma5_curr = np.mean(prices[-5:])
        ma10 = np.mean(prices)
        
        # 检查5日均线是否上穿10日均线
        crossover = ma5_prev < ma10 and ma5_curr > ma10
        
        return crossover
    
    def _create_buy_signal(
        self, 
        symbol: str, 
        market_data: Dict[str, Dict[datetime, Dict[str, float]]],
        sector: str
    ) -> Optional[Signal]:
        """
        创建买入信号
        
        Args:
            symbol: 股票代码
            market_data: 市场数据
            sector: 行业名称
            
        Returns:
            交易信号或None
        """
        # 获取当前价格
        data = market_data[symbol]
        dates = sorted(data.keys())
        if not dates:
            return None
            
        current_price = data[dates[-1]].get('close', 0)
        if current_price <= 0:
            return None
            
        # 创建交易信号
        signal = Signal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            price=current_price,
            quantity=100,  # 这里应该计算实际数量
            order_type=OrderType.LIMIT,
            confidence=0.8,
            expiration=datetime.now() + timedelta(days=1),
            metadata={
                'sector': sector,
                'strategy': self.__class__.__name__,
                'reason': f"行业强度选股: {sector} 行业领先股票"
            }
        )
        
        return signal
    
    def _process_exits(
        self,
        positions: Dict[str, Dict[str, Any]],
        market_data: Dict[str, Dict[datetime, Dict[str, float]]],
        sector_performance: Dict[str, float]
    ) -> List[Signal]:
        """
        处理当前持仓的退出信号
        
        Args:
            positions: 当前持仓
            market_data: 市场数据
            sector_performance: 行业表现
            
        Returns:
            退出信号列表
        """
        exit_signals = []
        
        for symbol, position in positions.items():
            # 跳过不在市场数据中的股票
            if symbol not in market_data:
                continue
                
            # 获取当前价格
            data = market_data[symbol]
            dates = sorted(data.keys())
            if not dates:
                continue
                
            current_price = data[dates[-1]].get('close', 0)
            if current_price <= 0:
                continue
                
            # 获取持仓信息
            entry_price = position.get('entry_price', current_price)
            entry_time = position.get('entry_time', datetime.now() - timedelta(days=30))
            sector = position.get('metadata', {}).get('sector', '')
            
            # 检查是否需要退出
            exit_signal = None
            reason = ""
            
            # 止盈检查
            if current_price >= entry_price * (1 + self.profit_take):
                reason = f"止盈: +{(current_price/entry_price - 1)*100:.2f}%"
                exit_signal = self._create_sell_signal(symbol, current_price, reason)
                
            # 止损检查
            elif current_price <= entry_price * (1 - self.stop_loss):
                reason = f"止损: {(current_price/entry_price - 1)*100:.2f}%"
                exit_signal = self._create_sell_signal(symbol, current_price, reason)
                
            # 行业强度减弱检查
            elif sector and sector in sector_performance:
                sector_strength = sector_performance[sector]
                if sector_strength < self.min_sector_momentum / 2:
                    reason = f"行业({sector})强度减弱: {sector_strength:.2f}"
                    exit_signal = self._create_sell_signal(symbol, current_price, reason)
                    
            # 持有时间过长检查
            elif (datetime.now() - entry_time).days > self.lookback_period * 1.5:
                # 如果超过目标期限且收益不明显，考虑退出
                if current_price < entry_price * 1.05:
                    reason = f"持有时间过长: {(datetime.now() - entry_time).days}天"
                    exit_signal = self._create_sell_signal(symbol, current_price, reason)
            
            # 添加退出信号
            if exit_signal:
                exit_signals.append(exit_signal)
                logger.info(f"为股票 {symbol} 生成退出信号: {reason}")
                
        return exit_signals
    
    def _create_sell_signal(
        self, 
        symbol: str, 
        price: float,
        reason: str
    ) -> Signal:
        """
        创建卖出信号
        
        Args:
            symbol: 股票代码
            price: 当前价格
            reason: 卖出原因
            
        Returns:
            交易信号
        """
        signal = Signal(
            symbol=symbol,
            signal_type=SignalType.SELL,
            price=price,
            quantity=0,  # 应由仓位管理设置实际数量，0表示全部
            order_type=OrderType.MARKET,
            confidence=0.9,
            expiration=datetime.now() + timedelta(days=1),
            metadata={
                'strategy': self.__class__.__name__,
                'reason': reason
            }
        )
        
        return signal
    
    def get_current_sectors(self) -> List[str]:
        """获取当前关注的行业"""
        return self.current_sectors
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return {
            "lookback_period": self.lookback_period,
            "sector_count": self.sector_count,
            "stocks_per_sector": self.stocks_per_sector,
            "min_sector_momentum": self.min_sector_momentum,
            "min_relative_strength": self.min_relative_strength,
            "profit_take": self.profit_take,
            "stop_loss": self.stop_loss,
            "enable_sector_timing": self.enable_sector_timing,
            "enable_stock_timing": self.enable_stock_timing
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """设置策略参数"""
        if 'lookback_period' in params:
            self.lookback_period = params['lookback_period']
        if 'sector_count' in params:
            self.sector_count = params['sector_count']
        if 'stocks_per_sector' in params:
            self.stocks_per_sector = params['stocks_per_sector']
        if 'min_sector_momentum' in params:
            self.min_sector_momentum = params['min_sector_momentum']
        if 'min_relative_strength' in params:
            self.min_relative_strength = params['min_relative_strength']
        if 'profit_take' in params:
            self.profit_take = params['profit_take']
        if 'stop_loss' in params:
            self.stop_loss = params['stop_loss']
        if 'enable_sector_timing' in params:
            self.enable_sector_timing = params['enable_sector_timing']
        if 'enable_stock_timing' in params:
            self.enable_stock_timing = params['enable_stock_timing']
            
        logger.info(f"更新行业策略参数: {params}")


# 行业轮动策略示例 - 继承自行业策略基类
class SectorRotationStrategy(SectorBaseStrategy):
    """
    行业轮动策略 - 追踪行业轮动并投资于强势行业
    
    这个策略:
    1. 识别当前经济周期阶段
    2. 寻找符合当前周期且动量强劲的行业
    3. 在这些行业中选择领先的个股
    4. 当行业轮动信号出现时调整持仓
    """
    
    def __init__(
        self,
        rotation_threshold: float = 0.03,
        min_holding_days: int = 15,
        max_holding_days: int = 60,
        cycle_timing_weight: float = 0.5,
        **kwargs
    ):
        """
        初始化行业轮动策略
        
        Args:
            rotation_threshold: 行业轮动信号阈值
            min_holding_days: 最小持有天数
            max_holding_days: 最大持有天数
            cycle_timing_weight: 经济周期择时权重
        """
        super().__init__(**kwargs)
        self.rotation_threshold = rotation_threshold
        self.min_holding_days = min_holding_days
        self.max_holding_days = max_holding_days
        self.cycle_timing_weight = cycle_timing_weight
        
        # 行业轮动特有状态
        self.current_cycle = "mid"  # 当前经济周期: early, mid, late, recession
        self.cycle_start_date = datetime.now() - timedelta(days=90)
        self.rotation_history = []
        
        logger.info(
            f"初始化行业轮动策略: rotation_threshold={rotation_threshold}, "
            f"cycle_timing_weight={cycle_timing_weight}"
        )
    
    def _update_economic_cycle(self, market_state: Dict[str, Any]) -> None:
        """
        更新当前经济周期状态
        
        Args:
            market_state: 市场状态数据
        """
        # 从市场状态中提取经济指标
        # 在实际应用中，这里应使用更复杂的经济周期判断逻辑
        # 包括GDP增速、PMI、失业率、通胀等
        
        indicators = market_state.get('economic_indicators', {})
        
        # 简化的周期判断逻辑 (示例)
        if 'pmi' in indicators and 'gdp_growth' in indicators and 'inflation' in indicators:
            pmi = indicators['pmi']
            gdp_growth = indicators['gdp_growth']
            inflation = indicators['inflation']
            
            if pmi < 50 and gdp_growth < 0:
                self.current_cycle = "recession"
            elif pmi > 55 and gdp_growth > 3 and inflation > 3:
                self.current_cycle = "late"
            elif pmi > 52 and gdp_growth > 2:
                self.current_cycle = "mid"
            else:
                self.current_cycle = "early"
                
            logger.debug(f"当前经济周期判断为: {self.current_cycle}")
    
    def _get_top_sectors(self, sector_performance: Dict[str, float]) -> List[str]:
        """
        根据动量和经济周期识别表现最好的行业
        
        重写基类方法，考虑经济周期因素
        
        Args:
            sector_performance: 行业表现指标字典
            
        Returns:
            顶级行业名称列表
        """
        # 周期偏好行业
        cycle_sectors = {
            "early": ["材料", "工业", "金融"],
            "mid": ["科技", "消费", "医药"],
            "late": ["能源", "必需消费", "医药"],
            "recession": ["必需消费", "公用事业", "医药"]
        }
        
        # 当前周期的偏好行业
        preferred_sectors = cycle_sectors.get(self.current_cycle, [])
        
        # 按最小动量过滤行业
        qualified_sectors = {
            sector: perf for sector, perf in sector_performance.items()
            if perf >= self.min_sector_momentum
        }
        
        # 调整行业得分 - 考虑经济周期
        adjusted_scores = {}
        for sector, score in qualified_sectors.items():
            # 如果是当前周期偏好行业，增加得分
            cycle_bonus = self.cycle_timing_weight if sector in preferred_sectors else 0
            adjusted_scores[sector] = score + cycle_bonus
        
        # 按调整后的得分排序(降序)
        sorted_sectors = sorted(
            adjusted_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 取前N个行业
        top_sectors = [
            sector for sector, _ in sorted_sectors[:self.sector_count]
        ]
        
        return top_sectors
    
    def generate_signals(
        self, 
        market_data: Dict[str, Dict[datetime, Dict[str, float]]], 
        positions: Dict[str, Dict[str, Any]] = None,
        market_state: Dict[str, Any] = None,
        **kwargs
    ) -> List[Signal]:
        """
        根据行业轮动分析生成交易信号
        
        Args:
            market_data: 按股票代码组织的市场数据
            positions: 当前持仓
            market_state: 当前市场状态
            
        Returns:
            交易信号列表
        """
        # 更新经济周期
        if market_state:
            self._update_economic_cycle(market_state)
            
        # 调用基类方法生成信号
        return super().generate_signals(market_data, positions, market_state, **kwargs)
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        params = super().get_parameters()
        params.update({
            "rotation_threshold": self.rotation_threshold,
            "min_holding_days": self.min_holding_days,
            "max_holding_days": self.max_holding_days,
            "cycle_timing_weight": self.cycle_timing_weight,
            "current_cycle": self.current_cycle
        })
        return params 