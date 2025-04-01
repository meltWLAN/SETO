import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

from seto_versal.data.manager import DataManager
from seto_versal.agents.factory import AgentFactory
from seto_versal.data.quality import DataQualityChecker

logger = logging.getLogger(__name__)

class QualityAwareSignalGenerator:
    """
    质量感知的交易信号生成器，在生成交易信号前进行数据质量验证。
    
    这个组件负责在调用交易代理生成信号前，确保使用的市场数据符合质量要求，
    避免因数据问题导致错误的交易决策。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化信号生成器。
        
        Args:
            config: 配置字典，包含数据质量和信号生成的参数
        """
        self.config = config or {}
        self.data_manager = None
        self.agent_factory = None
        self.quality_checker = DataQualityChecker(config)
        
        # 默认质量控制参数
        self.quality_threshold = self.config.get('quality_threshold', 0.8)  # 质量得分阈值
        self.quality_check_enabled = self.config.get('quality_check_enabled', True)  # 是否启用质量检查
        self.fallback_strategy = self.config.get('fallback_strategy', 'skip')  # 数据质量不足时的策略
        self.quality_log_enabled = self.config.get('quality_log_enabled', True)  # 是否记录质量日志
        
        # 质量检查统计
        self.quality_stats = {
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'quality_issues': {},
            'last_check_time': None
        }
    
    def initialize(self, data_manager: DataManager, agent_factory: AgentFactory):
        """
        使用数据管理器和代理工厂初始化信号生成器。
        
        Args:
            data_manager: 数据管理器实例
            agent_factory: 代理工厂实例
        """
        self.data_manager = data_manager
        self.agent_factory = agent_factory
        logger.info("质量感知信号生成器初始化完成")
    
    def _calculate_quality_score(self, validation_result: Dict[str, Any]) -> float:
        """
        根据验证结果计算数据质量得分。
        
        Args:
            validation_result: 数据质量验证结果字典
            
        Returns:
            float: 0到1之间的质量得分
        """
        if not validation_result.get('valid', False):
            # 检查各个验证项目的结果
            details = validation_result.get('details', {})
            
            # 计算每个检查项的权重
            weights = {
                'missing_values': 0.3,    # 缺失值检查权重
                'stale_data': 0.2,        # 数据新鲜度检查权重
                'price_jumps': 0.2,       # 价格跳跃检查权重
                'zero_volume': 0.1,       # 零交易量检查权重
                'consistency': 0.2         # 数据一致性检查权重
            }
            
            # 计算加权得分
            score = 0.0
            for check_name, weight in weights.items():
                if check_name in details:
                    check_result = details[check_name]
                    # 如果检查通过，加上权重分
                    if check_result.get('valid', False):
                        score += weight
                    # 对于缺失值，可以按比例给分
                    elif check_name == 'missing_values' and 'missing_pct' in check_result:
                        # 计算平均缺失率
                        missing_values = check_result.get('missing_pct', {})
                        if missing_values:
                            avg_missing = sum(missing_values.values()) / len(missing_values)
                            # 根据缺失率给部分分数
                            if avg_missing < self.quality_checker.threshold_missing_pct:
                                partial_score = (1 - avg_missing / self.quality_checker.threshold_missing_pct) * weight
                                score += partial_score
            
            return score
        else:
            # 如果整体验证通过，返回满分
            return 1.0
    
    def _log_quality_issue(self, symbol: str, timeframe: str, validation_result: Dict[str, Any]):
        """
        记录数据质量问题。
        
        Args:
            symbol: 交易品种代码
            timeframe: 时间周期
            validation_result: 数据质量验证结果
        """
        if not self.quality_log_enabled:
            return
            
        timestamp = datetime.now()
        
        # 更新统计信息
        self.quality_stats['total_checks'] += 1
        if validation_result.get('valid', False):
            self.quality_stats['passed_checks'] += 1
        else:
            self.quality_stats['failed_checks'] += 1
            
            # 记录具体问题类型
            details = validation_result.get('details', {})
            for check_name, check_result in details.items():
                if not check_result.get('valid', True):
                    if check_name not in self.quality_stats['quality_issues']:
                        self.quality_stats['quality_issues'][check_name] = 0
                    self.quality_stats['quality_issues'][check_name] += 1
        
        self.quality_stats['last_check_time'] = timestamp
        
        # 记录日志
        if not validation_result.get('valid', False):
            logger.warning(f"数据质量检查未通过: {symbol}/{timeframe}, "
                          f"得分: {self._calculate_quality_score(validation_result):.2f}")
            
            # 详细记录各个问题
            details = validation_result.get('details', {})
            failed_checks = [check for check, result in details.items() if not result.get('valid', True)]
            logger.debug(f"失败的检查项: {failed_checks}")
    
    def get_quality_checked_data(self, symbol: str, timeframe: str, 
                                start_date: Optional[Union[str, datetime]] = None,
                                end_date: Optional[Union[str, datetime]] = None) -> Tuple[pd.DataFrame, bool]:
        """
        获取经过质量检查的市场数据。
        
        Args:
            symbol: 交易品种代码
            timeframe: 时间周期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Tuple[pd.DataFrame, bool]: 市场数据和质量检查是否通过
        """
        if self.data_manager is None:
            logger.error("数据管理器未初始化")
            return pd.DataFrame(), False
        
        # 获取市场数据
        data = self.data_manager.get_historical_data(symbol, start_date, end_date, timeframe)
        
        if data is None or data.empty:
            logger.warning(f"未能获取数据: {symbol}/{timeframe}")
            return pd.DataFrame(), False
        
        # 如果禁用质量检查，直接返回数据
        if not self.quality_check_enabled:
            return data, True
        
        # 进行数据质量验证
        validation_result = self.quality_checker.validate_market_data(data)
        
        # 记录质量问题
        self._log_quality_issue(symbol, timeframe, validation_result)
        
        # 计算质量得分
        quality_score = self._calculate_quality_score(validation_result)
        
        # 检查是否满足质量阈值
        quality_passed = quality_score >= self.quality_threshold
        
        if not quality_passed:
            logger.warning(f"数据质量得分 ({quality_score:.2f}) 低于阈值 ({self.quality_threshold})")
            
            # 应用不同的后备策略
            if self.fallback_strategy == 'skip':
                # 跳过生成信号
                return pd.DataFrame(), False
            elif self.fallback_strategy == 'use_anyway':
                # 尽管质量较低，仍然使用数据
                logger.warning(f"尽管质量较低，仍然使用数据 ({symbol}/{timeframe})")
                return data, False
            elif self.fallback_strategy == 'fill':
                # 尝试填充/修复数据
                logger.info(f"尝试修复数据质量问题: {symbol}/{timeframe}")
                fixed_data = self._try_fix_data_issues(data, validation_result)
                return fixed_data, True
        
        return data, quality_passed
    
    def _try_fix_data_issues(self, data: pd.DataFrame, validation_result: Dict[str, Any]) -> pd.DataFrame:
        """
        尝试修复数据质量问题。
        
        Args:
            data: 原始市场数据
            validation_result: 数据质量验证结果
            
        Returns:
            pd.DataFrame: 修复后的数据
        """
        if data is None or data.empty:
            return data
            
        fixed_data = data.copy()
        details = validation_result.get('details', {})
        
        # 处理缺失值
        if 'missing_values' in details and not details['missing_values'].get('valid', True):
            logger.info("修复缺失值")
            # 填充缺失值
            # 对于OHLC价格，使用前向填充
            price_cols = ['open', 'high', 'low', 'close']
            present_price_cols = [col for col in price_cols if col in fixed_data.columns]
            if present_price_cols:
                fixed_data[present_price_cols] = fixed_data[present_price_cols].fillna(method='ffill')
                # 再次前向填充可能没有解决的缺失值
                fixed_data[present_price_cols] = fixed_data[present_price_cols].fillna(method='bfill')
            
            # 对于交易量，用0填充或均值填充
            if 'volume' in fixed_data.columns:
                fixed_data['volume'] = fixed_data['volume'].fillna(fixed_data['volume'].median())
        
        # 处理价格跳跃
        if 'price_jumps' in details and not details['price_jumps'].get('valid', True):
            logger.info("修复价格跳跃")
            if 'close' in fixed_data.columns:
                # 获取跳跃点
                jump_indices = details['price_jumps'].get('jump_indices', [])
                
                # 使用移动平均替换异常值
                if jump_indices:
                    window = 5
                    ma = fixed_data['close'].rolling(window=window, center=True).mean()
                    for idx in jump_indices:
                        if idx in fixed_data.index:
                            fixed_data.loc[idx, 'close'] = ma.loc[idx]
        
        return fixed_data
    
    def generate_signals(self, symbols: List[str], agent_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        为指定的交易品种生成交易信号，在生成前进行数据质量验证。
        
        Args:
            symbols: 交易品种代码列表
            agent_types: 代理类型列表，如果为None则使用所有可用代理
            
        Returns:
            pd.DataFrame: 包含交易信号的DataFrame
        """
        if self.agent_factory is None or self.data_manager is None:
            logger.error("代理工厂或数据管理器未初始化")
            return pd.DataFrame()
        
        all_signals = []
        
        # 获取所有代理
        agents = self.agent_factory.get_agents()
        if agent_types:
            agents = [agent for agent in agents if agent.__class__.__name__ in agent_types]
        
        if not agents:
            logger.warning("没有可用的交易代理")
            return pd.DataFrame()
        
        timeframe = self.config.get('signal_timeframe', 'day')
        lookback_days = self.config.get('lookback_days', 90)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # 为每个交易品种生成信号
        for symbol in symbols:
            # 获取质量检查后的数据
            market_data, quality_passed = self.get_quality_checked_data(
                symbol, timeframe, start_date, end_date
            )
            
            if market_data.empty or not quality_passed:
                if self.fallback_strategy != 'use_anyway':
                    logger.warning(f"由于数据质量问题，跳过 {symbol} 的信号生成")
                    continue
                else:
                    logger.warning(f"尽管存在数据质量问题，仍为 {symbol} 生成信号")
            
            # 对每个代理生成信号
            for agent in agents:
                try:
                    # 生成信号
                    signals = agent.generate_signals(market_data, symbol)
                    
                    if signals is not None and not signals.empty:
                        # 添加代理名称和时间戳
                        signals['agent'] = agent.__class__.__name__
                        signals['timestamp'] = datetime.now()
                        all_signals.append(signals)
                except Exception as e:
                    logger.error(f"代理 {agent.__class__.__name__} 生成信号出错: {str(e)}")
        
        # 合并所有信号
        if all_signals:
            return pd.concat(all_signals, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """
        获取数据质量统计信息。
        
        Returns:
            Dict[str, Any]: 质量统计信息
        """
        if self.quality_stats['total_checks'] > 0:
            self.quality_stats['pass_rate'] = self.quality_stats['passed_checks'] / self.quality_stats['total_checks']
        else:
            self.quality_stats['pass_rate'] = 0.0
            
        return self.quality_stats 