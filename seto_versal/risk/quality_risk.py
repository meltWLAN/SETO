import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

from seto_versal.data.manager import DataManager
from seto_versal.data.quality import DataQualityChecker

logger = logging.getLogger(__name__)

class DataQualityRiskManager:
    """
    数据质量风险管理器。
    
    这个组件负责评估数据质量风险，并提供风险缓解策略。
    它将数据质量问题视为交易风险的一部分，纳入整体风险管理框架。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化数据质量风险管理器。
        
        Args:
            config: 配置字典，包含风险管理参数
        """
        self.config = config or {}
        self.data_manager = None
        self.quality_checker = DataQualityChecker(config)
        
        # 风险参数
        self.max_quality_risk = self.config.get('max_quality_risk', 0.3)  # 最大允许的质量风险
        self.quality_risk_weight = self.config.get('quality_risk_weight', 0.2)  # 质量风险在总风险中的权重
        self.high_risk_threshold = self.config.get('high_risk_threshold', 0.7)  # 高风险阈值
        self.medium_risk_threshold = self.config.get('medium_risk_threshold', 0.4)  # 中风险阈值
        
        # 跟踪的数据质量风险
        self.portfolio_quality_risk = 0.0
        self.symbol_quality_risks = {}
        self.historical_risk = []
        self.last_assessment_time = None
    
    def initialize(self, data_manager: DataManager):
        """
        初始化数据质量风险管理器。
        
        Args:
            data_manager: 数据管理器实例
        """
        self.data_manager = data_manager
        logger.info("数据质量风险管理器初始化完成")
    
    def _calculate_data_quality_risk(self, validation_result: Dict[str, Any]) -> float:
        """
        根据数据质量验证结果计算风险水平。
        
        Args:
            validation_result: 数据质量验证结果
            
        Returns:
            float: 0到1之间的风险水平，1表示最高风险
        """
        if validation_result.get('valid', True):
            return 0.0  # 没有风险
        
        details = validation_result.get('details', {})
        
        # 风险评分因子
        risk_factors = {
            'missing_values': {
                'weight': 0.3,
                'risk': 0.0
            },
            'stale_data': {
                'weight': 0.25,
                'risk': 0.0
            },
            'price_jumps': {
                'weight': 0.2,
                'risk': 0.0
            },
            'zero_volume': {
                'weight': 0.1,
                'risk': 0.0
            },
            'consistency': {
                'weight': 0.15,
                'risk': 0.0
            }
        }
        
        # 计算每种风险因素的风险值
        for factor, factor_data in risk_factors.items():
            if factor in details:
                check_result = details[factor]
                
                if not check_result.get('valid', True):
                    # 根据不同的风险因素计算风险程度
                    if factor == 'missing_values' and 'missing_pct' in check_result:
                        # 平均缺失率越高风险越大
                        missing_pct = check_result.get('missing_pct', {})
                        if missing_pct:
                            avg_missing = sum(missing_pct.values()) / len(missing_pct)
                            # 根据阈值缩放到0-1
                            threshold = self.quality_checker.threshold_missing_pct
                            risk_factors[factor]['risk'] = min(1.0, avg_missing / (threshold * 2))
                    
                    elif factor == 'stale_data' and 'days_since_update' in check_result:
                        # 数据越陈旧风险越高
                        days = check_result.get('days_since_update', 0)
                        threshold = self.quality_checker.threshold_stale_days
                        risk_factors[factor]['risk'] = min(1.0, days / (threshold * 2))
                    
                    elif factor == 'price_jumps' and 'extreme_jumps' in check_result:
                        # 价格跳跃次数越多风险越高
                        jumps = check_result.get('extreme_jumps', 0)
                        # 假设超过10次跳跃为高风险
                        risk_factors[factor]['risk'] = min(1.0, jumps / 10)
                    
                    elif factor == 'zero_volume' and 'zero_volume_pct' in check_result:
                        # 零交易量比例越高风险越高
                        zero_pct = check_result.get('zero_volume_pct', 0)
                        threshold = self.quality_checker.threshold_volume_zero_pct
                        risk_factors[factor]['risk'] = min(1.0, zero_pct / (threshold * 2))
                    
                    elif factor == 'consistency' and 'symbols_with_missing_dates' in check_result:
                        # 数据不一致的股票越多风险越高
                        missing_symbols = check_result.get('symbols_with_missing_dates', 0)
                        total_symbols = 10  # 假设基准值
                        risk_factors[factor]['risk'] = min(1.0, missing_symbols / total_symbols)
                    
                    else:
                        # 默认风险为高
                        risk_factors[factor]['risk'] = 0.8
                        
        # 计算加权风险值
        total_risk = sum(factor_data['weight'] * factor_data['risk'] for factor_data in risk_factors.values())
        
        return total_risk
    
    def assess_portfolio_data_quality(self, symbols: List[str]) -> Dict[str, Any]:
        """
        评估投资组合中所有股票的数据质量风险。
        
        Args:
            symbols: 投资组合中的股票列表
            
        Returns:
            Dict[str, Any]: 包含风险评估结果的字典
        """
        if self.data_manager is None:
            logger.error("数据管理器未初始化")
            return {
                'valid': False,
                'error': '数据管理器未初始化',
                'quality_risk': 1.0  # 最高风险
            }
        
        # 获取最近的市场数据
        market_data = self.data_manager.get_recent_market_data(symbols)
        
        if market_data is None or market_data.empty:
            logger.warning("无法获取市场数据进行质量风险评估")
            return {
                'valid': False,
                'error': '无法获取市场数据',
                'quality_risk': 0.8  # 高风险
            }
        
        # 验证数据质量
        validation_result = self.quality_checker.validate_market_data(market_data)
        
        # 计算整体数据质量风险
        portfolio_risk = self._calculate_data_quality_risk(validation_result)
        
        # 更新风险状态
        self.portfolio_quality_risk = portfolio_risk
        self.last_assessment_time = datetime.now()
        
        # 添加到历史记录
        self.historical_risk.append({
            'timestamp': self.last_assessment_time,
            'risk': portfolio_risk,
            'details': validation_result
        })
        
        # 保持历史记录在合理大小
        max_history = self.config.get('max_risk_history', 100)
        if len(self.historical_risk) > max_history:
            self.historical_risk = self.historical_risk[-max_history:]
        
        # 按股票评估风险
        self.symbol_quality_risks = {}
        for symbol in symbols:
            symbol_data = market_data[market_data['symbol'] == symbol] if 'symbol' in market_data.columns else None
            
            if symbol_data is not None and not symbol_data.empty:
                symbol_validation = self.quality_checker.validate_market_data(symbol_data)
                symbol_risk = self._calculate_data_quality_risk(symbol_validation)
                self.symbol_quality_risks[symbol] = {
                    'risk': symbol_risk,
                    'risk_level': self._get_risk_level(symbol_risk),
                    'details': symbol_validation
                }
            else:
                # 没有数据视为高风险
                self.symbol_quality_risks[symbol] = {
                    'risk': 0.9,
                    'risk_level': 'high',
                    'details': {'valid': False, 'error': '无数据'}
                }
        
        return {
            'valid': True,
            'quality_risk': portfolio_risk,
            'risk_level': self._get_risk_level(portfolio_risk),
            'symbol_risks': self.symbol_quality_risks,
            'timestamp': self.last_assessment_time
        }
    
    def _get_risk_level(self, risk_value: float) -> str:
        """
        根据风险值获取风险级别。
        
        Args:
            risk_value: 0到1之间的风险值
            
        Returns:
            str: 风险级别，'low', 'medium', 或 'high'
        """
        if risk_value >= self.high_risk_threshold:
            return 'high'
        elif risk_value >= self.medium_risk_threshold:
            return 'medium'
        else:
            return 'low'
    
    def adjust_position_for_quality_risk(self, symbol: str, target_position: float) -> float:
        """
        根据数据质量风险调整目标持仓。
        
        Args:
            symbol: 股票代码
            target_position: 原目标持仓金额或比例
            
        Returns:
            float: 调整后的目标持仓
        """
        # 获取该股票的质量风险
        symbol_risk = self.symbol_quality_risks.get(symbol, {'risk': 0.0})['risk']
        
        # 如果数据质量风险超过阈值，减少持仓
        if symbol_risk > self.max_quality_risk:
            # 根据风险程度线性减少持仓
            risk_factor = max(0, 1 - (symbol_risk - self.max_quality_risk) / (1 - self.max_quality_risk))
            adjusted_position = target_position * risk_factor
            
            logger.info(f"由于数据质量风险 ({symbol_risk:.2f})，将 {symbol} 的目标持仓从 {target_position} 调整为 {adjusted_position}")
            
            return adjusted_position
        
        return target_position
    
    def get_total_risk_contribution(self) -> Dict[str, float]:
        """
        获取数据质量对总体风险的贡献。
        
        Returns:
            Dict[str, float]: 风险贡献
        """
        return {
            'quality_risk': self.portfolio_quality_risk,
            'risk_contribution': self.portfolio_quality_risk * self.quality_risk_weight,
            'weight': self.quality_risk_weight
        }
    
    def get_risk_alerts(self) -> List[Dict[str, Any]]:
        """
        获取数据质量风险警报。
        
        Returns:
            List[Dict[str, Any]]: 风险警报列表
        """
        alerts = []
        
        # 检查整体风险水平
        if self.portfolio_quality_risk >= self.high_risk_threshold:
            alerts.append({
                'level': 'high',
                'message': f'投资组合数据质量风险高 ({self.portfolio_quality_risk:.2f})',
                'timestamp': datetime.now()
            })
        
        # 检查个股风险
        high_risk_symbols = [
            symbol for symbol, data in self.symbol_quality_risks.items()
            if data['risk'] >= self.high_risk_threshold
        ]
        
        if high_risk_symbols:
            alerts.append({
                'level': 'high',
                'message': f'以下股票存在高数据质量风险: {", ".join(high_risk_symbols)}',
                'symbols': high_risk_symbols,
                'timestamp': datetime.now()
            })
        
        return alerts
    
    def get_recommended_actions(self) -> List[Dict[str, Any]]:
        """
        获取基于数据质量风险的建议操作。
        
        Returns:
            List[Dict[str, Any]]: 建议操作列表
        """
        actions = []
        
        # 如果整体质量风险较高
        if self.portfolio_quality_risk >= self.high_risk_threshold:
            actions.append({
                'action_type': 'reduce_exposure',
                'message': '由于数据质量风险高，建议降低整体市场敞口',
                'reduce_factor': 0.5,  # 建议减少50%的敞口
                'priority': 'high'
            })
        
        # 处理高风险股票
        high_risk_symbols = [
            symbol for symbol, data in self.symbol_quality_risks.items()
            if data['risk'] >= self.high_risk_threshold
        ]
        
        if high_risk_symbols:
            actions.append({
                'action_type': 'avoid_trading',
                'symbols': high_risk_symbols,
                'message': f'避免交易数据质量风险高的股票: {", ".join(high_risk_symbols)}',
                'priority': 'high'
            })
        
        # 处理中风险股票
        medium_risk_symbols = [
            symbol for symbol, data in self.symbol_quality_risks.items()
            if self.medium_risk_threshold <= data['risk'] < self.high_risk_threshold
        ]
        
        if medium_risk_symbols:
            actions.append({
                'action_type': 'reduce_positions',
                'symbols': medium_risk_symbols,
                'message': f'减少对数据质量风险中等的股票的敞口',
                'reduce_factor': 0.3,  # 建议减少30%的持仓
                'priority': 'medium'
            })
        
        return actions
    
    def get_quality_risk_metrics(self) -> Dict[str, Any]:
        """
        获取数据质量风险指标。
        
        Returns:
            Dict[str, Any]: 风险指标
        """
        # 计算风险趋势
        risk_trend = 'stable'
        if len(self.historical_risk) >= 2:
            recent_risks = [entry['risk'] for entry in self.historical_risk[-5:]]
            if len(recent_risks) >= 2:
                if recent_risks[-1] > recent_risks[0] * 1.1:
                    risk_trend = 'increasing'
                elif recent_risks[-1] < recent_risks[0] * 0.9:
                    risk_trend = 'decreasing'
        
        # 找出风险最高的因素
        factor_risks = {
            'missing_values': 0.0,
            'stale_data': 0.0,
            'price_jumps': 0.0,
            'zero_volume': 0.0,
            'consistency': 0.0
        }
        
        # 统计每种因素的风险
        for symbol_data in self.symbol_quality_risks.values():
            details = symbol_data.get('details', {}).get('details', {})
            for factor in factor_risks.keys():
                if factor in details and not details[factor].get('valid', True):
                    factor_risks[factor] += 1
        
        # 找出风险最高的因素
        highest_risk_factor = max(factor_risks.items(), key=lambda x: x[1])[0] if factor_risks else None
        
        return {
            'current_risk': self.portfolio_quality_risk,
            'risk_level': self._get_risk_level(self.portfolio_quality_risk),
            'risk_trend': risk_trend,
            'highest_risk_symbols': [
                symbol for symbol, data in sorted(
                    self.symbol_quality_risks.items(), 
                    key=lambda x: x[1]['risk'], 
                    reverse=True
                )[:5]
            ],
            'highest_risk_factor': highest_risk_factor,
            'last_assessment': self.last_assessment_time,
            'high_risk_count': sum(
                1 for data in self.symbol_quality_risks.values() 
                if data['risk'] >= self.high_risk_threshold
            ),
            'medium_risk_count': sum(
                1 for data in self.symbol_quality_risks.values()
                if self.medium_risk_threshold <= data['risk'] < self.high_risk_threshold
            ),
            'low_risk_count': sum(
                1 for data in self.symbol_quality_risks.values()
                if data['risk'] < self.medium_risk_threshold
            )
        } 