import logging
import random
from seto_versal.risk.quality_risk import DataQualityRiskManager
from datetime import datetime

logger = logging.getLogger(__name__)

class RiskManager:
    """Manager for risk control and position sizing"""
    
    def __init__(self, config=None):
        """
        Initialize the risk manager
        
        Args:
            config (dict): Risk management configuration
        """
        self.config = config
        self.mode = config.get('mode', 'backtest')
        self.mode_config = config.get('mode_configs', {}).get(self.mode, {})
        
        # Get mode-specific parameters
        self.risk_level = self.mode_config.get('risk_level', 'medium')
        self.max_drawdown = self.mode_config.get('max_drawdown', config.get('max_drawdown', 0.03))
        self.max_position_size = self.mode_config.get('max_position_size', config.get('max_position_size', 0.25))
        self.trailing_stop = self.mode_config.get('trailing_stop', config.get('trailing_stop', True))
        
        # Initialize risk metrics
        self.current_drawdown = 0.0
        self.peak_value = 0.0
        self.position_sizes = {}
        self.stop_losses = {}
        
        # 初始化数据质量风险管理器
        self.quality_risk_manager = DataQualityRiskManager(config)
        
        # 是否将数据质量风险纳入总体风险评估
        self.include_quality_risk = config.get('include_quality_risk', True) if config else True
        
    def initialize(self, data_manager, portfolio_manager):
        # 初始化数据质量风险管理器
        if self.quality_risk_manager is not None:
            self.quality_risk_manager.initialize(data_manager)
            logger.info("数据质量风险管理器初始化完成")
        
    def check_risk_limits(self, order, portfolio):
        """
        检查订单是否违反风险限制
        
        Args:
            order: 订单对象
            portfolio: 投资组合对象
            
        Returns:
            tuple: (是否通过检查, 风险问题列表)
        """
        try:
            risk_issues = []
            
            # 获取基本信息
            symbol = order.symbol
            portfolio_value = portfolio.get_total_value() if portfolio else 0
            current_positions = portfolio.get_positions() if portfolio else {}
            
            # 如果投资组合价值为0，可能是初始状态，放行订单
            if portfolio_value <= 0:
                return True, []
            
            # 计算订单价值
            price = order.price if hasattr(order, 'price') and order.price else 0
            quantity = order.quantity if hasattr(order, 'quantity') else 0
            order_value = abs(price * quantity)
            
            # 1. 检查单笔订单大小限制
            max_order_value = portfolio_value * self.max_position_size
            if order_value > max_order_value:
                risk_issues.append({
                    'type': 'order_size',
                    'message': f'订单大小 ({order_value:.2f}) 超过限制 ({max_order_value:.2f})',
                    'risk_level': 'high'
                })
            
            # 2. 检查持仓集中度限制
            if symbol in current_positions:
                current_position = current_positions[symbol]
                current_value = current_position.get('value', 0)
                # 计算新的持仓价值
                new_position_value = current_value + order_value
                position_ratio = new_position_value / portfolio_value
                
                if position_ratio > self.max_position_size:
                    risk_issues.append({
                        'type': 'position_concentration',
                        'message': f'持仓集中度 ({position_ratio:.2%}) 超过限制 ({self.max_position_size:.2%})',
                        'risk_level': 'medium'
                    })
            
            # 3. 检查回撤风险
            if self.current_drawdown > self.max_drawdown * 0.8:  # 接近最大回撤
                risk_issues.append({
                    'type': 'drawdown',
                    'message': f'当前回撤 ({self.current_drawdown:.2%}) 接近最大允许回撤 ({self.max_drawdown:.2%})',
                    'risk_level': 'medium'
                })
            
            # 4. 检查数据质量风险
            if self.include_quality_risk and hasattr(self, 'quality_risk_manager') and self.quality_risk_manager is not None:
                # 获取该股票的数据质量风险
                quality_risks = self.quality_risk_manager.symbol_quality_risks
                if symbol in quality_risks:
                    symbol_quality_risk = quality_risks[symbol]['risk']
                    risk_level = quality_risks[symbol]['risk_level']
                    
                    # 如果数据质量风险高，可能需要调整或拒绝订单
                    if symbol_quality_risk >= self.quality_risk_manager.high_risk_threshold:
                        risk_issues.append({
                            'type': 'data_quality',
                            'message': f'数据质量风险高 ({symbol_quality_risk:.2f})',
                            'risk_level': risk_level
                        })
                        
                        # 如果风险特别高，考虑拒绝订单
                        if symbol_quality_risk > 0.9:  # 极高风险阈值
                            logger.warning(f"由于极高的数据质量风险 ({symbol_quality_risk:.2f})，拒绝 {symbol} 的订单")
                            return False, risk_issues
            
            # 决定是否允许订单
            # 如果有高风险问题，拒绝订单
            high_risk_issues = [issue for issue in risk_issues if issue.get('risk_level') == 'high']
            if high_risk_issues:
                return False, risk_issues
            
            # 如果只有中低风险问题，可以考虑调整订单而不是拒绝
            return True, risk_issues
            
        except Exception as e:
            logger.error(f"检查风险限制出错: {e}")
            # 出错时保守处理，拒绝订单
            return False, [{'type': 'error', 'message': str(e), 'risk_level': 'high'}]
    
    def adjust_order_for_risk(self, order, portfolio):
        """
        根据风险评估调整订单大小
        
        Args:
            order: 订单对象
            portfolio: 投资组合对象
            
        Returns:
            order: 调整后的订单对象
        """
        try:
            # 检查是否需要调整
            passed, risk_issues = self.check_risk_limits(order, portfolio)
            
            # 如果完全通过风险检查，无需调整
            if passed and not risk_issues:
                return order
            
            # 如果没通过，但可以通过调整解决
            if not passed:
                # 找到最严重的风险问题
                most_severe_issue = max(risk_issues, key=lambda x: {'low': 1, 'medium': 2, 'high': 3}.get(x.get('risk_level', 'low'), 0))
                issue_type = most_severe_issue.get('type', '')
                
                # 根据风险类型调整订单
                if issue_type == 'order_size' or issue_type == 'position_concentration':
                    # 获取投资组合价值
                    portfolio_value = portfolio.get_total_value() if portfolio else 0
                    
                    # 计算安全的订单大小
                    safe_position_size = self.max_position_size * 0.8  # 留20%安全边际
                    safe_order_value = portfolio_value * safe_position_size
                    
                    # 获取当前价格
                    price = order.price if hasattr(order, 'price') and order.price else 0
                    if price > 0:
                        # 计算新的数量
                        new_quantity = int(safe_order_value / price)
                        original_quantity = order.quantity
                        
                        # 调整订单数量
                        order.quantity = new_quantity
                        logger.info(f"订单大小风险调整: {order.symbol} 数量从 {original_quantity} 调整为 {new_quantity}")
            
            # 根据数据质量风险调整订单大小
            if self.include_quality_risk and hasattr(self, 'quality_risk_manager') and self.quality_risk_manager is not None:
                symbol = order.symbol
                original_quantity = order.quantity
                
                # 应用数据质量风险调整
                adjusted_quantity = self.quality_risk_manager.adjust_position_for_quality_risk(
                    symbol, original_quantity
                )
                
                if adjusted_quantity != original_quantity:
                    logger.info(f"数据质量风险调整: {symbol} 数量从 {original_quantity} 调整为 {adjusted_quantity}")
                    order.quantity = adjusted_quantity
            
            return order
            
        except Exception as e:
            logger.error(f"调整订单风险出错: {e}")
            return order
            
    def calculate_position_size(self, symbol, price, portfolio_value):
        """
        Calculate position size based on risk parameters
        
        Args:
            symbol (str): Trading symbol
            price (float): Current price
            portfolio_value (float): Current portfolio value
            
        Returns:
            float: Position size in units
        """
        try:
            # Get base position size
            base_size = portfolio_value * self.max_position_size
            
            # Adjust for risk level
            if self.risk_level == 'low':
                base_size *= 0.8
            elif self.risk_level == 'high':
                base_size *= 1.2
                
            # Calculate units
            units = base_size / price
            
            # Round to appropriate precision
            units = round(units, 2)
            
            return units
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
            
    def update_stop_loss(self, symbol, position, current_price):
        """
        Update trailing stop loss for a position
        
        Args:
            symbol (str): Trading symbol
            position (dict): Position information
            current_price (float): Current price
            
        Returns:
            float: New stop loss price
        """
        try:
            if not self.trailing_stop:
                return position.get('stop_loss', 0)
                
            # Get current stop loss
            current_stop = position.get('stop_loss', 0)
            
            # Calculate new stop loss
            if position['value'] > 0:  # Long position
                new_stop = current_price * (1 - self.max_drawdown)
                if new_stop > current_stop:
                    current_stop = new_stop
            else:  # Short position
                new_stop = current_price * (1 + self.max_drawdown)
                if new_stop < current_stop or current_stop == 0:
                    current_stop = new_stop
                    
            return current_stop
            
        except Exception as e:
            logger.error(f"Error updating stop loss: {e}")
            return position.get('stop_loss', 0)
            
    def check_stop_loss(self, symbol, position, current_price):
        """
        Check if stop loss is triggered
        
        Args:
            symbol (str): Trading symbol
            position (dict): Position information
            current_price (float): Current price
            
        Returns:
            bool: True if stop loss is triggered, False otherwise
        """
        try:
            stop_loss = position.get('stop_loss', 0)
            if not stop_loss:
                return False
                
            if position['value'] > 0:  # Long position
                return current_price <= stop_loss
            else:  # Short position
                return current_price >= stop_loss
                
        except Exception as e:
            logger.error(f"Error checking stop loss: {e}")
            return False
            
    def get_risk_metrics(self):
        """Get current risk metrics"""
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'peak_value': self.peak_value,
            'risk_level': self.risk_level,
            'max_position_size': self.max_position_size
        }
        
    def reset_metrics(self):
        """Reset risk metrics"""
        self.current_drawdown = 0.0
        self.peak_value = 0.0
        self.position_sizes = {}
        self.stop_losses = {}
        
    def get_metrics(self):
        """
        获取当前的风险指标
        
        Returns:
            dict: 包含各种风险指标的字典
        """
        try:
            metrics = {
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown,
                'peak_value': self.peak_value,
                'risk_level': self.risk_level,
                'max_position_size': self.max_position_size,
                'sharpe_ratio': random.uniform(0.5, 2.5),  # 示例值，实际应基于历史数据计算
                'volatility': random.uniform(0.01, 0.05),  # 示例值，实际应基于历史数据计算
                'beta': random.uniform(0.8, 1.2),          # 示例值，实际应基于历史数据计算
                'risk_score': random.uniform(0, 100)       # 示例值，实际应基于多维度风险计算
            }
            
            # 添加数据质量风险指标
            if self.include_quality_risk and hasattr(self, 'quality_risk_manager') and self.quality_risk_manager is not None:
                quality_metrics = self.quality_risk_manager.get_quality_risk_metrics()
                metrics['data_quality'] = quality_metrics
                
                # 将数据质量风险整合到总体风险评分中
                if 'risk_score' in metrics and 'current_risk' in quality_metrics:
                    # 按权重融合风险分数
                    quality_weight = self.quality_risk_manager.quality_risk_weight
                    metrics['risk_score'] = metrics['risk_score'] * (1 - quality_weight) + \
                                           quality_metrics['current_risk'] * 100 * quality_weight
            
            return metrics
            
        except Exception as e:
            logger.error(f"获取风险指标出错: {e}")
            return {
                'error': str(e),
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown
            }

    def get_risk_metrics(self):
        """别名方法，用于向后兼容"""
        return self.get_metrics()

    def assess_portfolio_risk(self, portfolio):
        """
        评估投资组合的整体风险，包括数据质量风险
        
        Args:
            portfolio: 投资组合对象
            
        Returns:
            dict: 风险评估结果
        """
        try:
            # 初始化风险评估结果
            total_risk = 0.0
            risk_breakdown = {}
            
            # 从投资组合获取基本信息
            portfolio_value = portfolio.get_total_value() if portfolio else 0
            positions = portfolio.get_positions() if portfolio else {}
            symbols = list(positions.keys())
            
            # 计算基本风险指标
            # 1. 回撤风险
            if self.peak_value > 0:
                drawdown_risk = self.current_drawdown / self.max_drawdown
                total_risk += drawdown_risk * 0.3  # 权重为30%
                risk_breakdown['drawdown'] = drawdown_risk
            
            # 2. 集中度风险
            if positions and portfolio_value > 0:
                position_values = {symbol: pos.get('value', 0) for symbol, pos in positions.items()}
                concentration_risk = sum((value / portfolio_value) ** 2 for value in position_values.values())
                total_risk += concentration_risk * 0.2  # 权重为20%
                risk_breakdown['concentration'] = concentration_risk
            
            # 3. 基础市场风险（简化版）
            market_risk = 0.4  # 基础市场风险，根据实际情况调整
            total_risk += market_risk * 0.3  # 权重为30%
            risk_breakdown['market'] = market_risk
            
            # 评估数据质量风险
            if self.include_quality_risk and hasattr(self, 'quality_risk_manager') and self.quality_risk_manager is not None:
                quality_risk_result = self.quality_risk_manager.assess_portfolio_data_quality(symbols)
                
                # 如果风险水平高，记录警告
                if quality_risk_result.get('risk_level') == 'high':
                    logger.warning("投资组合数据质量风险较高，可能影响交易决策")
                    
                # 将数据质量风险纳入总体风险评估
                quality_risk_contribution = self.quality_risk_manager.get_total_risk_contribution()
                quality_risk = quality_risk_contribution['risk_contribution']
                total_risk += quality_risk
                
                # 添加到风险细分
                risk_breakdown['data_quality'] = quality_risk_contribution['quality_risk']
                
                # 获取高风险股票
                high_risk_symbols = [
                    symbol for symbol, data in self.quality_risk_manager.symbol_quality_risks.items()
                    if data['risk'] >= self.quality_risk_manager.high_risk_threshold
                ]
                
                if high_risk_symbols:
                    logger.warning(f"以下股票存在高数据质量风险: {', '.join(high_risk_symbols)}")
            
            # 计算整体风险等级
            risk_level = 'low'
            if total_risk >= 0.7:
                risk_level = 'high'
            elif total_risk >= 0.4:
                risk_level = 'medium'
            
            # 返回完整的风险评估结果
            return {
                'total_risk': total_risk,
                'risk_level': risk_level,
                'risk_breakdown': risk_breakdown,
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_value,
                'position_count': len(positions)
            }
            
        except Exception as e:
            logger.error(f"评估投资组合风险出错: {e}")
            return {
                'error': str(e),
                'total_risk': 0.5,  # 默认中等风险
                'risk_level': 'medium'
            }

    def get_risk_alerts(self):
        """
        获取当前系统的风险警报
        
        Returns:
            list: 风险警报列表
        """
        try:
            alerts = []
            
            # 1. 检查回撤风险
            if self.current_drawdown > self.max_drawdown * 0.8:
                alerts.append({
                    'type': 'drawdown',
                    'level': 'high' if self.current_drawdown > self.max_drawdown * 0.9 else 'medium',
                    'message': f'当前回撤 ({self.current_drawdown:.2%}) 接近最大允许回撤 ({self.max_drawdown:.2%})',
                    'timestamp': datetime.now()
                })
            
            # 2. 检查持仓集中度
            # 这里需要外部传入持仓信息，暂时跳过
            
            # 3. 添加数据质量风险警报
            if self.include_quality_risk and hasattr(self, 'quality_risk_manager') and self.quality_risk_manager is not None:
                quality_alerts = self.quality_risk_manager.get_risk_alerts()
                # 为质量警报添加统一类型标记
                for alert in quality_alerts:
                    alert['type'] = 'data_quality'
                    
                alerts.extend(quality_alerts)
            
            return alerts
            
        except Exception as e:
            logger.error(f"获取风险警报出错: {e}")
            return [{
                'type': 'error',
                'level': 'high',
                'message': f'获取风险警报时发生错误: {str(e)}',
                'timestamp': datetime.now()
            }]
    
    def get_recommended_actions(self):
        """
        获取基于当前风险状况的建议操作
        
        Returns:
            list: 建议操作列表
        """
        try:
            actions = []
            
            # 1. 回撤风险缓解
            if self.current_drawdown > self.max_drawdown * 0.8:
                actions.append({
                    'action_type': 'reduce_exposure',
                    'message': f'由于回撤风险 ({self.current_drawdown:.2%})，建议降低市场敞口',
                    'reduce_factor': min(0.5, self.current_drawdown / self.max_drawdown),
                    'priority': 'high' if self.current_drawdown > self.max_drawdown * 0.9 else 'medium'
                })
            
            # 2. 添加基于数据质量风险的建议操作
            if self.include_quality_risk and hasattr(self, 'quality_risk_manager') and self.quality_risk_manager is not None:
                quality_actions = self.quality_risk_manager.get_recommended_actions()
                actions.extend(quality_actions)
            
            return actions
            
        except Exception as e:
            logger.error(f"获取建议操作出错: {e}")
            return [{
                'action_type': 'error',
                'message': f'获取建议操作时发生错误: {str(e)}',
                'priority': 'high'
            }] 