#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Storyteller module for SETO-Versal
Generates human-readable narratives about system activity and trading decisions
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
import random

logger = logging.getLogger(__name__)

class Narrator:
    """
    Narrator/Storyteller for SETO-Versal
    
    Generates human-readable narratives about:
    - Daily market summaries
    - Trading decisions and rationales
    - Performance analysis
    - System insights
    """
    
    def __init__(self, config):
        """
        Initialize the narrator
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.language_style = config.get('narrative_style', 'balanced')  # balanced, technical, conversational
        self.detail_level = config.get('detail_level', 'medium')  # low, medium, high
        
        # Templates for different narrative types
        self._load_templates()
        
        # History of generated narratives
        self.narratives = []
        
        logger.info(f"Narrator initialized with {self.language_style} style")
    
    def _load_templates(self):
        """Load narrative templates"""
        # Market summary templates
        self.market_templates = {
            'bull': [
                "市场保持强势上涨，{top_sectors}领涨，资金流动活跃度高。",
                "多头气氛浓厚，大盘指数走高，{top_sectors}表现抢眼。",
                "市场维持上升趋势，成交量显著增加，{top_sectors}板块强劲。"
            ],
            'bear': [
                "市场承压下行，{bottom_sectors}跌幅较大，观望情绪浓厚。",
                "空头市场特征明显，指数走弱，{bottom_sectors}领跌。",
                "大盘延续弱势，交投清淡，{bottom_sectors}承压明显。"
            ],
            'sideways': [
                "市场震荡整理，{top_sectors}有所表现，{bottom_sectors}偏弱。",
                "指数窄幅波动，板块轮动明显，{top_sectors}相对活跃。",
                "大盘维持区间震荡，成交量温和，个股分化明显。"
            ]
        }
        
        # Agent activity templates
        self.agent_templates = {
            'fast_profit': [
                "{agent_name}今日追入{stock_code}，入场动因为{reason}。",
                "短线{agent_name}选择{stock_code}，基于{reason}的信号。",
                "{agent_name}对{stock_code}形成{confidence_text}买入信号，{reason}。"
            ],
            'trend': [
                "{agent_name}跟踪主升浪，建仓{stock_code}，理由是{reason}。",
                "趋势追踪{agent_name}选择了{stock_code}，看好其{reason}。",
                "{agent_name}基于{reason}，对{stock_code}形成中期趋势性买入。"
            ],
            'reversal': [
                "{agent_name}捕捉超跌反弹，买入{stock_code}，{reason}。",
                "反转信号触发，{agent_name}选择{stock_code}，基于{reason}。",
                "{agent_name}发现{stock_code}底部结构良好，{reason}，建议买入。"
            ],
            'sector_rotation': [
                "{agent_name}捕捉板块轮动，布局{stock_code}，{reason}。",
                "热点切换策略激活，{agent_name}选择{stock_code}，{reason}。",
                "{agent_name}基于行业轮动逻辑，关注{stock_code}，{reason}。"
            ],
            'defensive': [
                "{agent_name}启动防御策略，调整{stock_code}仓位，{reason}。",
                "风险防御{agent_name}对{stock_code}采取行动，基于{reason}。",
                "{agent_name}侦测到风险信号，针对{stock_code}{reason}。"
            ]
        }
        
        # Trade result templates
        self.trade_templates = {
            'win': [
                "{stock_code}交易获利了结，盈利{profit_pct:.2%}，{agent_name}的{strategy}策略奏效。",
                "成功兑现{stock_code}利润{profit_pct:.2%}，{strategy}策略判断正确。",
                "{agent_name}操作的{stock_code}获利{profit_pct:.2%}，{exit_reason}。"
            ],
            'loss': [
                "{stock_code}止损出局，亏损{profit_pct:.2%}，{agent_name}的{strategy}策略失效。",
                "执行止损规则，{stock_code}亏损{profit_pct:.2%}，{exit_reason}。",
                "{agent_name}认为{stock_code}不符预期，亏损{profit_pct:.2%}离场，{exit_reason}。"
            ],
            'breakeven': [
                "{stock_code}交易基本持平，{agent_name}决定调整策略。",
                "接近盈亏平衡点，{stock_code}交易平仓，重新评估。",
                "{agent_name}对{stock_code}的{strategy}策略需要优化，本次交易微利。"
            ]
        }
        
        # Performance templates
        self.performance_templates = {
            'improving': [
                "系统表现持续改善，{win_rate:.0%}的胜率显示策略契合当前市场。",
                "绩效曲线向上，近期{win_rate:.0%}胜率，最大回撤控制在{max_drawdown:.2%}内。",
                "系统正处于良性状态，{win_rate:.0%}的胜率带来{period_return:.2%}的阶段收益。"
            ],
            'deteriorating': [
                "系统绩效有所下滑，胜率降至{win_rate:.0%}，正在寻找适应性改进。",
                "近期策略表现不佳，胜率{win_rate:.0%}，回撤达到{max_drawdown:.2%}。",
                "市场环境变化导致系统有效性降低，正在进行参数再学习。"
            ],
            'stable': [
                "系统保持稳定表现，{win_rate:.0%}胜率，风险收益比保持在理想区间。",
                "策略组合正常运行，{win_rate:.0%}胜率，{period_return:.2%}阶段收益。",
                "系统各项指标处于正常区间，{portfolio_metrics}。"
            ]
        }
    
    def generate_daily_summary(self, market_state, agents, execution_results, portfolio):
        """
        Generate a daily summary narrative
        
        Args:
            market_state (MarketState): Current market state
            agents (list): List of agent instances
            execution_results (list): List of execution results
            portfolio (Portfolio): Portfolio instance
            
        Returns:
            str: Generated narrative
        """
        # Get market summary
        market_summary = market_state.get_market_summary()
        
        # Format date
        date_str = datetime.now().strftime('%Y年%m月%d日')
        
        # Generate market state description
        market_narrative = self._generate_market_narrative(market_summary)
        
        # Generate agent activity narrative
        agent_narrative = self._generate_agent_narrative(agents, execution_results)
        
        # Generate performance narrative
        performance_narrative = self._generate_performance_narrative(portfolio)
        
        # Combine narratives
        narrative = f"{date_str}交易总结：\n\n"
        narrative += f"1. 市场状况：{market_narrative}\n\n"
        narrative += f"2. 交易活动：{agent_narrative}\n\n"
        narrative += f"3. 系统表现：{performance_narrative}"
        
        # Record narrative
        self.narratives.append({
            'timestamp': datetime.now(),
            'type': 'daily_summary',
            'content': narrative
        })
        
        return narrative
    
    def generate_trade_narrative(self, trade_result, agent_name=None, strategy=None, exit_reason=None):
        """
        Generate a narrative for a specific trade
        
        Args:
            trade_result (dict): Trade result information
            agent_name (str, optional): Name of the agent responsible
            strategy (str, optional): Strategy name
            exit_reason (str, optional): Reason for exiting the trade
            
        Returns:
            str: Generated narrative
        """
        # Determine if win/loss/breakeven
        profit_pct = trade_result.get('profit_pct', 0)
        
        if profit_pct > 0.005:  # More than 0.5% profit
            templates = self.trade_templates['win']
        elif profit_pct < -0.005:  # More than 0.5% loss
            templates = self.trade_templates['loss']
        else:
            templates = self.trade_templates['breakeven']
        
        # Fill in template
        template = random.choice(templates)
        
        narrative = template.format(
            stock_code=trade_result.get('stock_code', 'unknown'),
            profit_pct=profit_pct,
            agent_name=agent_name or "系统",
            strategy=strategy or "交易",
            exit_reason=exit_reason or "按计划执行"
        )
        
        # Record narrative
        self.narratives.append({
            'timestamp': datetime.now(),
            'type': 'trade_narrative',
            'content': narrative
        })
        
        return narrative
    
    def generate_session_report(self, performance_data, portfolio):
        """
        Generate a comprehensive session report
        
        Args:
            performance_data (dict): Performance metrics and data
            portfolio (Portfolio): Portfolio instance
            
        Returns:
            str: Generated report
        """
        # Get portfolio summary
        summary = portfolio.get_performance_summary()
        
        # Generate report
        report = "SETO-Versal 交易系统会话报告\n"
        report += "=" * 50 + "\n\n"
        
        # Overall performance
        report += "整体表现\n"
        report += "-" * 20 + "\n"
        report += f"总资产: ¥{summary['total_value']:,.2f}\n"
        report += f"现金余额: ¥{summary['cash']:,.2f}\n"
        report += f"持仓市值: ¥{summary['positions_value']:,.2f}\n"
        report += f"总收益率: {summary['total_return']:.2%}\n"
        report += f"年化收益: {summary['annualized_return']:.2%}\n"
        report += f"最大回撤: {summary['max_drawdown']:.2%}\n"
        report += f"夏普比率: {summary['sharpe_ratio']:.2f}\n\n"
        
        # Trading statistics
        report += "交易统计\n"
        report += "-" * 20 + "\n"
        report += f"总交易次数: {summary['trade_count']}\n"
        report += f"盈利交易: {summary['win_count']}\n"
        report += f"亏损交易: {summary['loss_count']}\n"
        report += f"胜率: {summary['win_rate']:.2%}\n"
        report += f"当前持仓: {summary['positions_count']} 只股票\n\n"
        
        # Agent performance
        report += "智能体表现\n"
        report += "-" * 20 + "\n"
        
        if 'agent_performance' in performance_data:
            for agent, perf in performance_data['agent_performance'].items():
                report += f"{agent}: 贡献度 {perf.get('contribution', 0):.2%}, "
                report += f"胜率 {perf.get('win_rate', 0):.2%}, "
                report += f"平均收益 {perf.get('avg_return', 0):.2%}\n"
        else:
            report += "暂无足够数据评估各智能体表现\n"
        
        report += "\n"
        
        # System insights
        report += "系统洞察\n"
        report += "-" * 20 + "\n"
        
        # Generate some insights based on the data
        insights = self._generate_insights(summary, performance_data)
        for insight in insights:
            report += f"- {insight}\n"
        
        report += "\n"
        
        # Record report
        self.narratives.append({
            'timestamp': datetime.now(),
            'type': 'session_report',
            'content': report
        })
        
        return report
    
    def _generate_market_narrative(self, market_summary):
        """
        Generate a narrative about market conditions
        
        Args:
            market_summary (dict): Market summary information
            
        Returns:
            str: Generated narrative
        """
        regime = market_summary.get('market_regime', 'sideways')
        
        # Get top and bottom sectors
        top_sectors = market_summary.get('top_sectors', [])
        bottom_sectors = market_summary.get('bottom_sectors', [])
        
        top_sectors_text = "、".join(top_sectors[:2]) if top_sectors else "无明显强势板块"
        bottom_sectors_text = "、".join(bottom_sectors[:2]) if bottom_sectors else "无明显弱势板块"
        
        # Choose template based on market regime
        templates = self.market_templates.get(regime, self.market_templates['sideways'])
        template = random.choice(templates)
        
        # Fill in template
        narrative = template.format(
            top_sectors=top_sectors_text,
            bottom_sectors=bottom_sectors_text
        )
        
        # Add strength and volatility information
        strength = market_summary.get('market_strength', 0)
        volatility = market_summary.get('market_volatility', 'medium')
        
        if abs(strength) > 0.5:
            narrative += f" 市场强度{abs(strength):.1f}，"
            if strength > 0:
                narrative += "多头占优。"
            else:
                narrative += "空头占优。"
        
        if volatility != 'medium':
            if volatility == 'high':
                narrative += " 市场波动性较高，需谨慎操作。"
            else:
                narrative += " 市场波动性低，交投平淡。"
        
        return narrative
    
    def _generate_agent_narrative(self, agents, execution_results):
        """
        Generate a narrative about agent activity
        
        Args:
            agents (list): List of agent instances
            execution_results (list): List of execution results
            
        Returns:
            str: Generated narrative
        """
        if not execution_results:
            return "今日无交易执行。"
        
        narratives = []
        
        # Process buy executions
        buys = [r for r in execution_results if r.direction == 'BUY']
        if buys:
            for result in buys[:3]:  # Limit to top 3
                stock_code = result.stock_code
                
                # Find the agent responsible for this trade
                agent_name = "系统"
                agent_type = "fast_profit"  # Default
                reason = "综合因素分析"
                confidence_text = "中等强度"
                
                # Try to find more details about this trade
                # (In a real system, you would have more detailed tracking)
                
                # Choose template based on agent type
                templates = self.agent_templates.get(agent_type, self.agent_templates['fast_profit'])
                template = random.choice(templates)
                
                # Fill in template
                trade_narrative = template.format(
                    agent_name=agent_name,
                    stock_code=stock_code,
                    reason=reason,
                    confidence_text=confidence_text
                )
                
                narratives.append(trade_narrative)
        
        # Process sell executions
        sells = [r for r in execution_results if r.direction == 'SELL']
        if sells:
            for result in sells[:2]:  # Limit to top 2
                profit_pct = 0.0  # In a real system, you would calculate this
                
                # Generate trade narrative
                trade_narrative = self.generate_trade_narrative(
                    {'stock_code': result.stock_code, 'profit_pct': profit_pct},
                    agent_name="系统",
                    strategy="交易策略",
                    exit_reason="按计划执行"
                )
                
                narratives.append(trade_narrative)
        
        # Combine narratives
        if narratives:
            return " ".join(narratives)
        else:
            return "今日系统运行正常，无特殊交易活动。"
    
    def _generate_performance_narrative(self, portfolio):
        """
        Generate a narrative about system performance
        
        Args:
            portfolio (Portfolio): Portfolio instance
            
        Returns:
            str: Generated narrative
        """
        summary = portfolio.get_performance_summary()
        
        # Determine performance trend
        # (In a real system, you would compare to historical performance)
        trend = 'stable'  # Default to stable
        
        # Winning rate is good indicator
        win_rate = summary['win_rate']
        if win_rate >= 0.6:
            trend = 'improving'
        elif win_rate <= 0.4:
            trend = 'deteriorating'
        
        # Choose template based on trend
        templates = self.performance_templates.get(trend, self.performance_templates['stable'])
        template = random.choice(templates)
        
        # Calculate period return (last week)
        period_return = summary['total_return']  # Simplified
        
        # Create portfolio metrics text
        portfolio_metrics = f"胜率{win_rate:.0%}，资金曲线{trend_text(trend)}"
        
        # Fill in template
        narrative = template.format(
            win_rate=win_rate,
            max_drawdown=summary['max_drawdown'],
            period_return=period_return,
            portfolio_metrics=portfolio_metrics
        )
        
        return narrative
    
    def _generate_insights(self, summary, performance_data):
        """
        Generate insights based on performance data
        
        Args:
            summary (dict): Portfolio performance summary
            performance_data (dict): Additional performance data
            
        Returns:
            list: List of insight strings
        """
        insights = []
        
        # Win rate insights
        win_rate = summary['win_rate']
        if win_rate > 0.6:
            insights.append(f"当前胜率{win_rate:.0%}处于良好水平，系统状态健康")
        elif win_rate < 0.4:
            insights.append(f"当前胜率{win_rate:.0%}偏低，建议检查策略参数或市场环境变化")
        
        # Drawdown insights
        max_drawdown = summary['max_drawdown']
        if max_drawdown > 0.1:
            insights.append(f"最大回撤{max_drawdown:.2%}超过警戒线，系统已自动减小仓位")
        elif max_drawdown < 0.03:
            insights.append(f"最大回撤控制良好，仅{max_drawdown:.2%}，风控系统有效")
        
        # Add more insights based on available data
        if summary['positions_count'] > 10:
            insights.append(f"当前持仓{summary['positions_count']}只股票，分散度较高")
        
        # If not enough insights, add a general one
        if len(insights) < 2:
            insights.append("系统运行状态正常，各项指标处于预期范围内")
        
        return insights

    def generate_cycle_narrative(self, market_state, decisions, results):
        """
        Generate a narrative for a single trading cycle
        
        Args:
            market_state (MarketState): Current market state
            decisions (list): Filtered trade decisions
            results (list): Execution results
            
        Returns:
            str: Narrative text
        """
        # Simple implementation - would be more nuanced in a real system
        if not results:
            return "No trading activity in this cycle."
            
        # Count buys and sells
        buys = sum(1 for r in results if r.get('decision_type') == 'buy')
        sells = sum(1 for r in results if r.get('decision_type') == 'sell')
        
        # Get market regime
        regime = market_state.get_market_regime()
        regime_desc = {
            'bull': 'bullish',
            'bear': 'bearish',
            'sideways': 'sideways',
            'unknown': 'unclear'
        }.get(regime, 'uncertain')
        
        # Generate narrative
        narrative = f"In a {regime_desc} market, executed {len(results)} trades ({buys} buys, {sells} sells)."
        
        return narrative

def trend_text(trend):
    """Helper function to convert trend to Chinese text"""
    if trend == 'improving':
        return "稳步上升"
    elif trend == 'deteriorating':
        return "有所下滑"
    else:
        return "保持平稳" 