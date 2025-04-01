#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal GUI主窗口
"""

import sys
import logging
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QTableWidget, QTableWidgetItem, QTabWidget,
                            QMessageBox, QHeaderView, QGroupBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QIcon

from seto_versal.market.state import MarketState
from seto_versal.agents.factory import AgentFactory
from seto_versal.evolution.engine import EvolutionEngine
from seto_versal.risk.manager import RiskManager
from seto_versal.gui.international_data_tab import InternationalDataTab

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """SETO-Versal GUI主窗口"""
    
    def __init__(self):
        """初始化主窗口"""
        super().__init__()
        self.setWindowTitle("SETO-Versal 交易系统")
        self.setMinimumSize(1200, 800)
        
        # 设置样式表
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f5f5f5;
            }
            QTableWidget {
                gridline-color: #d0d0d0;
                background-color: white;
                alternate-background-color: #f9f9f9;
                selection-background-color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #e0e0e0;
                padding: 4px;
                border: 1px solid #d0d0d0;
                font-weight: bold;
            }
            QLabel {
                color: #333333;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                padding: 6px 12px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QPushButton:pressed {
                background-color: #2a66c8;
            }
            QComboBox {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
            }
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #d0d0d0;
            }
        """)
        
        # 初始化组件
        self.market_state = None
        self.agent_factory = None
        self.evolution_engine = None
        self.risk_manager = None
        
        # 创建UI
        self._create_ui()
        
        # 设置定时器更新
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_status)
        self.update_timer.start(1000)  # 每秒更新一次
        
    def _create_ui(self):
        """创建用户界面"""
        # 创建中央部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 创建顶部栏
        top_bar = self._create_top_bar()
        main_layout.addLayout(top_bar)
        
        # 创建标签页
        self.tab_widget = QTabWidget(self)
        main_layout.addWidget(self.tab_widget)
        
        # 添加标签页
        self._create_dashboard_tab()
        self._create_market_tab()
        self._create_agents_tab()
        self._create_positions_tab()
        self._create_risk_tab()
        self._create_signals_tab()
        self._create_international_data_tab()
        
    def _create_top_bar(self):
        """创建顶部控制栏"""
        layout = QHBoxLayout()
        
        # 模式选择器
        mode_label = QLabel("模式:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["回测", "模拟", "实盘"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        
        # 开始/停止按钮
        self.start_button = QPushButton("启动")
        self.start_button.clicked.connect(self._on_start_clicked)
        
        # 状态标签
        self.status_label = QLabel("系统状态: 已停止")
        
        # 添加部件到布局
        layout.addWidget(mode_label)
        layout.addWidget(self.mode_combo)
        layout.addWidget(self.start_button)
        layout.addStretch()
        layout.addWidget(self.status_label)
        
        return layout
        
    def _create_dashboard_tab(self):
        """创建仪表盘标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 添加顶部指标面板
        indicators_layout = QHBoxLayout()
        indicators_layout.setSpacing(20)
        
        # 市场状态指标
        market_indicator = QWidget()
        market_indicator.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
        """)
        market_indicator_layout = QVBoxLayout(market_indicator)
        market_indicator_layout.setContentsMargins(15, 15, 15, 15)
        
        market_indicator_title = QLabel("市场状态")
        market_indicator_title.setStyleSheet("font-weight: bold; color: #666666;")
        self.market_status_label = QLabel("开盘")
        self.market_status_label.setStyleSheet("font-size: 16pt; color: #4a86e8; font-weight: bold;")
        
        market_indicator_layout.addWidget(market_indicator_title)
        market_indicator_layout.addWidget(self.market_status_label)
        
        # 代理指标
        agent_indicator = QWidget()
        agent_indicator.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
        """)
        agent_indicator_layout = QVBoxLayout(agent_indicator)
        agent_indicator_layout.setContentsMargins(15, 15, 15, 15)
        
        agent_indicator_title = QLabel("活跃代理")
        agent_indicator_title.setStyleSheet("font-weight: bold; color: #666666;")
        self.active_agents_label = QLabel("5")
        self.active_agents_label.setStyleSheet("font-size: 16pt; color: #4a86e8; font-weight: bold;")
        
        agent_indicator_layout.addWidget(agent_indicator_title)
        agent_indicator_layout.addWidget(self.active_agents_label)
        
        # 持仓指标
        position_indicator = QWidget()
        position_indicator.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
        """)
        position_indicator_layout = QVBoxLayout(position_indicator)
        position_indicator_layout.setContentsMargins(15, 15, 15, 15)
        
        position_indicator_title = QLabel("当前持仓")
        position_indicator_title.setStyleSheet("font-weight: bold; color: #666666;")
        self.position_count_label = QLabel("5")
        self.position_count_label.setStyleSheet("font-size: 16pt; color: #4a86e8; font-weight: bold;")
        
        position_indicator_layout.addWidget(position_indicator_title)
        position_indicator_layout.addWidget(self.position_count_label)
        
        # 风险指标
        risk_indicator = QWidget()
        risk_indicator.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
        """)
        risk_indicator_layout = QVBoxLayout(risk_indicator)
        risk_indicator_layout.setContentsMargins(15, 15, 15, 15)
        
        risk_indicator_title = QLabel("风险评分")
        risk_indicator_title.setStyleSheet("font-weight: bold; color: #666666;")
        self.risk_score_label = QLabel("27")
        self.risk_score_label.setStyleSheet("font-size: 16pt; color: #4a86e8; font-weight: bold;")
        
        risk_indicator_layout.addWidget(risk_indicator_title)
        risk_indicator_layout.addWidget(self.risk_score_label)
        
        # 添加所有指标到布局
        indicators_layout.addWidget(market_indicator)
        indicators_layout.addWidget(agent_indicator)
        indicators_layout.addWidget(position_indicator)
        indicators_layout.addWidget(risk_indicator)
        
        layout.addLayout(indicators_layout)
        
        # 市场行情摘要
        market_group = QWidget()
        market_group.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
        """)
        market_layout = QVBoxLayout(market_group)
        market_layout.setContentsMargins(15, 15, 15, 15)
        
        market_header = QLabel("市场行情")
        market_header.setStyleSheet("font-weight: bold; font-size: 14pt; color: #333333;")
        market_layout.addWidget(market_header)
        
        self.market_table = QTableWidget()
        self.market_table.setColumnCount(4)
        self.market_table.setRowCount(5)  # 仪表盘只显示前5支股票
        self.market_table.setHorizontalHeaderLabels(["股票代码", "价格", "涨跌幅", "成交量"])
        self.market_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.market_table.setAlternatingRowColors(True)
        self.market_table.setShowGrid(True)
        self.market_table.setStyleSheet("""
            QTableWidget {
                border: none;
            }
        """)
        market_layout.addWidget(self.market_table)
        
        layout.addWidget(market_group)
        
        self.tab_widget.addTab(widget, "仪表盘")
        
    def _create_market_tab(self):
        """创建市场行情标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 市场状态，将它移到交易建议面板之后
        market_group = QWidget()
        market_group.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
        """)
        market_layout = QVBoxLayout(market_group)
        market_layout.setContentsMargins(15, 15, 15, 15)
        
        market_header = QLabel("市场行情")
        market_header.setStyleSheet("font-weight: bold; font-size: 14pt; color: #333333;")
        market_layout.addWidget(market_header)
        
        self.market_table = QTableWidget()
        self.market_table.setColumnCount(4)
        self.market_table.setHorizontalHeaderLabels(["股票代码", "价格", "涨跌幅", "成交量"])
        self.market_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.market_table.setAlternatingRowColors(True)
        self.market_table.setShowGrid(True)
        self.market_table.setStyleSheet("""
            QTableWidget {
                border: none;
            }
        """)
        market_layout.addWidget(self.market_table)
        
        layout.addWidget(market_group)
        
        self.tab_widget.addTab(widget, "市场行情")
        
    def _create_agents_tab(self):
        """创建代理标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 代理表格
        self.agents_table = QTableWidget()
        self.agents_table.setColumnCount(5)
        self.agents_table.setHorizontalHeaderLabels(["名称", "类型", "状态", "绩效", "权重"])
        self.agents_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.agents_table)
        
        self.tab_widget.addTab(widget, "代理")
        
    def _create_positions_tab(self):
        """创建持仓标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 持仓表格
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(6)
        self.positions_table.setHorizontalHeaderLabels(["股票代码", "数量", "入场价", "当前价", "盈亏", "风险"])
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.positions_table)
        
        self.tab_widget.addTab(widget, "持仓")
        
    def _create_risk_tab(self):
        """创建风险标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 风险指标
        risk_group = QWidget()
        risk_layout = QVBoxLayout(risk_group)
        
        risk_header = QLabel("风险指标")
        risk_header.setFont(QFont("", 12, QFont.Weight.Bold))
        risk_layout.addWidget(risk_header)
        
        self.risk_table = QTableWidget()
        self.risk_table.setColumnCount(3)
        self.risk_table.setHorizontalHeaderLabels(["指标", "当前值", "限制"])
        self.risk_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        risk_layout.addWidget(self.risk_table)
        
        layout.addWidget(risk_group)
        layout.addStretch()
        
        self.tab_widget.addTab(widget, "风险")
        
    def _create_signals_tab(self):
        """创建交易信号标签页"""
        signals_tab = QWidget()
        self.tab_widget.addTab(signals_tab, "交易信号")
        
        # 主布局
        main_layout = QVBoxLayout(signals_tab)
        
        # 信号更新时间
        header_layout = QHBoxLayout()
        self.signal_update_time_label = QLabel("最后更新: 未更新")
        refresh_button = QPushButton("刷新信号")
        refresh_button.clicked.connect(self._update_trading_signals)
        header_layout.addWidget(self.signal_update_time_label)
        header_layout.addStretch()
        header_layout.addWidget(refresh_button)
        main_layout.addLayout(header_layout)
        
        # 买入信号区域
        buy_group = QGroupBox("买入建议")
        buy_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 6px; margin-top: 12px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        self.buy_signals_layout = QVBoxLayout(buy_group)
        self.buy_signals_layout.setSpacing(8)
        self.buy_signals_widgets = []
        
        # 卖出信号区域
        sell_group = QGroupBox("卖出建议")
        sell_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 6px; margin-top: 12px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        self.sell_signals_layout = QVBoxLayout(sell_group)
        self.sell_signals_layout.setSpacing(8)
        self.sell_signals_widgets = []
        
        # 添加到主布局
        main_layout.addWidget(buy_group)
        main_layout.addWidget(sell_group)
        main_layout.addStretch()
        
        # 初始信号
        self._update_trading_signals()
        
    def _create_international_data_tab(self):
        """创建国际市场数据选项卡"""
        # 获取数据管理器
        data_manager = None
        if hasattr(self, 'engine') and hasattr(self.engine, 'data_manager'):
            data_manager = self.engine.data_manager
            
        if data_manager is None:
            # 如果没有数据管理器，创建一个临时的
            from seto_versal.data.enhanced_manager import EnhancedDataManager
            try:
                data_manager = EnhancedDataManager()
            except Exception as e:
                QMessageBox.warning(self, "警告", f"无法初始化数据管理器: {str(e)}")
                return
                
        # 创建国际市场数据选项卡
        try:
            self.international_data_tab = InternationalDataTab(data_manager)
            self.tab_widget.addTab(self.international_data_tab, "国际市场数据")
        except Exception as e:
            logger.error(f"创建国际市场数据选项卡失败: {e}")
            QMessageBox.warning(self, "警告", f"创建国际市场数据选项卡失败: {str(e)}")
            
    def _update_trading_signals(self):
        """更新交易信号"""
        try:
            # 清理现有信号部件
            for widget in self.buy_signals_widgets:
                widget.deleteLater()
            for widget in self.sell_signals_widgets:
                widget.deleteLater()
                
            self.buy_signals_widgets = []
            self.sell_signals_widgets = []
            
            # 获取系统建议
            buy_signals = []
            sell_signals = []
            
            # 首先从市场状态获取实时信号
            if self.market_state and hasattr(self.market_state, 'get_trading_signals'):
                signals = self.market_state.get_trading_signals()
                if signals:
                    buy_signals = signals.get('buy', [])
                    sell_signals = signals.get('sell', [])
            
            # 如果市场状态没有提供信号，尝试从代理获取
            if (not buy_signals and not sell_signals) and self.agent_factory:
                # 从所有代理获取建议
                agents = []
                if hasattr(self.agent_factory, 'get_all_agents'):
                    agents = self.agent_factory.get_all_agents()
                elif hasattr(self.agent_factory, 'get_agents'):
                    agents = self.agent_factory.get_agents()
                    
                for agent in agents:
                    if hasattr(agent, 'get_recommendations'):
                        agent_signals = agent.get_recommendations()
                        if agent_signals:
                            # 分类为买入和卖出信号
                            for signal in agent_signals:
                                if signal.get('action', '').lower() == 'buy':
                                    buy_signals.append(signal)
                                elif signal.get('action', '').lower() == 'sell':
                                    sell_signals.append(signal)
            
            # 如果没有建议，添加示例数据
            if not buy_signals and not sell_signals:
                buy_signals = self._get_sample_buy_signals()
                sell_signals = self._get_sample_sell_signals()
                
            # 添加买入信号
            for signal in buy_signals[:5]:  # 最多显示5个
                widget = self._create_signal_widget(signal, 'buy')
                self.buy_signals_layout.addWidget(widget)
                self.buy_signals_widgets.append(widget)
                
            # 添加卖出信号
            for signal in sell_signals[:5]:  # 最多显示5个
                widget = self._create_signal_widget(signal, 'sell')
                self.sell_signals_layout.addWidget(widget)
                self.sell_signals_widgets.append(widget)
                
            # 更新最新更新时间
            current_time = datetime.now().strftime('%H:%M:%S')
            self.signal_update_time_label.setText(f"最后更新: {current_time}")
                
        except Exception as e:
            logger.error(f"更新交易信号失败: {e}")
            
    def _create_signal_widget(self, signal, signal_type):
        """创建单个信号部件"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        symbol = signal.get('symbol', 'Unknown')
        confidence = signal.get('confidence', 0.0)
        reason = signal.get('reason', '')
        timestamp = signal.get('timestamp', '')
        
        if not reason:
            if signal_type == 'buy':
                reason = "看好后市上涨趋势"
            else:
                reason = "规避下跌风险"
                
        price = signal.get('price', 0.0)
        if price <= 0 and self.market_state:
            price = self.market_state.get_price(symbol) or 0.0
            
        # 股票代码和价格
        stock_label = QLabel(f"{symbol}  {price:.2f}")
        stock_label.setStyleSheet(f"font-weight: bold; color: {'#e63946' if signal_type == 'buy' else '#2a9d8f'};")
        
        # 建议原因
        reason_label = QLabel(reason)
        reason_label.setStyleSheet("color: #666666;")
        
        # 信心指数
        confidence_label = QLabel(f"{int(confidence * 100)}%")
        confidence_label.setStyleSheet(f"color: {'#e63946' if signal_type == 'buy' else '#2a9d8f'};")
        
        layout.addWidget(stock_label)
        layout.addWidget(reason_label)
        layout.addStretch()
        layout.addWidget(confidence_label)
        
        # 设置整体样式
        widget.setStyleSheet(f"QWidget {{ background-color: {'#fff5f5' if signal_type == 'buy' else '#f0f8f5'}; border-radius: 4px; }}")
        
        # 添加点击事件
        widget.mousePressEvent = lambda event: self._show_signal_details(signal, signal_type)
        widget.setCursor(Qt.CursorShape.PointingHandCursor)
        
        return widget
        
    def _show_signal_details(self, signal, signal_type):
        """显示信号详细信息"""
        try:
            symbol = signal.get('symbol', 'Unknown')
            price = signal.get('price', 0.0)
            confidence = signal.get('confidence', 0.0)
            reason = signal.get('reason', '')
            timestamp = signal.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            action_text = "买入" if signal_type == 'buy' else "卖出"
            
            # 创建详细信息对话框
            dialog = QMessageBox(self)
            dialog.setWindowTitle(f"交易信号详情 - {symbol}")
            
            # 设置图标
            if signal_type == 'buy':
                dialog.setIconPixmap(QIcon.fromTheme("go-up").pixmap(32, 32))
            else:
                dialog.setIconPixmap(QIcon.fromTheme("go-down").pixmap(32, 32))
            
            # 构建详细信息
            detail_text = f"""
            <h3>交易信号详情</h3>
            <p><b>建议操作:</b> <span style="color: {'#e63946' if signal_type == 'buy' else '#2a9d8f'};">{action_text}</span></p>
            <p><b>股票代码:</b> {symbol}</p>
            <p><b>当前价格:</b> {price:.2f}</p>
            <p><b>信心指数:</b> {int(confidence * 100)}%</p>
            <p><b>建议理由:</b> {reason}</p>
            <p><b>信号时间:</b> {timestamp}</p>
            <p><b>技术分析:</b> {'MACD指标金叉，KDJ指标显示超卖，建议逢低买入' if signal_type == 'buy' else 'MACD指标死叉，KDJ指标显示超买，建议逢高卖出'}</p>
            """
            
            dialog.setText(detail_text)
            dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
            dialog.exec()
            
        except Exception as e:
            logger.error(f"显示信号详情失败: {e}")
        
    def _get_sample_buy_signals(self):
        """获取示例买入建议"""
        return [
            {
                'symbol': '600519.SH',
                'action': 'buy',
                'price': 1800.00,
                'confidence': 0.85,
                'reason': '突破阻力位，趋势向好'
            },
            {
                'symbol': '000651.SZ',
                'action': 'buy',
                'price': 55.60,
                'confidence': 0.78,
                'reason': '基本面优秀，成交量放大'
            },
            {
                'symbol': '600036.SH',
                'action': 'buy',
                'price': 36.25,
                'confidence': 0.72,
                'reason': '行业龙头，业绩稳定增长'
            },
            {
                'symbol': '002415.SZ',
                'action': 'buy',
                'price': 42.10,
                'confidence': 0.68,
                'reason': '技术指标MACD金叉'
            }
        ]
        
    def _get_sample_sell_signals(self):
        """获取示例卖出建议"""
        return [
            {
                'symbol': '601318.SH',
                'action': 'sell',
                'price': 42.50,
                'confidence': 0.81,
                'reason': '触及阻力位，短期回调风险'
            },
            {
                'symbol': '000625.SZ',
                'action': 'sell',
                'price': 20.35,
                'confidence': 0.75,
                'reason': '成交量萎缩，上涨动能不足'
            },
            {
                'symbol': '002230.SZ',
                'action': 'sell',
                'price': 58.90,
                'confidence': 0.69,
                'reason': '技术指标KDJ死叉'
            }
        ]
        
    def _on_mode_changed(self, mode):
        """处理模式变更"""
        logger.info(f"模式已更改为: {mode}")
        # TODO: 实现模式变更逻辑
        
    def _on_start_clicked(self):
        """处理开始/停止按钮点击"""
        if self.start_button.text() == "启动":
            self._start_system()
        else:
            self._stop_system()
            
    def _start_system(self):
        """启动交易系统"""
        try:
            # 初始化组件
            self.market_state = MarketState(self._get_config())
            self.agent_factory = AgentFactory(self._get_config())
            self.evolution_engine = EvolutionEngine(self._get_config())
            self.risk_manager = RiskManager(self._get_config())
            
            # 更新UI
            self.start_button.setText("停止")
            self.status_label.setText("系统状态: 运行中")
            
            QMessageBox.information(self, "成功", "交易系统启动成功！")
            
        except Exception as e:
            logger.error(f"启动系统失败: {e}")
            QMessageBox.critical(self, "错误", f"启动系统失败: {str(e)}")
            
    def _stop_system(self):
        """停止交易系统"""
        try:
            # 清理组件
            self.market_state = None
            self.agent_factory = None
            self.evolution_engine = None
            self.risk_manager = None
            
            # 更新UI
            self.start_button.setText("启动")
            self.status_label.setText("系统状态: 已停止")
            
            QMessageBox.information(self, "成功", "交易系统已停止！")
            
        except Exception as e:
            logger.error(f"停止系统失败: {e}")
            QMessageBox.critical(self, "错误", f"停止系统失败: {str(e)}")
            
    def _update_status(self):
        """更新系统状态"""
        if self.market_state is None:
            # 默认值
            self.market_status_label.setText("休市")
            self.active_agents_label.setText("0")
            self.position_count_label.setText("0")
            self.risk_score_label.setText("0")
            return
            
        try:
            # 更新顶部指标
            if self.market_state.is_market_open():
                self.market_status_label.setText("开盘")
                self.market_status_label.setStyleSheet("font-size: 16pt; color: green; font-weight: bold;")
            else:
                self.market_status_label.setText("休市")
                self.market_status_label.setStyleSheet("font-size: 16pt; color: gray; font-weight: bold;")
                
            # 更新代理数量
            agents = []
            if self.agent_factory:
                # 尝试不同的获取方法
                if hasattr(self.agent_factory, 'get_all_agents'):
                    agents = self.agent_factory.get_all_agents()
                elif hasattr(self.agent_factory, 'get_agents'):
                    agents = self.agent_factory.get_agents()
            self.active_agents_label.setText(str(len(agents)))
            
            # 更新持仓数量
            positions = {}
            if hasattr(self.market_state, "executor") and self.market_state.executor:
                positions = self.market_state.executor.get_positions()
            self.position_count_label.setText(str(len(positions)))
            
            # 更新风险评分
            risk_score = 0
            if self.risk_manager:
                metrics = self.risk_manager.get_metrics()
                if metrics and "risk_score" in metrics:
                    risk_score = int(metrics["risk_score"])
            self.risk_score_label.setText(str(risk_score))
            if risk_score < 30:
                self.risk_score_label.setStyleSheet("font-size: 16pt; color: green; font-weight: bold;")
            elif risk_score < 70:
                self.risk_score_label.setStyleSheet("font-size: 16pt; color: orange; font-weight: bold;")
            else:
                self.risk_score_label.setStyleSheet("font-size: 16pt; color: red; font-weight: bold;")
            
            # 更新市场数据
            self._update_market_data()
            
            # 更新代理状态
            self._update_agent_status()
            
            # 更新持仓
            self._update_positions()
            
            # 更新风险指标
            self._update_risk_metrics()
            
            # 更新状态标签
            if self.market_state.is_market_open():
                self.status_label.setText(f"系统状态: 运行中 | 市场开盘 | {datetime.now().strftime('%H:%M:%S')}")
            else:
                self.status_label.setText(f"系统状态: 运行中 | 市场休市 | {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"更新状态失败: {e}")
            
    def _update_market_data(self):
        """更新市场数据表格"""
        try:
            # 使用系统数据管理模块获取数据
            if self.market_state:
                # 获取当前市场数据
                market_data = self.market_state.get_market_data()
                
                if not market_data:
                    # 如果没有数据，使用示例数据
                    self._add_sample_market_data()
                    return
                
                # 设置表格行数
                symbols = list(market_data.keys())
                self.market_table.setRowCount(min(len(symbols), 20))  # 最多显示20个股票
                
                # 更新表格数据
                for i, symbol in enumerate(symbols[:20]):
                    # 获取股票数据
                    data = market_data[symbol]
                    price = data.get('price', 0.0)
                    
                    # 获取昨日收盘价
                    if 'prev_close' in data:
                        price_prev = data['prev_close']
                    else:
                        price_prev = self.market_state.get_price_previous(symbol) or price
                    
                    volume = data.get('volume', 0)
                    
                    # 计算涨跌幅
                    if price_prev and price_prev > 0:
                        change_pct = (price - price_prev) / price_prev * 100
                        change_text = f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%"
                    else:
                        change_text = "0.00%"
                        change_pct = 0
                    
                    # 设置表格项
                    self.market_table.setItem(i, 0, QTableWidgetItem(symbol))
                    self.market_table.setItem(i, 1, QTableWidgetItem(f"{price:.2f}"))
                    self.market_table.setItem(i, 2, QTableWidgetItem(change_text))
                    self.market_table.setItem(i, 3, QTableWidgetItem(f"{volume:,}"))
                    
                    # 设置涨跌幅颜色
                    if change_pct > 0:
                        self.market_table.item(i, 2).setForeground(Qt.GlobalColor.red)
                    elif change_pct < 0:
                        self.market_table.item(i, 2).setForeground(Qt.GlobalColor.green)
            else:
                # 如果市场状态不可用，显示示例数据
                self._add_sample_market_data()
                
        except Exception as e:
            logger.error(f"更新市场数据失败: {e}")
            # 出错时显示示例数据
            self._add_sample_market_data()
        
    def _update_agent_status(self):
        """更新代理状态表格"""
        try:
            # 获取代理列表
            agents = []
            if self.agent_factory:
                # 尝试不同的获取方法
                if hasattr(self.agent_factory, 'get_all_agents'):
                    agents = self.agent_factory.get_all_agents()
                elif hasattr(self.agent_factory, 'get_agents'):
                    agents = self.agent_factory.get_agents()
            
            if not agents:
                # 如果没有代理，添加示例数据
                self._add_sample_agent_data()
                return
                
            # 设置表格行数
            self.agents_table.setRowCount(len(agents))
            
            # 更新表格数据
            for i, agent in enumerate(agents):
                # 获取代理信息
                name = agent.name if hasattr(agent, 'name') else f"Agent-{i+1}"
                agent_type = agent.type if hasattr(agent, 'type') else type(agent).__name__.replace('Agent', '').lower()
                
                # 获取代理状态
                is_active = getattr(agent, 'is_active', True)
                status = "运行中" if is_active else "暂停"
                
                # 获取绩效指标
                metrics = {}
                if hasattr(agent, 'get_performance_metrics'):
                    metrics = agent.get_performance_metrics()
                
                if metrics and 'total_return' in metrics:
                    performance = f"{'+' if metrics['total_return'] >= 0 else ''}{metrics['total_return']:.2f}%"
                    perf_value = metrics['total_return']
                else:
                    performance = "N/A"
                    perf_value = 0
                
                weight = getattr(agent, 'weight', 1.0)
                
                # 设置表格项
                self.agents_table.setItem(i, 0, QTableWidgetItem(name))
                self.agents_table.setItem(i, 1, QTableWidgetItem(agent_type))
                self.agents_table.setItem(i, 2, QTableWidgetItem(status))
                self.agents_table.setItem(i, 3, QTableWidgetItem(performance))
                self.agents_table.setItem(i, 4, QTableWidgetItem(f"{weight:.2f}"))
                
                # 设置绩效颜色
                if perf_value > 0:
                    self.agents_table.item(i, 3).setForeground(Qt.GlobalColor.red)
                elif perf_value < 0:
                    self.agents_table.item(i, 3).setForeground(Qt.GlobalColor.green)
        except Exception as e:
            logger.error(f"更新代理状态失败: {e}")
            # 出错时显示示例数据
            self._add_sample_agent_data()
        
    def _update_positions(self):
        """更新持仓表格"""
        try:
            # 获取持仓信息
            positions = {}
            
            # 从市场状态中获取执行器中的持仓信息
            if self.market_state:
                if hasattr(self.market_state, "executor") and self.market_state.executor:
                    positions = self.market_state.executor.get_positions()
                
                # 如果没有持仓模块，尝试直接从市场状态获取
                elif hasattr(self.market_state, "positions"):
                    positions = self.market_state.positions
                    
                # 如果还是没有持仓数据，尝试从交易历史重建
                elif hasattr(self.market_state, "get_trade_history"):
                    trades = self.market_state.get_trade_history()
                    if trades:
                        positions = self._build_positions_from_trades(trades)
            
            if not positions:
                # 如果没有持仓，添加示例数据
                self._add_sample_position_data()
                return
                
            # 设置表格行数
            self.positions_table.setRowCount(len(positions))
            
            # 更新表格数据
            for i, (symbol, position) in enumerate(positions.items()):
                # 获取持仓信息 - 适应不同格式的持仓数据
                if isinstance(position, dict):
                    quantity = position.get('quantity', 0)
                    entry_price = position.get('entry_price', 0.0)
                else:
                    # 如果不是字典类型，尝试作为对象处理
                    quantity = getattr(position, 'quantity', 0)
                    entry_price = getattr(position, 'entry_price', 0.0)
                
                # 获取当前价格
                current_price = 0.0
                if self.market_state:
                    current_price = self.market_state.get_price(symbol) or entry_price
                
                # 计算盈亏
                if entry_price > 0 and quantity > 0:
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    pnl_text = f"{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%"
                else:
                    pnl_text = "0.00%"
                    pnl_pct = 0
                
                # 评估风险
                if pnl_pct < -5:
                    risk = "高"
                elif pnl_pct < -2:
                    risk = "中"
                else:
                    risk = "低"
                
                # 设置表格项
                self.positions_table.setItem(i, 0, QTableWidgetItem(symbol))
                self.positions_table.setItem(i, 1, QTableWidgetItem(f"{quantity:,}"))
                self.positions_table.setItem(i, 2, QTableWidgetItem(f"{entry_price:.2f}"))
                self.positions_table.setItem(i, 3, QTableWidgetItem(f"{current_price:.2f}"))
                self.positions_table.setItem(i, 4, QTableWidgetItem(pnl_text))
                self.positions_table.setItem(i, 5, QTableWidgetItem(risk))
                
                # 设置盈亏颜色
                if pnl_pct > 0:
                    self.positions_table.item(i, 4).setForeground(Qt.GlobalColor.red)
                elif pnl_pct < 0:
                    self.positions_table.item(i, 4).setForeground(Qt.GlobalColor.green)
                
                # 设置风险颜色
                if risk == "高":
                    self.positions_table.item(i, 5).setForeground(Qt.GlobalColor.red)
                elif risk == "中":
                    self.positions_table.item(i, 5).setForeground(Qt.GlobalColor.yellow)
                else:
                    self.positions_table.item(i, 5).setForeground(Qt.GlobalColor.green)
        except Exception as e:
            logger.error(f"更新持仓失败: {e}")
            # 出错时显示示例数据
            self._add_sample_position_data()
        
    def _build_positions_from_trades(self, trades):
        """根据交易历史构建持仓"""
        positions = {}
        for trade in trades:
            symbol = trade.get('symbol')
            action = trade.get('action', '').lower()
            quantity = trade.get('quantity', 0)
            price = trade.get('price', 0)
            
            if not symbol or not action or not quantity:
                continue
                
            if symbol not in positions:
                positions[symbol] = {'quantity': 0, 'entry_price': 0, 'cost': 0}
                
            pos = positions[symbol]
            
            if action == 'buy':
                # 更新持仓成本
                total_cost = pos['cost'] + (price * quantity)
                total_quantity = pos['quantity'] + quantity
                if total_quantity > 0:
                    pos['entry_price'] = total_cost / total_quantity
                pos['quantity'] = total_quantity
                pos['cost'] = total_cost
            elif action == 'sell':
                # 减少持仓
                pos['quantity'] -= quantity
                if pos['quantity'] <= 0:
                    # 清仓，重置成本
                    pos['quantity'] = 0
                    pos['cost'] = 0
                    pos['entry_price'] = 0
        
        # 移除数量为0的持仓
        return {k: v for k, v in positions.items() if v['quantity'] > 0}
        
    def _update_risk_metrics(self):
        """更新风险指标表格"""
        try:
            # 获取风险指标
            metrics = {}
            if self.risk_manager:
                # 尝试使用风险管理器的API获取指标
                if hasattr(self.risk_manager, 'get_metrics'):
                    metrics = self.risk_manager.get_metrics()
                elif hasattr(self.risk_manager, 'get_risk_metrics'):
                    metrics = self.risk_manager.get_risk_metrics()
            
            # 如果系统没有风险管理器或无法获取指标
            if not metrics and self.market_state:
                # 尝试从市场状态获取性能指标
                if hasattr(self.market_state, 'get_performance_metrics'):
                    perf_metrics = self.market_state.get_performance_metrics()
                    if perf_metrics:
                        # 从性能指标中提取风险相关指标
                        metrics = {
                            'max_drawdown': perf_metrics.get('max_drawdown', 0.0),
                            'sharpe_ratio': perf_metrics.get('sharpe_ratio', 0.0),
                            'volatility': perf_metrics.get('volatility', 0.0),
                            'beta': perf_metrics.get('beta', 1.0),
                        }
            
            if not metrics:
                # 如果没有指标，添加示例数据
                self._add_sample_risk_data()
                return
                
            # 准备数据
            risk_data = []
            
            # 添加最大回撤
            if "max_drawdown" in metrics:
                max_dd = metrics["max_drawdown"]
                max_dd_limit = 0.05
                if self.risk_manager and hasattr(self.risk_manager, 'max_drawdown'):
                    max_dd_limit = self.risk_manager.max_drawdown
                risk_data.append(("最大回撤", f"{max_dd:.2%}", f"{max_dd_limit:.2%}"))
            
            # 添加夏普比率
            if "sharpe_ratio" in metrics:
                sharpe = metrics["sharpe_ratio"]
                risk_data.append(("夏普比率", f"{sharpe:.2f}", "1.5"))
            
            # 添加波动率
            if "volatility" in metrics:
                vol = metrics["volatility"]
                risk_data.append(("波动率", f"{vol:.2%}", "15%"))
            
            # 添加Beta
            if "beta" in metrics:
                beta = metrics["beta"]
                risk_data.append(("Beta", f"{beta:.2f}", "1.0"))
            
            # 添加最大持仓比例
            if "max_position_size" in metrics:
                max_pos = metrics["max_position_size"]
                max_pos_limit = 0.3
                if self.risk_manager and hasattr(self.risk_manager, 'max_position_size'):
                    max_pos_limit = self.risk_manager.max_position_size
                risk_data.append(("最大持仓", f"{max_pos:.2%}", f"{max_pos_limit:.2%}"))
                
            # 添加风险评分
            if "risk_score" in metrics:
                risk_score = metrics["risk_score"]
                risk_data.append(("风险评分", f"{risk_score:.0f}", "100"))
                
                # 更新仪表盘风险评分
                if hasattr(self, 'risk_score_label'):
                    self.risk_score_label.setText(f"{risk_score:.0f}")
                    if risk_score < 30:
                        self.risk_score_label.setStyleSheet("font-size: 16pt; color: green; font-weight: bold;")
                    elif risk_score < 70:
                        self.risk_score_label.setStyleSheet("font-size: 16pt; color: orange; font-weight: bold;")
                    else:
                        self.risk_score_label.setStyleSheet("font-size: 16pt; color: red; font-weight: bold;")
            
            # 设置表格行数
            self.risk_table.setRowCount(len(risk_data))
            
            # 更新表格数据
            for i, (metric, value, limit) in enumerate(risk_data):
                self.risk_table.setItem(i, 0, QTableWidgetItem(metric))
                self.risk_table.setItem(i, 1, QTableWidgetItem(value))
                self.risk_table.setItem(i, 2, QTableWidgetItem(limit))
        except Exception as e:
            logger.error(f"更新风险指标失败: {e}")
            # 出错时显示示例数据
            self._add_sample_risk_data()
    
    def _add_sample_market_data(self):
        """添加示例市场数据"""
        sample_data = [
            ("000001.SZ", "10.25", "+2.5%", "1,234,567"),
            ("000002.SZ", "15.80", "-1.2%", "987,654"),
            ("000003.SZ", "25.60", "+3.8%", "2,345,678"),
            ("000004.SZ", "8.90", "-0.5%", "456,789"),
            ("000005.SZ", "12.35", "+1.8%", "789,123")
        ]
        
        self.market_table.setRowCount(len(sample_data))
        for i, (code, price, change, volume) in enumerate(sample_data):
            self.market_table.setItem(i, 0, QTableWidgetItem(code))
            self.market_table.setItem(i, 1, QTableWidgetItem(price))
            self.market_table.setItem(i, 2, QTableWidgetItem(change))
            self.market_table.setItem(i, 3, QTableWidgetItem(volume))
            
            # 设置涨跌幅颜色
            if "+" in change:
                self.market_table.item(i, 2).setForeground(Qt.GlobalColor.red)
            elif "-" in change:
                self.market_table.item(i, 2).setForeground(Qt.GlobalColor.green)
            
    def _add_sample_agent_data(self):
        """添加示例代理数据"""
        sample_data = [
            ("趋势跟踪", "趋势", "运行中", "+15.2%", "1.0"),
            ("动量交易", "动量", "运行中", "+8.5%", "0.8"),
            ("均值回归", "回归", "暂停", "-2.3%", "0.5"),
            ("波动套利", "套利", "运行中", "+5.7%", "0.7"),
            ("事件驱动", "事件", "运行中", "+12.1%", "0.9")
        ]
        
        self.agents_table.setRowCount(len(sample_data))
        for i, (name, type_, status, perf, weight) in enumerate(sample_data):
            self.agents_table.setItem(i, 0, QTableWidgetItem(name))
            self.agents_table.setItem(i, 1, QTableWidgetItem(type_))
            self.agents_table.setItem(i, 2, QTableWidgetItem(status))
            self.agents_table.setItem(i, 3, QTableWidgetItem(perf))
            self.agents_table.setItem(i, 4, QTableWidgetItem(weight))
            
            # 设置绩效颜色
            if "+" in perf:
                self.agents_table.item(i, 3).setForeground(Qt.GlobalColor.red)
            elif "-" in perf:
                self.agents_table.item(i, 3).setForeground(Qt.GlobalColor.green)
            
    def _add_sample_position_data(self):
        """添加示例持仓数据"""
        sample_data = [
            ("000001.SZ", "1000", "10.00", "10.25", "+2.5%", "低"),
            ("000002.SZ", "2000", "16.00", "15.80", "-1.2%", "中"),
            ("000003.SZ", "1500", "24.50", "25.60", "+4.5%", "低"),
            ("000004.SZ", "3000", "9.00", "8.90", "-1.1%", "高"),
            ("000005.SZ", "2500", "12.10", "12.35", "+2.1%", "中")
        ]
        
        self.positions_table.setRowCount(len(sample_data))
        for i, (code, qty, entry, current, pnl, risk) in enumerate(sample_data):
            self.positions_table.setItem(i, 0, QTableWidgetItem(code))
            self.positions_table.setItem(i, 1, QTableWidgetItem(qty))
            self.positions_table.setItem(i, 2, QTableWidgetItem(entry))
            self.positions_table.setItem(i, 3, QTableWidgetItem(current))
            self.positions_table.setItem(i, 4, QTableWidgetItem(pnl))
            self.positions_table.setItem(i, 5, QTableWidgetItem(risk))
            
            # 设置盈亏颜色
            if "+" in pnl:
                self.positions_table.item(i, 4).setForeground(Qt.GlobalColor.red)
            elif "-" in pnl:
                self.positions_table.item(i, 4).setForeground(Qt.GlobalColor.green)
                
            # 设置风险颜色
            if risk == "高":
                self.positions_table.item(i, 5).setForeground(Qt.GlobalColor.red)
            elif risk == "中":
                self.positions_table.item(i, 5).setForeground(Qt.GlobalColor.yellow)
            else:
                self.positions_table.item(i, 5).setForeground(Qt.GlobalColor.green)
            
    def _add_sample_risk_data(self):
        """添加示例风险数据"""
        sample_data = [
            ("最大回撤", "2.5%", "5%"),
            ("夏普比率", "1.8", "1.5"),
            ("波动率", "12.5%", "15%"),
            ("Beta", "0.85", "1.0"),
            ("信息比率", "1.2", "1.0")
        ]
        
        self.risk_table.setRowCount(len(sample_data))
        for i, (metric, value, limit) in enumerate(sample_data):
            self.risk_table.setItem(i, 0, QTableWidgetItem(metric))
            self.risk_table.setItem(i, 1, QTableWidgetItem(value))
            self.risk_table.setItem(i, 2, QTableWidgetItem(limit))
        
    def _get_config(self):
        """获取系统配置"""
        mode_map = {
            "回测": "backtest",
            "模拟": "paper",
            "实盘": "live"
        }
        return {
            'mode': mode_map[self.mode_combo.currentText()],
            'mode_configs': {
                'backtest': {
                    'update_interval': 300,
                    'use_cache': True,
                    'risk_level': 'high',
                    'max_drawdown': 0.05,
                    'max_position_size': 0.3,
                    'trailing_stop': True
                },
                'paper': {
                    'update_interval': 60,
                    'use_cache': True,
                    'risk_level': 'medium',
                    'max_drawdown': 0.03,
                    'max_position_size': 0.25,
                    'trailing_stop': True
                },
                'live': {
                    'update_interval': 30,
                    'use_cache': False,
                    'risk_level': 'low',
                    'max_drawdown': 0.02,
                    'max_position_size': 0.2,
                    'trailing_stop': True
                }
            },
            'datasource': 'tushare',
            'universe': 'hs300',
            'trading_hours': {
                'open': '09:30',
                'close': '15:00',
                'lunch_start': '11:30',
                'lunch_end': '13:00'
            },
            'agents': [
                {
                    'type': 'trend',
                    'name': 'trend_agent_1',
                    'confidence_threshold': 0.7,
                    'max_positions': 5,
                    'weight': 1.0,
                    'lookback_period': 20,
                    'trend_threshold': 0.05
                }
            ]
        } 