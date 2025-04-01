#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal 交易系统 PyQt6 主窗口
"""

import os
import sys
import time
import logging
import datetime
from pathlib import Path
import random

from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QTabWidget, 
                            QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                            QTableWidget, QTableWidgetItem, QStatusBar, 
                            QGroupBox, QGridLayout, QTextEdit, QComboBox,
                            QMessageBox, QSplitter, QFrame, QHeaderView)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QSize
from PyQt6.QtGui import QFont, QIcon, QColor

# 导入市场状态
from seto_versal.market.state import MarketState

# 设置日志
logger = logging.getLogger(__name__)

class MarketStateDisplay(QWidget):
    """市场状态显示组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # 市场状态标题
        status_label = QLabel("市场状态", self)
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(14)
        status_label.setFont(font)
        self.layout.addWidget(status_label)
        
        # 市场状态信息组
        info_group = QGroupBox("市场数据", self)
        info_layout = QGridLayout()
        
        # 市场开闭状态
        self.market_status_label = QLabel("状态: 未知", self)
        info_layout.addWidget(QLabel("市场状态:"), 0, 0)
        info_layout.addWidget(self.market_status_label, 0, 1)
        
        # 上涨/下跌股票数
        self.up_stocks_label = QLabel("0", self)
        self.down_stocks_label = QLabel("0", self)
        info_layout.addWidget(QLabel("上涨股票:"), 1, 0)
        info_layout.addWidget(self.up_stocks_label, 1, 1)
        info_layout.addWidget(QLabel("下跌股票:"), 2, 0)
        info_layout.addWidget(self.down_stocks_label, 2, 1)
        
        # 平均涨跌幅
        self.avg_change_label = QLabel("0.00%", self)
        info_layout.addWidget(QLabel("平均涨跌:"), 3, 0)
        info_layout.addWidget(self.avg_change_label, 3, 1)
        
        # 成交量
        self.volume_label = QLabel("0", self)
        info_layout.addWidget(QLabel("总成交量:"), 4, 0)
        info_layout.addWidget(self.volume_label, 4, 1)
        
        info_group.setLayout(info_layout)
        self.layout.addWidget(info_group)
        
        # 添加一些空间
        self.layout.addStretch(1)

    def update_market_data(self, market_data):
        """更新市场数据显示"""
        self.market_status_label.setText(market_data.get('market_status', '未知'))
        self.up_stocks_label.setText(str(market_data.get('up_count', 0)))
        self.down_stocks_label.setText(str(market_data.get('down_count', 0)))
        self.avg_change_label.setText(f"{market_data.get('avg_change', 0):.2f}%")
        self.volume_label.setText(f"{market_data.get('total_volume', 0):,}")


class StockListWidget(QWidget):
    """股票列表组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("可交易股票", self)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(14)
        title_label.setFont(font)
        self.layout.addWidget(title_label)
        
        # 股票列表表格
        self.stock_table = QTableWidget(0, 4, self)
        self.stock_table.setHorizontalHeaderLabels(["代码", "名称", "价格", "涨跌幅"])
        self.stock_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.stock_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.stock_table.setAlternatingRowColors(True)
        
        # 设置列宽
        self.stock_table.setColumnWidth(0, 80)
        self.stock_table.setColumnWidth(1, 120)
        self.stock_table.setColumnWidth(2, 80)
        self.stock_table.setColumnWidth(3, 80)
        
        self.layout.addWidget(self.stock_table)

    def update_stock_list(self, stocks):
        """更新股票列表"""
        self.stock_table.setRowCount(0)
        
        for row, stock in enumerate(stocks):
            self.stock_table.insertRow(row)
            
            # 股票代码
            code_item = QTableWidgetItem(stock.get('symbol', ''))
            self.stock_table.setItem(row, 0, code_item)
            
            # 股票名称
            name_item = QTableWidgetItem(stock.get('name', '未知'))
            self.stock_table.setItem(row, 1, name_item)
            
            # 价格
            price_item = QTableWidgetItem(f"{stock.get('price', 0):.2f}")
            self.stock_table.setItem(row, 2, price_item)
            
            # 涨跌幅
            change = stock.get('change_pct', 0)
            change_item = QTableWidgetItem(f"{change:.2f}%")
            
            # 根据涨跌设置颜色
            if change > 0:
                change_item.setForeground(QColor(255, 0, 0))  # 红色表示上涨
            elif change < 0:
                change_item.setForeground(QColor(0, 128, 0))  # 绿色表示下跌
                
            self.stock_table.setItem(row, 3, change_item)


class TradePanel(QWidget):
    """交易面板组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("交易面板", self)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(14)
        title_label.setFont(font)
        self.layout.addWidget(title_label)
        
        # 交易面板表单
        trade_form = QGroupBox("交易表单", self)
        form_layout = QGridLayout()
        
        # 股票选择
        form_layout.addWidget(QLabel("股票:"), 0, 0)
        self.stock_combo = QComboBox(self)
        form_layout.addWidget(self.stock_combo, 0, 1)
        
        # 交易数量
        form_layout.addWidget(QLabel("数量:"), 1, 0)
        self.quantity_combo = QComboBox(self)
        for q in [100, 200, 300, 400, 500, 1000, 2000, 5000]:
            self.quantity_combo.addItem(str(q))
        form_layout.addWidget(self.quantity_combo, 1, 1)
        
        # 买入按钮
        self.buy_button = QPushButton("买入", self)
        self.buy_button.setStyleSheet("background-color: red; color: white;")
        form_layout.addWidget(self.buy_button, 2, 0)
        
        # 卖出按钮
        self.sell_button = QPushButton("卖出", self)
        self.sell_button.setStyleSheet("background-color: green; color: white;")
        form_layout.addWidget(self.sell_button, 2, 1)
        
        trade_form.setLayout(form_layout)
        self.layout.addWidget(trade_form)
        
        # 交易日志
        log_group = QGroupBox("交易日志", self)
        log_layout = QVBoxLayout()
        self.trade_log = QTextEdit(self)
        self.trade_log.setReadOnly(True)
        log_layout.addWidget(self.trade_log)
        log_group.setLayout(log_layout)
        
        self.layout.addWidget(log_group)
        
        # 连接信号
        self.buy_button.clicked.connect(self.buy_stock)
        self.sell_button.clicked.connect(self.sell_stock)
    
    def update_stock_list(self, stocks):
        """更新股票下拉列表"""
        self.stock_combo.clear()
        for stock in stocks:
            display_text = f"{stock.get('symbol', '')} - {stock.get('name', '未知')}"
            self.stock_combo.addItem(display_text, stock.get('symbol', ''))
    
    def add_log_entry(self, text):
        """添加交易日志条目"""
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        self.trade_log.append(f"[{time_str}] {text}")
    
    def buy_stock(self):
        """买入股票"""
        if self.stock_combo.currentIndex() < 0:
            return
            
        symbol = self.stock_combo.currentData()
        quantity = int(self.quantity_combo.currentText())
        
        # 这里应该调用实际的交易函数
        self.add_log_entry(f"买入 {symbol} {quantity}股")
    
    def sell_stock(self):
        """卖出股票"""
        if self.stock_combo.currentIndex() < 0:
            return
            
        symbol = self.stock_combo.currentData()
        quantity = int(self.quantity_combo.currentText())
        
        # 这里应该调用实际的交易函数
        self.add_log_entry(f"卖出 {symbol} {quantity}股")


class PortfolioPanel(QWidget):
    """持仓面板组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("持仓情况", self)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(14)
        title_label.setFont(font)
        self.layout.addWidget(title_label)
        
        # 账户信息
        account_group = QGroupBox("账户信息", self)
        account_layout = QGridLayout()
        
        # 总资产
        self.total_assets_label = QLabel("0.00", self)
        account_layout.addWidget(QLabel("总资产:"), 0, 0)
        account_layout.addWidget(self.total_assets_label, 0, 1)
        
        # 可用资金
        self.available_cash_label = QLabel("0.00", self)
        account_layout.addWidget(QLabel("可用资金:"), 1, 0)
        account_layout.addWidget(self.available_cash_label, 1, 1)
        
        # 持仓市值
        self.position_value_label = QLabel("0.00", self)
        account_layout.addWidget(QLabel("持仓市值:"), 2, 0)
        account_layout.addWidget(self.position_value_label, 2, 1)
        
        # 当日盈亏
        self.daily_pnl_label = QLabel("0.00", self)
        account_layout.addWidget(QLabel("当日盈亏:"), 3, 0)
        account_layout.addWidget(self.daily_pnl_label, 3, 1)
        
        account_group.setLayout(account_layout)
        self.layout.addWidget(account_group)
        
        # 持仓列表
        position_group = QGroupBox("持仓列表", self)
        position_layout = QVBoxLayout()
        
        self.position_table = QTableWidget(0, 5, self)
        self.position_table.setHorizontalHeaderLabels(["代码", "名称", "持仓量", "成本价", "当前价"])
        self.position_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.position_table.setAlternatingRowColors(True)
        
        # 设置列宽
        self.position_table.setColumnWidth(0, 80)
        self.position_table.setColumnWidth(1, 120)
        self.position_table.setColumnWidth(2, 80)
        self.position_table.setColumnWidth(3, 80)
        self.position_table.setColumnWidth(4, 80)
        
        position_layout.addWidget(self.position_table)
        position_group.setLayout(position_layout)
        
        self.layout.addWidget(position_group)
    
    def update_portfolio(self, account_info, positions):
        """更新持仓信息"""
        # 更新账户信息
        self.total_assets_label.setText(f"{account_info.get('total_assets', 0):.2f}")
        self.available_cash_label.setText(f"{account_info.get('available_cash', 0):.2f}")
        self.position_value_label.setText(f"{account_info.get('position_value', 0):.2f}")
        
        # 更新当日盈亏，根据盈亏设置颜色
        daily_pnl = account_info.get('daily_pnl', 0)
        self.daily_pnl_label.setText(f"{daily_pnl:.2f}")
        if daily_pnl > 0:
            self.daily_pnl_label.setStyleSheet("color: red;")
        elif daily_pnl < 0:
            self.daily_pnl_label.setStyleSheet("color: green;")
        else:
            self.daily_pnl_label.setStyleSheet("")
        
        # 更新持仓列表
        self.position_table.setRowCount(0)
        
        for row, position in enumerate(positions):
            self.position_table.insertRow(row)
            
            # 股票代码
            code_item = QTableWidgetItem(position.get('symbol', ''))
            self.position_table.setItem(row, 0, code_item)
            
            # 股票名称
            name_item = QTableWidgetItem(position.get('name', '未知'))
            self.position_table.setItem(row, 1, name_item)
            
            # 持仓量
            volume_item = QTableWidgetItem(str(position.get('volume', 0)))
            self.position_table.setItem(row, 2, volume_item)
            
            # 成本价
            cost_item = QTableWidgetItem(f"{position.get('cost', 0):.2f}")
            self.position_table.setItem(row, 3, cost_item)
            
            # 当前价
            price_item = QTableWidgetItem(f"{position.get('price', 0):.2f}")
            self.position_table.setItem(row, 4, price_item)


class MarketAnalysisPanel(QFrame):
    """市场分析面板"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """初始化UI组件"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建标题标签
        title_label = QLabel("市场智能分析", self)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(16)
        title_label.setFont(font)
        main_layout.addWidget(title_label)
        
        # 创建垂直分割器
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 上部分：行业分析
        sector_frame = QFrame()
        sector_layout = QVBoxLayout(sector_frame)
        
        sector_group = QGroupBox("行业板块表现")
        sector_inner = QVBoxLayout()
        
        # 创建行业表格
        self.sector_table = QTableWidget(0, 4)
        self.sector_table.setHorizontalHeaderLabels(["行业名称", "涨跌幅", "龙头股", "龙头涨跌幅"])
        self.sector_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.sector_table.setAlternatingRowColors(True)
        self.sector_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        sector_inner.addWidget(self.sector_table)
        sector_group.setLayout(sector_inner)
        sector_layout.addWidget(sector_group)
        
        splitter.addWidget(sector_frame)
        
        # 下部分：市场热点与机会
        market_frame = QFrame()
        market_layout = QVBoxLayout(market_frame)
        
        market_group = QGroupBox("市场热点与投资机会")
        market_inner = QVBoxLayout()
        
        # 创建市场分析文本区域
        self.market_analysis_text = QTextEdit()
        self.market_analysis_text.setReadOnly(True)
        self.market_analysis_text.setHtml("""
            <h3>市场概况分析</h3>
            <p>当前市场整体呈现震荡整理态势，上涨个股152只，下跌个股143只。</p>
            
            <h3>市场热点板块</h3>
            <p>当前市场热点集中在<span style="color:red;">新能源</span>和<span style="color:red;">科技</span>板块，
            呈现出明显的轮动特征。新能源板块受政策利好刺激，整体上涨3.25%；
            科技板块在半导体和AI应用带动下，整体涨幅达2.78%。</p>
            
            <h3>投资机会分析</h3>
            <p>短期投资机会：关注新能源板块龙头企业宁德时代(300750.SZ)，该股突破前期高点，成交量明显放大。</p>
            <p>中期投资机会：科技板块中芯片设计公司有望受益于国产替代进程加速，建议关注华为鲲鹏生态链相关企业。</p>
            <p>长期投资价值：医药创新药企业在经历调整后估值趋于合理，未来发展空间广阔。</p>
            
            <h3>风险提示</h3>
            <p>地产板块持续走弱，短期内仍有下行压力；金融板块受利率影响，表现相对低迷。</p>
        """)
        
        market_inner.addWidget(self.market_analysis_text)
        market_group.setLayout(market_inner)
        market_layout.addWidget(market_group)
        
        splitter.addWidget(market_frame)
        
        # 设置分割比例
        splitter.setSizes([300, 400])
        
        main_layout.addWidget(splitter)
    
    def update_analysis(self, sectors, market_analysis):
        """更新行业分析和市场热点数据"""
        # 更新行业表格
        self.sector_table.setRowCount(len(sectors))
        
        for row, sector in enumerate(sectors):
            # 设置行业名称
            name_item = QTableWidgetItem(sector["name"])
            self.sector_table.setItem(row, 0, name_item)
            
            # 设置涨跌幅
            change_item = QTableWidgetItem(f"{sector['change_pct']:+.2f}%")
            if sector["change_pct"] > 0:
                change_item.setForeground(QColor(255, 0, 0))  # 红色表示上涨
            elif sector["change_pct"] < 0:
                change_item.setForeground(QColor(0, 128, 0))  # 绿色表示下跌
            self.sector_table.setItem(row, 1, change_item)
            
            # 设置龙头股
            leading_item = QTableWidgetItem(sector["leading_stocks"])
            self.sector_table.setItem(row, 2, leading_item)
            
            # 设置龙头涨跌幅
            leading_change_item = QTableWidgetItem(f"{sector['leading_change']:+.2f}%")
            if sector["leading_change"] > 0:
                leading_change_item.setForeground(QColor(255, 0, 0))  # 红色表示上涨
            elif sector["leading_change"] < 0:
                leading_change_item.setForeground(QColor(0, 128, 0))  # 绿色表示下跌
            self.sector_table.setItem(row, 3, leading_change_item)
        
        # 更新市场分析文本
        self.market_analysis_text.setHtml(market_analysis)


class SetoMainWindow(QMainWindow):
    """SETO-Versal 交易系统主窗口"""
    def __init__(self):
        super().__init__()
        
        # 设置窗口标题和大小
        self.setWindowTitle("SETO-Versal 量化交易系统")
        self.resize(1024, 768)
        
        # 初始化市场状态对象
        self.market_state = None
        
        # 初始化UI
        self.init_ui()
        
        # 设置定时器，定期更新市场数据
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_market_data)
        self.update_timer.start(1000)  # 每1秒更新一次
        
        # 记录启动日志
        logger.info("SETO-Versal PyQt6 GUI 已启动")
    
    def init_ui(self):
        """初始化UI组件"""
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建标题标签
        title_label = QLabel("SETO-Versal 智能量化交易系统", self)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(18)
        title_label.setFont(font)
        main_layout.addWidget(title_label)
        
        # 创建标签页控件
        tab_widget = QTabWidget()
        
        # 1. 创建主交易页面
        trading_page = QWidget()
        trading_layout = QVBoxLayout(trading_page)
        
        # 创建交易页面的分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧市场状态面板
        self.market_panel = MarketStateDisplay()
        splitter.addWidget(self.market_panel)
        
        # 中间股票列表
        self.stock_list = StockListWidget()
        splitter.addWidget(self.stock_list)
        
        # 右侧交易面板
        self.trade_panel = TradePanel()
        splitter.addWidget(self.trade_panel)
        
        # 设置分割比例
        splitter.setSizes([200, 400, 300])
        
        trading_layout.addWidget(splitter)
        tab_widget.addTab(trading_page, "市场交易")
        
        # 2. 创建持仓页面
        portfolio_page = QWidget()
        portfolio_layout = QVBoxLayout(portfolio_page)
        
        # 添加持仓面板
        self.portfolio_panel = PortfolioPanel()
        portfolio_layout.addWidget(self.portfolio_panel)
        
        tab_widget.addTab(portfolio_page, "持仓管理")
        
        # 3. 创建智能分析页面
        analysis_page = QWidget()
        analysis_layout = QVBoxLayout(analysis_page)
        
        # 添加市场分析面板
        self.analysis_panel = MarketAnalysisPanel()
        analysis_layout.addWidget(self.analysis_panel)
        
        tab_widget.addTab(analysis_page, "智能分析")
        
        # 4. 创建算法交易页面
        algo_page = QWidget()
        algo_layout = QVBoxLayout(algo_page)
        
        # 添加算法交易标题
        algo_title = QLabel("智能算法交易", self)
        algo_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(14)
        algo_title.setFont(font)
        algo_layout.addWidget(algo_title)
        
        # 创建算法交易分割器
        algo_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧算法选择面板
        algo_select_frame = QFrame()
        algo_select_layout = QVBoxLayout(algo_select_frame)
        
        algo_select_group = QGroupBox("智能算法策略")
        algo_select_inner = QVBoxLayout()
        
        # 算法列表
        self.algo_list = QTableWidget(6, 3)
        self.algo_list.setHorizontalHeaderLabels(["策略名称", "风险等级", "状态"])
        self.algo_list.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.algo_list.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.algo_list.setAlternatingRowColors(True)
        
        # 设置列宽
        self.algo_list.setColumnWidth(0, 150)
        self.algo_list.setColumnWidth(1, 80)
        self.algo_list.setColumnWidth(2, 80)
        
        # 填充示例数据
        algos = [
            ("量化动量跟踪", "中等", "运行中"),
            ("AI智能选股", "较高", "已启用"),
            ("智能板块轮动", "中等", "已启用"),
            ("多因子Alpha", "较高", "已启用"),
            ("趋势突破", "较低", "已禁用"),
            ("行业龙头捕捉", "中等", "已启用")
        ]
        
        for row, (name, risk, status) in enumerate(algos):
            self.algo_list.setItem(row, 0, QTableWidgetItem(name))
            self.algo_list.setItem(row, 1, QTableWidgetItem(risk))
            status_item = QTableWidgetItem(status)
            if status == "运行中":
                status_item.setForeground(QColor(0, 128, 0)) # 绿色
            elif status == "已启用":
                status_item.setForeground(QColor(0, 0, 255)) # 蓝色
            else:
                status_item.setForeground(QColor(128, 128, 128)) # 灰色
            self.algo_list.setItem(row, 2, status_item)
            
        algo_select_inner.addWidget(self.algo_list)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        self.algo_start_btn = QPushButton("启动")
        self.algo_stop_btn = QPushButton("停止")
        self.algo_config_btn = QPushButton("配置")
        button_layout.addWidget(self.algo_start_btn)
        button_layout.addWidget(self.algo_stop_btn)
        button_layout.addWidget(self.algo_config_btn)
        
        algo_select_inner.addLayout(button_layout)
        algo_select_group.setLayout(algo_select_inner)
        algo_select_layout.addWidget(algo_select_group)
        
        algo_splitter.addWidget(algo_select_frame)
        
        # 右侧算法状态面板
        algo_status_frame = QFrame()
        algo_status_layout = QVBoxLayout(algo_status_frame)
        
        status_group = QGroupBox("算法运行状态")
        status_inner = QVBoxLayout()
        
        # 算法状态文本
        self.algo_status_text = QTextEdit()
        self.algo_status_text.setReadOnly(True)
        self.algo_status_text.setHtml("""
            <h3>智能算法运行状态</h3>
            <p><b>量化动量跟踪:</b> 正在监控市场动量变化</p>
            <p style="color:green;">- 已捕捉到新能源板块强势动量信号</p>
            <p style="color:green;">- 半导体行业呈现加速上涨态势</p>
            <p style="color:red;">- 传统金融板块动量减弱</p>
            
            <h3>AI智能选股结果 (Top 5)</h3>
            <ol>
                <li>600519.SH - 贵州茅台 (置信度: 89.5%)</li>
                <li>300750.SZ - 宁德时代 (置信度: 87.2%)</li>
                <li>601899.SH - 紫金矿业 (置信度: 85.4%)</li>
                <li>000858.SZ - 五粮液 (置信度: 82.7%)</li>
                <li>600036.SH - 招商银行 (置信度: 81.3%)</li>
            </ol>
            
            <h3>近期系统进化</h3>
            <p>系统在过去7天内完成了2次策略参数优化，均值回报率提升2.7%</p>
            <p>市场环境识别准确率: 92.5% (+1.3%)</p>
        """)
        
        status_inner.addWidget(self.algo_status_text)
        status_group.setLayout(status_inner)
        algo_status_layout.addWidget(status_group)
        
        algo_splitter.addWidget(algo_status_frame)
        
        # 设置分割比例
        algo_splitter.setSizes([300, 500])
        
        algo_layout.addWidget(algo_splitter)
        tab_widget.addTab(algo_page, "算法交易")
        
        # 5. 创建风险控制页面
        risk_page = QWidget()
        risk_layout = QVBoxLayout(risk_page)
        
        # 风险控制标题
        risk_title = QLabel("风险控制中心", self)
        risk_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(14)
        risk_title.setFont(font)
        risk_layout.addWidget(risk_title)
        
        # 创建风险控制分割器
        risk_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 上部分：风险指标面板
        risk_metrics_frame = QFrame()
        risk_metrics_layout = QVBoxLayout(risk_metrics_frame)
        
        metrics_group = QGroupBox("风险度量指标")
        metrics_inner = QGridLayout()
        
        # 添加各种风险指标
        metrics = [
            ("最大回撤", "7.85%", 0, 0),
            ("波动率", "12.36%", 0, 2),
            ("夏普比率", "1.72", 1, 0),
            ("索提诺比率", "2.05", 1, 2),
            ("贝塔系数", "0.83", 2, 0),
            ("Alpha系数", "3.25%", 2, 2),
            ("信息比率", "1.15", 3, 0),
            ("VaR (95%)", "2.34%", 3, 2)
        ]
        
        for label, value, row, col in metrics:
            metrics_inner.addWidget(QLabel(f"{label}:"), row, col)
            value_label = QLabel(value)
            if label == "最大回撤" or label == "波动率" or label == "VaR (95%)":
                if float(value.rstrip('%')) > 10:
                    value_label.setStyleSheet("color: red;")
                elif float(value.rstrip('%')) > 5:
                    value_label.setStyleSheet("color: orange;")
                else:
                    value_label.setStyleSheet("color: green;")
            elif label in ["夏普比率", "索提诺比率", "信息比率"]:
                if float(value) > 1.5:
                    value_label.setStyleSheet("color: green;")
                elif float(value) > 1.0:
                    value_label.setStyleSheet("color: orange;")
                else:
                    value_label.setStyleSheet("color: red;")
            
            metrics_inner.addWidget(value_label, row, col + 1)
            
        metrics_group.setLayout(metrics_inner)
        risk_metrics_layout.addWidget(metrics_group)
        
        risk_splitter.addWidget(risk_metrics_frame)
        
        # 下部分：资金管理和风控规则
        risk_rules_frame = QFrame()
        risk_rules_layout = QVBoxLayout(risk_rules_frame)
        
        rules_group = QGroupBox("智能风控规则")
        rules_inner = QVBoxLayout()
        
        # 风控规则表格
        self.risk_rules_table = QTableWidget(5, 3)
        self.risk_rules_table.setHorizontalHeaderLabels(["规则名称", "阈值", "状态"])
        self.risk_rules_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.risk_rules_table.setAlternatingRowColors(True)
        
        # 设置列宽
        self.risk_rules_table.setColumnWidth(0, 200)
        self.risk_rules_table.setColumnWidth(1, 150)
        self.risk_rules_table.setColumnWidth(2, 100)
        
        # 填充风控规则
        rules = [
            ("单日最大亏损限制", "总资产的 2%", "激活"),
            ("单个持仓最大占比", "总资产的 10%", "激活"),
            ("止损策略", "成本价的 -8%", "激活"),
            ("止盈策略", "成本价的 +20%", "已禁用"),
            ("波动率过滤", "超过 35% 不交易", "激活")
        ]
        
        for row, (name, threshold, status) in enumerate(rules):
            self.risk_rules_table.setItem(row, 0, QTableWidgetItem(name))
            self.risk_rules_table.setItem(row, 1, QTableWidgetItem(threshold))
            status_item = QTableWidgetItem(status)
            if status == "激活":
                status_item.setForeground(QColor(0, 128, 0))  # 绿色
            else:
                status_item.setForeground(QColor(128, 128, 128))  # 灰色
            self.risk_rules_table.setItem(row, 2, status_item)
            
        rules_inner.addWidget(self.risk_rules_table)
        
        # 风控规则控制
        rule_buttons = QHBoxLayout()
        self.add_rule_btn = QPushButton("添加规则")
        self.edit_rule_btn = QPushButton("编辑规则")
        self.del_rule_btn = QPushButton("删除规则")
        rule_buttons.addWidget(self.add_rule_btn)
        rule_buttons.addWidget(self.edit_rule_btn)
        rule_buttons.addWidget(self.del_rule_btn)
        
        rules_inner.addLayout(rule_buttons)
        rules_group.setLayout(rules_inner)
        risk_rules_layout.addWidget(rules_group)
        
        risk_splitter.addWidget(risk_rules_frame)
        
        # 设置分割比例
        risk_splitter.setSizes([300, 400])
        
        risk_layout.addWidget(risk_splitter)
        tab_widget.addTab(risk_page, "风险控制")
        
        # 6. 创建系统进化页面
        evolution_page = QWidget()
        evolution_layout = QVBoxLayout(evolution_page)
        
        # 系统进化标题
        evolution_title = QLabel("系统进化与优化", self)
        evolution_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(14)
        evolution_title.setFont(font)
        evolution_layout.addWidget(evolution_title)
        
        # 创建系统进化分割器
        evolution_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧进化参数面板
        evolution_params_frame = QFrame()
        evolution_params_layout = QVBoxLayout(evolution_params_frame)
        
        params_group = QGroupBox("进化参数设置")
        params_inner = QVBoxLayout()
        
        # 参数表格
        self.evolution_params_table = QTableWidget(7, 2)
        self.evolution_params_table.setHorizontalHeaderLabels(["参数名称", "当前值"])
        self.evolution_params_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.evolution_params_table.setAlternatingRowColors(True)
        
        # 设置列宽
        self.evolution_params_table.setColumnWidth(0, 200)
        self.evolution_params_table.setColumnWidth(1, 100)
        
        # 填充进化参数
        params = [
            ("进化周期", "每周日 23:00"),
            ("种群大小", "200"),
            ("变异率", "0.15"),
            ("交叉率", "0.85"),
            ("适应度函数", "夏普比率"),
            ("最大迭代次数", "50"),
            ("收敛阈值", "0.001")
        ]
        
        for row, (name, value) in enumerate(params):
            self.evolution_params_table.setItem(row, 0, QTableWidgetItem(name))
            self.evolution_params_table.setItem(row, 1, QTableWidgetItem(value))
            
        params_inner.addWidget(self.evolution_params_table)
        
        # 控制按钮
        params_buttons = QHBoxLayout()
        self.start_evolution_btn = QPushButton("开始进化")
        self.stop_evolution_btn = QPushButton("停止进化")
        self.start_evolution_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.stop_evolution_btn.setStyleSheet("background-color: #F44336; color: white;")
        params_buttons.addWidget(self.start_evolution_btn)
        params_buttons.addWidget(self.stop_evolution_btn)
        
        params_inner.addLayout(params_buttons)
        params_group.setLayout(params_inner)
        evolution_params_layout.addWidget(params_group)
        
        evolution_splitter.addWidget(evolution_params_frame)
        
        # 右侧进化历史和结果面板
        evolution_history_frame = QFrame()
        evolution_history_layout = QVBoxLayout(evolution_history_frame)
        
        history_group = QGroupBox("进化历史与性能提升")
        history_inner = QVBoxLayout()
        
        # 进化历史文本
        self.evolution_history_text = QTextEdit()
        self.evolution_history_text.setReadOnly(True)
        self.evolution_history_text.setHtml("""
            <h3>系统进化历史记录</h3>
            <table border="1" cellspacing="0" cellpadding="5" style="width:100%">
                <tr style="background-color:#f0f0f0">
                    <th>日期</th>
                    <th>迭代次数</th>
                    <th>性能提升</th>
                </tr>
                <tr>
                    <td>2025-03-30</td>
                    <td>42</td>
                    <td style="color:green">+3.25%</td>
                </tr>
                <tr>
                    <td>2025-03-23</td>
                    <td>38</td>
                    <td style="color:green">+2.87%</td>
                </tr>
                <tr>
                    <td>2025-03-16</td>
                    <td>45</td>
                    <td style="color:green">+1.94%</td>
                </tr>
                <tr>
                    <td>2025-03-09</td>
                    <td>30</td>
                    <td style="color:red">-0.42%</td>
                </tr>
                <tr>
                    <td>2025-03-02</td>
                    <td>48</td>
                    <td style="color:green">+4.18%</td>
                </tr>
            </table>
            
            <h3>当前最优策略组合</h3>
            <p>
                <b>组合名称:</b> AlphaPlus_v3.7<br>
                <b>参数组合:</b> {momentum: 0.72, mean_reversion: 0.45, trend_following: 0.68}<br>
                <b>风险系数:</b> 0.58 (中等)<br>
                <b>回测夏普比率:</b> 2.34<br>
                <b>最大回撤:</b> 5.28%<br>
                <b>年化收益率:</b> 28.7%
            </p>
            
            <h3>系统自动发现的市场规律</h3>
            <ul>
                <li>周四尾盘通常呈现上涨概率较高 (置信度: 78.4%)</li>
                <li>低波动小盘股在高市场不确定性下表现更佳 (置信度: 82.1%)</li>
                <li>金融板块与能源板块呈现负相关关系 (置信度: 75.6%)</li>
            </ul>
        """)
        
        history_inner.addWidget(self.evolution_history_text)
        history_group.setLayout(history_inner)
        evolution_history_layout.addWidget(history_group)
        
        evolution_splitter.addWidget(evolution_history_frame)
        
        # 设置分割比例
        evolution_splitter.setSizes([300, 500])
        
        evolution_layout.addWidget(evolution_splitter)
        tab_widget.addTab(evolution_page, "系统进化")
        
        # 将标签页控件添加到主布局
        main_layout.addWidget(tab_widget)
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("系统就绪")
    
    def initialize_market_state(self, data_dir='data/market', universe='test'):
        """初始化市场状态"""
        try:
            self.market_state = MarketState(mode='paper', data_dir=data_dir, universe=universe)
            logger.info(f"市场状态初始化成功，加载了 {len(self.market_state.symbols)} 只股票")
            self.status_bar.showMessage(f"市场数据已加载，共 {len(self.market_state.symbols)} 只股票")
            
            # 初始更新
            self.update_market_data()
        except Exception as e:
            logger.error(f"市场状态初始化失败: {e}")
            self.status_bar.showMessage("市场数据加载失败")
            QMessageBox.critical(self, "错误", f"市场状态初始化失败: {e}")
    
    def update_market_data(self):
        """更新市场数据和所有面板"""
        try:
            # 检查市场是否已初始化
            if self.market_state is None:
                return

            # 更新市场状态
            self.market_state.update()
            
            # 获取市场摘要信息
            is_open = self.market_state.is_market_open()
            market_summary = {
                "is_open": is_open,
                "total_stocks": 300,  # 模拟数据
                "up_stocks": 165,     # 模拟数据
                "down_stocks": 125,   # 模拟数据
                "average_change": 0.78  # 模拟数据
            }
            
            # 更新市场状态面板
            self.market_panel.update_market_data(market_summary)
            
            # 获取可交易的股票列表
            tradable_symbols = []
            try:
                tradable_symbols = self.market_state.get_tradable_symbols()
            except Exception as e:
                logger.error(f"获取可交易股票失败: {e}")
                
            # 构建股票数据列表
            stock_data = []
            for symbol in tradable_symbols[:40]:  # 限制显示数量
                try:
                    # 获取股票价格和变化百分比
                    price = self.market_state.get_latest_price(symbol)
                    change_pct = random.uniform(-5.0, 5.0)  # 模拟数据
                    
                    # 获取股票名称
                    name = f"股票{symbol.split('.')[0]}"  # 模拟数据
                    
                    stock_data.append({
                        "symbol": symbol,
                        "name": name,
                        "price": price,
                        "change_pct": change_pct
                    })
                except Exception as e:
                    logger.error(f"获取股票 {symbol}.SH 数据失败: {e}")
            
            # 更新股票列表
            self.stock_list.update_stock_list(stock_data)
            
            # 更新交易面板
            self.trade_panel.update_stock_list(stock_data)
            
            # 模拟账户信息
            account_info = {
                "total_assets": 1025800.50,
                "available_cash": 425680.75,
                "position_value": 600119.75,
                "daily_pnl": 15680.25
            }
            
            # 模拟持仓信息
            positions = []
            for i in range(min(8, len(stock_data))):
                # 随机选择一些股票作为持仓
                stock = stock_data[i]
                volume = random.randint(100, 5000) // 100 * 100  # 随机持仓量，按手取整
                cost_price = stock["price"] * (1 + random.uniform(-0.1, 0.1))  # 模拟成本价
                
                positions.append({
                    "symbol": stock["symbol"],
                    "name": stock["name"],
                    "volume": volume,
                    "cost_price": cost_price,
                    "current_price": stock["price"]
                })
                
            # 更新持仓面板
            self.portfolio_panel.update_portfolio(account_info, positions)
            
            # 更新智能分析面板
            self.update_intelligent_panels(market_summary, stock_data, positions)
            
            # 更新状态栏
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_message = f"最后更新: {current_time} | 市场状态: {'开盘' if is_open else '休市'}"
            self.status_bar.showMessage(status_message)
            
        except Exception as e:
            logger.error(f"更新市场数据时发生错误: {e}")
            self.status_bar.showMessage(f"更新失败: {str(e)}")

    def update_intelligent_panels(self, market_summary, stock_data, positions):
        """更新所有智能面板的数据"""
        try:
            # 1. 更新市场分析面板
            # 模拟行业板块数据
            sectors = [
                {"name": "金融", "change_pct": -1.25, "leading_stocks": "招商银行, 工商银行", "leading_change": -0.85},
                {"name": "医药", "change_pct": 2.35, "leading_stocks": "恒瑞医药, 迈瑞医疗", "leading_change": 3.25},
                {"name": "科技", "change_pct": 3.82, "leading_stocks": "华为, 腾讯", "leading_change": 4.56},
                {"name": "新能源", "change_pct": 5.17, "leading_stocks": "宁德时代, 比亚迪", "leading_change": 6.23},
                {"name": "消费", "change_pct": 0.65, "leading_stocks": "贵州茅台, 五粮液", "leading_change": 1.12},
                {"name": "地产", "change_pct": -2.43, "leading_stocks": "万科A, 保利发展", "leading_change": -1.85}
            ]
            
            # 模拟市场分析文本
            market_analysis = f"""
                <h3>市场概况分析</h3>
                <p>当前市场整体呈现{
                    "震荡上行" if market_summary["average_change"] > 0.5 else 
                    "震荡下行" if market_summary["average_change"] < -0.5 else "震荡整理"
                }态势，上涨个股{market_summary["up_stocks"]}只，下跌个股{market_summary["down_stocks"]}只。</p>
                
                <h3>市场热点板块</h3>
                <p>当前市场热点集中在<span style="color:red;">新能源</span>和<span style="color:red;">科技</span>板块，
                呈现出明显的轮动特征。新能源板块受政策利好刺激，整体上涨{sectors[3]["change_pct"]}%；
                科技板块在半导体和AI应用带动下，整体涨幅达{sectors[2]["change_pct"]}%。</p>
                
                <h3>投资机会分析</h3>
                <p>短期投资机会：关注新能源板块龙头企业宁德时代(300750.SZ)，该股突破前期高点，成交量明显放大。</p>
                <p>中期投资机会：科技板块中芯片设计公司有望受益于国产替代进程加速，建议关注华为鲲鹏生态链相关企业。</p>
                <p>长期投资价值：医药创新药企业在经历调整后估值趋于合理，未来发展空间广阔。</p>
                
                <h3>风险提示</h3>
                <p>地产板块持续走弱，短期内仍有下行压力；金融板块受利率影响，表现相对低迷。</p>
            """
            
            if hasattr(self, 'analysis_panel'):
                self.analysis_panel.update_analysis(sectors, market_analysis)
                
            # 2. 更新算法交易面板状态
            # 模拟算法监测到的市场信号
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            
            # 模拟市场信号
            if random.random() > 0.7:  # 30%概率生成新信号
                # 确保stock_data有足够的数据用于取样
                sample_size = min(3, len(stock_data))
                if sample_size > 0:
                    signal_stocks = random.sample([s["symbol"] for s in stock_data], sample_size)
                    signal_types = ["买入", "卖出", "持有"]
                    signal_strengths = ["强", "中", "弱"]
                    
                    signals = []
                    for i in range(sample_size):
                        stock = signal_stocks[i]
                        signal_type = random.choice(signal_types)
                        signal_strength = random.choice(signal_strengths)
                        signals.append(f"{stock}: {signal_type}信号({signal_strength})")
                    
                    # 确保signals数组有足够的元素
                    while len(signals) < 3:
                        signals.append("无信号")
                    
                    signal_text = f"""
                        <h3>最新交易信号 ({current_time})</h3>
                        <p style="color:green;">● {signals[0]}</p>
                        <p style="color:red;">● {signals[1]}</p>
                        <p style="color:blue;">● {signals[2]}</p>
                        
                        <h3>智能算法运行状态</h3>
                        <p><b>量化动量跟踪:</b> 正在监控市场动量变化</p>
                        <p style="color:green;">- 已捕捉到新能源板块强势动量信号</p>
                        <p style="color:green;">- 半导体行业呈现加速上涨态势</p>
                        <p style="color:red;">- 传统金融板块动量减弱</p>
                        
                        <h3>AI智能选股结果 (Top 5)</h3>
                        <ol>
                            <li>600519.SH - 贵州茅台 (置信度: 89.5%)</li>
                            <li>300750.SZ - 宁德时代 (置信度: 87.2%)</li>
                            <li>601899.SH - 紫金矿业 (置信度: 85.4%)</li>
                            <li>000858.SZ - 五粮液 (置信度: 82.7%)</li>
                            <li>600036.SH - 招商银行 (置信度: 81.3%)</li>
                        </ol>
                        
                        <h3>近期系统进化</h3>
                        <p>系统在过去7天内完成了2次策略参数优化，均值回报率提升2.7%</p>
                        <p>市场环境识别准确率: 92.5% (+1.3%)</p>
                    """
                    
                    if hasattr(self, 'algo_status_text'):
                        self.algo_status_text.setHtml(signal_text)
                
            # 3. 更新风险控制面板
            # 模拟风险变化情况
            value_at_risk = 2.34 + random.uniform(-0.2, 0.2)
            max_drawdown = 7.85 + random.uniform(-0.3, 0.3)
            volatility = 12.36 + random.uniform(-0.5, 0.5)
            
            # 更新风险指标
            if hasattr(self, 'risk_rules_table'):
                # 更新特定单元格内容
                for i, metric in enumerate([("最大回撤", f"{max_drawdown:.2f}%"),
                                         ("波动率", f"{volatility:.2f}%"),
                                         ("VaR (95%)", f"{value_at_risk:.2f}%")]):
                    row = i // 2
                    col = 0 if i % 2 == 0 else 2
                    
                    # 创建标签和值，但不更新UI (这里仅示例，实际实现需要更具体的UI更新逻辑)
                    label, value = metric
                    
                    # 根据值设置颜色
                    color = "green"
                    if label == "最大回撤" or label == "波动率" or label == "VaR (95%)":
                        metric_value = float(value.rstrip('%'))
                        if metric_value > 10:
                            color = "red"
                        elif metric_value > 5:
                            color = "orange"
                        else:
                            color = "green"
                    
                    # 这里演示更新风险指标，实际实现需要对应具体的UI组件
                    # 例如: self.risk_metrics[i].setText(value)
                    # self.risk_metrics[i].setStyleSheet(f"color: {color};")
            
            # 4. 更新系统进化面板
            # 在实际应用中，这里会收集系统进化的历史数据和当前进化状态
            # 本例中简单模拟一些数据更新
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            iteration = random.randint(30, 50)
            performance_change = random.uniform(-1.0, 5.0)
            
            # 设置参数值
            momentum_val = 0.72
            mean_reversion_val = 0.45
            trend_following_val = 0.68
            
            evolution_history = f"""
                <h3>系统进化历史记录</h3>
                <table border="1" cellspacing="0" cellpadding="5" style="width:100%">
                    <tr style="background-color:#f0f0f0">
                        <th>日期</th>
                        <th>迭代次数</th>
                        <th>性能提升</th>
                    </tr>
                    <tr>
                        <td>{current_date}</td>
                        <td>{iteration}</td>
                        <td style="color:{'green' if performance_change > 0 else 'red'}">
                            {'+' if performance_change > 0 else ''}{performance_change:.2f}%
                        </td>
                    </tr>
                    <tr>
                        <td>2025-03-30</td>
                        <td>42</td>
                        <td style="color:green">+3.25%</td>
                    </tr>
                    <tr>
                        <td>2025-03-23</td>
                        <td>38</td>
                        <td style="color:green">+2.87%</td>
                    </tr>
                    <tr>
                        <td>2025-03-16</td>
                        <td>45</td>
                        <td style="color:green">+1.94%</td>
                    </tr>
                    <tr>
                        <td>2025-03-09</td>
                        <td>30</td>
                        <td style="color:red">-0.42%</td>
                    </tr>
                </table>
                
                <h3>当前最优策略组合</h3>
                <p>
                    <b>组合名称:</b> AlphaPlus_v3.7<br>
                    <b>参数组合:</b> {{momentum: {momentum_val}, mean_reversion: {mean_reversion_val}, trend_following: {trend_following_val}}}<br>
                    <b>风险系数:</b> 0.58 (中等)<br>
                    <b>回测夏普比率:</b> 2.34<br>
                    <b>最大回撤:</b> 5.28%<br>
                    <b>年化收益率:</b> 28.7%
                </p>
                
                <h3>系统自动发现的市场规律</h3>
                <ul>
                    <li>周四尾盘通常呈现上涨概率较高 (置信度: 78.4%)</li>
                    <li>低波动小盘股在高市场不确定性下表现更佳 (置信度: 82.1%)</li>
                    <li>金融板块与能源板块呈现负相关关系 (置信度: 75.6%)</li>
                </ul>
            """
            
            if hasattr(self, 'evolution_history_text'):
                self.evolution_history_text.setHtml(evolution_history)
                
        except Exception as e:
            logger.error(f"更新智能面板时发生错误: {e}")


def start_gui():
    """启动PyQt6 GUI界面"""
    try:
        # 创建应用
        app = QApplication(sys.argv)
        
        # 设置应用样式
        app.setStyle("Fusion")
        
        # 创建主窗口
        window = SetoMainWindow()
        
        # 初始化市场状态
        try:
            # 创建市场状态对象
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'market')
            window.market_state = MarketState(mode="paper", data_dir=data_dir, universe="sample")
            
            # 初始化市场数据
            window.market_state.initialize_data()
            
            # 记录日志
            logger.info(f"市场状态已初始化 (模式: paper, 数据目录: {data_dir})")
        except Exception as e:
            logger.error(f"初始化市场状态时发生错误: {e}")
            window.status_bar.showMessage(f"市场状态初始化失败: {str(e)}")
            
        # 显示窗口
        window.show()
        
        # 开始事件循环
        return app.exec()
    except Exception as e:
        logger.error(f"启动GUI时发生错误: {e}")
        raise


if __name__ == "__main__":
    start_gui() 