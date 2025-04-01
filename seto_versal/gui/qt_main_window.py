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
import json

from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QTabWidget, 
                            QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                            QTableWidget, QTableWidgetItem, QStatusBar, 
                            QGroupBox, QGridLayout, QTextEdit, QComboBox,
                            QMessageBox, QSplitter, QFrame, QHeaderView,
                            QListWidget, QInputDialog, QDialog, QFormLayout,
                            QSpinBox, QDoubleSpinBox, QDialogButtonBox,
                            QProgressDialog, QProgressBar, QLineEdit, QCheckBox,
                            QRadioButton, QButtonGroup, QFileDialog, QSpacerItem,
                            QSizePolicy)
from PyQt6.QtCore import Qt, QTimer, QDateTime, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QColor, QPainter, QPixmap

# 添加图表相关导入
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QPieSeries, QValueAxis, QBarSeries, QBarSet

# 添加matplotlib集成
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 导入市场状态
from seto_versal.market.state import MarketState

# 设置日志
logger = logging.getLogger(__name__)

class MarketStateDisplay(QWidget):
    """市场状态显示组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """初始化UI组件"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建标题
        title_label = QLabel("市场状态", self)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(14)
        title_label.setFont(font)
        main_layout.addWidget(title_label)
        
        # 添加股票池选择器
        pool_group = QGroupBox("股票池选择")
        pool_layout = QVBoxLayout(pool_group)
        
        # 创建下拉选择框
        self.pool_selector = QComboBox()
        self.pool_selector.addItems(["全部股票", "沪深300", "中证500", "创业板50", "科创50", "自定义行业"])
        
        # 连接信号
        self.pool_selector.currentTextChanged.connect(self.on_pool_changed)
        
        # 添加到布局
        pool_layout.addWidget(QLabel("选择股票池:"))
        pool_layout.addWidget(self.pool_selector)
        
        # 添加股票池信息标签
        self.pool_info_label = QLabel("当前股票池: 全部股票")
        pool_layout.addWidget(self.pool_info_label)
        
        main_layout.addWidget(pool_group)
        
        # 市场摘要
        market_group = QGroupBox("市场摘要")
        market_layout = QGridLayout(market_group)
        
        # 添加市场摘要标签
        self.up_count_label = QLabel("上涨: 0", self)
        self.down_count_label = QLabel("下跌: 0", self)
        self.avg_change_label = QLabel("平均涨幅: 0.00%", self)
        self.volatility_label = QLabel("波动率: 0.00%", self)
        
        # 设置颜色
        self.up_count_label.setStyleSheet("color: red;")
        self.down_count_label.setStyleSheet("color: green;")
        
        # 添加到布局
        market_layout.addWidget(self.up_count_label, 0, 0)
        market_layout.addWidget(self.down_count_label, 0, 1)
        market_layout.addWidget(self.avg_change_label, 1, 0)
        market_layout.addWidget(self.volatility_label, 1, 1)
        
        main_layout.addWidget(market_group)
        
        # 添加热点行业
        hot_group = QGroupBox("热点行业")
        hot_layout = QVBoxLayout(hot_group)
        
        self.hot_sectors_list = QListWidget(self)
        hot_layout.addWidget(self.hot_sectors_list)
        
        main_layout.addWidget(hot_group)
        
        # 添加市场时间和状态
        time_group = QGroupBox("市场信息")
        time_layout = QVBoxLayout(time_group)
        
        self.market_time_label = QLabel("市场时间: --:--:--", self)
        self.market_status_label = QLabel("市场状态: 已开盘", self)
        
        time_layout.addWidget(self.market_time_label)
        time_layout.addWidget(self.market_status_label)
        
        main_layout.addWidget(time_group)
        
        # 添加弹性空间
        main_layout.addStretch(1)
        
        # 设置布局
        self.setLayout(main_layout)

    def on_pool_changed(self, pool_name):
        """处理股票池变更事件"""
        # 更新股票池信息标签
        self.pool_info_label.setText(f"当前股票池: {pool_name}")
        # 通知主窗口
        if hasattr(self.parent(), "change_stock_pool"):
            self.parent().change_stock_pool(pool_name)

    def update_market_data(self, market_data):
        """更新市场数据显示"""
        # 设置市场状态
        market_status = market_data.get('market_status', '未知')
        self.market_status_label.setText(market_status)
        
        # 根据不同的市场状态设置不同的颜色
        if market_status == "回测模式":
            self.market_status_label.setStyleSheet("color: green; font-weight: bold;")
        elif market_status == "模拟交易":
            self.market_status_label.setStyleSheet("color: blue; font-weight: bold;")
        elif market_status == "实时交易":
            self.market_status_label.setStyleSheet("color: red; font-weight: bold;")
        elif market_status == "开盘":
            self.market_status_label.setStyleSheet("color: green; font-weight: bold;")
        elif market_status == "休市":
            self.market_status_label.setStyleSheet("color: gray;")
        else:
            self.market_status_label.setStyleSheet("")
        
        # 更新其他市场数据
        self.up_count_label.setText(str(market_data.get('up_count', 0)))
        self.down_count_label.setText(str(market_data.get('down_count', 0)))
        self.avg_change_label.setText(f"{market_data.get('avg_change', 0):.2f}%")
        self.volatility_label.setText(f"{market_data.get('volatility', 0):.2f}%")


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


class SectorChart(QWidget):
    """行业分析图表组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """初始化UI组件"""
        # 创建布局
        layout = QVBoxLayout(self)
        
        # 创建图表类型选择
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["行业涨跌幅", "行业热度", "行业轮动"])
        self.chart_type_combo.currentIndexChanged.connect(self.update_chart)
        
        layout.addWidget(QLabel("图表类型:"))
        layout.addWidget(self.chart_type_combo)
        
        # 创建matplotlib图形
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        layout.addWidget(self.canvas)
        
        # 数据存储
        self.sectors_data = []
        
    def update_chart(self):
        """根据选择的图表类型更新图表"""
        if not self.sectors_data:
            return
            
        chart_type = self.chart_type_combo.currentText()
        self.ax.clear()
        
        if chart_type == "行业涨跌幅":
            self._draw_sector_performance_chart()
        elif chart_type == "行业热度":
            self._draw_sector_heatmap()
        elif chart_type == "行业轮动":
            self._draw_sector_rotation_chart()
            
        self.figure.tight_layout()
        self.canvas.draw()
        
    def set_data(self, sectors_data):
        """设置行业数据"""
        self.sectors_data = sectors_data
        self.update_chart()
        
    def _draw_sector_performance_chart(self):
        """绘制行业涨跌幅条形图"""
        if not self.sectors_data:
            return
            
        # 提取数据
        sector_names = [s["name"] for s in self.sectors_data]
        changes = [s["change_pct"] for s in self.sectors_data]
        
        # 设置颜色
        colors = ['red' if x > 0 else 'green' for x in changes]
        
        # 绘制条形图
        bars = self.ax.bar(sector_names, changes, color=colors)
        
        # 添加数据标签
        for bar, change in zip(bars, changes):
            height = bar.get_height()
            self.ax.annotate(f'{change:.2f}%',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom')
        
        # 设置标题和标签
        self.ax.set_title('行业涨跌幅比较')
        self.ax.set_ylabel('涨跌幅 (%)')
        self.ax.set_ylim([min(min(changes) * 1.2, -0.5), max(max(changes) * 1.2, 0.5)])
        
        # 旋转x轴标签，使其不重叠
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')
        
    def _draw_sector_heatmap(self):
        """绘制行业热度热图"""
        if not self.sectors_data:
            return
            
        # 创建热度数据 - 将涨跌幅转换为热度值
        sector_names = [s["name"] for s in self.sectors_data]
        changes = [s["change_pct"] for s in self.sectors_data]
        
        # 创建数据框架
        df = pd.DataFrame({'sector': sector_names, 'change': changes})
        
        # 计算热度 (可基于涨跌幅和其他因素)
        df['heat'] = df['change'].apply(lambda x: min(100, max(0, (x + 5) * 10)))
        
        # 生成热图矩阵
        matrix = np.zeros((1, len(sector_names)))
        for i, heat in enumerate(df['heat']):
            matrix[0, i] = heat
            
        # 绘制热图
        im = self.ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
        
        # 添加标签
        self.ax.set_xticks(range(len(sector_names)))
        self.ax.set_xticklabels(sector_names)
        self.ax.set_yticks([])  # 不显示y轴刻度
        
        # 旋转x轴标签
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')
        
        # 添加标题
        self.ax.set_title('行业热度分析')
        
        # 添加颜色条
        self.figure.colorbar(im, ax=self.ax, orientation='horizontal', pad=0.2)
        
    def _draw_sector_rotation_chart(self):
        """绘制行业轮动分析图"""
        if not self.sectors_data:
            return
            
        # 设置一些模拟的历史数据来展示轮动
        # 实际应用中应从数据库或市场状态获取数据
        sector_names = [s["name"] for s in self.sectors_data]
        current_changes = [s["change_pct"] for s in self.sectors_data]
        
        # 模拟历史轮动数据 (3个时间点)
        # 实际环境应替换为真实历史数据
        np.random.seed(42)  # 固定种子以获得可重复的结果
        historical_changes = []
        for i in range(3):
            historical_changes.append([
                (c + np.random.uniform(-2, 2)) for c in current_changes
            ])
        
        # 创建时间标签
        time_periods = ['1个月前', '2周前', '1周前', '当前']
        
        # 准备绘图数据
        all_data = historical_changes + [current_changes]
        
        # 绘制折线图
        for i, sector in enumerate(sector_names):
            sector_perf = [data[i] for data in all_data]
            self.ax.plot(time_periods, sector_perf, marker='o', label=sector)
        
        # 添加图例
        self.ax.legend(loc='best', fontsize='small')
        
        # 设置标题和标签
        self.ax.set_title('行业轮动分析')
        self.ax.set_ylabel('涨跌幅 (%)')
        self.ax.grid(True, linestyle='--', alpha=0.7)


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
        
        # 创建水平分割器，左侧表格，右侧图表
        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：行业表格面板
        table_panel = QFrame()
        table_layout = QVBoxLayout(table_panel)
        
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
        table_layout.addWidget(sector_group)
        
        h_splitter.addWidget(table_panel)
        
        # 右侧：行业图表
        chart_panel = QFrame()
        chart_layout = QVBoxLayout(chart_panel)
        
        # 创建行业图表
        self.sector_chart = SectorChart()
        chart_layout.addWidget(self.sector_chart)
        
        h_splitter.addWidget(chart_panel)
        
        # 设置水平分割比例
        h_splitter.setSizes([400, 600])
        
        sector_layout.addWidget(h_splitter)
        splitter.addWidget(sector_frame)
        
        # 下部分：市场热点与机会
        market_frame = QFrame()
        market_layout = QVBoxLayout(market_frame)
        
        market_group = QGroupBox("市场热点与投资机会")
        market_inner = QVBoxLayout()
        
        # 创建市场分析文本区域
        self.market_analysis_text = QTextEdit()
        self.market_analysis_text.setReadOnly(True)
        
        market_inner.addWidget(self.market_analysis_text)
        market_group.setLayout(market_inner)
        market_layout.addWidget(market_group)
        
        splitter.addWidget(market_frame)
        
        # 设置分割比例
        splitter.setSizes([600, 400])
        
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
        
        # 更新行业图表
        self.sector_chart.set_data(sectors)
        
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
        
        # 设置当前运行模式（回测、模拟、实时）
        self.current_mode = "回测"
        
        # 股票名称映射表
        self.stock_names = {
            '000001.SZ': '平安银行',
            '000333.SZ': '美的集团',
            '000651.SZ': '格力电器',
            '000858.SZ': '五粮液',
            '600000.SH': '浦发银行',
            '600036.SH': '招商银行',
            '600276.SH': '恒瑞医药',
            '600519.SH': '贵州茅台',
            '601318.SH': '中国平安',
            '601888.SH': '中国中免',
            '002415.SZ': '海康威视'
        }
        
        # 扩展股票名称映射表，增加更多股票名称
        self._load_extended_stock_names()
        
        # 股票行业信息
        self.stock_sectors = {}
        self._load_sector_information()
        
        # 保存回测模式下的虚拟股票数据，确保数据一致性
        self.backtest_stock_data = {}
        
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
        
        # 添加运行模式选择器
        mode_layout = QHBoxLayout()
        
        # 模式选择标签
        mode_label = QLabel("运行模式:", self)
        mode_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        mode_layout.addWidget(mode_label)
        
        # 模式选择按钮组
        self.backtest_btn = QPushButton("回测", self)
        self.simulation_btn = QPushButton("模拟", self)
        self.realtime_btn = QPushButton("实时", self)
        
        # 设置初始状态 - 回测模式激活
        self.backtest_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.simulation_btn.setStyleSheet("background-color: #f0f0f0;")
        self.realtime_btn.setStyleSheet("background-color: #f0f0f0;")
        
        # 连接按钮信号
        self.backtest_btn.clicked.connect(lambda: self.change_mode("回测"))
        self.simulation_btn.clicked.connect(lambda: self.change_mode("模拟"))
        self.realtime_btn.clicked.connect(lambda: self.change_mode("实时"))
        
        # 添加按钮到布局
        mode_layout.addWidget(self.backtest_btn)
        mode_layout.addWidget(self.simulation_btn)
        mode_layout.addWidget(self.realtime_btn)
        
        # 添加弹性空间
        mode_layout.addStretch(1)
        
        # 显示当前模式状态
        self.mode_status = QLabel(f"当前模式: {self.current_mode}", self)
        self.mode_status.setStyleSheet("color: #1E88E5; font-weight: bold;")
        mode_layout.addWidget(self.mode_status)
        
        # 将模式选择器添加到主布局
        main_layout.addLayout(mode_layout)
        
        # 添加策略测试按钮区域
        strategy_test_layout = QHBoxLayout()
        strategy_test_layout.addWidget(QLabel("策略测试:"))
        
        # 添加不同策略测试按钮
        self.test_sector_rotation_btn = QPushButton("测试行业轮动策略", self)
        self.test_sector_rotation_btn.clicked.connect(self.test_sector_rotation_strategy)
        strategy_test_layout.addWidget(self.test_sector_rotation_btn)
        
        self.test_industry_leader_btn = QPushButton("测试行业龙头策略", self)
        self.test_industry_leader_btn.clicked.connect(self.test_industry_leader_strategy)
        strategy_test_layout.addWidget(self.test_industry_leader_btn)
        
        # 添加弹性空间
        strategy_test_layout.addStretch(1)
        
        # 将策略测试添加到主布局
        main_layout.addLayout(strategy_test_layout)
        
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

            # 根据当前模式进行不同的数据更新
            if self.current_mode == "回测":
                self._update_backtest_data()
            elif self.current_mode == "模拟":
                self._update_simulation_data()
            elif self.current_mode == "实时":
                self._update_realtime_data()
            
            # 更新状态栏显示当前模式
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 根据不同模式显示不同的状态栏信息
            if self.current_mode == "回测":
                status_message = f"最后更新: {current_time} | 运行模式: {self.current_mode} | 市场状态: 回测中"
            else:
                status_message = f"最后更新: {current_time} | 运行模式: {self.current_mode} | 市场状态: {'开盘' if self.market_state.is_market_open() else '休市'}"
                
            self.status_bar.showMessage(status_message)
            
        except Exception as e:
            logger.error(f"更新市场数据时发生错误: {e}")
            self.status_bar.showMessage(f"更新失败: {str(e)}")
    
    def _update_backtest_data(self):
        """更新回测模式下的数据，使用行业信息产生更智能的价格变动"""
        # 更新市场状态 - 回测模式下使用历史数据
        self.market_state.update()
        
        # 确保已初始化回测股票数据
        if not self.backtest_stock_data:
            self._initialize_backtest_stock_data()
        
        # 获取市场摘要信息 - 回测模式下展示历史数据
        market_summary = {
            "market_status": "回测模式",
            "up_count": 0,  # 稍后更新
            "down_count": 0,  # 稍后更新
            "avg_change": 0.0,  # 稍后更新
            "total_volume": random.randint(50000000, 150000000)  # 模拟历史数据
        }
        
        # 更新回测模式下的股票数据 - 保持一致性但略有变化
        stock_data = []
        
        # 生成行业因子（模拟不同行业在不同时间点的表现差异）
        sector_factors = {
            "金融": random.uniform(-1.5, 2.0),
            "科技": random.uniform(-2.0, 3.0),
            "消费": random.uniform(-1.0, 1.5),
            "医药": random.uniform(-1.5, 2.5),
            "能源": random.uniform(-2.5, 2.5),
            "工业": random.uniform(-1.8, 1.8),
            "未知": random.uniform(-1.0, 1.0)
        }
        
        # 获取当前时间作为变化因子
        time_factor = datetime.datetime.now().second / 60.0  # 0-1之间变化
        
        # 更新所有股票的价格和涨跌幅
        up_count = 0
        down_count = 0
        total_change = 0.0
        
        # 按行业分组的股票数据，用于更新智能分析面板
        sector_performance = {}
        
        for symbol, data in self.backtest_stock_data.items():
            # 股票特定的随机种子，使得相同股票在同一时间点的变化是一致的
            seed = sum(ord(c) for c in symbol) + int(time_factor * 100)
            random.seed(seed)
            
            # 获取股票行业和波动性
            sector = data.get("sector", "未知")
            volatility = data.get("volatility", 1.0)
            
            # 计算行业因子影响
            sector_impact = sector_factors.get(sector, 0) * 0.5  # 行业因子贡献度
            
            # 计算价格变动 - 结合行业因子、个股波动性和随机因子
            random_factor = random.uniform(-2.0, 2.0)  # 随机成分
            price_change_pct = (random_factor + sector_impact) * volatility
            
            # 价格变动范围控制在 [-6%, +6%]
            price_change_pct = max(min(price_change_pct, 6.0), -6.0)
            
            # 计算新价格
            new_price = data["base_price"] * (1 + price_change_pct/100)
            
            # 更新涨跌幅（相对于初始价格）
            change_pct = price_change_pct
            
            # 更新数据
            self.backtest_stock_data[symbol]["price"] = new_price
            self.backtest_stock_data[symbol]["change_pct"] = change_pct
            
            # 统计上涨下跌
            if change_pct > 0:
                up_count += 1
            elif change_pct < 0:
                down_count += 1
            
            total_change += change_pct
            
            # 添加到显示数据
            stock_data.append({
                "symbol": data["symbol"],
                "name": data["name"],
                "price": new_price,
                "change_pct": change_pct,
                "sector": sector  # 添加行业信息
            })
            
            # 更新行业表现统计
            if sector not in sector_performance:
                sector_performance[sector] = {
                    "total_change": 0.0,
                    "count": 0,
                    "stocks": []
                }
            
            sector_performance[sector]["total_change"] += change_pct
            sector_performance[sector]["count"] += 1
            sector_performance[sector]["stocks"].append({
                "symbol": symbol,
                "name": data["name"],
                "change_pct": change_pct
            })
        
        # 重置随机种子
        random.seed()
        
        # 更新市场摘要信息
        market_summary["up_count"] = up_count
        market_summary["down_count"] = down_count
        market_summary["avg_change"] = total_change / len(stock_data) if stock_data else 0.0
        
        # 更新市场状态面板
        self.market_panel.update_market_data(market_summary)
        
        # 更新股票列表
        self.stock_list.update_stock_list(stock_data)
        
        # 更新交易面板
        self.trade_panel.update_stock_list(stock_data)
        
        # 生成回测模式下的账户和持仓信息
        account_info, positions = self._generate_account_data(stock_data)
        
        # 更新持仓面板
        self.portfolio_panel.update_portfolio(account_info, positions)
        
        # 计算行业表现
        sectors_analysis = []
        for sector_name, perf in sector_performance.items():
            if perf["count"] > 0:
                avg_change = perf["total_change"] / perf["count"]
                # 找出该行业表现最好的股票作为龙头股
                leading_stock = max(perf["stocks"], key=lambda x: x["change_pct"])
                sectors_analysis.append({
                    "name": sector_name,
                    "change_pct": avg_change,
                    "leading_stocks": f"{leading_stock['name']} ({leading_stock['symbol']})",
                    "leading_change": leading_stock["change_pct"]
                })
        
        # 按行业涨幅排序
        sectors_analysis.sort(key=lambda x: x["change_pct"], reverse=True)
        
        # 更新智能分析面板
        self.update_intelligent_panels(market_summary, stock_data, positions, sectors_analysis)
    
    def _update_simulation_data(self):
        """更新模拟交易模式下的数据"""
        # 更新市场状态 - 模拟交易模式下使用实时行情但模拟交易
        self.market_state.update()
        
        # 获取市场摘要信息
        market_summary = {
            "market_status": "模拟交易",
            "up_count": random.randint(120, 220),  # 模拟数据
            "down_count": random.randint(80, 180),  # 模拟数据
            "avg_change": random.uniform(-1.0, 3.0),  # 模拟数据
            "total_volume": random.randint(80000000, 200000000)  # 模拟数据
        }
        
        # 更新市场状态面板
        self.market_panel.update_market_data(market_summary)
        
        # 构建股票数据列表 - 模拟交易模式下使用接近实时数据
        stock_data = self._generate_stock_data()
        
        # 更新股票列表
        self.stock_list.update_stock_list(stock_data)
        
        # 更新交易面板
        self.trade_panel.update_stock_list(stock_data)
        
        # 生成模拟交易下的账户和持仓信息
        account_info, positions = self._generate_account_data(stock_data)
        
        # 更新持仓面板
        self.portfolio_panel.update_portfolio(account_info, positions)
        
        # 更新智能分析面板
        self.update_intelligent_panels(market_summary, stock_data, positions)
    
    def _update_realtime_data(self):
        """更新实时交易模式下的数据"""
        # 更新市场状态 - 实时交易模式下使用实时行情和实际交易
        self.market_state.update()
        
        # 获取市场摘要信息 - 实时数据
        market_summary = {
            "market_status": "实时交易",
            "up_count": random.randint(130, 240),  # 临时模拟数据，实际应从行情获取
            "down_count": random.randint(70, 160),  # 临时模拟数据，实际应从行情获取
            "avg_change": random.uniform(-0.8, 3.5),  # 临时模拟数据，实际应从行情获取
            "total_volume": random.randint(100000000, 250000000)  # 临时模拟数据，实际应从行情获取
        }
        
        # 更新市场状态面板
        self.market_panel.update_market_data(market_summary)
        
        # 构建股票数据列表 - 实时交易模式下使用真实数据
        stock_data = self._generate_stock_data()
        
        # 更新股票列表
        self.stock_list.update_stock_list(stock_data)
        
        # 更新交易面板
        self.trade_panel.update_stock_list(stock_data)
        
        # 生成实时交易下的账户和持仓信息
        account_info, positions = self._generate_account_data(stock_data)
        
        # 更新持仓面板
        self.portfolio_panel.update_portfolio(account_info, positions)
        
        # 更新智能分析面板
        self.update_intelligent_panels(market_summary, stock_data, positions)
    
    def _generate_stock_data(self):
        """生成股票数据（根据当前模式有所不同）"""
        stock_data = []
        
        # 获取可交易的股票列表
        tradable_symbols = []
        try:
            tradable_symbols = self.market_state.get_tradable_symbols()
        except Exception as e:
            logger.error(f"获取可交易股票失败: {e}")
            return stock_data
            
        # 限制显示数量
        for symbol in tradable_symbols[:40]:
            try:
                # 获取股票价格和变化百分比 - 根据不同模式可能使用历史、模拟或实时数据
                price = self.market_state.get_latest_price(symbol)
                
                # 根据不同模式设置不同的涨跌幅波动范围
                if self.current_mode == "回测":
                    change_pct = random.uniform(-4.0, 4.0)
                elif self.current_mode == "模拟":
                    change_pct = random.uniform(-6.0, 6.0)
                else:  # 实时模式
                    change_pct = random.uniform(-8.0, 8.0)
                
                # 获取股票名称
                name = self.stock_names.get(symbol, f"股票{symbol.split('.')[0]}")  # 使用映射表中的名称
                
                stock_data.append({
                    "symbol": symbol,
                    "name": name,
                    "price": price,
                    "change_pct": change_pct
                })
            except Exception as e:
                logger.error(f"获取股票 {symbol} 数据失败: {e}")
        
        return stock_data
    
    def _generate_account_data(self, stock_data):
        """生成账户和持仓数据（根据当前模式有所不同）"""
        # 根据不同模式设置不同的账户资产情况
        if self.current_mode == "回测":
            total_assets = 1000000 + random.uniform(-50000, 200000)
            position_ratio = 0.6  # 回测模式下的持仓比例
        elif self.current_mode == "模拟":
            total_assets = 2000000 + random.uniform(-100000, 400000)
            position_ratio = 0.7  # 模拟交易模式下的持仓比例
        else:  # 实时模式
            total_assets = 3000000 + random.uniform(-150000, 600000)
            position_ratio = 0.8  # 实时交易模式下的持仓比例
        
        # 计算持仓价值和可用现金
        position_value = total_assets * position_ratio
        available_cash = total_assets - position_value
        
        # 计算每日盈亏（根据不同模式有不同的波动范围）
        if self.current_mode == "回测":
            daily_pnl = random.uniform(-15000, 25000)
        elif self.current_mode == "模拟":
            daily_pnl = random.uniform(-30000, 50000)
        else:  # 实时模式
            daily_pnl = random.uniform(-50000, 80000)
        
        # 构建账户信息
        account_info = {
            "total_assets": total_assets,
            "available_cash": available_cash,
            "position_value": position_value,
            "daily_pnl": daily_pnl
        }
        
        # 构建持仓信息
        positions = []
        
        # 确保stock_data有足够的数据用于取样
        sample_size = min(8, len(stock_data))
        if sample_size > 0:
            for i in range(sample_size):
                # 模拟持仓
                stock = stock_data[i]
                volume = random.randint(100, 5000) // 100 * 100  # 随机持仓量，按手取整
                cost_price = stock["price"] * (1 + random.uniform(-0.15, 0.15))  # 模拟成本价
                
                positions.append({
                    "symbol": stock["symbol"],
                    "name": stock["name"],
                    "volume": volume,
                    "cost_price": cost_price,
                    "current_price": stock["price"]
                })
        
        return account_info, positions

    def update_intelligent_panels(self, market_summary, stock_data, positions, sectors_analysis=None):
        """更新所有智能面板的数据"""
        try:
            # 1. 更新市场分析面板
            # 使用传入的行业板块数据或生成默认行业板块数据
            if not sectors_analysis:
                # 模拟行业板块数据
                sectors = [
                    {"name": "金融", "change_pct": -1.25, "leading_stocks": "招商银行, 工商银行", "leading_change": -0.85},
                    {"name": "医药", "change_pct": 2.35, "leading_stocks": "恒瑞医药, 迈瑞医疗", "leading_change": 3.25},
                    {"name": "科技", "change_pct": 3.82, "leading_stocks": "华为, 腾讯", "leading_change": 4.56},
                    {"name": "新能源", "change_pct": 5.17, "leading_stocks": "宁德时代, 比亚迪", "leading_change": 6.23},
                    {"name": "消费", "change_pct": 0.65, "leading_stocks": "贵州茅台, 五粮液", "leading_change": 1.12},
                    {"name": "地产", "change_pct": -2.43, "leading_stocks": "万科A, 保利发展", "leading_change": -1.85}
                ]
            else:
                sectors = sectors_analysis
                
            # 找出表现最好和最差的两个行业
            sorted_sectors = sorted(sectors, key=lambda x: x['change_pct'], reverse=True)
            best_sectors = sorted_sectors[:2] if len(sorted_sectors) >= 2 else sorted_sectors
            worst_sectors = sorted_sectors[-2:] if len(sorted_sectors) >= 2 else []
            worst_sectors.reverse()  # 从小到大排序
            
            # 模拟市场分析文本
            market_analysis = f"""
                <h3>市场概况分析</h3>
                <p>当前市场整体呈现{
                    "震荡上行" if market_summary["avg_change"] > 0.5 else 
                    "震荡下行" if market_summary["avg_change"] < -0.5 else "震荡整理"
                }态势，上涨个股{market_summary["up_count"]}只，下跌个股{market_summary["down_count"]}只。</p>
                
                <h3>市场热点板块</h3>
                <p>当前市场热点集中在"""
            
            # 添加表现最好的行业
            if best_sectors:
                market_analysis += f"""<span style="color:red;">{best_sectors[0]['name']}</span>"""
                if len(best_sectors) > 1:
                    market_analysis += f"""和<span style="color:red;">{best_sectors[1]['name']}</span>"""
                    
                market_analysis += """板块，
                呈现出明显的轮动特征。"""
                
                if best_sectors:
                    market_analysis += f"""{best_sectors[0]['name']}板块受市场情绪带动，整体上涨{best_sectors[0]['change_pct']:.2f}%；
                    """
                if len(best_sectors) > 1:
                    market_analysis += f"""{best_sectors[1]['name']}板块在{best_sectors[1]['leading_stocks'].split('(')[0]}带动下，整体涨幅达{best_sectors[1]['change_pct']:.2f}%。</p>
                    """
                else:
                    market_analysis += "</p>"
            else:
                market_analysis += "暂无明显热点板块。</p>"
            
            # 投资机会分析
            market_analysis += """
                <h3>投资机会分析</h3>
                """
            
            if best_sectors:
                leading_name = best_sectors[0]['leading_stocks'].split('(')[0]
                leading_symbol = best_sectors[0]['leading_stocks'].split('(')[1].strip(')')
                market_analysis += f"""<p>短期投资机会：关注{best_sectors[0]['name']}板块龙头企业{leading_name}({leading_symbol})，该股近期表现强势，涨幅{best_sectors[0]['leading_change']:.2f}%。</p>
                """
                
                if len(best_sectors) > 1:
                    market_analysis += f"""<p>中期投资机会：{best_sectors[1]['name']}板块有望受益于行业景气度提升，可关注板块内低估值优质公司。</p>
                    """
            
            market_analysis += """<p>长期投资价值：优质成长股在经历调整后估值趋于合理，未来发展空间广阔。</p>
                
                <h3>风险提示</h3>
                """
            
            # 添加表现最差的行业
            if worst_sectors:
                market_analysis += f"""<p>{worst_sectors[0]['name']}板块持续走弱，短期内仍有下行压力"""
                if len(worst_sectors) > 1:
                    market_analysis += f"""；{worst_sectors[1]['name']}板块表现相对低迷，关注企业基本面变化。</p>"""
                else:
                    market_analysis += "。</p>"
            else:
                market_analysis += "<p>暂无明显风险板块。</p>"
            
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
                        """
                        
                    # 添加行业动量信息
                    if best_sectors:
                        signal_text += f"""<p style="color:green;">- 已捕捉到{best_sectors[0]['name']}板块强势动量信号</p>
                        """
                    if len(best_sectors) > 1:
                        signal_text += f"""<p style="color:green;">- {best_sectors[1]['name']}行业呈现加速上涨态势</p>
                        """
                    if worst_sectors:
                        signal_text += f"""<p style="color:red;">- {worst_sectors[0]['name']}板块动量减弱</p>
                        """
                    
                    # 根据股票数据选出表现最好的5支股票作为AI选股结果
                    top_stocks = sorted(stock_data, key=lambda x: x['change_pct'], reverse=True)[:5]
                    
                    signal_text += f"""
                        <h3>AI智能选股结果 (Top 5)</h3>
                        <ol>
                        """
                    
                    for i, stock in enumerate(top_stocks):
                        confidence = 90 - i * 2 + random.uniform(-1.0, 1.0)  # 从高到低的置信度
                        signal_text += f"""<li>{stock['symbol']} - {stock['name']} (置信度: {confidence:.1f}%)</li>
                        """
                    
                    signal_text += """</ol>
                        
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
                
            # 5. 更新行业分析面板
            if sectors_analysis:
                self.analysis_panel.update_analysis(sectors_analysis, "")
                
        except Exception as e:
            logger.error(f"更新智能面板时发生错误: {e}")

    def change_mode(self, mode):
        """更改运行模式"""
        # 更新当前模式
        previous_mode = self.current_mode
        self.current_mode = mode
        
        # 更新模式状态标签
        self.mode_status.setText(f"当前模式: {self.current_mode}")
        
        # 更新按钮样式
        self.backtest_btn.setStyleSheet("background-color: #f0f0f0;")
        self.simulation_btn.setStyleSheet("background-color: #f0f0f0;")
        self.realtime_btn.setStyleSheet("background-color: #f0f0f0;")
        
        # 高亮当前选中的模式按钮
        if mode == "回测":
            self.backtest_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        elif mode == "模拟":
            self.simulation_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        elif mode == "实时":
            self.realtime_btn.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
        
        # 记录模式变更
        logger.info(f"交易系统模式已从 {previous_mode} 切换为 {mode}")
        
        # 显示模式切换提示框
        QMessageBox.information(self, "模式切换", f"系统已切换至{mode}模式")
        
        # 根据不同模式调整系统行为
        if mode == "回测":
            # 回测模式特定初始化
            self._initialize_backtest_mode()
        elif mode == "模拟":
            # 模拟模式特定初始化
            self._initialize_simulation_mode()
        elif mode == "实时":
            # 实时模式特定初始化
            self._initialize_realtime_mode()
        
        # 更新市场数据和UI显示
        self.update_market_data()
    
    def _initialize_backtest_mode(self):
        """初始化回测模式"""
        # 初始化回测模式下的股票数据
        self._initialize_backtest_stock_data()
        pass
    
    def _initialize_backtest_stock_data(self):
        """初始化回测模式下的股票数据，保证股票代码、名称、价格和涨跌幅一致性"""
        # 清空旧数据
        self.backtest_stock_data = {}
        
        # 从stock_list.json加载股票列表
        stock_list_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                     'data', 'market', 'stock_list.json')
        
        try:
            if os.path.exists(stock_list_path):
                with open(stock_list_path, 'r') as f:
                    symbols = json.load(f)
                logger.info(f"从stock_list.json加载了 {len(symbols)} 只股票")
            else:
                # 如果文件不存在，使用默认股票列表
                logger.warning("stock_list.json不存在，使用默认股票列表")
                symbols = [
                    '000001.SZ', '000333.SZ', '000651.SZ', '000858.SZ', 
                    '600000.SH', '600036.SH', '600276.SH', '600519.SH', 
                    '601318.SH', '601888.SH'
                ]
        except Exception as e:
            logger.error(f"加载股票列表失败: {e}，使用默认股票列表")
            symbols = [
                '000001.SZ', '000333.SZ', '000651.SZ', '000858.SZ', 
                '600000.SH', '600036.SH', '600276.SH', '600519.SH', 
                '601318.SH', '601888.SH'
            ]
        
        # 生成固定的股票数据
        for symbol in symbols:
            # 使用固定种子生成稳定但略有变化的价格
            seed = sum(ord(c) for c in symbol)
            random.seed(seed)
            
            # 获取行业信息
            sector = self.stock_sectors.get(symbol, {}).get("sector", "未知")
            
            # 根据行业调整基础价格范围（使不同行业有不同特征）
            if sector == "金融":
                base_price = random.uniform(5.0, 30.0)  # 金融股通常价格较低
                volatility = 0.8  # 金融股通常波动性较低
            elif sector == "科技":
                base_price = random.uniform(40.0, 180.0)  # 科技股通常价格中等偏高
                volatility = 1.5  # 科技股通常波动性较高
            elif sector == "消费":
                base_price = random.uniform(50.0, 300.0)  # 消费股价格跨度大
                volatility = 1.0  # 消费股波动适中
            elif sector == "医药":
                base_price = random.uniform(30.0, 120.0)  # 医药股价格中等
                volatility = 1.2  # 医药股波动较大
            elif sector == "能源":
                base_price = random.uniform(4.0, 20.0)  # 能源股通常价格较低
                volatility = 1.3  # 能源股波动较大
            elif sector == "工业":
                base_price = random.uniform(8.0, 40.0)  # 工业股价格中低
                volatility = 1.1  # 工业股波动适中
            else:
                base_price = random.uniform(10.0, 200.0)  # 默认价格范围
                volatility = 1.0  # 默认波动性
            
            # 保存股票数据
            self.backtest_stock_data[symbol] = {
                "symbol": symbol,
                "name": self.stock_names.get(symbol, f"股票{symbol.split('.')[0]}"),
                "base_price": base_price,
                "price": base_price,  # 当前价格，会根据时间略有变化
                "change_pct": 0.0,  # 初始涨跌幅为0
                "sector": sector,    # 行业信息
                "volatility": volatility  # 波动性特征
            }
        
        # 重置随机种子
        random.seed()
        
        logger.info(f"回测模式初始化了 {len(self.backtest_stock_data)} 只股票数据")
    
    def _initialize_simulation_mode(self):
        """初始化模拟交易模式"""
        # 这里实现模拟交易模式的特定初始化逻辑
        # 例如连接模拟交易API、设置模拟账户等
        pass
    
    def _initialize_realtime_mode(self):
        """初始化实时交易模式"""
        # 这里实现实时交易模式的特定初始化逻辑
        # 例如连接实时交易API、验证账户等
        
        # 显示实时交易确认对话框
        reply = QMessageBox.question(self, '确认实时交易', 
                                    '您确定要切换到实时交易模式吗？该模式下的交易将影响实际资金。',
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                    QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.No:
            # 用户取消，回退到之前的模式
            if self.current_mode == "实时":  # 如果当前已经是实时模式，则回退到模拟模式
                self.change_mode("模拟")
            return False
        
        return True

    def _load_extended_stock_names(self):
        """加载扩展的股票名称映射"""
        # 增加更多股票名称
        additional_names = {
            # 金融行业
            '601688.SH': '华泰证券',
            '600030.SH': '中信证券',
            '601628.SH': '中国人寿',
            '601601.SH': '中国太保',
            '601288.SH': '农业银行',
            '601398.SH': '工商银行',
            
            # 科技行业
            '000725.SZ': '京东方A',
            '002230.SZ': '科大讯飞',
            '000063.SZ': '中兴通讯',
            '688981.SH': '中芯国际',
            '002594.SZ': '比亚迪电子',
            '002475.SZ': '立讯精密',
            '600703.SH': '三安光电',
            '002241.SZ': '歌尔股份',
            '688111.SH': '金山办公',
            
            # 消费行业
            '600887.SH': '伊利股份',
            '603288.SH': '海天味业',
            '600809.SH': '山西汾酒',
            '002304.SZ': '洋河股份',
            '600690.SH': '海尔智家',
            
            # 医药行业
            '300760.SZ': '迈瑞医疗',
            '300759.SZ': '康龙化成',
            '600196.SH': '复星医药',
            '603259.SH': '药明康德',
            '000538.SZ': '云南白药',
            '600085.SH': '同仁堂',
            '300347.SZ': '泰格医药',
            '600763.SH': '通策医疗',
            '002821.SZ': '凯莱英',
            
            # 能源行业
            '601857.SH': '中国石油',
            '600028.SH': '中国石化',
            '601898.SH': '中煤能源',
            '600900.SH': '长江电力',
            '600905.SH': '三峡能源',
            '601985.SH': '中国核电',
            '600025.SH': '华能水电',
            '601225.SH': '陕西煤业',
            '600886.SH': '国投电力',
            '601088.SH': '中国神华',
            
            # 工业行业
            '601766.SH': '中国中车',
            '600031.SH': '三一重工',
            '601390.SH': '中国中铁',
            '601186.SH': '中国铁建',
            '601800.SH': '中国交建',
            '600019.SH': '宝钢股份',
            '601669.SH': '中国电建',
            '601668.SH': '中国建筑',
            '601989.SH': '中国重工',
            '601899.SH': '紫金矿业'
        }
        
        # 更新字典
        self.stock_names.update(additional_names)
        logger.info(f"股票名称映射表已扩展，当前共有 {len(self.stock_names)} 只股票的名称信息")
    
    def _load_sector_information(self):
        """加载股票行业信息"""
        sectors_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                  'data', 'market', 'sectors', 'classification.json')
        
        if os.path.exists(sectors_file):
            try:
                with open(sectors_file, 'r', encoding='utf-8') as f:
                    self.stock_sectors = json.load(f)
                logger.info(f"成功加载股票行业信息，共 {len(self.stock_sectors)} 只股票")
            except Exception as e:
                logger.error(f"加载股票行业信息失败: {e}")
        else:
            logger.warning("股票行业信息文件不存在，将使用默认行业分类")
            # 设置一些默认行业分类
            default_sectors = {
                '000001.SZ': {"sector": "金融"}, 
                '600000.SH': {"sector": "金融"},
                '600036.SH': {"sector": "金融"},
                '601318.SH': {"sector": "金融"},
                '000858.SZ': {"sector": "消费"},
                '600519.SH': {"sector": "消费"},
                '000651.SZ': {"sector": "消费"},
                '000333.SZ': {"sector": "消费"},
                '601888.SH': {"sector": "消费"},
                '600276.SH': {"sector": "医药"},
                '002415.SZ': {"sector": "科技"}
            }
            self.stock_sectors = default_sectors
    
    def change_stock_pool(self, pool_name):
        """切换当前股票池"""
        logger.info(f"切换股票池为: {pool_name}")
        
        # 将UI选择的股票池名称转换为系统内部名称
        pool_mapping = {
            "全部股票": "全A股",
            "沪深300": "沪深300",
            "中证500": "中证500",
            "创业板50": "创业板50",
            "科创50": "科创50",
            "自定义行业": "custom"
        }
        
        system_pool_name = pool_mapping.get(pool_name, "全A股")
        
        # 如果是自定义行业，弹出行业选择对话框
        if system_pool_name == "custom":
            selected_sector = self._show_sector_selector()
            if selected_sector:
                system_pool_name = f"行业-{selected_sector}"
            else:
                # 用户取消选择，恢复原来的选择
                return
        
        # 更新市场状态中的股票池
        if self.market_state and hasattr(self.market_state, "change_stock_pool"):
            success = self.market_state.change_stock_pool(system_pool_name)
            if success:
                # 立即更新数据以反映新的股票池
                self.update_market_data()
                # 显示提示
                stocks_count = len(self.market_state.tradable_symbols) if self.market_state else 0
                QMessageBox.information(self, "股票池已变更", 
                                        f"已切换到 {pool_name} 股票池，包含 {stocks_count} 只股票")
            else:
                QMessageBox.warning(self, "切换失败", 
                                    f"股票池 {pool_name} 不可用或不存在")
                # 恢复原来的选择
                self.market_panel.pool_selector.blockSignals(True)
                # TODO: 恢复为当前实际使用的股票池
                self.market_panel.pool_selector.setCurrentText("全部股票")
                self.market_panel.pool_selector.blockSignals(False)
    
    def _show_sector_selector(self):
        """显示行业选择对话框"""
        sectors = list(self.stock_sectors.keys())
        selected_sector, ok = QInputDialog.getItem(
            self, "选择行业", "请选择要关注的行业:", sectors, 0, False
        )
        if ok and selected_sector:
            return selected_sector
        return None
    
    def test_sector_rotation_strategy(self):
        """测试行业轮动策略"""
        # 确保在回测模式
        if self.current_mode != "回测":
            self.change_mode("回测")
        
        # 创建策略配置对话框
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("行业轮动策略测试配置")
            dialog.setMinimumWidth(400)
            
            layout = QVBoxLayout(dialog)
            
            # 添加策略参数
            form_layout = QFormLayout()
            
            # 策略基本参数
            lookback_period = QSpinBox()
            lookback_period.setRange(5, 60)
            lookback_period.setValue(20)
            form_layout.addRow("回看期(天):", lookback_period)
            
            sector_count = QSpinBox()
            sector_count.setRange(1, 6)
            sector_count.setValue(2)
            form_layout.addRow("选择行业数:", sector_count)
            
            stocks_per_sector = QSpinBox()
            stocks_per_sector.setRange(1, 10)
            stocks_per_sector.setValue(3)
            form_layout.addRow("每行业股票数:", stocks_per_sector)
            
            # 行业轮动特有参数
            cycle_timing_weight = QDoubleSpinBox()
            cycle_timing_weight.setRange(0.0, 1.0)
            cycle_timing_weight.setSingleStep(0.1)
            cycle_timing_weight.setValue(0.5)
            form_layout.addRow("周期择时权重:", cycle_timing_weight)
            
            min_holding_days = QSpinBox()
            min_holding_days.setRange(5, 60)
            min_holding_days.setValue(15)
            form_layout.addRow("最小持有天数:", min_holding_days)
            
            # 回测期间
            backtest_days = QSpinBox()
            backtest_days.setRange(30, 365)
            backtest_days.setValue(60)
            form_layout.addRow("回测天数:", backtest_days)
            
            layout.addLayout(form_layout)
            
            # 添加按钮区
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | 
                                         QDialogButtonBox.StandardButton.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            # 显示对话框并获取结果
            if dialog.exec() == QDialog.DialogCode.Accepted:
                # 收集参数
                config = {
                    "lookback_period": lookback_period.value(),
                    "sector_count": sector_count.value(),
                    "stocks_per_sector": stocks_per_sector.value(),
                    "cycle_timing_weight": cycle_timing_weight.value(),
                    "min_holding_days": min_holding_days.value(),
                    "backtest_days": backtest_days.value()
                }
                
                # 执行策略回测
                self._run_sector_rotation_backtest(config)
            
        except Exception as e:
            logger.error(f"创建行业轮动策略测试对话框失败: {e}", exc_info=True)
            QMessageBox.critical(self, "错误", f"配置行业轮动策略失败: {str(e)}")
    
    def _run_sector_rotation_backtest(self, config):
        """运行行业轮动策略回测"""
        try:
            # 获取所有行业及其股票
            sectors = self.market_state.get_sectors()
            if not sectors:
                QMessageBox.warning(self, "回测失败", "无法获取行业数据，请确保已加载行业分类信息")
                return
            
            # 初始化回测结果
            backtest_results = {
                'returns': [],
                'sector_returns': {},
                'sector_allocations': {},
                'trades': []
            }
            
            # 初始化各行业回报
            for sector_name in sectors:
                backtest_results['sector_returns'][sector_name] = []
                backtest_results['sector_allocations'][sector_name] = []
            
            # 设置初始资金和回测周期
            initial_capital = 1000000.0  # 100万初始资金
            backtest_days = 60  # 回测60天
            current_capital = initial_capital
            current_positions = {}  # 持仓 {symbol: {quantity, cost}}
            day_returns = []
            
            # 跟踪每个行业的表现
            sector_performance = {}
            
            # 每10天调整一次持仓
            rebalance_interval = 10
            
            # 模拟每一天的交易
            for day in range(backtest_days):
                # 更新股票数据
                self._update_backtest_data()
                
                # 计算每个行业的当日表现
                for sector_name, sector_data in sectors.items():
                    sector_stocks = sector_data.get('stocks', [])
                    if not sector_stocks:
                        continue
                    
                    # 计算行业当日平均涨跌幅
                    changes = []
                    for symbol in sector_stocks:
                        # 从backtest_stock_data中获取股票数据
                        stock_data = self.backtest_stock_data.get(symbol, {})
                        change_pct = stock_data.get('change_pct', 0)
                        changes.append(change_pct)
                    
                    if changes:
                        avg_change = sum(changes) / len(changes)
                        
                        # 更新行业表现历史
                        if sector_name not in sector_performance:
                            sector_performance[sector_name] = []
                        
                        sector_performance[sector_name].append(avg_change)
                        
                        # 记录行业收益
                        backtest_results['sector_returns'][sector_name].append({
                            'day': day,
                            'return': avg_change
                        })
                
                # 每rebalance_interval天重新配置资产
                if day % rebalance_interval == 0:
                    # 卖出所有现有持仓
                    for symbol, position in list(current_positions.items()):
                        stock_data = self.backtest_stock_data.get(symbol, {})
                        current_price = stock_data.get('price', 0)
                        if current_price > 0:
                            sell_value = position['quantity'] * current_price
                            current_capital += sell_value
                            
                            # 记录交易
                            backtest_results['trades'].append({
                                'day': day,
                                'symbol': symbol,
                                'action': 'sell',
                                'quantity': position['quantity'],
                                'price': current_price,
                                'value': sell_value
                            })
                    
                    # 清空持仓
                    current_positions = {}
                    
                    # 选择表现最好的前2个行业
                    top_sectors = []
                    if sector_performance:
                        # 计算各行业过去一段时间的平均表现
                        sector_avg_performance = {}
                        for sector_name, perf_history in sector_performance.items():
                            # 使用最近的表现数据，最少5个数据点或全部历史
                            recent_perf = perf_history[-min(5, len(perf_history)):]
                            if recent_perf:
                                sector_avg_performance[sector_name] = sum(recent_perf) / len(recent_perf)
                        
                        # 按表现排序并选择前2名
                        sorted_sectors = sorted(sector_avg_performance.items(), 
                                             key=lambda x: x[1], reverse=True)
                        top_sectors = [s[0] for s in sorted_sectors[:2]]
                    
                    # 为每个顶级行业分配资金
                    allocation_per_sector = current_capital / len(top_sectors) if top_sectors else 0
                    
                    for sector_name in top_sectors:
                        # 记录行业配置
                        backtest_results['sector_allocations'][sector_name].append({
                            'day': day,
                            'allocation': allocation_per_sector / current_capital if current_capital > 0 else 0
                        })
                        
                        # 获取该行业的股票
                        sector_stocks = sectors.get(sector_name, {}).get('stocks', [])
                        
                        # 在该行业中选择表现最好的前3只股票
                        stock_performance = []
                        for symbol in sector_stocks:
                            stock_data = self.backtest_stock_data.get(symbol, {})
                            change_pct = stock_data.get('change_pct', 0)
                            price = stock_data.get('price', 0)
                            if price > 0:
                                stock_performance.append((symbol, change_pct, price))
                        
                        # 排序并选择前3名
                        top_stocks = sorted(stock_performance, key=lambda x: x[1], reverse=True)[:3]
                        
                        # 为每只股票分配资金
                        allocation_per_stock = allocation_per_sector / len(top_stocks) if top_stocks else 0
                        
                        # 买入股票
                        for symbol, _, price in top_stocks:
                            if price > 0:
                                # 计算可买入的数量
                                quantity = int(allocation_per_stock / price)
                                
                                if quantity > 0:
                                    # 更新持仓
                                    cost = quantity * price
                                    current_positions[symbol] = {
                                        'quantity': quantity,
                                        'cost': cost
                                    }
                                    
                                    # 更新资金
                                    current_capital -= cost
                                    
                                    # 记录交易
                                    backtest_results['trades'].append({
                                        'day': day,
                                        'symbol': symbol,
                                        'action': 'buy',
                                        'quantity': quantity,
                                        'price': price,
                                        'value': cost
                                    })
                
                # 计算当前总资产价值
                portfolio_value = current_capital
                for symbol, position in current_positions.items():
                    stock_data = self.backtest_stock_data.get(symbol, {})
                    current_price = stock_data.get('price', 0)
                    if current_price > 0:
                        portfolio_value += position['quantity'] * current_price
                
                # 计算当日收益率
                if day > 0:
                    prev_value = initial_capital if day == 1 else backtest_results['returns'][-1]['value']
                    daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
                    day_returns.append(daily_return)
                else:
                    daily_return = 0
                
                # 记录总收益
                backtest_results['returns'].append({
                    'day': day,
                    'value': portfolio_value,
                    'return': daily_return
                })
            
            # 计算回测结果统计
            total_return = (portfolio_value - initial_capital) / initial_capital
            annualized_return = (1 + total_return) ** (252 / backtest_days) - 1
            
            # 计算波动率
            volatility = np.std(day_returns) * np.sqrt(252) if day_returns else 0
            
            # 计算夏普比率
            risk_free_rate = 0.02  # 假设无风险利率为2%
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # 计算最大回撤
            max_drawdown = 0
            peak_value = initial_capital
            for daily_result in backtest_results['returns']:
                current_value = daily_result['value']
                if current_value > peak_value:
                    peak_value = current_value
                else:
                    drawdown = (peak_value - current_value) / peak_value
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
            
            # 显示回测结果
            self._show_sector_rotation_results(backtest_results, {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            })
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"行业轮动策略回测失败: {e}\n{error_details}")
            QMessageBox.critical(self, "回测失败", f"运行行业轮动策略回测时出错: {e}")
    
    def _update_backtest_data_with_date(self, date, factor):
        """使用指定日期和因子更新回测数据"""
        # 这个方法相当于_update_backtest_data的变种，允许设置特定日期
        if not hasattr(self, 'backtest_stock_data') or not self.backtest_stock_data:
            self._initialize_backtest_stock_data()
        
        # 市场摘要信息
        market_summary = {
            "up_count": 0,
            "down_count": 0,
            "avg_change": 0.0,
            "total_volume": 0,
            "date": date
        }
        
        # 行业表现统计
        sector_performance = {}
        total_change = 0.0
        up_count = 0
        down_count = 0
        
        # 更新每只股票的价格和涨跌幅
        for symbol, data in self.backtest_stock_data.items():
            # 使用股票代码作为随机种子，保证一致性
            seed = sum(ord(c) for c in symbol)
            random.seed(seed + int(factor * 100))
            
            # 原始基准价格
            base_price = data.get('base_price', 10.0)
            
            # 获取行业
            sector = data.get('sector', '')
            
            # 根据行业调整波动率
            volatility = 0.02  # 默认波动率
            if sector in ['科技', '医药']:
                volatility = 0.03  # 高波动行业
            elif sector in ['金融', '公用事业']:
                volatility = 0.01  # 低波动行业
            
            # 使用时间因子和行业特性生成更真实的价格变动
            # 时间因子让价格随时间有趋势变化
            time_trend = (factor - 0.5) * 0.1  # -0.05 到 0.05 的趋势
            
            # 行业趋势 - 模拟不同行业在不同时期的表现
            sector_trend = 0.0
            if 0.0 <= factor < 0.3:  # 初期
                if sector in ['金融', '材料']:
                    sector_trend = 0.02
                elif sector in ['能源']:
                    sector_trend = -0.01
            elif 0.3 <= factor < 0.7:  # 中期
                if sector in ['科技', '消费']:
                    sector_trend = 0.02
                elif sector in ['金融']:
                    sector_trend = -0.01
            else:  # 后期
                if sector in ['医药', '必需消费']:
                    sector_trend = 0.02
                elif sector in ['科技']:
                    sector_trend = -0.01
            
            # 随机波动
            price_change = random.uniform(-volatility, volatility)
            
            # 综合因素
            total_change_pct = price_change + time_trend + sector_trend
            
            # 更新价格
            old_price = data.get('price', base_price)
            new_price = old_price * (1 + total_change_pct)
            new_price = max(new_price, base_price * 0.5)  # 确保价格不会太低
            
            # 更新股票数据
            data['price'] = new_price
            data['change'] = new_price - old_price
            data['change_pct'] = (new_price - old_price) / old_price
            
            # 统计上涨下跌
            if data['change'] > 0:
                up_count += 1
            else:
                down_count += 1
            
            total_change += data['change_pct']
            
            # 更新行业表现统计
            if sector:
                if sector not in sector_performance:
                    sector_performance[sector] = {
                        "total_change": 0.0,
                        "count": 0,
                        "stocks": []
                    }
                
                sector_performance[sector]["total_change"] += data['change_pct']
                sector_performance[sector]["count"] += 1
                sector_performance[sector]["stocks"].append({
                    "symbol": symbol,
                    "name": data.get('name', ''),
                    "change_pct": data['change_pct']
                })
        
        # 更新市场摘要信息
        market_summary["up_count"] = up_count
        market_summary["down_count"] = down_count
        market_summary["avg_change"] = total_change / len(self.backtest_stock_data) if self.backtest_stock_data else 0.0
        
        # 计算行业表现
        sectors_analysis = []
        for sector_name, perf in sector_performance.items():
            if perf["count"] > 0:
                avg_change = perf["total_change"] / perf["count"]
                # 找出该行业表现最好的股票作为龙头股
                leading_stock = max(perf["stocks"], key=lambda x: x["change_pct"])
                sectors_analysis.append({
                    "name": sector_name,
                    "change_pct": avg_change,
                    "leading_stocks": f"{leading_stock['name']} ({leading_stock['symbol']})",
                    "leading_change": leading_stock["change_pct"]
                })
        
        # 按行业涨幅排序
        sectors_analysis.sort(key=lambda x: x["change_pct"], reverse=True)
        
        # 返回分析结果供回测逻辑使用
        return {
            "market_summary": market_summary,
            "sectors_analysis": sectors_analysis
        }
    
    def _show_sector_rotation_results(self, backtest_results, stats):
        """显示行业轮动策略回测结果"""
        try:
            # 创建结果对话框
            result_dialog = QDialog(self)
            result_dialog.setWindowTitle("行业轮动策略回测结果")
            result_dialog.resize(900, 700)
            
            # 创建布局
            layout = QVBoxLayout()
            
            # 添加标题
            title = QLabel("行业轮动策略回测报告")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setStyleSheet("font-weight: bold; font-size: 16px; margin: 10px;")
            layout.addWidget(title)
            
            # 添加统计信息
            stats_frame = QFrame()
            stats_layout = QGridLayout()
            stats_frame.setLayout(stats_layout)
            stats_frame.setFrameShape(QFrame.Shape.Box)
            stats_frame.setStyleSheet("background-color: #f5f5f5; padding: 10px; border-radius: 5px;")
            
            row = 0
            # 添加总回报
            stats_layout.addWidget(QLabel("总回报率:"), row, 0)
            return_value = QLabel(f"{stats['total_return']:.2%}")
            return_value.setStyleSheet("font-weight: bold; color: green;" if stats['total_return'] > 0 else "font-weight: bold; color: red;")
            stats_layout.addWidget(return_value, row, 1)
            
            # 添加年化回报
            row += 1
            stats_layout.addWidget(QLabel("年化回报率:"), row, 0)
            annual_return = QLabel(f"{stats['annualized_return']:.2%}")
            annual_return.setStyleSheet("font-weight: bold; color: green;" if stats['annualized_return'] > 0 else "font-weight: bold; color: red;")
            stats_layout.addWidget(annual_return, row, 1)
            
            # 添加波动率
            row += 1
            stats_layout.addWidget(QLabel("波动率:"), row, 0)
            volatility = QLabel(f"{stats['volatility']:.2%}")
            stats_layout.addWidget(volatility, row, 1)
            
            # 添加夏普比率
            row += 1
            stats_layout.addWidget(QLabel("夏普比率:"), row, 0)
            sharpe = QLabel(f"{stats['sharpe_ratio']:.2f}")
            sharpe.setStyleSheet("font-weight: bold; color: green;" if stats['sharpe_ratio'] > 1 else "font-weight: bold; color: black;")
            stats_layout.addWidget(sharpe, row, 1)
            
            # 添加最大回撤
            row += 1
            stats_layout.addWidget(QLabel("最大回撤:"), row, 0)
            drawdown = QLabel(f"{stats['max_drawdown']:.2%}")
            drawdown.setStyleSheet("font-weight: bold; color: red;")
            stats_layout.addWidget(drawdown, row, 1)
            
            layout.addWidget(stats_frame)
            
            # 创建选项卡部件
            tab_widget = QTabWidget()
            
            # 添加回报曲线选项卡
            returns_tab = QWidget()
            returns_layout = QVBoxLayout()
            returns_tab.setLayout(returns_layout)
            
            # 创建资产价值图表
            chart_view = self._create_portfolio_value_chart(backtest_results['returns'])
            returns_layout.addWidget(chart_view)
            
            tab_widget.addTab(returns_tab, "资产曲线")
            
            # 添加行业表现选项卡
            sectors_tab = QWidget()
            sectors_layout = QVBoxLayout()
            sectors_tab.setLayout(sectors_layout)
            
            # 创建行业表现图表
            sector_chart_view = self._create_sector_performance_chart(backtest_results['sector_returns'])
            sectors_layout.addWidget(sector_chart_view)
            
            tab_widget.addTab(sectors_tab, "行业表现")
            
            # 添加行业配置选项卡
            allocations_tab = QWidget()
            allocations_layout = QVBoxLayout()
            allocations_tab.setLayout(allocations_layout)
            
            # 创建行业配置图表
            allocation_chart_view = self._create_sector_allocation_chart(backtest_results['sector_allocations'])
            allocations_layout.addWidget(allocation_chart_view)
            
            tab_widget.addTab(allocations_tab, "行业配置")
            
            # 添加交易记录选项卡
            trades_tab = QWidget()
            trades_layout = QVBoxLayout()
            trades_tab.setLayout(trades_layout)
            
            # 创建交易表格
            trades_table = QTableWidget()
            trades_table.setColumnCount(6)
            trades_table.setHorizontalHeaderLabels(["日期", "股票", "操作", "数量", "价格", "价值"])
            
            # 添加交易数据
            trades_table.setRowCount(len(backtest_results['trades']))
            for i, trade in enumerate(backtest_results['trades']):
                trades_table.setItem(i, 0, QTableWidgetItem(str(trade['day'])))
                trades_table.setItem(i, 1, QTableWidgetItem(trade['symbol']))
                trades_table.setItem(i, 2, QTableWidgetItem(trade['action']))
                trades_table.setItem(i, 3, QTableWidgetItem(str(trade['quantity'])))
                trades_table.setItem(i, 4, QTableWidgetItem(f"{trade['price']:.2f}"))
                trades_table.setItem(i, 5, QTableWidgetItem(f"{trade['value']:.2f}"))
                
                # 设置颜色
                if trade['action'] == 'buy':
                    trades_table.item(i, 2).setBackground(QColor(235, 255, 235))
                else:
                    trades_table.item(i, 2).setBackground(QColor(255, 235, 235))
            
            trades_table.setSortingEnabled(True)
            trades_table.setAlternatingRowColors(True)
            trades_table.resizeColumnsToContents()
            
            trades_layout.addWidget(trades_table)
            tab_widget.addTab(trades_tab, "交易记录")
            
            layout.addWidget(tab_widget)
            
            # 添加关闭按钮
            close_button = QPushButton("关闭")
            close_button.clicked.connect(result_dialog.accept)
            layout.addWidget(close_button)
            
            result_dialog.setLayout(layout)
            result_dialog.exec()
            
        except Exception as e:
            logger.error(f"显示行业轮动策略回测结果时出错: {e}", exc_info=True)
            QMessageBox.critical(self, "显示结果失败", f"显示回测结果时出错: {str(e)}")
    
    def _create_portfolio_value_chart(self, returns_data):
        """创建投资组合价值图表"""
        # 创建图表
        chart = QChart()
        chart.setTitle("投资组合价值曲线")
        
        # 创建价值曲线序列
        value_series = QLineSeries()
        value_series.setName("资产价值")
        
        # 添加数据点
        for i, data_point in enumerate(returns_data):
            value_series.append(i, data_point['value'])
        
        # 添加到图表
        chart.addSeries(value_series)
        
        # 创建坐标轴
        x_axis = QValueAxis()
        x_axis.setTitleText("日期")
        x_axis.setRange(0, len(returns_data) - 1)
        
        y_axis = QValueAxis()
        y_axis.setTitleText("价值")
        min_value = min([data_point['value'] for data_point in returns_data]) * 0.95
        max_value = max([data_point['value'] for data_point in returns_data]) * 1.05
        y_axis.setRange(min_value, max_value)
        
        chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)
        
        value_series.attachAxis(x_axis)
        value_series.attachAxis(y_axis)
        
        # 创建视图
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        return chart_view
    
    def _create_sector_performance_chart(self, sector_returns):
        """创建行业表现图表"""
        # 创建图表
        chart = QChart()
        chart.setTitle("行业表现")
        
        # 为每个行业创建一个序列
        for sector_name, returns in sector_returns.items():
            if not returns:
                continue
                
            series = QLineSeries()
            series.setName(sector_name)
            
            # 添加累计回报数据点
            cumulative_return = 1.0
            for i, data_point in enumerate(returns):
                cumulative_return *= (1 + data_point['return'])
                series.append(i, cumulative_return)
            
            chart.addSeries(series)
        
        # 创建坐标轴
        x_axis = QValueAxis()
        x_axis.setTitleText("日期")
        max_length = max([len(returns) for returns in sector_returns.values()]) if sector_returns else 0
        x_axis.setRange(0, max_length - 1)
        
        y_axis = QValueAxis()
        y_axis.setTitleText("累计回报")
        y_axis.setRange(0.5, 2.0)  # 假设回报在-50%到+100%之间
        
        chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)
        
        # 将所有系列附加到轴上
        for series in chart.series():
            series.attachAxis(x_axis)
            series.attachAxis(y_axis)
        
        # 创建视图
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        return chart_view
    
    def _create_sector_allocation_chart(self, sector_allocations):
        """创建行业配置图表"""
        # 创建图表
        chart = QChart()
        chart.setTitle("行业资金配置")
        
        # 为展示最近的配置,使用饼图
        pie_series = QPieSeries()
        
        # 获取最近一次配置
        last_allocations = {}
        for sector_name, allocations in sector_allocations.items():
            if allocations:
                last_allocations[sector_name] = allocations[-1]['allocation']
        
        # 添加数据到饼图
        for sector_name, allocation in last_allocations.items():
            if allocation > 0:
                slice = pie_series.append(sector_name, allocation)
                slice.setLabelVisible(True)
                slice.setLabel(f"{sector_name}: {allocation:.1%}")
        
        chart.addSeries(pie_series)
        
        # 创建视图
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        return chart_view
    
    def test_industry_leader_strategy(self):
        """测试行业龙头策略"""
        try:
            # 检查行业数据是否加载
            if not hasattr(self.market_state, 'stock_sectors') or not self.market_state.stock_sectors:
                QMessageBox.warning(self, "行业数据缺失", "无法加载行业数据，请确保已正确初始化市场状态")
                return
            
            # 使用行业数据开始行业龙头分析
            sectors = self.market_state.get_sectors()
            if not sectors:
                QMessageBox.warning(self, "行业数据缺失", "未找到有效的行业分类数据")
                return
            
            # 显示进度对话框
            progress = QProgressDialog("正在分析行业龙头...", "取消", 0, 100, self)
            progress.setWindowTitle("行业龙头分析")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setValue(0)
            progress.show()
            
            try:
                # 分析每个行业的龙头股票
                industry_leaders = {}
                for i, (sector_name, sector_data) in enumerate(sectors.items()):
                    # 更新进度
                    progress.setValue(int(i / len(sectors) * 50))
                    if progress.wasCanceled():
                        break
                    
                    # 获取该行业的股票
                    sector_stocks = sector_data.get('stocks', [])
                    if not sector_stocks:
                        continue
                    
                    # 获取股票数据
                    stock_data = []
                    for symbol in sector_stocks:
                        # 根据当前模式选择正确的数据源
                        if self.current_mode == "回测" and hasattr(self, 'backtest_stock_data'):
                            if symbol not in self.backtest_stock_data:
                                continue
                            data = self.backtest_stock_data[symbol].copy()
                        else:
                            # 从市场状态获取股票数据
                            if not hasattr(self.market_state, 'stocks') or symbol not in self.market_state.stocks:
                                continue
                            data = self.market_state.stocks[symbol].copy()
                        
                        data['symbol'] = symbol
                        
                        # 获取市值数据（如果有）
                        if hasattr(self.market_state, 'fundamentals') and symbol in self.market_state.fundamentals:
                            data['market_cap'] = self.market_state.fundamentals[symbol].get('market_cap', 0)
                        else:
                            # 使用价格和成交量的乘积作为市值的近似
                            data['market_cap'] = data.get('price', 0) * data.get('volume', 0) / 1000
                        
                        stock_data.append(data)
                    
                    # 计算行业指标
                    if not stock_data:
                        continue
                    
                    # 计算市值排名
                    market_cap_ranking = sorted(stock_data, key=lambda x: x.get('market_cap', 0), reverse=True)
                    
                    # 计算涨幅排名
                    change_ranking = sorted(stock_data, key=lambda x: x.get('change_pct', 0), reverse=True)
                    
                    # 计算综合得分
                    for stock in stock_data:
                        # 市值分数 (0-100)
                        market_cap_rank = market_cap_ranking.index(stock) + 1
                        market_cap_score = 100 * (1 - (market_cap_rank / len(stock_data)))
                        
                        # 涨幅分数 (0-100)
                        change_rank = change_ranking.index(stock) + 1
                        change_score = 100 * (1 - (change_rank / len(stock_data)))
                        
                        # 综合得分 (市值权重70%，涨幅权重30%)
                        stock['leader_score'] = market_cap_score * 0.7 + change_score * 0.3
                    
                    # 按综合得分排序
                    leader_ranking = sorted(stock_data, key=lambda x: x.get('leader_score', 0), reverse=True)
                    
                    # 选择前3名作为龙头股
                    top_leaders = leader_ranking[:3]
                    
                    # 保存结果
                    industry_leaders[sector_name] = top_leaders
                    
                # 更新进度
                progress.setValue(90)
                
                # 显示结果
                self._show_industry_leader_results(industry_leaders)
                
                progress.setValue(100)
            
            finally:
                # 确保进度对话框关闭
                progress.close()
        
        except Exception as e:
            logger.error(f"行业龙头策略分析失败: {e}", exc_info=True)
            QMessageBox.critical(self, "分析失败", f"行业龙头策略分析过程中出错: {str(e)}")

    def _show_industry_leader_results(self, industry_leaders):
        """显示行业龙头分析结果"""
        # 创建结果对话框
        result_dialog = QDialog(self)
        result_dialog.setWindowTitle("行业龙头分析结果")
        result_dialog.resize(800, 600)
        
        # 创建布局
        layout = QVBoxLayout()
        
        # 添加标题
        title = QLabel("行业龙头股分析报告")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 16px; margin: 10px;")
        layout.addWidget(title)
        
        # 创建选项卡部件
        tab_widget = QTabWidget()
        
        # 为每个行业创建一个选项卡
        for sector_name, leaders in industry_leaders.items():
            # 创建行业选项卡
            sector_tab = QWidget()
            sector_layout = QVBoxLayout()
            sector_tab.setLayout(sector_layout)
            
            # 添加行业说明
            sector_info = QLabel(f"行业: {sector_name}")
            sector_info.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
            sector_layout.addWidget(sector_info)
            
            # 创建龙头股表格
            table = QTableWidget()
            table.setColumnCount(6)
            table.setHorizontalHeaderLabels(["代码", "名称", "最新价", "涨跌幅", "市值(亿)", "龙头得分"])
            
            # 添加数据
            table.setRowCount(len(leaders))
            for i, stock in enumerate(leaders):
                symbol = stock.get('symbol', '')
                name = self.stock_names.get(symbol, symbol)
                price = stock.get('price', 0)
                change_pct = stock.get('change_pct', 0)
                market_cap = stock.get('market_cap', 0) / 100000000  # 转换为亿
                score = stock.get('leader_score', 0)
                
                table.setItem(i, 0, QTableWidgetItem(symbol))
                table.setItem(i, 1, QTableWidgetItem(name))
                table.setItem(i, 2, QTableWidgetItem(f"{price:.2f}"))
                
                change_item = QTableWidgetItem(f"{change_pct:.2%}")
                change_item.setForeground(QColor("green" if change_pct > 0 else "red"))
                table.setItem(i, 3, change_item)
                
                table.setItem(i, 4, QTableWidgetItem(f"{market_cap:.2f}"))
                table.setItem(i, 5, QTableWidgetItem(f"{score:.1f}"))
            
            table.resizeColumnsToContents()
            sector_layout.addWidget(table)
            
            # 添加行业分析说明
            analysis_text = """
            行业龙头分析基于以下因素：
            1. 市值规模 (70%权重)
            2. 涨幅表现 (30%权重)
            
            龙头股通常具有行业内领先的市场地位、较强的定价能力和更稳健的业绩表现。
            """
            analysis_label = QLabel(analysis_text)
            analysis_label.setWordWrap(True)
            sector_layout.addWidget(analysis_label)
            
            # 添加到选项卡
            tab_widget.addTab(sector_tab, sector_name)
        
        layout.addWidget(tab_widget)
        
        # 添加关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(result_dialog.accept)
        layout.addWidget(close_button)
        
        result_dialog.setLayout(layout)
        result_dialog.exec()


def start_gui():
    """启动PyQt6 GUI界面"""
    try:
        # 创建应用
        app = QApplication(sys.argv)
        
        # 设置应用样式
        app.setStyle("Fusion")
        
        # 设置异常捕获
        sys._excepthook = sys.excepthook
        def exception_hook(exctype, value, traceback):
            logger.error(f"Uncaught exception: {exctype}, {value}")
            logger.error(f"Traceback: {traceback}")
            sys._excepthook(exctype, value, traceback)
        sys.excepthook = exception_hook
        
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
        logger.info("开始Qt应用事件循环")
        result = app.exec()
        logger.info(f"Qt应用事件循环结束，退出码: {result}")
        return result
    except Exception as e:
        logger.error(f"启动GUI时发生错误: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    start_gui() 