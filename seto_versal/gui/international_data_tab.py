#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
国际市场数据显示Tab

用于在GUI中展示国际市场（美股、港股等）的行情数据
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
                           QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                           QHeaderView, QSplitter, QFrame, QGridLayout, QTabWidget,
                           QDateEdit, QLineEdit, QCheckBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QDate
from PyQt5.QtGui import QColor, QFont, QIcon

from seto_versal.data.manager import TimeFrame
from seto_versal.data.enhanced_manager import EnhancedDataManager

logger = logging.getLogger(__name__)

class InternationalDataTab(QWidget):
    """国际市场数据显示Tab"""
    
    def __init__(self, data_manager: EnhancedDataManager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.current_symbol = ""
        self.current_market = ""
        self.current_timeframe = TimeFrame.DAY_1
        self.timer = QTimer(self)
        
        self._create_ui()
        self._connect_signals()
        
        # 初始化数据
        self._init_data()
        
    def _create_ui(self):
        """创建用户界面"""
        main_layout = QVBoxLayout(self)
        
        # 控制区域
        control_frame = QFrame(self)
        control_frame.setFrameShape(QFrame.StyledPanel)
        control_layout = QHBoxLayout(control_frame)
        
        # 市场选择
        market_label = QLabel("市场:")
        self.market_combo = QComboBox()
        
        # 品种选择
        symbol_label = QLabel("品种:")
        self.symbol_combo = QComboBox()
        
        # 时间周期选择
        timeframe_label = QLabel("周期:")
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1分钟", "5分钟", "15分钟", "30分钟", "1小时", "日线", "周线", "月线"])
        self.timeframe_combo.setCurrentIndex(5)  # 默认日线
        
        # 日期选择
        date_label = QLabel("开始日期:")
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate().addDays(-30))
        
        # 刷新按钮
        self.refresh_btn = QPushButton("刷新数据")
        
        # 实时更新复选框
        self.realtime_check = QCheckBox("实时更新")
        self.realtime_check.setChecked(True)
        
        # 保存按钮
        self.save_btn = QPushButton("保存数据")
        
        # 添加到控制布局
        control_layout.addWidget(market_label)
        control_layout.addWidget(self.market_combo)
        control_layout.addWidget(symbol_label)
        control_layout.addWidget(self.symbol_combo)
        control_layout.addWidget(timeframe_label)
        control_layout.addWidget(self.timeframe_combo)
        control_layout.addWidget(date_label)
        control_layout.addWidget(self.date_edit)
        control_layout.addWidget(self.refresh_btn)
        control_layout.addWidget(self.realtime_check)
        control_layout.addWidget(self.save_btn)
        control_layout.addStretch()
        
        # 数据表格
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(7)
        self.data_table.setHorizontalHeaderLabels(["时间", "开盘", "最高", "最低", "收盘", "涨跌幅", "成交量"])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.verticalHeader().setVisible(False)
        self.data_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.data_table.setAlternatingRowColors(True)
        
        # 添加到主布局
        main_layout.addWidget(control_frame)
        main_layout.addWidget(self.data_table)
        
    def _connect_signals(self):
        """连接信号和槽"""
        self.market_combo.currentIndexChanged.connect(self._on_market_changed)
        self.symbol_combo.currentIndexChanged.connect(self._on_symbol_changed)
        self.timeframe_combo.currentIndexChanged.connect(self._on_timeframe_changed)
        self.refresh_btn.clicked.connect(self._refresh_data)
        self.realtime_check.stateChanged.connect(self._toggle_realtime)
        self.save_btn.clicked.connect(self._save_data)
        self.timer.timeout.connect(self._refresh_latest_data)
        
    def _init_data(self):
        """初始化数据"""
        # 获取可用的数据源
        available_sources = self.data_manager.get_available_sources()
        if 'yahoo' not in available_sources:
            QMessageBox.warning(self, "警告", "Yahoo Finance数据源不可用，请检查配置或安装yfinance")
            return
            
        # 获取国际市场品种
        self.market_symbols = self.data_manager.get_international_symbols()
        if not self.market_symbols:
            QMessageBox.warning(self, "警告", "无法获取国际市场品种列表")
            return
            
        # 填充市场选择下拉框
        self.market_combo.addItems(sorted(self.market_symbols.keys()))
        
        # 设置定时器，每10秒刷新一次
        if self.realtime_check.isChecked():
            self.timer.start(10000)
        
    def _on_market_changed(self, index):
        """市场改变时的处理"""
        if index < 0:
            return
            
        self.current_market = self.market_combo.currentText()
        self.symbol_combo.clear()
        
        # 填充该市场的所有品种
        if self.current_market in self.market_symbols:
            symbols = sorted(self.market_symbols[self.current_market])
            self.symbol_combo.addItems(symbols)
            
    def _on_symbol_changed(self, index):
        """品种改变时的处理"""
        if index < 0:
            return
            
        self.current_symbol = self.symbol_combo.currentText()
        self._refresh_data()
        
    def _on_timeframe_changed(self, index):
        """时间周期改变时的处理"""
        if index < 0:
            return
            
        timeframe_map = {
            0: TimeFrame.MINUTE_1,
            1: TimeFrame.MINUTE_5,
            2: TimeFrame.MINUTE_15,
            3: TimeFrame.MINUTE_30,
            4: TimeFrame.HOUR_1,
            5: TimeFrame.DAY_1,
            6: TimeFrame.WEEK_1,
            7: TimeFrame.MONTH_1
        }
        
        self.current_timeframe = timeframe_map.get(index, TimeFrame.DAY_1)
        self._refresh_data()
        
    def _refresh_data(self):
        """刷新数据"""
        if not self.current_symbol:
            return
            
        try:
            # 获取开始日期
            start_date = self.date_edit.date().toPyDate()
            start_datetime = datetime.combine(start_date, datetime.min.time())
            
            # 获取历史数据
            data = self.data_manager.get_historical_international_data(
                symbol=self.current_symbol,
                timeframe=self.current_timeframe,
                start_time=start_datetime,
                source='yahoo'
            )
            
            # 显示数据
            self._display_data(data)
            
        except Exception as e:
            logger.error(f"刷新数据失败: {e}")
            QMessageBox.critical(self, "错误", f"获取数据失败: {str(e)}")
            
    def _refresh_latest_data(self):
        """刷新最新数据"""
        if not self.current_symbol or not self.realtime_check.isChecked():
            return
            
        try:
            # 获取最新数据
            data = self.data_manager.get_latest_international_data(
                symbol=self.current_symbol,
                timeframe=self.current_timeframe,
                source='yahoo'
            )
            
            # 如果有新数据，更新表格
            if not data.empty:
                # 如果表格为空，显示所有数据
                if self.data_table.rowCount() == 0:
                    self._display_data(data)
                else:
                    # 否则，添加新数据到表格顶部
                    latest_timestamp = data['timestamp'].iloc[-1]
                    
                    # 检查是否已有该时间戳的数据
                    for row in range(self.data_table.rowCount()):
                        time_item = self.data_table.item(row, 0)
                        if time_item and time_item.text() == latest_timestamp.strftime('%Y-%m-%d %H:%M:%S'):
                            # 更新现有行的数据
                            self._update_table_row(row, data.iloc[-1])
                            return
                            
                    # 如果是新数据，插入到表格顶部
                    self.data_table.insertRow(0)
                    self._update_table_row(0, data.iloc[-1])
                    
        except Exception as e:
            logger.error(f"刷新最新数据失败: {e}")
            
    def _display_data(self, data: pd.DataFrame):
        """显示数据到表格
        
        Args:
            data: 包含市场数据的DataFrame
        """
        if data.empty:
            self.data_table.setRowCount(0)
            return
            
        # 确保数据包含必要的列
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"数据缺少必要的列: {col}")
                return
                
        # 计算涨跌幅
        data['change_pct'] = 0.0
        if len(data) > 1:
            for i in range(1, len(data)):
                if data['close'].iloc[i-1] != 0:
                    data.loc[data.index[i], 'change_pct'] = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1] * 100
                    
        # 按时间倒序排序
        data = data.sort_values('timestamp', ascending=False).reset_index(drop=True)
                
        # 填充表格
        self.data_table.setRowCount(len(data))
        
        for row, (_, item_data) in enumerate(data.iterrows()):
            # 格式化时间
            timestamp = item_data['timestamp']
            if isinstance(timestamp, str):
                time_str = timestamp
            else:
                time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                
            # 创建时间单元格
            time_item = QTableWidgetItem(time_str)
            self.data_table.setItem(row, 0, time_item)
            
            # 价格和成交量数据
            open_item = QTableWidgetItem(f"{item_data['open']:.2f}")
            high_item = QTableWidgetItem(f"{item_data['high']:.2f}")
            low_item = QTableWidgetItem(f"{item_data['low']:.2f}")
            close_item = QTableWidgetItem(f"{item_data['close']:.2f}")
            
            # 涨跌幅
            change_pct = item_data.get('change_pct', 0.0)
            change_item = QTableWidgetItem(f"{change_pct:.2f}%")
            
            # 根据涨跌设置颜色
            if change_pct > 0:
                change_item.setForeground(QColor('red'))
                close_item.setForeground(QColor('red'))
            elif change_pct < 0:
                change_item.setForeground(QColor('green'))
                close_item.setForeground(QColor('green'))
                
            # 成交量
            volume = item_data['volume']
            volume_str = f"{volume:,.0f}"
            volume_item = QTableWidgetItem(volume_str)
            
            # 设置对齐方式
            for item in [open_item, high_item, low_item, close_item, change_item, volume_item]:
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                
            # 添加到表格
            self.data_table.setItem(row, 1, open_item)
            self.data_table.setItem(row, 2, high_item)
            self.data_table.setItem(row, 3, low_item)
            self.data_table.setItem(row, 4, close_item)
            self.data_table.setItem(row, 5, change_item)
            self.data_table.setItem(row, 6, volume_item)
            
    def _update_table_row(self, row: int, data: pd.Series):
        """更新表格中的一行
        
        Args:
            row: 行索引
            data: 行数据
        """
        # 格式化时间
        timestamp = data['timestamp']
        if isinstance(timestamp, str):
            time_str = timestamp
        else:
            time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
        # 设置时间单元格
        time_item = QTableWidgetItem(time_str)
        self.data_table.setItem(row, 0, time_item)
        
        # 价格和成交量数据
        open_item = QTableWidgetItem(f"{data['open']:.2f}")
        high_item = QTableWidgetItem(f"{data['high']:.2f}")
        low_item = QTableWidgetItem(f"{data['low']:.2f}")
        close_item = QTableWidgetItem(f"{data['close']:.2f}")
        
        # 涨跌幅，如果有前一行，计算涨跌幅
        change_pct = 0.0
        if row < self.data_table.rowCount() - 1:
            prev_close_item = self.data_table.item(row + 1, 4)
            if prev_close_item:
                prev_close = float(prev_close_item.text())
                if prev_close > 0:
                    change_pct = (data['close'] - prev_close) / prev_close * 100
                    
        change_item = QTableWidgetItem(f"{change_pct:.2f}%")
        
        # 根据涨跌设置颜色
        if change_pct > 0:
            change_item.setForeground(QColor('red'))
            close_item.setForeground(QColor('red'))
        elif change_pct < 0:
            change_item.setForeground(QColor('green'))
            close_item.setForeground(QColor('green'))
            
        # 成交量
        volume = data['volume']
        volume_str = f"{volume:,.0f}"
        volume_item = QTableWidgetItem(volume_str)
        
        # 设置对齐方式
        for item in [open_item, high_item, low_item, close_item, change_item, volume_item]:
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
        # 添加到表格
        self.data_table.setItem(row, 1, open_item)
        self.data_table.setItem(row, 2, high_item)
        self.data_table.setItem(row, 3, low_item)
        self.data_table.setItem(row, 4, close_item)
        self.data_table.setItem(row, 5, change_item)
        self.data_table.setItem(row, 6, volume_item)
        
    def _toggle_realtime(self, state):
        """切换实时更新状态"""
        if state == Qt.Checked:
            self.timer.start(10000)  # 10秒刷新一次
        else:
            self.timer.stop()
            
    def _save_data(self):
        """保存当前数据到CSV文件"""
        if not self.current_symbol:
            QMessageBox.warning(self, "警告", "请先选择交易品种")
            return
            
        try:
            # 构造文件名
            filename = f"{self.current_symbol}_{self.current_timeframe.value}_{datetime.now().strftime('%Y%m%d')}.csv"
            
            # 获取表格数据
            rows = self.data_table.rowCount()
            cols = self.data_table.columnCount()
            
            if rows == 0:
                QMessageBox.warning(self, "警告", "没有数据可保存")
                return
                
            # 创建DataFrame存储数据
            data = []
            headers = ["时间", "开盘", "最高", "最低", "收盘", "涨跌幅", "成交量"]
            
            for row in range(rows):
                row_data = {}
                for col in range(cols):
                    item = self.data_table.item(row, col)
                    if item:
                        # 移除涨跌幅的百分号
                        value = item.text()
                        if col == 5:  # 涨跌幅列
                            value = value.replace('%', '')
                        row_data[headers[col]] = value
                data.append(row_data)
                
            df = pd.DataFrame(data)
            
            # 保存到CSV
            df.to_csv(filename, index=False)
            QMessageBox.information(self, "成功", f"数据已保存到 {filename}")
            
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            QMessageBox.critical(self, "错误", f"保存数据失败: {str(e)}")
            
    def closeEvent(self, event):
        """关闭事件处理，停止定时器"""
        self.timer.stop()
        super().closeEvent(event) 