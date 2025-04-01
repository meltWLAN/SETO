import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, font
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seto_versal.market.state import MarketState

class DashboardGUI:
    def __init__(self, market):
        """初始化仪表板GUI"""
        self.market = market
        self.logger = logging.getLogger(__name__)
        self.market_data = None
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("SETO-Versal 交易仪表板")
        self.root.geometry("1200x800")
        
        # 设置中文字体
        self.default_font = font.Font(family="Microsoft YaHei", size=10)
        self.title_font = font.Font(family="Microsoft YaHei", size=12, weight="bold")
        
        # 配置ttk样式
        style = ttk.Style()
        style.configure("TLabel", font=self.default_font)
        style.configure("TButton", font=self.default_font)
        style.configure("TLabelframe", font=self.title_font)
        style.configure("TLabelframe.Label", font=self.title_font)
        style.configure("Treeview", font=self.default_font)
        style.configure("Treeview.Heading", font=self.default_font)
        
        # 创建主容器
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建各个部分
        self._create_market_data_section()
        self._create_trade_history_section()
        self._create_metrics_section()
        
        # 启动更新循环
        self._update_display()
        
    def _create_market_data_section(self):
        """创建市场数据显示部分"""
        market_frame = ttk.LabelFrame(self.main_container, text="市场数据", padding="5")
        market_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # 创建市场数据标签
        self.price_label = ttk.Label(market_frame, text="价格: --")
        self.price_label.grid(row=0, column=0, padx=5)
        
        self.volume_label = ttk.Label(market_frame, text="成交量: --")
        self.volume_label.grid(row=0, column=1, padx=5)
        
        self.timestamp_label = ttk.Label(market_frame, text="最后更新: --")
        self.timestamp_label.grid(row=0, column=2, padx=5)
        
    def _create_trade_history_section(self):
        """创建交易历史显示部分"""
        history_frame = ttk.LabelFrame(self.main_container, text="交易历史", padding="5")
        history_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # 创建交易历史表格
        columns = ("时间", "类型", "价格", "数量", "总额")
        self.trade_tree = ttk.Treeview(history_frame, columns=columns, show="headings")
        
        # 设置列标题
        for col in columns:
            self.trade_tree.heading(col, text=col)
            self.trade_tree.column(col, width=100)
            
        self.trade_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.trade_tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.trade_tree.configure(yscrollcommand=scrollbar.set)
        
    def _create_metrics_section(self):
        """创建性能指标显示部分"""
        metrics_frame = ttk.LabelFrame(self.main_container, text="性能指标", padding="5")
        metrics_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # 创建指标标签
        self.pnl_label = ttk.Label(metrics_frame, text="盈亏: --")
        self.pnl_label.grid(row=0, column=0, padx=5)
        
        self.win_rate_label = ttk.Label(metrics_frame, text="胜率: --")
        self.win_rate_label.grid(row=0, column=1, padx=5)
        
        self.trades_label = ttk.Label(metrics_frame, text="总交易数: --")
        self.trades_label.grid(row=0, column=2, padx=5)
        
    def _update_market_data(self):
        """更新市场数据显示"""
        if self.market_data:
            self.price_label.config(text=f"价格: {self.market_data.get('price', '--')}")
            self.volume_label.config(text=f"成交量: {self.market_data.get('volume', '--')}")
            self.timestamp_label.config(text=f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
    def _update_trade_history(self):
        """更新交易历史显示"""
        try:
            # 清除现有项目
            for item in self.trade_tree.get_children():
                self.trade_tree.delete(item)
                
            # 从市场状态获取交易历史
            trades = self.market.get_state().get_trade_history()
            
            # 添加交易到表格
            for trade in trades:
                self.trade_tree.insert("", 0, values=(
                    trade.get('timestamp', '--'),
                    trade.get('type', '--'),
                    trade.get('price', '--'),
                    trade.get('volume', '--'),
                    trade.get('total', '--')
                ))
                
        except Exception as e:
            self.logger.error(f"更新交易历史时出错: {str(e)}")
            
    def _update_metrics(self):
        """更新性能指标显示"""
        try:
            metrics = self.market.get_state().get_performance_metrics()
            
            self.pnl_label.config(text=f"盈亏: {metrics.get('pnl', '--')}")
            self.win_rate_label.config(text=f"胜率: {metrics.get('win_rate', '--')}%")
            self.trades_label.config(text=f"总交易数: {metrics.get('total_trades', '--')}")
            
        except Exception as e:
            self.logger.error(f"更新性能指标时出错: {str(e)}")
            
    def _update_display(self):
        """更新显示数据"""
        try:
            # 获取当前市场状态
            market_state = self.market.get_state()
            
            # 更新市场数据
            self.market_data = market_state.get_market_data()
            
            # 更新显示
            self._update_market_data()
            self._update_trade_history()
            self._update_metrics()
            
        except Exception as e:
            self.logger.error(f"更新显示时出错: {str(e)}")
            self._show_error("更新显示错误", str(e))
            
        # 安排下一次更新
        self.root.after(1000, self._update_display)
        
    def _show_error(self, title, message):
        """显示错误消息对话框"""
        messagebox.showerror(title, message)
        
    def run(self):
        """启动GUI主循环"""
        self.root.mainloop() 