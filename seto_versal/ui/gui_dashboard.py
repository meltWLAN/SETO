#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal 交易系统界面
提供基于Tkinter的桌面GUI界面，用于运行和监控交易系统
"""

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np

# 导入SETO-Versal系统组件
from seto_versal.main import SetoVersal
from seto_versal.utils.config import load_config

class SetoGUI:
    """基于Tkinter的交易系统GUI界面"""
    
    def __init__(self, master):
        """初始化GUI界面"""
        self.master = master
        self.master.title("SETO-Versal 交易系统")
        self.master.geometry("1200x800")
        self.master.minsize(800, 600)
        
        # 系统状态
        self.seto = None
        self.running = False
        self.thread = None
        self.config = None
        self.config_path = tk.StringVar(value="config.yaml")
        self.mode = tk.StringVar(value="backtest")
        
        # 添加交易推荐列表
        self.trading_recommendations = []
        
        # 创建主框架
        self.setup_ui()
        
        # 更新状态
        self.update_status()
        
    def setup_ui(self):
        """设置UI组件"""
        # 创建菜单栏
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开配置文件", command=self.open_config)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.master.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 创建左侧控制面板框架
        left_frame = ttk.Frame(self.master, padding=10, relief="ridge", borderwidth=2)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # 控制面板标题
        ttk.Label(left_frame, text="控制面板", font=("Arial", 14, "bold")).pack(pady=10)
        
        # 配置文件选择
        ttk.Label(left_frame, text="配置文件路径:").pack(anchor=tk.W, pady=(10, 0))
        config_frame = ttk.Frame(left_frame)
        config_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(config_frame, textvariable=self.config_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(config_frame, text="浏览", command=self.browse_config).pack(side=tk.RIGHT, padx=5)
        
        # 运行模式选择
        ttk.Label(left_frame, text="运行模式:").pack(anchor=tk.W, pady=(10, 0))
        mode_frame = ttk.Frame(left_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        modes = ["回测", "模拟", "实盘", "测试"]
        mode_values = {"回测": "backtest", "模拟": "paper", "实盘": "live", "测试": "test"}
        self.mode_values = mode_values
        mode_cb = ttk.Combobox(mode_frame, textvariable=self.mode, values=modes, state="readonly")
        mode_cb.pack(fill=tk.X)
        mode_cb.current(0)
        
        # 加载配置按钮
        ttk.Button(left_frame, text="加载配置", command=self.load_config_btn).pack(fill=tk.X, pady=10)
        
        # 系统控制按钮
        buttons_frame = ttk.Frame(left_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        self.start_btn = ttk.Button(buttons_frame, text="启动系统", command=self.start_system)
        self.start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.stop_btn = ttk.Button(buttons_frame, text="停止系统", command=self.stop_system, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # 系统状态
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(left_frame, text="系统状态:").pack(anchor=tk.W)
        self.status_label = ttk.Label(left_frame, text="已停止", foreground="red")
        self.status_label.pack(anchor=tk.W, pady=5)
        
        # 系统信息
        self.info_frame = ttk.LabelFrame(left_frame, text="系统信息", padding=10)
        self.info_frame.pack(fill=tk.X, pady=10, expand=True)
        self.info_labels = {}
        for key in ["版本", "市场", "更新间隔", "风险等级"]:
            frame = ttk.Frame(self.info_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=f"{key}:").pack(side=tk.LEFT)
            self.info_labels[key] = ttk.Label(frame, text="--")
            self.info_labels[key].pack(side=tk.RIGHT)
        
        # 创建右侧主内容区域
        right_frame = ttk.Frame(self.master)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建选项卡
        self.tabs = ttk.Notebook(right_frame)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        
        # 系统概览选项卡
        overview_tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(overview_tab, text="系统概览")
        
        # 系统概览 - 指标面板
        metrics_frame = ttk.Frame(overview_tab)
        metrics_frame.pack(fill=tk.X, pady=10)
        
        # 创建四个指标框
        metrics = [
            {"label": "账户余额", "value": "¥100,000"},
            {"label": "今日盈亏", "value": "¥500 (0.5%)"},
            {"label": "持仓数量", "value": "3"},
            {"label": "智能体数量", "value": "5"}
        ]
        
        for i, metric in enumerate(metrics):
            frame = ttk.LabelFrame(metrics_frame, text=metric["label"], padding=10)
            frame.grid(row=0, column=i, padx=5, sticky=tk.EW)
            metrics_frame.columnconfigure(i, weight=1)
            ttk.Label(frame, text=metric["value"], font=("Arial", 12, "bold")).pack()
        
        # 市场状态
        market_frame = ttk.LabelFrame(overview_tab, text="市场状态", padding=10)
        market_frame.pack(fill=tk.X, pady=10)
        self.market_status = ttk.Label(market_frame, text="未连接")
        self.market_status.pack()
        
        # 添加图表
        chart_frame = ttk.LabelFrame(overview_tab, text="账户权益曲线", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 创建matplotlib图表
        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 绘制示例图表
        self.update_chart()
        
        # 交易记录选项卡
        trades_tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(trades_tab, text="交易记录")
        
        # 创建交易记录表格
        trades_frame = ttk.Frame(trades_tab)
        trades_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建Treeview
        columns = ("日期", "股票", "操作", "价格", "数量", "金额", "智能体")
        self.trades_tree = ttk.Treeview(trades_frame, columns=columns, show="headings")
        
        # 设置列标题
        for col in columns:
            self.trades_tree.heading(col, text=col)
            self.trades_tree.column(col, width=100)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(trades_frame, orient=tk.VERTICAL, command=self.trades_tree.yview)
        self.trades_tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.trades_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 添加示例数据
        self.update_trades()
        
        # 智能体状态选项卡
        agents_tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(agents_tab, text="智能体状态")
        
        # 创建智能体状态表格
        agents_frame = ttk.Frame(agents_tab)
        agents_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建Treeview
        columns = ("智能体", "交易次数", "胜率", "盈亏比", "累计收益")
        self.agents_tree = ttk.Treeview(agents_frame, columns=columns, show="headings")
        
        # 设置列标题
        for col in columns:
            self.agents_tree.heading(col, text=col)
            self.agents_tree.column(col, width=100)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(agents_frame, orient=tk.VERTICAL, command=self.agents_tree.yview)
        self.agents_tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.agents_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 添加示例数据
        self.update_agents()
        
        # 添加日志选项卡
        logs_tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(logs_tab, text="系统日志")
        
        # 创建日志文本框
        self.log_text = tk.Text(logs_tab, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # 添加滚动条
        log_scrollbar = ttk.Scrollbar(logs_tab, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 在日志中添加一些起始消息
        self.log("SETO-Versal 交易系统已启动")
        self.log("等待配置加载...")
        
        # 添加交易推荐选项卡
        recommendations_tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(recommendations_tab, text="交易推荐")
        
        # 创建推荐标题
        ttk.Label(recommendations_tab, text="系统交易推荐", 
                 font=("Arial", 16, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        # 创建买入推荐框架
        buy_frame = ttk.LabelFrame(recommendations_tab, text="买入推荐", padding=10)
        buy_frame.pack(fill=tk.X, pady=10)
        
        # 创建买入推荐表格
        buy_columns = ("股票代码", "股票名称", "当前价格", "推荐理由", "预期目标价", "信心指数", "推荐智能体")
        self.buy_tree = ttk.Treeview(buy_frame, columns=buy_columns, show="headings", height=5)
        
        # 设置列标题
        for col in buy_columns:
            self.buy_tree.heading(col, text=col)
            self.buy_tree.column(col, width=100)
        
        # 添加滚动条
        buy_scrollbar = ttk.Scrollbar(buy_frame, orient=tk.VERTICAL, command=self.buy_tree.yview)
        self.buy_tree.configure(yscrollcommand=buy_scrollbar.set)
        
        # 布局
        self.buy_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        buy_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建卖出推荐框架
        sell_frame = ttk.LabelFrame(recommendations_tab, text="卖出推荐", padding=10)
        sell_frame.pack(fill=tk.X, pady=10)
        
        # 创建卖出推荐表格
        sell_columns = ("股票代码", "股票名称", "当前价格", "推荐理由", "持有收益", "信心指数", "推荐智能体")
        self.sell_tree = ttk.Treeview(sell_frame, columns=sell_columns, show="headings", height=5)
        
        # 设置列标题
        for col in sell_columns:
            self.sell_tree.heading(col, text=col)
            self.sell_tree.column(col, width=100)
        
        # 添加滚动条
        sell_scrollbar = ttk.Scrollbar(sell_frame, orient=tk.VERTICAL, command=self.sell_tree.yview)
        self.sell_tree.configure(yscrollcommand=sell_scrollbar.set)
        
        # 布局
        self.sell_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sell_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 手动刷新按钮
        refresh_frame = ttk.Frame(recommendations_tab)
        refresh_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(refresh_frame, text="刷新交易推荐", 
                  command=self.refresh_recommendations).pack(side=tk.RIGHT)
        
        # 推荐更新时间标签
        self.last_recommendation_time = ttk.Label(refresh_frame, 
                                                text="上次更新: 尚未更新")
        self.last_recommendation_time.pack(side=tk.LEFT)
        
        # 确保窗口关闭时停止系统
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def update_chart(self):
        """更新图表"""
        # 创建示例数据
        dates = pd.date_range(start='2023-01-01', end='2023-01-30')
        equity = 100000 + np.array(range(30)) * 100 + np.random.randn(30) * 500
        benchmark = 100000 + np.array(range(30)) * 80 + np.random.randn(30) * 400
        
        # 清除当前图表
        self.ax.clear()
        
        # 绘制新图表
        self.ax.plot(dates, equity, label='账户权益', color='blue')
        self.ax.plot(dates, benchmark, label='基准', color='red')
        
        # 添加标签和图例
        self.ax.set_xlabel('日期')
        self.ax.set_ylabel('权益')
        self.ax.set_title('账户权益曲线')
        self.ax.legend()
        
        # 旋转日期标签
        self.figure.autofmt_xdate()
        
        # 更新画布
        self.canvas.draw()
    
    def update_trades(self):
        """更新交易记录"""
        # 清除现有数据
        for item in self.trades_tree.get_children():
            self.trades_tree.delete(item)
        
        # 示例交易数据
        trades_data = [
            ("2023-01-01", "000001.SZ", "买入", "¥12.50", "1000", "¥12,500", "趋势跟踪"),
            ("2023-01-02", "600000.SH", "买入", "¥8.30", "2000", "¥16,600", "逆势交易"),
            ("2023-01-03", "000001.SZ", "卖出", "¥13.10", "1000", "¥13,100", "趋势跟踪"),
            ("2023-01-04", "601318.SH", "买入", "¥45.20", "500", "¥22,600", "防御型"),
        ]
        
        # 添加数据
        for trade in trades_data:
            self.trades_tree.insert("", tk.END, values=trade)
    
    def update_agents(self):
        """更新智能体状态"""
        # 清除现有数据
        for item in self.agents_tree.get_children():
            self.agents_tree.delete(item)
        
        # 示例智能体数据
        agents_data = [
            ("趋势跟踪", "15", "60%", "1.5", "¥2,500"),
            ("逆势交易", "8", "75%", "1.2", "¥1,800"),
            ("快速获利", "23", "52%", "0.9", "¥1,200"),
            ("防御型", "5", "80%", "2.1", "¥2,600"),
            ("板块轮动", "12", "67%", "1.3", "¥2,100"),
        ]
        
        # 添加数据
        for agent in agents_data:
            self.agents_tree.insert("", tk.END, values=agent)
    
    def browse_config(self):
        """浏览选择配置文件"""
        filename = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("YAML文件", "*.yaml"), ("所有文件", "*.*")]
        )
        if filename:
            self.config_path.set(filename)
    
    def open_config(self):
        """打开配置文件进行编辑"""
        try:
            import subprocess
            import os
            import platform
            
            config_path = self.config_path.get()
            
            if platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', config_path))
            elif platform.system() == 'Windows':  # Windows
                os.startfile(config_path)
            else:  # linux
                subprocess.call(('xdg-open', config_path))
                
            self.log(f"已打开配置文件: {config_path}")
        except Exception as e:
            messagebox.showerror("错误", f"无法打开配置文件: {e}")
            self.log(f"打开配置文件失败: {e}")
    
    def load_config_btn(self):
        """加载配置按钮处理"""
        try:
            config_path = self.config_path.get()
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
            self.config = load_config(config_path)
            
            # 更新系统信息
            if self.config:
                self.info_labels["版本"].config(text=self.config['system'].get('version', '--'))
                self.info_labels["市场"].config(text=self.config['market'].get('market_type', '--'))
                self.info_labels["更新间隔"].config(text=str(self.config['market'].get('update_interval', '--')))
                self.info_labels["风险等级"].config(text=self.config['philosophy'].get('risk_level', '--'))
            
            messagebox.showinfo("成功", "配置加载成功!")
            self.log(f"配置已加载: {config_path}")
            
        except Exception as e:
            messagebox.showerror("配置加载错误", str(e))
            self.log(f"配置加载错误: {e}")
    
    def start_system(self):
        """启动交易系统"""
        if self.running:
            messagebox.showinfo("系统已在运行", "SETO-Versal系统已经在运行，请先停止")
            return
        
        if not self.config:
            messagebox.showwarning("未加载配置", "请先加载配置文件")
            return
        
        # 获取当前模式
        mode = self.mode.get()
        mode_value = self.mode_values.get(mode, "backtest")
        
        # 启动测试模式
        if mode_value == "test":
            self.running = True
            self.log(f"系统以测试模式启动")
            self._run_test_mode()
            return
        
        try:
            # 创建临时配置文件
            import tempfile
            import yaml
            
            # 添加缺失的必要配置项
            self.config['system'] = self.config.get('system', {})
            self.config['system']['mode'] = mode_value
            
            self.config['market'] = self.config.get('market', {})
            self.config['market']['mode'] = mode_value
            
            if 'update_interval' not in self.config['market']:
                self.config['market']['update_interval'] = 5
                
            if 'agents' not in self.config:
                self.config['agents'] = {'enabled': ['trend', 'reversal', 'defensive']}
                
            if 'evolution' not in self.config:
                self.config['evolution'] = {'enabled': True, 'frequency': 'daily'}
                
            if 'philosophy' not in self.config:
                self.config['philosophy'] = {'risk_tolerance': 'medium'}
                
            if 'feedback' not in self.config:
                self.config['feedback'] = {'performance_metrics': ['sharpe', 'drawdown', 'win_rate']}
            
            # 创建临时配置文件
            temp_config = tempfile.NamedTemporaryFile(mode='w+', suffix='.yaml', delete=False)
            yaml.dump(self.config, temp_config)
            temp_config.flush()
            temp_config_path = temp_config.name
            temp_config.close()
            
            # 保存配置路径以便后续删除
            self.temp_config_path = temp_config_path
            
            # 启动系统
            self.log(f"系统启动中，模式: {mode}")
            
            # 创建SetoVersal实例
            self.seto = SetoVersal(config_path=temp_config_path)
            
            # 更新UI显示
            self.info_labels["版本"].config(text=self.config.get('system', {}).get('version', '0.9.0'))
            self.info_labels["市场"].config(text=self.config.get('market', {}).get('exchange', 'ALL'))
            
            update_interval = self.config.get('market', {}).get('update_interval', 5)
            self.info_labels["更新间隔"].config(text=str(update_interval))
            
            risk_level = self.config.get('philosophy', {}).get('risk_tolerance', 'medium')
            self.info_labels["风险等级"].config(text=risk_level)
            
            # 在新线程中启动系统
            self.running = True
            self.thread = threading.Thread(target=self._system_loop)
            self.thread.daemon = True
            self.thread.start()
            
            # 更新UI状态
            self.update_status()
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("启动失败", f"系统启动失败: {str(e)}")
            self.log(f"系统启动失败: {str(e)}")
            
    def stop_system(self):
        """停止交易系统"""
        if not self.running:
            return
        
        try:
            self.running = False
            self.log("系统停止中...")
            
            # 如果有实例，调用停止方法
            if self.seto:
                self.seto.stop()
                
            # 等待线程结束
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5.0)
                
            # 清理临时配置文件
            if hasattr(self, 'temp_config_path') and os.path.exists(self.temp_config_path):
                try:
                    os.unlink(self.temp_config_path)
                except Exception as e:
                    self.log(f"删除临时配置文件失败: {str(e)}")
            
            # 更新UI
            self.update_status()
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
            self.log("系统已停止")
        
        except Exception as e:
            messagebox.showerror("停止失败", f"系统停止失败: {str(e)}")
            self.log(f"系统停止失败: {str(e)}")
    
    def update_status(self):
        """更新UI状态"""
        if self.running:
            self.status_label.config(text="运行中", foreground="green")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.market_status.config(text="已连接" if self.seto and self.seto.market_state.is_market_open() else "已连接 (市场闭市)")
        else:
            self.status_label.config(text="已停止", foreground="red")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.market_status.config(text="未连接")
    
    def on_closing(self):
        """窗口关闭时的处理"""
        if self.running:
            if messagebox.askyesno("退出", "系统正在运行，确定要退出吗？"):
                self.stop_system()
                self.master.destroy()
        else:
            self.master.destroy()
    
    def _run_test_mode(self):
        """在测试模式下运行系统，使用模拟数据"""
        # 设置为运行状态
        self.running = True
        self.update_status()
        
        # 更新UI信息
        self.info_labels["版本"].config(text="测试版")
        self.info_labels["市场"].config(text="模拟")
        self.info_labels["更新间隔"].config(text="5")
        self.info_labels["风险等级"].config(text="中等")
        
        # 在新线程中运行测试循环
        self.thread = threading.Thread(target=self._test_loop)
        self.thread.daemon = True
        self.thread.start()
        
        messagebox.showinfo("系统启动", "测试模式已启动!")
        self.log("系统已启动，运行模式: 测试")
    
    def _test_loop(self):
        """测试模式的主循环"""
        try:
            while self.running:
                # 生成一些测试数据
                self._generate_test_data()
                
                # 更新UI
                self.master.after(0, self._update_test_ui)
                
                # 等待一段时间
                time.sleep(5)
        except Exception as e:
            error_msg = str(e)
            self.master.after(0, lambda msg=error_msg: self.log(f"测试模式错误: {msg}"))
            self.running = False
            self.master.after(0, self.update_status)
    
    def _generate_test_data(self):
        """生成测试数据"""
        # 这个方法在测试线程中调用，生成模拟数据
        current_time = datetime.now()
        
        # 定义更多股票代码和名称作为测试数据源
        test_symbols = [
            ("000001.SZ", "平安银行"), ("600000.SH", "浦发银行"), ("601318.SH", "中国平安"),
            ("600519.SH", "贵州茅台"), ("000651.SZ", "格力电器"), ("600036.SH", "招商银行"),
            ("601166.SH", "兴业银行"), ("000858.SZ", "五粮液"), ("000333.SZ", "美的集团"),
            ("600276.SH", "恒瑞医药"), ("601888.SH", "中国中免"), ("600009.SH", "上海机场"),
            ("002415.SZ", "海康威视"), ("600887.SH", "伊利股份"), ("600019.SH", "宝钢股份"),
            ("600028.SH", "中国石化"), ("601857.SH", "中国石油"), ("603288.SH", "海天味业"),
            ("000538.SZ", "云南白药"), ("000661.SZ", "长春高新"), ("002230.SZ", "科大讯飞"),
            ("603501.SH", "韦尔股份"), ("688005.SH", "容百科技"), ("000063.SZ", "中兴通讯"),
            ("002475.SZ", "立讯精密"), ("600050.SH", "中国联通"), ("601728.SH", "中国电信")
        ]
        
        reasons = [
            "突破20日均线，成交量放大",
            "RSI超卖反弹信号",
            "MACD底背离",
            "30日线支撑有效",
            "量价齐升，突破压力位",
            "突破下降趋势线",
            "形成黄金交叉",
            "突破三角形态",
            "底部放量",
            "头肩底形态确认",
            "跳空突破前期高点",
            "缩量回调到支撑位",
            "布林带收口后向上突破",
            "KDJ超卖区金叉",
            "BOLL带下轨支撑有效",
            "五浪上升结构完成",
            "主力资金流入明显"
        ]
        
        agents = ["趋势跟踪", "反转交易", "快速获利", "防御型", "板块轮动", "震荡突破", "量能分析", "突破交易"]
        
        # 随机选择3-7个买入推荐
        self.test_buy_recommendations = []
        num_buys = np.random.randint(3, 8)
        indices = np.random.choice(len(test_symbols), num_buys, replace=False)
        
        for i in indices:
            symbol, name = test_symbols[i]
            price = np.random.random() * 100 + 10
            target = price * (1 + np.random.random() * 0.1)
            conf = np.random.random() * 0.3 + 0.6
            
            self.test_buy_recommendations.append({
                "symbol": symbol,
                "name": name,
                "price": price,
                "reason": reasons[np.random.randint(0, len(reasons))],
                "target_price": target,
                "confidence": conf,
                "agent": agents[np.random.randint(0, len(agents))]
            })
        
        # 随机生成一些卖出推荐
        self.test_sell_recommendations = []
        
        # 随机选择1-4个卖出推荐
        num_sells = np.random.randint(1, 5)
        indices = np.random.choice(len(test_symbols), num_sells, replace=False)
        
        for i in indices:
            symbol, name = test_symbols[i]
            price = np.random.random() * 100 + 10
            profit = f"+{np.random.random()*10:.1f}%" if np.random.random() > 0.3 else f"-{np.random.random()*5:.1f}%"
            conf = np.random.random() * 0.3 + 0.6
            
            self.test_sell_recommendations.append({
                "symbol": symbol,
                "name": name,
                "price": price,
                "reason": reasons[np.random.randint(0, len(reasons))],
                "profit": profit,
                "confidence": conf,
                "agent": agents[np.random.randint(0, len(agents))]
            })
        
        # 记录日志
        self.master.after(0, lambda: self.log(f"生成测试数据: {num_buys}个买入推荐, {num_sells}个卖出推荐"))
    
    def _update_test_ui(self):
        """更新测试模式的UI"""
        # 清空现有推荐
        for item in self.buy_tree.get_children():
            self.buy_tree.delete(item)
            
        for item in self.sell_tree.get_children():
            self.sell_tree.delete(item)
        
        # 更新交易推荐列表
        self.trading_recommendations = {
            "buy": self.test_buy_recommendations,
            "sell": self.test_sell_recommendations,
            "timestamp": datetime.now()
        }
        
        # 添加买入推荐到表格
        for rec in self.test_buy_recommendations:
            self.buy_tree.insert("", tk.END, values=(
                rec["symbol"],
                rec["name"],
                f"¥{rec['price']:.2f}",
                rec["reason"],
                f"¥{rec['target_price']:.2f}",
                f"{rec['confidence']*100:.0f}%",
                rec["agent"]
            ))
        
        # 添加卖出推荐到表格
        for rec in self.test_sell_recommendations:
            self.sell_tree.insert("", tk.END, values=(
                rec["symbol"],
                rec["name"],
                f"¥{rec['price']:.2f}",
                rec["reason"],
                rec["profit"],
                f"{rec['confidence']*100:.0f}%",
                rec["agent"]
            ))
        
        # 更新时间戳
        self.last_recommendation_time.config(
            text=f"上次更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # 随机生成详细系统日志信息
        actions = [
            ("分析市场状态", ["检查大盘指数趋势", "计算行业板块强弱", "评估市场情绪指标", "分析市场资金流向"]),
            ("检查交易机会", ["筛选突破个股", "识别趋势反转信号", "寻找超跌反弹机会", "发现成交量异常股"]),
            ("更新技术指标", ["更新MACD指标", "计算RSI超买超卖", "分析KDJ交叉信号", "检查布林带状态"]),
            ("执行风险评估", ["计算持仓风险敞口", "评估市场波动风险", "检查止损位置有效性", "分析系统风险敞口"]),
            ("调整智能体配置", ["增加趋势跟踪权重", "降低反转策略权重", "激活区间交易模式", "调整技术指标参数"]),
            ("扫描市场异动", ["检测大宗交易", "监控盘中异常波动", "分析高管增减持", "监测机构调研动向"])
        ]
        
        # 随机选择1-2个动作
        num_actions = np.random.randint(1, 3)
        for _ in range(num_actions):
            action_group = actions[np.random.randint(0, len(actions))]
            main_action = action_group[0]
            sub_actions = action_group[1]
            
            # 记录主要动作
            self.log(f"{main_action}...")
            
            # 随机选择1-3个子动作
            num_sub_actions = np.random.randint(1, 4)
            sub_indices = np.random.choice(len(sub_actions), num_sub_actions, replace=False)
            for j in sub_indices:
                sub_action = sub_actions[j]
                self.log(f"  - {sub_action}")
            
            # 随机暂停一小段时间，模拟处理过程
            time.sleep(np.random.random() * 0.5)
        
        # 随机更新市场状态
        market_states = ["活跃", "震荡", "上涨", "下跌", "横盘整理", "高波动"]
        market_state = market_states[np.random.randint(0, len(market_states))]
        self.market_status.config(text=f"测试模式 - 市场状态: {market_state}")
        
        # 更新图表
        self.update_chart()

    def _system_loop(self):
        """在新线程中运行系统循环"""
        try:
            while self.running:
                if hasattr(self.seto, 'market_state') and hasattr(self.seto.market_state, 'is_market_open'):
                    # 检查市场是否开放
                    market_open = self.seto.market_state.is_market_open()
                    
                    if market_open:
                        # 运行一个交易周期
                        try:
                            narrative = self.seto.run_once()
                            # 在主线程中更新日志
                            self.master.after(0, lambda n=narrative: self.log(n))
                            
                            # 获取交易推荐
                            self.master.after(0, self._get_recommendations)
                        except Exception as e:
                            error_msg = str(e)
                            self.master.after(0, lambda msg=error_msg: self.log(f"交易周期执行错误: {msg}"))
                    else:
                        # 市场关闭，记录日志
                        self.master.after(0, lambda: self.log("市场当前关闭，等待开盘..."))
                
                # 更新状态显示
                self.master.after(0, self.update_status)
                
                # 暂停一段时间
                update_interval = self.config.get('market', {}).get('update_interval', 5)
                time.sleep(update_interval)
                
        except Exception as e:
            error_msg = str(e)
            self.master.after(0, lambda msg=error_msg: self.log(f"系统运行错误: {msg}"))
            self.running = False
            self.master.after(0, self.update_status)
    
    def _get_recommendations(self):
        """获取并更新交易推荐"""
        try:
            # 清空现有推荐
            for item in self.buy_tree.get_children():
                self.buy_tree.delete(item)
                
            for item in self.sell_tree.get_children():
                self.sell_tree.delete(item)
            
            # 获取买入推荐
            buy_recommendations = []
            if hasattr(self.seto, 'coordinator') and hasattr(self.seto.coordinator, 'get_buy_recommendations'):
                buy_recommendations = self.seto.coordinator.get_buy_recommendations()
            else:
                # 使用模拟数据
                buy_recommendations = self._generate_fake_buy_recommendations()
            
            # 获取卖出推荐
            sell_recommendations = []
            if hasattr(self.seto, 'coordinator') and hasattr(self.seto.coordinator, 'get_sell_recommendations'):
                sell_recommendations = self.seto.coordinator.get_sell_recommendations()
            else:
                # 使用模拟数据
                sell_recommendations = self._generate_fake_sell_recommendations()
            
            # 更新推荐列表
            self.trading_recommendations = {
                "buy": buy_recommendations,
                "sell": sell_recommendations,
                "timestamp": datetime.now()
            }
            
            # 添加买入推荐到表格
            for rec in buy_recommendations:
                # 确保rec是字典并使用字典访问方式
                if not isinstance(rec, dict):
                    continue
                    
                self.buy_tree.insert("", tk.END, values=(
                    rec.get("symbol", ""),
                    rec.get("name", ""),
                    f"¥{rec.get('price', 0):.2f}",
                    rec.get("reason", ""),
                    f"¥{rec.get('target_price', 0):.2f}",
                    f"{rec.get('confidence', 0)*100:.0f}%",
                    rec.get("agent", "")
                ))
            
            # 添加卖出推荐到表格
            for rec in sell_recommendations:
                # 确保rec是字典并使用字典访问方式
                if not isinstance(rec, dict):
                    continue
                    
                self.sell_tree.insert("", tk.END, values=(
                    rec.get("symbol", ""),
                    rec.get("name", ""),
                    f"¥{rec.get('price', 0):.2f}",
                    rec.get("reason", ""),
                    rec.get("profit", "0%"),
                    f"{rec.get('confidence', 0)*100:.0f}%",
                    rec.get("agent", "")
                ))
            
            # 更新时间戳
            self.last_recommendation_time.config(
                text=f"上次更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
        except Exception as e:
            self.log(f"获取交易推荐失败: {e}")
    
    def _generate_fake_buy_recommendations(self):
        """生成模拟买入推荐用于测试"""
        return [
            {
                "symbol": "000001.SZ",
                "name": "平安银行",
                "price": 12.45,
                "reason": "突破20日均线，成交量放大",
                "target_price": 13.50,
                "confidence": 0.85,
                "agent": "趋势跟踪"
            },
            {
                "symbol": "600000.SH",
                "name": "浦发银行",
                "price": 8.76,
                "reason": "RSI超卖反弹信号",
                "target_price": 9.50,
                "confidence": 0.75,
                "agent": "反转交易"
            }
        ]
    
    def _generate_fake_sell_recommendations(self):
        """生成模拟卖出推荐用于测试"""
        return [
            {
                "symbol": "601318.SH", 
                "name": "中国平安",
                "price": 45.60,
                "reason": "MACD死叉，量能萎缩",
                "profit": "+8.5%",
                "confidence": 0.82,
                "agent": "趋势跟踪"
            }
        ]
        
    def refresh_recommendations(self):
        """手动刷新交易推荐"""
        self._get_recommendations()
        self.log("已手动刷新交易推荐")

def run_gui():
    """运行GUI界面"""
    root = tk.Tk()
    app = SetoGUI(root)
    root.mainloop()

if __name__ == "__main__":
    run_gui() 