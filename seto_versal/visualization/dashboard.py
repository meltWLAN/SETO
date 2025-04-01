#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal 交易系统界面
提供基于Streamlit的交互式仪表板，用于运行和监控交易系统
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yaml
import sys
import threading
import time
from datetime import datetime

# 导入SETO-Versal系统组件
from seto_versal.main import SetoVersal
from seto_versal.utils.config import load_config

class Dashboard:
    """基于Streamlit的交易系统界面"""
    
    def __init__(self):
        """初始化仪表板"""
        self.seto = None
        self.running = False
        self.thread = None
        self.config = None
        
    def load_config(self, config_path='config.yaml'):
        """加载配置文件"""
        try:
            self.config = load_config(config_path)
            return True
        except Exception as e:
            st.error(f"配置加载错误: {e}")
            return False
            
    def start_system(self, config_path='config.yaml', mode=None):
        """初始化并启动交易系统"""
        try:
            # 创建SETO实例
            self.seto = SetoVersal(config_path)
            
            # 覆盖运行模式
            if mode:
                self.seto.config['system']['mode'] = mode
                
            # 在新线程中运行系统
            self.running = True
            self.thread = threading.Thread(target=self._run_system)
            self.thread.daemon = True
            self.thread.start()
            
            return True
        except Exception as e:
            st.error(f"系统启动错误: {e}")
            return False
    
    def _run_system(self):
        """在后台线程中运行SETO系统"""
        try:
            while self.running:
                if self.seto.market_state.is_market_open():
                    narrative = self.seto.run_once()
                    # 记录到运行日志
                    with open("logs/dashboard_run.log", "a") as f:
                        f.write(f"{datetime.now()}: {narrative}\n")
                    
                    # 运行进化逻辑
                    if self.seto.market_state.should_evolve():
                        self.seto.evolution_engine.evolve(
                            self.seto.agents,
                            {'feedback_analyzer': self.seto.feedback_analyzer}
                        )
                
                time.sleep(self.seto.config['market']['update_interval'])
        except Exception as e:
            with open("logs/dashboard_error.log", "a") as f:
                f.write(f"{datetime.now()}: Error in system thread: {e}\n")
            self.running = False
    
    def stop_system(self):
        """停止交易系统"""
        if self.seto and self.running:
            self.running = False
            self.seto.stop()
            if self.thread:
                self.thread.join(timeout=2.0)
            return True
        return False
    
    def render(self):
        """渲染Streamlit界面"""
        st.set_page_config(
            page_title="SETO-Versal 交易系统",
            page_icon="📈",
            layout="wide"
        )
        
        # 头部
        st.title("SETO-Versal 交易系统")
        st.markdown("*自进化交易智能体系统*")
        
        # 侧边栏 - 控制面板
        with st.sidebar:
            st.header("控制面板")
            
            # 配置选择
            config_path = st.text_input("配置文件路径", "config.yaml")
            
            # 运行模式
            mode = st.selectbox(
                "运行模式",
                options=["backtest", "paper", "live"],
                index=0
            )
            
            # 加载配置按钮
            if st.button("加载配置"):
                if self.load_config(config_path):
                    st.success("配置加载成功")
                    # 显示一些关键配置信息
                    if self.config:
                        st.info(f"系统版本: {self.config['system'].get('version', '未知')}")
                        st.info(f"市场: {self.config['market'].get('market_type', '未知')}")
            
            # 启动/停止按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("启动系统", disabled=self.running):
                    if self.start_system(config_path, mode):
                        st.success("系统已启动")
            with col2:
                if st.button("停止系统", disabled=not self.running):
                    if self.stop_system():
                        st.success("系统已停止")
            
            # 系统状态
            st.subheader("系统状态")
            if self.running:
                st.success("● 运行中")
            else:
                st.error("● 已停止")
        
        # 主界面 - 仪表盘与监控
        tab1, tab2, tab3 = st.tabs(["系统概览", "交易记录", "智能体状态"])
        
        # 系统概览标签页
        with tab1:
            st.header("系统概览")
            
            # 系统指标
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="账户余额", value="¥100,000", delta="0")
            with col2:
                st.metric(label="今日盈亏", value="¥500", delta="0.5%")
            with col3:
                st.metric(label="持仓数量", value="3")
            with col4:
                st.metric(label="智能体数量", value="5")
            
            # 市场状态
            st.subheader("市场状态")
            if self.seto and hasattr(self.seto, 'market_state'):
                is_open = self.seto.market_state.is_market_open()
                st.info("市场状态: " + ("开盘" if is_open else "闭市"))
            else:
                st.info("市场状态: 未连接")
            
            # 策略性能图表 (示例数据)
            st.subheader("策略性能")
            
            # 创建示例数据
            dates = pd.date_range(start='2023-01-01', end='2023-01-30')
            equity = 100000 + pd.Series(range(30)) * 100 + pd.Series(pd.np.random.randn(30) * 500)
            benchmark = 100000 + pd.Series(range(30)) * 80 + pd.Series(pd.np.random.randn(30) * 400)
            
            df = pd.DataFrame({
                'date': dates,
                'equity': equity,
                'benchmark': benchmark
            })
            
            # 绘制权益曲线
            fig = px.line(df, x='date', y=['equity', 'benchmark'], 
                title='账户权益曲线',
                labels={'value': '权益', 'date': '日期', 'variable': '类型'},
                color_discrete_map={'equity': 'blue', 'benchmark': 'red'})
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 交易记录标签页
        with tab2:
            st.header("交易记录")
            
            # 示例交易数据
            data = {
                "日期": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
                "股票": ["000001.SZ", "600000.SH", "000001.SZ", "601318.SH"],
                "操作": ["买入", "买入", "卖出", "买入"],
                "价格": [12.5, 8.3, 13.1, 45.2],
                "数量": [1000, 2000, 1000, 500],
                "金额": [12500, 16600, 13100, 22600],
                "智能体": ["趋势跟踪", "逆势交易", "趋势跟踪", "防御型"],
            }
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        
        # 智能体状态标签页
        with tab3:
            st.header("智能体状态")
            
            # 智能体性能指标
            data = {
                "智能体": ["趋势跟踪", "逆势交易", "快速获利", "防御型", "板块轮动"],
                "交易次数": [15, 8, 23, 5, 12],
                "胜率": [0.6, 0.75, 0.52, 0.8, 0.67],
                "盈亏比": [1.5, 1.2, 0.9, 2.1, 1.3],
                "累计收益": [2500, 1800, 1200, 2600, 2100],
            }
            
            df = pd.DataFrame(data)
            
            # 显示数据表
            st.dataframe(df, use_container_width=True)
            
            # 胜率可视化
            fig = px.bar(df, x='智能体', y='胜率', title='智能体胜率比较',
                        color='胜率', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
            
            # 累计收益可视化
            fig = px.bar(df, x='智能体', y='累计收益', title='智能体累计收益',
                        color='累计收益', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

def run_dashboard():
    """运行仪表板"""
    dashboard = Dashboard()
    dashboard.render()

if __name__ == "__main__":
    run_dashboard() 