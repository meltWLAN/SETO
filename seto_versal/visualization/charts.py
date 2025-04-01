#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图表生成模块，提供各种交易系统性能和结果的可视化图表。
支持多种图表类型和展示方式。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta


class ChartType(Enum):
    """图表类型枚举"""
    LINE = "line"
    BAR = "bar"
    CANDLESTICK = "candlestick"
    AREA = "area"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    PIE = "pie"
    RADAR = "radar"


def create_performance_chart(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "策略收益曲线",
    figsize: Tuple[int, int] = (12, 6),
    chart_type: ChartType = ChartType.LINE,
    colors: List[str] = ["#0066CC", "#FF9900"],
    cumulative: bool = True,
    log_scale: bool = False,
    include_drawdown: bool = False,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    创建策略性能图表，展示收益曲线及可选的基准比较
    
    Args:
        returns: 策略收益率序列，索引为日期
        benchmark_returns: 基准收益率序列，可选
        title: 图表标题
        figsize: 图表尺寸
        chart_type: 图表类型
        colors: 颜色列表，第一个用于策略，第二个用于基准
        cumulative: 是否显示累计收益
        log_scale: 是否使用对数比例
        include_drawdown: 是否包含回撤子图
        save_path: 保存路径，如果提供则保存图表
        
    Returns:
        matplotlib Figure对象
    """
    # 验证输入
    if returns is None or len(returns) == 0:
        raise ValueError("收益率数据不能为空")
    
    # 如果是累计收益，计算累计收益率
    if cumulative:
        returns_plot = (1 + returns).cumprod() - 1
        if benchmark_returns is not None:
            benchmark_plot = (1 + benchmark_returns).cumprod() - 1
    else:
        returns_plot = returns
        benchmark_plot = benchmark_returns
    
    # 创建图表
    if include_drawdown:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    # 绘制策略收益
    if chart_type == ChartType.LINE:
        ax1.plot(returns_plot.index, returns_plot.values * 100, label='策略收益', color=colors[0], linewidth=2)
    elif chart_type == ChartType.AREA:
        ax1.fill_between(returns_plot.index, returns_plot.values * 100, 0, alpha=0.3, color=colors[0], label='策略收益')
    
    # 添加基准比较
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        ax1.plot(benchmark_plot.index, benchmark_plot.values * 100, label='基准收益', color=colors[1], linewidth=1.5, linestyle='--')
    
    # 设置x轴格式
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 设置y轴格式
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter())
    
    # 添加网格和图例
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')
    
    # 设置标题和标签
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('收益率 (%)', fontsize=12)
    
    # 设置对数比例
    if log_scale:
        ax1.set_yscale('symlog')
    
    # 如果包含回撤子图，绘制回撤
    if include_drawdown:
        # 计算回撤
        if cumulative:
            dd = returns_plot.div(returns_plot.cummax()) - 1
        else:
            cum_returns = (1 + returns).cumprod() - 1
            dd = cum_returns.div(cum_returns.cummax()) - 1
        
        # 绘制回撤
        ax2.fill_between(dd.index, dd.values * 100, 0, color='red', alpha=0.3, label='回撤')
        ax2.set_ylabel('回撤 (%)', fontsize=12)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_drawdown_chart(
    returns: pd.Series,
    title: str = "策略回撤分析",
    figsize: Tuple[int, int] = (12, 6),
    colors: List[str] = ["#FF3333", "#0066CC"],
    include_underwater: bool = True,
    show_max_drawdown: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    创建策略回撤分析图表
    
    Args:
        returns: 策略收益率序列，索引为日期
        title: 图表标题
        figsize: 图表尺寸
        colors: 颜色列表
        include_underwater: 是否包含水下时间分析
        show_max_drawdown: 是否显示最大回撤
        save_path: 保存路径，如果提供则保存图表
        
    Returns:
        matplotlib Figure对象
    """
    # 验证输入
    if returns is None or len(returns) == 0:
        raise ValueError("收益率数据不能为空")
    
    # 计算累计收益和回撤
    cum_returns = (1 + returns).cumprod() - 1
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns / running_max) - 1
    
    # 创建图表
    if include_underwater:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    # 绘制回撤
    ax1.fill_between(drawdown.index, drawdown.values * 100, 0, color=colors[0], alpha=0.5)
    ax1.set_ylabel('回撤 (%)', fontsize=12)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    
    # 设置x轴格式
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 添加最大回撤标记
    if show_max_drawdown:
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax1.scatter(max_dd_date, max_dd * 100, color='black', s=50, zorder=5)
        ax1.annotate(f'最大回撤: {max_dd*100:.2f}%',
                     xy=(max_dd_date, max_dd * 100),
                     xytext=(max_dd_date + pd.Timedelta(days=30), max_dd * 100 * 0.5),
                     arrowprops=dict(arrowstyle="->", color='black'),
                     fontsize=10, fontweight='bold')
    
    # 如果包含水下时间分析
    if include_underwater:
        # 计算水下时间
        underwater = drawdown < 0
        underwater_periods = []
        current_period = {'start': None, 'end': None, 'duration': 0}
        
        for date, value in underwater.items():
            if value:
                if current_period['start'] is None:
                    current_period['start'] = date
            else:
                if current_period['start'] is not None:
                    current_period['end'] = date
                    current_period['duration'] = (current_period['end'] - current_period['start']).days
                    underwater_periods.append(current_period.copy())
                    current_period = {'start': None, 'end': None, 'duration': 0}
        
        # 如果最后一个周期还没有结束
        if current_period['start'] is not None:
            current_period['end'] = underwater.index[-1]
            current_period['duration'] = (current_period['end'] - current_period['start']).days
            underwater_periods.append(current_period)
        
        # 绘制水下时间柱状图
        if underwater_periods:
            durations = [period['duration'] for period in underwater_periods]
            starts = [period['start'] for period in underwater_periods]
            ax2.bar(starts, durations, width=10, color=colors[1], alpha=0.7)
            ax2.set_ylabel('水下天数', fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # 添加最长水下期标记
            if durations:
                max_duration_idx = np.argmax(durations)
                max_duration = durations[max_duration_idx]
                max_duration_start = starts[max_duration_idx]
                ax2.annotate(f'最长水下期: {max_duration}天',
                             xy=(max_duration_start, max_duration),
                             xytext=(max_duration_start + pd.Timedelta(days=30), max_duration * 0.8),
                             arrowprops=dict(arrowstyle="->", color='black'),
                             fontsize=10, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_returns_distribution(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "收益率分布",
    figsize: Tuple[int, int] = (12, 6),
    bins: int = 50,
    colors: List[str] = ["#0066CC", "#FF9900"],
    include_stats: bool = True,
    show_normal: bool = True,
    freq: str = 'D',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    创建收益率分布直方图
    
    Args:
        returns: 策略收益率序列，索引为日期
        benchmark_returns: 基准收益率序列，可选
        title: 图表标题
        figsize: 图表尺寸
        bins: 直方图的箱数
        colors: 颜色列表
        include_stats: 是否包含统计信息
        show_normal: 是否显示正态分布曲线
        freq: 收益率频率，用于标题，例如'D'表示日度，'M'表示月度
        save_path: 保存路径，如果提供则保存图表
        
    Returns:
        matplotlib Figure对象
    """
    # 验证输入
    if returns is None or len(returns) == 0:
        raise ValueError("收益率数据不能为空")
    
    # 将收益率转换为百分比
    returns_pct = returns * 100
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制收益率分布
    sns.histplot(returns_pct, bins=bins, kde=True, color=colors[0], alpha=0.7, 
                 label='策略收益率', ax=ax)
    
    # 添加基准比较
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        benchmark_pct = benchmark_returns * 100
        sns.histplot(benchmark_pct, bins=bins, kde=True, color=colors[1], alpha=0.5, 
                    label='基准收益率', ax=ax)
    
    # 添加正态分布曲线
    if show_normal:
        x = np.linspace(min(returns_pct), max(returns_pct), 1000)
        y = np.exp(-(x - np.mean(returns_pct))**2 / (2 * np.var(returns_pct))) / np.sqrt(2 * np.pi * np.var(returns_pct))
        y = y * (np.max(np.histogram(returns_pct, bins=bins)[0]) / np.max(y))
        ax.plot(x, y, color='red', linestyle='--', alpha=0.7, label='正态分布')
    
    # 添加统计信息
    if include_stats:
        stats_text = f"均值: {np.mean(returns_pct):.2f}%\n"
        stats_text += f"标准差: {np.std(returns_pct):.2f}%\n"
        stats_text += f"偏度: {pd.Series(returns_pct).skew():.2f}\n"
        stats_text += f"峰度: {pd.Series(returns_pct).kurtosis():.2f}\n"
        stats_text += f"最小值: {np.min(returns_pct):.2f}%\n"
        stats_text += f"最大值: {np.max(returns_pct):.2f}%"
        
        # 使用文本框显示统计信息
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 设置轴标签和标题
    freq_map = {'D': '日', 'W': '周', 'M': '月', 'Q': '季', 'Y': '年'}
    freq_label = freq_map.get(freq, freq)
    ax.set_xlabel(f'{freq_label}收益率 (%)', fontsize=12)
    ax.set_ylabel('频率', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 添加网格和图例
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_monthly_returns_heatmap(
    returns: pd.Series,
    title: str = "月度收益率热图",
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "RdYlGn",
    annot: bool = True,
    fmt: str = ".2f",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    创建月度收益率热图
    
    Args:
        returns: 策略收益率序列，索引为日期
        title: 图表标题
        figsize: 图表尺寸
        cmap: 颜色映射
        annot: 是否在单元格中显示数值
        fmt: 数值格式字符串
        save_path: 保存路径，如果提供则保存图表
        
    Returns:
        matplotlib Figure对象
    """
    # 验证输入
    if returns is None or len(returns) == 0:
        raise ValueError("收益率数据不能为空")
    
    # 确保索引为日期类型
    if not isinstance(returns.index, pd.DatetimeIndex):
        try:
            returns.index = pd.to_datetime(returns.index)
        except:
            raise ValueError("收益率数据的索引必须是可转换为日期的类型")
    
    # 计算月度收益率
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # 创建年月矩阵
    monthly_returns_matrix = pd.DataFrame()
    for year in range(monthly_returns.index.year.min(), monthly_returns.index.year.max() + 1):
        year_returns = monthly_returns[monthly_returns.index.year == year]
        monthly_dict = {}
        for month in range(1, 13):
            month_data = year_returns[year_returns.index.month == month]
            if not month_data.empty:
                monthly_dict[month] = month_data.iloc[0] * 100  # 转换为百分比
            else:
                monthly_dict[month] = np.nan
        monthly_returns_matrix = pd.concat([monthly_returns_matrix, pd.DataFrame(monthly_dict, index=[year])])
    
    # 设置月份列名
    monthly_returns_matrix.columns = ['一月', '二月', '三月', '四月', '五月', '六月',
                                      '七月', '八月', '九月', '十月', '十一月', '十二月']
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热图
    sns.heatmap(monthly_returns_matrix, cmap=cmap, annot=annot, fmt=fmt, linewidths=0.5,
                cbar_kws={'label': '收益率 (%)'},
                annot_kws={"size": 9},
                ax=ax)
    
    # 添加年度收益率
    yearly_returns = monthly_returns_matrix.mean(axis=1).values
    for i, year_return in enumerate(yearly_returns):
        ax.text(monthly_returns_matrix.shape[1] + 0.5, i + 0.5, f"{year_return:.2f}%", 
                va='center', ha='left', fontweight='bold')
    
    # 添加月度平均收益率
    monthly_avg = monthly_returns_matrix.mean(axis=0).values
    for i, month_avg in enumerate(monthly_avg):
        ax.text(i + 0.5, monthly_returns_matrix.shape[0] + 0.5, f"{month_avg:.2f}%", 
                va='top', ha='center', fontweight='bold')
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_benchmark_comparison(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    title: str = "策略与基准比较",
    figsize: Tuple[int, int] = (12, 6),
    colors: List[str] = ["#0066CC", "#FF9900"],
    metrics: List[str] = ["cumulative_returns", "rolling_beta", "rolling_alpha"],
    rolling_window: int = 60,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    创建策略与基准比较图表
    
    Args:
        returns: 策略收益率序列，索引为日期
        benchmark_returns: 基准收益率序列
        title: 图表标题
        figsize: 图表尺寸
        colors: 颜色列表
        metrics: 要显示的指标列表
        rolling_window: 滚动窗口大小（天数）
        save_path: 保存路径，如果提供则保存图表
        
    Returns:
        matplotlib Figure对象
    """
    # 验证输入
    if returns is None or len(returns) == 0 or benchmark_returns is None or len(benchmark_returns) == 0:
        raise ValueError("策略和基准收益率数据不能为空")
    
    # 计算各种指标
    metrics_data = {}
    
    # 累计收益
    if "cumulative_returns" in metrics:
        metrics_data["累计收益"] = {
            "strategy": (1 + returns).cumprod() - 1,
            "benchmark": (1 + benchmark_returns).cumprod() - 1
        }
    
    # 滚动beta
    if "rolling_beta" in metrics:
        # 对齐数据
        common_index = returns.index.intersection(benchmark_returns.index)
        aligned_returns = returns.loc[common_index]
        aligned_benchmark = benchmark_returns.loc[common_index]
        
        # 计算滚动beta
        rolling_beta = pd.Series(index=common_index)
        for i in range(rolling_window, len(common_index)):
            window_returns = aligned_returns.iloc[i-rolling_window:i]
            window_benchmark = aligned_benchmark.iloc[i-rolling_window:i]
            cov = np.cov(window_returns, window_benchmark)[0, 1]
            var = np.var(window_benchmark)
            beta = cov / var if var != 0 else np.nan
            rolling_beta.iloc[i] = beta
        
        metrics_data["滚动Beta"] = {
            "data": rolling_beta
        }
    
    # 滚动alpha
    if "rolling_alpha" in metrics:
        # 对齐数据
        common_index = returns.index.intersection(benchmark_returns.index)
        aligned_returns = returns.loc[common_index]
        aligned_benchmark = benchmark_returns.loc[common_index]
        
        # 计算滚动alpha
        rolling_alpha = pd.Series(index=common_index)
        for i in range(rolling_window, len(common_index)):
            window_returns = aligned_returns.iloc[i-rolling_window:i]
            window_benchmark = aligned_benchmark.iloc[i-rolling_window:i]
            # 使用CAPM模型计算alpha
            beta = np.cov(window_returns, window_benchmark)[0, 1] / np.var(window_benchmark)
            mean_return = window_returns.mean()
            mean_benchmark = window_benchmark.mean()
            alpha = mean_return - beta * mean_benchmark
            rolling_alpha.iloc[i] = alpha * 252  # 年化alpha
        
        metrics_data["滚动Alpha"] = {
            "data": rolling_alpha * 100  # 转换为百分比
        }
    
    # 确定子图数量
    num_plots = len(metrics_data)
    
    # 创建图表
    fig, axes = plt.subplots(num_plots, 1, figsize=(figsize[0], figsize[1] * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]
    
    # 绘制各指标
    for i, (metric_name, metric_data) in enumerate(metrics_data.items()):
        ax = axes[i]
        
        if metric_name == "累计收益":
            ax.plot(metric_data["strategy"].index, metric_data["strategy"].values * 100, 
                    label='策略', color=colors[0], linewidth=2)
            ax.plot(metric_data["benchmark"].index, metric_data["benchmark"].values * 100, 
                    label='基准', color=colors[1], linewidth=1.5, linestyle='--')
            ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        else:
            ax.plot(metric_data["data"].index, metric_data["data"].values, 
                    label=metric_name, color=colors[0], linewidth=1.5)
            
            # 对于beta，添加1.0的基准线
            if metric_name == "滚动Beta":
                ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            
            # 对于alpha，添加0.0的基准线
            if metric_name == "滚动Alpha":
                ax.axhline(y=0.0, color='black', linestyle='--', alpha=0.5)
                ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        
        # 设置格式
        ax.set_ylabel(metric_name, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper left')
        
        # 设置日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # 设置标题和底部标签
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    axes[-1].set_xlabel('日期', fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 