#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化模块提供交易系统结果的图表、报表和仪表板展示功能。
"""

# 定义模块导出的内容
__all__ = []

# 尝试导入GUI仪表板
try:
    from seto_versal.visualization.gui_dashboard import SetoGUI, run_gui
    __all__.extend(['SetoGUI', 'run_gui'])
except ImportError:
    pass

# 尝试导入网页仪表板
try:
    from seto_versal.visualization.dashboard import Dashboard
    __all__.append('Dashboard')
except ImportError:
    pass

# 尝试导入图表功能
try:
    from seto_versal.visualization.charts import (
        ChartType, create_performance_chart, create_drawdown_chart,
        create_returns_distribution, create_monthly_returns_heatmap,
        create_benchmark_comparison
    )
    __all__.extend([
        'ChartType',
        'create_performance_chart',
        'create_drawdown_chart',
        'create_returns_distribution',
        'create_monthly_returns_heatmap',
        'create_benchmark_comparison'
    ])
except ImportError:
    pass

# 尝试导入报表功能
try:
    from seto_versal.visualization.reports import (
        create_performance_report, create_strategy_report,
        create_risk_report, export_report
    )
    __all__.extend([
        'create_performance_report',
        'create_strategy_report',
        'create_risk_report',
        'export_report'
    ])
except ImportError:
    pass 