#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据管理模块用于获取、处理和存储市场数据。
提供统一接口访问不同数据源的市场数据。
包含数据缓存、处理流水线和增强型数据管理功能。
"""

# 导入原始数据管理器组件
from seto_versal.data.manager import DataManager, DataSource, TimeFrame

# 定义__all__列表，初始包含基本组件
__all__ = ['DataManager', 'DataSource', 'TimeFrame']

# 使用try-except包装导入新模块，以兼容旧版本
try:
    # 导入数据缓存组件
    from seto_versal.data.cache import DataCache, CacheLevel, CachePolicy
    __all__.extend(['DataCache', 'CacheLevel', 'CachePolicy'])
except ImportError:
    pass

try:
    # 导入数据处理组件
    from seto_versal.data.processor import (
        DataProcessor, ProcessorType, DataCleaner, 
        FeatureGenerator, ProcessingPipeline
    )
    __all__.extend([
        'DataProcessor', 'ProcessorType', 'DataCleaner', 
        'FeatureGenerator', 'ProcessingPipeline'
    ])
except ImportError:
    pass

try:
    # 导入增强型数据管理器
    from seto_versal.data.enhanced_manager import EnhancedDataManager
    __all__.append('EnhancedDataManager')
except ImportError:
    pass

try:
    # 导入数据系统设置
    from seto_versal.data.setup import DataSystemSetup
    __all__.append('DataSystemSetup')
except ImportError:
    pass 