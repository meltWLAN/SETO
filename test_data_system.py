#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal 数据管理系统测试脚本

用于测试数据预处理模块的功能和性能
"""

import os
import logging
import time
from datetime import datetime
import shutil

# 导入数据预处理模块
from seto_versal.data.preprocess import DataPreprocessor, run_test

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("data_system_test")

def setup_test_environment():
    """
    准备测试环境
    
    Returns:
        bool: 是否成功
    """
    logger.info("准备测试环境")
    
    # 创建测试目录
    test_dir = "data/test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    os.makedirs(test_dir, exist_ok=True)
    
    logger.info(f"创建测试目录: {test_dir}")
    return True

def cleanup_test_environment():
    """
    清理测试环境
    
    Returns:
        bool: 是否成功
    """
    logger.info("清理测试环境")
    
    # 选择性地清理测试目录
    # 为了查看结果，暂时保留测试目录
    # test_dir = "data/test"
    # if os.path.exists(test_dir):
    #     shutil.rmtree(test_dir)
    
    logger.info("测试环境清理完成（保留测试目录以供查看）")
    return True

def performance_test(processor):
    """
    性能测试
    
    Args:
        processor: 数据预处理器实例
        
    Returns:
        dict: 性能测试结果
    """
    logger.info("开始性能测试")
    
    results = {
        'data_creation_time': 0,
        'verification_time': 0
    }
    
    # 测试数据创建性能
    start_time = time.time()
    processor.test_data_creation()
    end_time = time.time()
    results['data_creation_time'] = end_time - start_time
    
    # 测试数据验证性能
    start_time = time.time()
    processor.verify_data()
    end_time = time.time()
    results['verification_time'] = end_time - start_time
    
    logger.info(f"性能测试结果: 数据创建耗时 {results['data_creation_time']:.2f}秒, "
              f"数据验证耗时 {results['verification_time']:.2f}秒")
    
    return results

def integration_test():
    """
    集成测试
    
    Returns:
        bool: 测试是否成功
    """
    logger.info("开始集成测试")
    
    # 模拟 MarketState 使用生成的数据
    try:
        # 加载股票列表
        import json
        stock_list_path = os.path.join("data", "market", "stock_list.json")
        
        if not os.path.exists(stock_list_path):
            logger.error(f"股票列表文件不存在: {stock_list_path}")
            return False
        
        with open(stock_list_path, 'r') as f:
            stocks = json.load(f)
        
        logger.info(f"加载股票列表成功，共 {len(stocks)} 只股票")
        
        # 模拟加载价格数据
        import pandas as pd
        successful_loads = 0
        
        for symbol in stocks:
            price_file = os.path.join("data", "market", "price", symbol, "daily.csv")
            try:
                df = pd.read_csv(price_file)
                if not df.empty:
                    successful_loads += 1
            except Exception as e:
                logger.warning(f"加载 {symbol} 价格数据失败: {e}")
        
        logger.info(f"成功加载 {successful_loads}/{len(stocks)} 只股票的价格数据")
        
        # 模拟加载基本面数据
        successful_loads = 0
        
        for symbol in stocks:
            fundamental_file = os.path.join("data", "market", "fundamental", symbol, "fundamental.json")
            try:
                with open(fundamental_file, 'r') as f:
                    data = json.load(f)
                if data:
                    successful_loads += 1
            except Exception as e:
                logger.warning(f"加载 {symbol} 基本面数据失败: {e}")
        
        logger.info(f"成功加载 {successful_loads}/{len(stocks)} 只股票的基本面数据")
        
        # 判断测试是否成功
        success = successful_loads > 0
        
        if success:
            logger.info("集成测试成功")
        else:
            logger.error("集成测试失败")
        
        return success
        
    except Exception as e:
        logger.error(f"集成测试失败: {e}")
        return False

def run_all_tests():
    """
    运行所有测试
    
    Returns:
        bool: 测试是否全部成功
    """
    logger.info("===== SETO-Versal 数据管理系统测试 =====")
    start_time = time.time()
    
    # 记录测试结果
    results = {
        'setup': False,
        'basic': False,
        'performance': {},
        'integration': False,
        'overall': False
    }
    
    try:
        # 1. 准备测试环境
        results['setup'] = setup_test_environment()
        if not results['setup']:
            logger.error("测试环境准备失败，终止测试")
            return False
        
        # 2. 运行基本功能测试
        logger.info("开始基本功能测试")
        
        # 创建专用于测试的配置
        config = {
            'data_dir': 'data/market',
            'test_mode': True
        }
        
        # 初始化预处理器
        processor = DataPreprocessor(config)
        
        # 运行基本功能测试
        results['basic'] = processor.test_data_creation() and processor.verify_data() is not None
        
        if not results['basic']:
            logger.error("基本功能测试失败，继续其他测试")
        
        # 3. 运行性能测试
        results['performance'] = performance_test(processor)
        
        # 4. 运行集成测试
        results['integration'] = integration_test()
        
        # 5. 显示测试摘要
        end_time = time.time()
        results['overall'] = results['basic'] and results['integration']
        
        logger.info("===== 测试摘要 =====")
        logger.info(f"测试环境准备: {'成功' if results['setup'] else '失败'}")
        logger.info(f"基本功能测试: {'成功' if results['basic'] else '失败'}")
        logger.info(f"性能测试: 数据创建耗时 {results['performance'].get('data_creation_time', 0):.2f}秒, "
                  f"数据验证耗时 {results['performance'].get('verification_time', 0):.2f}秒")
        logger.info(f"集成测试: {'成功' if results['integration'] else '失败'}")
        logger.info(f"总体结果: {'成功' if results['overall'] else '失败'}")
        logger.info(f"总耗时: {end_time - start_time:.2f}秒")
        
        # 6. 清理测试环境
        cleanup_test_environment()
        
        # 显示数据预处理摘要
        print("\n" + processor.generate_summary())
        
        return results['overall']
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        cleanup_test_environment()
        return False

if __name__ == "__main__":
    run_all_tests() 