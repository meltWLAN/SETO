#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal 数据预处理模块

负责在系统启动前获取、验证和存储必要的市场数据。
提供简化的测试功能来验证数据管理系统的可行性。
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """数据预处理器 - 测试版"""
    
    def __init__(self, config=None):
        """初始化数据预处理器"""
        self.config = config or {}
        self.data_dir = self.config.get('data_dir', 'data/market')
        
        # 创建目录结构
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'price'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'fundamental'), exist_ok=True)
        
        # 初始化状态
        self.status = {
            'initialized': datetime.now().isoformat(),
            'last_update': None,
            'test_mode': True
        }
        
        logger.info(f"数据预处理器初始化完成，数据目录: {self.data_dir}")
    
    def test_data_creation(self):
        """
        测试数据创建功能
        
        Returns:
            bool: 测试是否成功
        """
        logger.info("开始测试数据创建")
        
        try:
            # 1. 创建样本股票列表
            self._create_sample_stock_list()
            
            # 2. 创建样本价格数据
            self._create_sample_price_data()
            
            # 3. 创建样本基本面数据
            self._create_sample_fundamental_data()
            
            # 更新状态
            self.status['last_update'] = datetime.now().isoformat()
            self.status['test_status'] = 'completed'
            self._save_status()
            
            logger.info("数据创建测试完成")
            return True
            
        except Exception as e:
            logger.error(f"数据创建测试失败: {e}")
            self.status['test_status'] = 'failed'
            self.status['error'] = str(e)
            self._save_status()
            return False
    
    def _create_sample_stock_list(self):
        """创建样本股票列表"""
        # 样本股票
        sample_stocks = [
            '000001.SZ', '000333.SZ', '000651.SZ', '000858.SZ',
            '600000.SH', '600036.SH', '600276.SH', '600519.SH',
            '601318.SH', '601888.SH'
        ]
        
        # 保存股票列表
        with open(os.path.join(self.data_dir, 'stock_list.json'), 'w') as f:
            json.dump(sample_stocks, f)
        
        logger.info(f"创建样本股票列表，共 {len(sample_stocks)} 只股票")
    
    def _create_sample_price_data(self):
        """创建样本价格数据"""
        # 样本股票
        with open(os.path.join(self.data_dir, 'stock_list.json'), 'r') as f:
            sample_stocks = json.load(f)
        
        # 生成日期范围 (90天)
        end_date = datetime.now()
        dates = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(90)]
        dates.reverse()  # 从旧到新排序
        
        for symbol in sample_stocks:
            # 为每只股票创建目录
            stock_dir = os.path.join(self.data_dir, 'price', symbol)
            os.makedirs(stock_dir, exist_ok=True)
            
            # 初始价格和随机种子
            base_price = np.random.uniform(10, 100)
            np.random.seed(int(symbol.split('.')[0][-4:]))
            
            # 生成价格数据
            price_data = []
            price = base_price
            volume = np.random.uniform(1000000, 5000000)
            
            for date in dates:
                # 随机价格变动 (-2% ~ +2%)
                change_pct = np.random.uniform(-0.02, 0.02)
                price = price * (1 + change_pct)
                
                # 随机成交量变动 (-20% ~ +20%)
                vol_change = np.random.uniform(0.8, 1.2)
                daily_volume = volume * vol_change
                
                # 构建日线数据
                daily_data = {
                    'date': date,
                    'open': round(price * (1 - np.random.uniform(0, 0.01)), 2),
                    'high': round(price * (1 + np.random.uniform(0, 0.02)), 2),
                    'low': round(price * (1 - np.random.uniform(0, 0.02)), 2),
                    'close': round(price, 2),
                    'volume': int(daily_volume),
                    'symbol': symbol
                }
                
                price_data.append(daily_data)
            
            # 创建DataFrame并保存
            df = pd.DataFrame(price_data)
            df.to_csv(os.path.join(stock_dir, 'daily.csv'), index=False)
            
            logger.info(f"创建 {symbol} 的样本价格数据，共 {len(df)} 条记录")
    
    def _create_sample_fundamental_data(self):
        """创建样本基本面数据"""
        # 样本股票
        with open(os.path.join(self.data_dir, 'stock_list.json'), 'r') as f:
            sample_stocks = json.load(f)
        
        for symbol in sample_stocks:
            # 为每只股票创建目录
            stock_dir = os.path.join(self.data_dir, 'fundamental', symbol)
            os.makedirs(stock_dir, exist_ok=True)
            
            # 设置随机种子以保持一致性
            np.random.seed(int(symbol.split('.')[0][-4:]))
            
            # 创建基本面数据
            fundamental_data = {
                'symbol': symbol,
                'pe_ratio': round(np.random.uniform(10, 30), 2),
                'pb_ratio': round(np.random.uniform(1, 5), 2),
                'roe': round(np.random.uniform(0.05, 0.25), 4),
                'revenue_growth': round(np.random.uniform(0.05, 0.35), 4),
                'profit_margin': round(np.random.uniform(0.05, 0.25), 4),
                'debt_to_equity': round(np.random.uniform(0.2, 1.5), 2),
                'current_ratio': round(np.random.uniform(1.0, 3.0), 2),
                'dividend_yield': round(np.random.uniform(0, 0.05), 4),
                'market_cap': round(np.random.uniform(1, 100) * 1e9, 2),
                'updated_at': datetime.now().strftime('%Y-%m-%d')
            }
            
            # 保存为JSON文件
            with open(os.path.join(stock_dir, 'fundamental.json'), 'w') as f:
                json.dump(fundamental_data, f, indent=2)
            
            logger.info(f"创建 {symbol} 的样本基本面数据")
    
    def _save_status(self):
        """保存处理状态"""
        try:
            status_path = os.path.join(self.data_dir, 'preprocess_status.json')
            with open(status_path, 'w') as f:
                json.dump(self.status, f, indent=2)
            logger.debug(f"已保存预处理状态到 {status_path}")
        except Exception as e:
            logger.error(f"保存预处理状态失败: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前处理状态"""
        return self.status

    def verify_data(self):
        """
        验证数据完整性
        
        Returns:
            Dict: 验证结果
        """
        logger.info("开始验证数据")
        
        result = {
            'stock_list_exists': False,
            'price_data_exists': False,
            'fundamental_data_exists': False,
            'data_count': {
                'stocks': 0,
                'price_files': 0,
                'fundamental_files': 0
            }
        }
        
        # 检查股票列表
        stock_list_path = os.path.join(self.data_dir, 'stock_list.json')
        result['stock_list_exists'] = os.path.exists(stock_list_path)
        
        if result['stock_list_exists']:
            # 读取股票列表
            with open(stock_list_path, 'r') as f:
                stocks = json.load(f)
            
            result['data_count']['stocks'] = len(stocks)
            
            # 检查每只股票的数据
            for symbol in stocks:
                # 价格数据
                price_file = os.path.join(self.data_dir, 'price', symbol, 'daily.csv')
                if os.path.exists(price_file):
                    result['data_count']['price_files'] += 1
                
                # 基本面数据
                fundamental_file = os.path.join(self.data_dir, 'fundamental', symbol, 'fundamental.json')
                if os.path.exists(fundamental_file):
                    result['data_count']['fundamental_files'] += 1
            
            # 判断数据是否存在
            result['price_data_exists'] = result['data_count']['price_files'] > 0
            result['fundamental_data_exists'] = result['data_count']['fundamental_files'] > 0
        
        # 更新状态
        self.status['verification'] = result
        self.status['verified_at'] = datetime.now().isoformat()
        self._save_status()
        
        logger.info(f"数据验证完成: 发现 {result['data_count']['stocks']} 只股票, " 
                  f"{result['data_count']['price_files']} 个价格文件, "
                  f"{result['data_count']['fundamental_files']} 个基本面数据文件")
        
        return result
    
    def generate_summary(self) -> str:
        """生成数据摘要"""
        verification = self.status.get('verification', {})
        data_count = verification.get('data_count', {})
        
        summary = [
            f"数据预处理测试摘要 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]",
            f"===================================================",
            f"测试模式: {self.status.get('test_mode', False)}",
            f"数据目录: {self.data_dir}",
            f"股票数量: {data_count.get('stocks', 0)}",
            f"价格数据文件: {data_count.get('price_files', 0)}",
            f"基本面数据文件: {data_count.get('fundamental_files', 0)}",
            f"",
            f"数据完整性检查:",
            f"  - 股票列表: {'✓' if verification.get('stock_list_exists', False) else '✗'}",
            f"  - 价格数据: {'✓' if verification.get('price_data_exists', False) else '✗'}",
            f"  - 基本面数据: {'✓' if verification.get('fundamental_data_exists', False) else '✗'}",
            f"",
            f"上次更新: {self.status.get('last_update', '未更新')}",
            f"===================================================",
        ]
        
        return "\n".join(summary)


def run_test():
    """
    运行数据预处理测试
    
    Returns:
        bool: 测试是否成功
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("开始数据预处理测试")
    
    try:
        # 创建测试配置
        config = {
            'data_dir': 'data/market',
            'test_mode': True
        }
        
        # 初始化预处理器
        processor = DataPreprocessor(config)
        
        # 运行测试数据创建
        if not processor.test_data_creation():
            logger.error("测试数据创建失败")
            return False
        
        # 验证数据
        verification = processor.verify_data()
        
        # 显示摘要
        summary = processor.generate_summary()
        print(summary)
        
        # 判断测试是否成功
        success = (verification.get('stock_list_exists', False) and 
                 verification.get('price_data_exists', False) and 
                 verification.get('fundamental_data_exists', False))
        
        if success:
            logger.info("数据预处理测试成功")
        else:
            logger.warning("数据预处理测试未完全成功，请检查详细日志")
        
        return success
    
    except Exception as e:
        logger.error(f"数据预处理测试失败: {e}")
        return False


if __name__ == "__main__":
    run_test() 