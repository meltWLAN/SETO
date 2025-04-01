#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据管理模块的测试文件
"""

import unittest
import os
import pandas as pd
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from seto_versal.data.manager import TimeFrame, CSVDataSource, DataManager


class TestTimeFrame(unittest.TestCase):
    """TimeFrame枚举的测试用例"""
    
    def test_string_conversion(self):
        """测试字符串转换"""
        self.assertEqual(str(TimeFrame.MINUTE_1), "1m")
        self.assertEqual(str(TimeFrame.HOUR_1), "1h")
        self.assertEqual(str(TimeFrame.DAY_1), "1d")
    
    def test_from_string(self):
        """测试从字符串创建枚举"""
        self.assertEqual(TimeFrame.from_string("1m"), TimeFrame.MINUTE_1)
        self.assertEqual(TimeFrame.from_string("1h"), TimeFrame.HOUR_1)
        self.assertEqual(TimeFrame.from_string("1d"), TimeFrame.DAY_1)
        
        with self.assertRaises(ValueError):
            TimeFrame.from_string("invalid")
    
    def test_to_minutes(self):
        """测试转换为分钟数"""
        self.assertEqual(TimeFrame.MINUTE_1.to_minutes(), 1)
        self.assertEqual(TimeFrame.MINUTE_5.to_minutes(), 5)
        self.assertEqual(TimeFrame.HOUR_1.to_minutes(), 60)
        self.assertEqual(TimeFrame.DAY_1.to_minutes(), 1440)


class TestCSVDataSource(unittest.TestCase):
    """CSVDataSource类的测试用例"""
    
    def setUp(self):
        """创建测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试数据
        self.create_test_data()
        
        # 创建数据源
        config = {'data_path': self.temp_dir}
        self.data_source = CSVDataSource("test_csv", config)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_data(self):
        """创建测试数据文件"""
        # 创建AAPL日线数据
        aapl_daily = pd.DataFrame({
            'timestamp': [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3)
            ],
            'open': [150.0, 151.0, 152.0],
            'high': [155.0, 156.0, 157.0],
            'low': [149.0, 150.0, 151.0],
            'close': [153.0, 154.0, 155.0],
            'volume': [1000000, 1100000, 1200000]
        })
        
        aapl_daily.to_csv(os.path.join(self.temp_dir, "AAPL_1d.csv"), index=False)
        
        # 创建MSFT小时数据
        msft_hourly = pd.DataFrame({
            'timestamp': [
                datetime(2023, 1, 1, 10, 0),
                datetime(2023, 1, 1, 11, 0),
                datetime(2023, 1, 1, 12, 0)
            ],
            'open': [250.0, 251.0, 252.0],
            'high': [255.0, 256.0, 257.0],
            'low': [249.0, 250.0, 251.0],
            'close': [253.0, 254.0, 255.0],
            'volume': [500000, 510000, 520000]
        })
        
        msft_hourly.to_csv(os.path.join(self.temp_dir, "MSFT_1h.csv"), index=False)
    
    def test_connect(self):
        """测试连接功能"""
        self.assertTrue(self.data_source.connect())
        
        # 测试不存在的目录
        invalid_source = CSVDataSource("invalid", {'data_path': '/non/existent/path'})
        self.assertFalse(invalid_source.connect())
    
    def test_get_symbols(self):
        """测试获取交易品种列表"""
        symbols = self.data_source.get_symbols()
        self.assertEqual(len(symbols), 2)
        self.assertIn("AAPL", symbols)
        self.assertIn("MSFT", symbols)
    
    def test_validate_symbol(self):
        """测试验证交易品种"""
        self.assertTrue(self.data_source.validate_symbol("AAPL"))
        self.assertTrue(self.data_source.validate_symbol("MSFT"))
        self.assertFalse(self.data_source.validate_symbol("INVALID"))
    
    def test_validate_timeframe(self):
        """测试验证时间周期"""
        self.assertTrue(self.data_source.validate_timeframe(TimeFrame.DAY_1))
        self.assertTrue(self.data_source.validate_timeframe(TimeFrame.HOUR_1))
        self.assertFalse(self.data_source.validate_timeframe(TimeFrame.MONTH_1))
    
    def test_get_historical_data(self):
        """测试获取历史数据"""
        # 获取AAPL日线数据
        data = self.data_source.get_historical_data(
            "AAPL", 
            TimeFrame.DAY_1, 
            datetime(2023, 1, 1),
            datetime(2023, 1, 3)
        )
        
        self.assertEqual(len(data), 3)
        self.assertEqual(data.iloc[0]['open'], 150.0)
        self.assertEqual(data.iloc[-1]['close'], 155.0)
        
        # 测试时间范围筛选
        data = self.data_source.get_historical_data(
            "AAPL", 
            TimeFrame.DAY_1, 
            datetime(2023, 1, 2),
            datetime(2023, 1, 3)
        )
        
        self.assertEqual(len(data), 2)
        self.assertEqual(data.iloc[0]['open'], 151.0)
        
        # 测试数量限制
        data = self.data_source.get_historical_data(
            "AAPL", 
            TimeFrame.DAY_1, 
            datetime(2023, 1, 1),
            datetime(2023, 1, 3),
            limit=1
        )
        
        self.assertEqual(len(data), 1)
        
        # 测试不支持的交易品种
        data = self.data_source.get_historical_data(
            "INVALID", 
            TimeFrame.DAY_1, 
            datetime(2023, 1, 1)
        )
        
        self.assertTrue(data.empty)
        
        # 测试不支持的时间周期
        data = self.data_source.get_historical_data(
            "AAPL", 
            TimeFrame.MONTH_1, 
            datetime(2023, 1, 1)
        )
        
        self.assertTrue(data.empty)
    
    def test_get_latest_data(self):
        """测试获取最新数据"""
        # 获取MSFT小时最新数据
        data = self.data_source.get_latest_data("MSFT", TimeFrame.HOUR_1)
        
        self.assertEqual(len(data), 1)
        self.assertEqual(data.iloc[0]['close'], 255.0)
        
        # 测试不支持的交易品种
        data = self.data_source.get_latest_data("INVALID", TimeFrame.HOUR_1)
        self.assertTrue(data.empty)


class TestDataManager(unittest.TestCase):
    """DataManager类的测试用例"""
    
    def setUp(self):
        """创建测试环境"""
        # 创建数据管理器
        self.manager = DataManager()
        
        # 创建模拟数据源
        self.mock_source = MagicMock()
        self.mock_source.name = "mock_source"
        self.mock_source.is_connected.return_value = True
        
        # 添加模拟数据源
        self.manager.add_data_source(self.mock_source, is_default=True)
    
    def test_add_remove_data_source(self):
        """测试添加和移除数据源"""
        # 添加数据源
        mock_source2 = MagicMock()
        mock_source2.name = "mock_source2"
        
        self.assertTrue(self.manager.add_data_source(mock_source2))
        self.assertEqual(len(self.manager.data_sources), 2)
        
        # 添加同名数据源应该失败
        mock_source3 = MagicMock()
        mock_source3.name = "mock_source"
        
        self.assertFalse(self.manager.add_data_source(mock_source3))
        self.assertEqual(len(self.manager.data_sources), 2)
        
        # 移除数据源
        self.assertTrue(self.manager.remove_data_source("mock_source2"))
        self.assertEqual(len(self.manager.data_sources), 1)
        
        # 移除不存在的数据源应该失败
        self.assertFalse(self.manager.remove_data_source("non_existent"))
    
    def test_get_data_source(self):
        """测试获取数据源"""
        # 获取默认数据源
        source = self.manager.get_data_source()
        self.assertEqual(source, self.mock_source)
        
        # 获取指定数据源
        source = self.manager.get_data_source("mock_source")
        self.assertEqual(source, self.mock_source)
        
        # 获取不存在的数据源
        source = self.manager.get_data_source("non_existent")
        self.assertIsNone(source)
    
    def test_get_historical_data(self):
        """测试获取历史数据"""
        # 创建模拟返回数据
        mock_data = pd.DataFrame({
            'open': [100, 101],
            'close': [102, 103]
        })
        self.mock_source.get_historical_data.return_value = mock_data
        
        # 获取历史数据
        data = self.manager.get_historical_data(
            "AAPL",
            TimeFrame.DAY_1,
            datetime(2023, 1, 1)
        )
        
        # 验证调用和结果
        self.mock_source.get_historical_data.assert_called_once()
        self.assertEqual(len(data), 2)
        
        # 测试数据源异常
        self.mock_source.get_historical_data.side_effect = Exception("Test error")
        
        data = self.manager.get_historical_data(
            "AAPL",
            TimeFrame.DAY_1,
            datetime(2023, 1, 1)
        )
        
        self.assertTrue(data.empty)
    
    def test_get_latest_data(self):
        """测试获取最新数据"""
        # 创建模拟返回数据
        mock_data = pd.DataFrame({
            'open': [100],
            'close': [102]
        })
        self.mock_source.get_latest_data.return_value = mock_data
        
        # 获取最新数据
        data = self.manager.get_latest_data("AAPL", TimeFrame.DAY_1)
        
        # 验证调用和结果
        self.mock_source.get_latest_data.assert_called_once()
        self.assertEqual(len(data), 1)
    
    def test_get_available_symbols(self):
        """测试获取可用交易品种"""
        # 创建模拟返回数据
        self.mock_source.get_symbols.return_value = ["AAPL", "MSFT"]
        
        # 获取交易品种列表
        symbols = self.manager.get_available_symbols()
        
        # 验证调用和结果
        self.mock_source.get_symbols.assert_called_once()
        self.assertEqual(len(symbols), 2)
        self.assertIn("AAPL", symbols)
    
    def test_resample_data(self):
        """测试数据重采样"""
        # 创建测试数据 - 小时数据
        hourly_data = pd.DataFrame({
            'timestamp': [
                datetime(2023, 1, 1, 10, 0),
                datetime(2023, 1, 1, 11, 0),
                datetime(2023, 1, 1, 12, 0),
                datetime(2023, 1, 1, 13, 0)
            ],
            'open': [100.0, 101.0, 102.0, 103.0],
            'high': [105.0, 106.0, 107.0, 108.0],
            'low': [99.0, 100.0, 101.0, 102.0],
            'close': [103.0, 104.0, 105.0, 106.0],
            'volume': [1000, 1100, 1200, 1300]
        })
        
        # 重采样到4小时数据
        resampled = self.manager.resample_data(
            hourly_data,
            TimeFrame.HOUR_1,
            TimeFrame.HOUR_4
        )
        
        # 验证结果
        self.assertEqual(len(resampled), 1)
        self.assertEqual(resampled.iloc[0]['open'], 100.0)  # 第一个open
        self.assertEqual(resampled.iloc[0]['high'], 108.0)  # 最高high
        self.assertEqual(resampled.iloc[0]['low'], 99.0)    # 最低low
        self.assertEqual(resampled.iloc[0]['close'], 106.0) # 最后一个close
        self.assertEqual(resampled.iloc[0]['volume'], 4600) # 所有volume之和
        
        # 测试不支持的降采样
        resampled = self.manager.resample_data(
            hourly_data,
            TimeFrame.HOUR_1,
            TimeFrame.MINUTE_1
        )
        
        self.assertTrue(resampled.empty)


if __name__ == '__main__':
    unittest.main() 