#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置管理器测试文件
"""

import os
import json
import yaml
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from seto_versal.config.manager import ConfigManager, ConfigSchema, ConfigError


class TestConfigSchema(unittest.TestCase):
    """测试配置模式"""
    
    def test_risk_controller_schema(self):
        """测试风险控制器配置模式"""
        schema = ConfigSchema.risk_controller()
        
        # 验证模式结构
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertIn("name", schema["properties"])
        self.assertIn("max_drawdown_percent", schema["properties"])
        self.assertIn("required", schema)
        self.assertIn("name", schema["required"])
    
    def test_system_schema(self):
        """测试系统整体配置模式"""
        schema = ConfigSchema.system()
        
        # 验证模式结构
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertIn("system_name", schema["properties"])
        self.assertIn("components", schema["properties"])
        self.assertIn("required", schema)
        self.assertIn("system_name", schema["required"])
        self.assertIn("definitions", schema)


class TestConfigManager(unittest.TestCase):
    """测试配置管理器"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时配置文件
        self.temp_dir = tempfile.mkdtemp()
        self.yaml_config_file = os.path.join(self.temp_dir, "config.yaml")
        self.json_config_file = os.path.join(self.temp_dir, "config.json")
        
        # 示例配置
        self.sample_config = {
            "system_name": "SETO-Versal-Test",
            "log_level": "INFO",
            "data_dir": "test_data",
            "components": {
                "risk_controller": {
                    "name": "test_risk",
                    "max_drawdown_percent": 5.0,
                    "max_position_percent": 10.0,
                    "initial_risk_level": "LOW"
                },
                "backtest_engine": {
                    "name": "test_backtest",
                    "initial_capital": 10000.0,
                    "commission_rate": 0.001
                }
            }
        }
        
        # 创建YAML配置文件
        with open(self.yaml_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.sample_config, f)
        
        # 创建JSON配置文件
        with open(self.json_config_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_config, f, indent=4)
        
        # 创建配置管理器
        self.config_manager = ConfigManager()
    
    def tearDown(self):
        """清理测试环境"""
        # 删除临时文件
        if os.path.exists(self.yaml_config_file):
            os.remove(self.yaml_config_file)
        if os.path.exists(self.json_config_file):
            os.remove(self.json_config_file)
        
        # 删除临时目录
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_load_yaml_config(self):
        """测试加载YAML配置文件"""
        self.config_manager.load_config(self.yaml_config_file)
        
        # 验证配置已加载
        config = self.config_manager.get_system_config()
        self.assertEqual(config["system_name"], "SETO-Versal-Test")
        self.assertEqual(config["log_level"], "INFO")
        
        # 验证组件配置
        risk_config = self.config_manager.get_component_config("risk_controller")
        self.assertEqual(risk_config["name"], "test_risk")
        self.assertEqual(risk_config["max_drawdown_percent"], 5.0)
    
    def test_load_json_config(self):
        """测试加载JSON配置文件"""
        self.config_manager.load_config(self.json_config_file)
        
        # 验证配置已加载
        config = self.config_manager.get_system_config()
        self.assertEqual(config["system_name"], "SETO-Versal-Test")
        
        # 验证组件配置
        backtest_config = self.config_manager.get_component_config("backtest_engine")
        self.assertEqual(backtest_config["name"], "test_backtest")
        self.assertEqual(backtest_config["initial_capital"], 10000.0)
    
    def test_load_invalid_file(self):
        """测试加载不存在的配置文件"""
        with self.assertRaises(ConfigError):
            self.config_manager.load_config("non_existent_file.yaml")
    
    def test_save_config(self):
        """测试保存配置到文件"""
        # 加载现有配置
        self.config_manager.load_config(self.yaml_config_file)
        
        # 修改配置
        self.config_manager.update_system_config({"log_level": "DEBUG"})
        
        # 保存到新文件
        new_config_file = os.path.join(self.temp_dir, "new_config.yaml")
        self.config_manager.save_config(new_config_file)
        
        # 验证文件已创建
        self.assertTrue(os.path.exists(new_config_file))
        
        # 加载新文件验证
        new_manager = ConfigManager(new_config_file)
        self.assertEqual(new_manager.get_system_config()["log_level"], "DEBUG")
        
        # 清理
        if os.path.exists(new_config_file):
            os.remove(new_config_file)
    
    def test_update_component_config(self):
        """测试更新组件配置"""
        # 加载现有配置
        self.config_manager.load_config(self.yaml_config_file)
        
        # 更新风险控制器配置
        self.config_manager.update_component_config("risk_controller", {
            "max_drawdown_percent": 8.0,
            "min_cash_reserve_percent": 15.0
        })
        
        # 验证配置已更新
        risk_config = self.config_manager.get_component_config("risk_controller")
        self.assertEqual(risk_config["max_drawdown_percent"], 8.0)
        self.assertEqual(risk_config["min_cash_reserve_percent"], 15.0)
        self.assertEqual(risk_config["name"], "test_risk")  # 原始值应保留
    
    def test_update_invalid_component(self):
        """测试更新不存在的组件配置"""
        # 加载现有配置
        self.config_manager.load_config(self.yaml_config_file)
        
        # 尝试更新不存在的组件
        with self.assertRaises(ConfigError):
            self.config_manager.update_component_config("non_existent_component", {"key": "value"})
    
    def test_update_invalid_component_config(self):
        """测试用无效配置更新组件"""
        # 加载现有配置
        self.config_manager.load_config(self.yaml_config_file)
        
        # 尝试用无效配置更新组件（超出范围的百分比值）
        with self.assertRaises(ConfigError):
            self.config_manager.update_component_config("risk_controller", {
                "max_drawdown_percent": 150.0  # 超过最大值100
            })
    
    def test_create_default_config(self):
        """测试创建默认配置"""
        default_config = self.config_manager.create_default_config()
        
        # 验证默认配置
        self.assertEqual(default_config["system_name"], "SETO-Versal")
        self.assertIn("components", default_config)
        self.assertIn("risk_controller", default_config["components"])
        self.assertIn("strategy_manager", default_config["components"])
    
    @patch.dict('os.environ', {
        'SETO_LOG_LEVEL': 'DEBUG',
        'SETO_COMPONENTS_RISK_CONTROLLER_MAX_DRAWDOWN_PERCENT': '7.5'
    })
    def test_apply_environment_overrides(self):
        """测试应用环境变量覆盖配置"""
        # 加载现有配置
        self.config_manager.load_config(self.yaml_config_file)
        
        # 应用环境变量覆盖
        self.config_manager.apply_environment_overrides()
        
        # 验证配置已被环境变量覆盖
        config = self.config_manager.get_system_config()
        self.assertEqual(config["log_level"], "DEBUG")
        
        risk_config = self.config_manager.get_component_config("risk_controller")
        self.assertEqual(risk_config["max_drawdown_percent"], 7.5)
    
    def test_get_cached_config(self):
        """测试获取缓存的组件配置"""
        # 加载现有配置
        self.config_manager.load_config(self.yaml_config_file)
        
        # 获取配置（第一次应该从源加载）
        risk_config1 = self.config_manager.get_cached_config("risk_controller")
        self.assertEqual(risk_config1["name"], "test_risk")
        
        # 修改缓存中的配置（不应影响原始配置）
        risk_config1["name"] = "modified_name"
        
        # 再次获取配置（应该从缓存获取）
        risk_config2 = self.config_manager.get_cached_config("risk_controller")
        self.assertEqual(risk_config2["name"], "test_risk")  # 应该是原始值
        
        # 更新组件配置（应清除缓存）
        self.config_manager.update_component_config("risk_controller", {"name": "new_name"})
        
        # 再次获取配置（应该获取更新后的值）
        risk_config3 = self.config_manager.get_cached_config("risk_controller")
        self.assertEqual(risk_config3["name"], "new_name")


if __name__ == '__main__':
    unittest.main() 