#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置管理器实现，提供统一的配置加载、验证和管理接口。
"""

import os
import json
import yaml
import logging
import copy
import jsonschema
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Type, Set, Callable
from dataclasses import dataclass, field


class ConfigError(Exception):
    """配置错误异常"""
    pass


class ConfigSchema:
    """配置模式定义，用于验证配置项"""
    
    @staticmethod
    def risk_controller() -> Dict[str, Any]:
        """风险控制器配置模式"""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "max_drawdown_percent": {"type": "number", "minimum": 0, "maximum": 100},
                "max_position_percent": {"type": "number", "minimum": 0, "maximum": 100},
                "initial_risk_level": {
                    "type": "string",
                    "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
                },
                "risk_rules_file": {"type": "string"},
                "min_cash_reserve_percent": {"type": "number", "minimum": 0, "maximum": 100},
                "max_sector_exposure_percent": {"type": "number", "minimum": 0, "maximum": 100},
                "state_file": {"type": "string"}
            },
            "required": ["name"],
            "additionalProperties": True
        }
    
    @staticmethod
    def evolution_manager() -> Dict[str, Any]:
        """进化管理器配置模式"""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "population_size": {"type": "integer", "minimum": 5},
                "generations": {"type": "integer", "minimum": 1},
                "mutation_rate": {"type": "number", "minimum": 0, "maximum": 1},
                "crossover_rate": {"type": "number", "minimum": 0, "maximum": 1},
                "selection_size": {"type": "integer", "minimum": 2},
                "elite_size": {"type": "integer", "minimum": 0},
                "fitness_function": {"type": "string"},
                "strategy_template": {"type": "string"},
                "parameter_ranges": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "min": {"type": ["number", "integer"]},
                            "max": {"type": ["number", "integer"]},
                            "type": {"type": "string", "enum": ["integer", "float"]}
                        },
                        "required": ["min", "max", "type"]
                    }
                },
                "evaluation_period": {"type": "string"},
                "output_dir": {"type": "string"},
                "random_seed": {"type": "integer"}
            },
            "required": ["name", "population_size", "generations", "parameter_ranges"],
            "additionalProperties": True
        }
    
    @staticmethod
    def strategy_manager() -> Dict[str, Any]:
        """策略管理器配置模式"""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "strategies_dir": {"type": "string"},
                "active_strategies": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "default_settings": {"type": "object"},
                "state_file": {"type": "string"},
                "max_active_strategies": {"type": "integer", "minimum": 1}
            },
            "required": ["name"],
            "additionalProperties": True
        }
    
    @staticmethod
    def data_manager() -> Dict[str, Any]:
        """数据管理器配置模式"""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "data_sources": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "api_key": {"type": "string"},
                            "api_secret": {"type": "string"},
                            "base_url": {"type": "string"},
                            "cache_dir": {"type": "string"},
                            "cache_expiry_seconds": {"type": "integer", "minimum": 0}
                        },
                        "required": ["type"],
                        "additionalProperties": True
                    }
                },
                "default_source": {"type": "string"},
                "local_data_dir": {"type": "string"},
                "download_missing": {"type": "boolean"}
            },
            "required": ["name"],
            "additionalProperties": True
        }
    
    @staticmethod
    def backtest_engine() -> Dict[str, Any]:
        """回测引擎配置模式"""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "initial_capital": {"type": "number", "minimum": 0},
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
                "symbols": {
                    "type": "array", 
                    "items": {"type": "string"}
                },
                "timeframe": {"type": "string"},
                "commission_rate": {"type": "number", "minimum": 0},
                "slippage_model": {"type": "string"},
                "data_source": {"type": "string"},
                "execution_model": {"type": "string"},
                "report_file": {"type": "string"},
                "detail_level": {"type": "string", "enum": ["low", "medium", "high"]},
                "plot_results": {"type": "boolean"}
            },
            "required": ["name", "initial_capital", "start_date", "end_date", "symbols"],
            "additionalProperties": True
        }
    
    @staticmethod
    def execution_client() -> Dict[str, Any]:
        """执行客户端配置模式"""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "client_type": {"type": "string"},
                "api_key": {"type": "string"},
                "api_secret": {"type": "string"},
                "base_url": {"type": "string"},
                "initial_balance": {"type": "number", "minimum": 0},
                "commission_rate": {"type": "number", "minimum": 0},
                "slippage_model": {"type": "string"},
                "order_latency_ms": {"type": "integer", "minimum": 0},
                "fill_latency_ms": {"type": "integer", "minimum": 0},
                "use_testnet": {"type": "boolean"}
            },
            "required": ["name", "client_type"],
            "additionalProperties": True
        }
    
    @staticmethod
    def system() -> Dict[str, Any]:
        """系统整体配置模式"""
        return {
            "type": "object",
            "properties": {
                "system_name": {"type": "string"},
                "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                "log_file": {"type": "string"},
                "data_dir": {"type": "string"},
                "output_dir": {"type": "string"},
                "temp_dir": {"type": "string"},
                "timezone": {"type": "string"},
                "max_threads": {"type": "integer", "minimum": 1},
                "debug_mode": {"type": "boolean"},
                "components": {
                    "type": "object",
                    "properties": {
                        "risk_controller": {"$ref": "#/definitions/risk_controller"},
                        "evolution_manager": {"$ref": "#/definitions/evolution_manager"},
                        "strategy_manager": {"$ref": "#/definitions/strategy_manager"},
                        "data_manager": {"$ref": "#/definitions/data_manager"},
                        "backtest_engine": {"$ref": "#/definitions/backtest_engine"},
                        "execution_client": {"$ref": "#/definitions/execution_client"}
                    },
                    "additionalProperties": False
                }
            },
            "required": ["system_name"],
            "additionalProperties": True,
            "definitions": {
                "risk_controller": ConfigSchema.risk_controller(),
                "evolution_manager": ConfigSchema.evolution_manager(),
                "strategy_manager": ConfigSchema.strategy_manager(),
                "data_manager": ConfigSchema.data_manager(),
                "backtest_engine": ConfigSchema.backtest_engine(),
                "execution_client": ConfigSchema.execution_client()
            }
        }


class ConfigManager:
    """配置管理器，负责加载、验证和管理系统各组件的配置"""
    
    def __init__(self, config_file: str = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，可选。如果提供，将加载该配置文件。
        """
        # 初始化日志记录器
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化配置存储
        self.config = {}
        self.component_configs = {}
        
        # 配置缓存
        self._cached_configs = {}
        
        # 加载配置
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """
        从文件加载配置
        
        Args:
            config_file: 配置文件路径
        
        Raises:
            ConfigError: 配置文件不存在或格式错误
        """
        if not os.path.exists(config_file):
            raise ConfigError(f"配置文件不存在: {config_file}")
        
        try:
            _, ext = os.path.splitext(config_file)
            
            if ext.lower() in ['.yml', '.yaml']:
                with open(config_file, 'r', encoding='utf-8') as file:
                    self.config = yaml.safe_load(file)
            elif ext.lower() == '.json':
                with open(config_file, 'r', encoding='utf-8') as file:
                    self.config = json.load(file)
            else:
                raise ConfigError(f"不支持的配置文件格式: {ext}")
                
            # 验证配置
            self.validate_config()
            
            # 提取组件配置
            self._extract_component_configs()
            
            self.logger.info(f"已加载配置文件: {config_file}")
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigError(f"配置文件格式错误: {str(e)}")
        except Exception as e:
            raise ConfigError(f"加载配置文件时发生错误: {str(e)}")
    
    def save_config(self, config_file: str) -> None:
        """
        保存配置到文件
        
        Args:
            config_file: 配置文件路径
        
        Raises:
            ConfigError: 保存配置文件失败
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)
            
            _, ext = os.path.splitext(config_file)
            
            if ext.lower() in ['.yml', '.yaml']:
                with open(config_file, 'w', encoding='utf-8') as file:
                    yaml.dump(self.config, file, default_flow_style=False)
            elif ext.lower() == '.json':
                with open(config_file, 'w', encoding='utf-8') as file:
                    json.dump(self.config, file, indent=4)
            else:
                raise ConfigError(f"不支持的配置文件格式: {ext}")
                
            self.logger.info(f"已保存配置到文件: {config_file}")
        except Exception as e:
            raise ConfigError(f"保存配置文件失败: {str(e)}")
    
    def validate_config(self) -> None:
        """
        验证配置是否符合模式
        
        Raises:
            ConfigError: 配置验证失败
        """
        try:
            # 验证整体系统配置
            jsonschema.validate(self.config, ConfigSchema.system())
            self.logger.info("配置验证通过")
        except jsonschema.exceptions.ValidationError as e:
            raise ConfigError(f"配置验证失败: {str(e)}")
    
    def _extract_component_configs(self) -> None:
        """
        从整体配置中提取各组件的配置
        """
        # 获取组件配置
        components = self.config.get('components', {})
        
        # 提取各组件配置，添加默认值
        self.component_configs = {
            'risk_controller': self._prepare_component_config(components.get('risk_controller', {}), 'risk_controller'),
            'evolution_manager': self._prepare_component_config(components.get('evolution_manager', {}), 'evolution_manager'),
            'strategy_manager': self._prepare_component_config(components.get('strategy_manager', {}), 'strategy_manager'),
            'data_manager': self._prepare_component_config(components.get('data_manager', {}), 'data_manager'),
            'backtest_engine': self._prepare_component_config(components.get('backtest_engine', {}), 'backtest_engine'),
            'execution_client': self._prepare_component_config(components.get('execution_client', {}), 'execution_client')
        }
    
    def _prepare_component_config(self, component_config: Dict[str, Any], component_type: str) -> Dict[str, Any]:
        """
        准备组件配置，添加默认值和通用配置
        
        Args:
            component_config: 组件特定配置
            component_type: 组件类型
        
        Returns:
            完整的组件配置
        """
        # 获取通用配置
        common_config = {
            'data_dir': self.config.get('data_dir', './data'),
            'output_dir': self.config.get('output_dir', './output'),
            'log_level': self.config.get('log_level', 'INFO')
        }
        
        # 合并配置（通用配置可被组件特定配置覆盖）
        config = {**common_config, **component_config}
        
        # 确保组件有名称
        if 'name' not in config:
            config['name'] = f"default_{component_type}"
        
        return config
    
    def get_component_config(self, component_type: str) -> Dict[str, Any]:
        """
        获取指定组件的配置
        
        Args:
            component_type: 组件类型
        
        Returns:
            组件配置
        
        Raises:
            ConfigError: 组件类型不存在
        """
        if component_type not in self.component_configs:
            raise ConfigError(f"不支持的组件类型: {component_type}")
        
        # 返回配置的深复制，防止外部修改
        return copy.deepcopy(self.component_configs[component_type])
    
    def update_component_config(self, component_type: str, config_updates: Dict[str, Any]) -> None:
        """
        更新组件配置
        
        Args:
            component_type: 组件类型
            config_updates: 配置更新
        
        Raises:
            ConfigError: 组件类型不存在或配置验证失败
        """
        if component_type not in self.component_configs:
            raise ConfigError(f"不支持的组件类型: {component_type}")
        
        # 更新配置
        component_config = self.component_configs[component_type]
        updated_config = {**component_config, **config_updates}
        
        # 验证更新后的配置
        schema_method = getattr(ConfigSchema, component_type, None)
        if schema_method:
            try:
                jsonschema.validate(updated_config, schema_method())
            except jsonschema.exceptions.ValidationError as e:
                raise ConfigError(f"配置验证失败: {str(e)}")
        
        # 更新组件配置
        self.component_configs[component_type] = updated_config
        
        # 更新系统整体配置
        if 'components' not in self.config:
            self.config['components'] = {}
        self.config['components'][component_type] = updated_config
        
        # 清除缓存
        if component_type in self._cached_configs:
            del self._cached_configs[component_type]
        
        self.logger.info(f"已更新组件配置: {component_type}")
    
    def get_system_config(self) -> Dict[str, Any]:
        """
        获取系统整体配置
        
        Returns:
            系统配置
        """
        # 返回配置的深复制，防止外部修改
        return copy.deepcopy(self.config)
    
    def update_system_config(self, config_updates: Dict[str, Any]) -> None:
        """
        更新系统整体配置
        
        Args:
            config_updates: 配置更新
        
        Raises:
            ConfigError: 配置验证失败
        """
        # 更新配置
        updated_config = {**self.config, **config_updates}
        
        # 验证更新后的配置
        try:
            jsonschema.validate(updated_config, ConfigSchema.system())
        except jsonschema.exceptions.ValidationError as e:
            raise ConfigError(f"配置验证失败: {str(e)}")
        
        # 更新系统配置
        self.config = updated_config
        
        # 重新提取组件配置
        self._extract_component_configs()
        
        # 清除所有缓存
        self._cached_configs.clear()
        
        self.logger.info("已更新系统配置")
    
    def create_default_config(self) -> Dict[str, Any]:
        """
        创建默认系统配置
        
        Returns:
            默认系统配置
        """
        default_config = {
            "system_name": "SETO-Versal",
            "log_level": "INFO",
            "log_file": "logs/seto-versal.log",
            "data_dir": "data",
            "output_dir": "output",
            "temp_dir": "temp",
            "timezone": "UTC",
            "max_threads": 4,
            "debug_mode": False,
            "components": {
                "risk_controller": {
                    "name": "default_risk",
                    "max_drawdown_percent": 10.0,
                    "max_position_percent": 10.0,
                    "initial_risk_level": "MEDIUM",
                    "risk_rules_file": "config/risk_rules.yaml",
                    "min_cash_reserve_percent": 20.0,
                    "max_sector_exposure_percent": 30.0
                },
                "evolution_manager": {
                    "name": "default_evolution",
                    "population_size": 50,
                    "generations": 20,
                    "mutation_rate": 0.1,
                    "crossover_rate": 0.8,
                    "selection_size": 10,
                    "elite_size": 5,
                    "fitness_function": "sharpe_ratio",
                    "evaluation_period": "1Y",
                    "output_dir": "output/evolution"
                },
                "strategy_manager": {
                    "name": "default_strategy",
                    "strategies_dir": "strategies",
                    "max_active_strategies": 10
                },
                "data_manager": {
                    "name": "default_data",
                    "local_data_dir": "data/market",
                    "download_missing": True
                },
                "backtest_engine": {
                    "name": "default_backtest",
                    "initial_capital": 100000.0,
                    "commission_rate": 0.001,
                    "slippage_model": "fixed",
                    "detail_level": "medium",
                    "plot_results": True
                },
                "execution_client": {
                    "name": "default_execution",
                    "client_type": "simulator",
                    "initial_balance": 100000.0,
                    "commission_rate": 0.001,
                    "use_testnet": True
                }
            }
        }
        
        return default_config
    
    def apply_environment_overrides(self) -> None:
        """
        应用环境变量覆盖配置
        """
        # 环境变量前缀
        env_prefix = "SETO_"
        
        # 遍历环境变量
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # 去掉前缀
                config_key = key[len(env_prefix):].lower()
                
                # 替换分隔符
                path_parts = config_key.split('_')
                
                # 递归更新配置
                self._update_nested_config(self.config, path_parts, value)
        
        # 重新提取组件配置
        self._extract_component_configs()
        
        # 清除所有缓存
        self._cached_configs.clear()
    
    def _update_nested_config(self, config: Dict[str, Any], path_parts: List[str], value: str) -> None:
        """
        递归更新嵌套配置
        
        Args:
            config: 配置字典
            path_parts: 配置路径部分
            value: 配置值
        """
        if len(path_parts) == 1:
            # 转换值类型
            typed_value = self._convert_value_type(value)
            config[path_parts[0]] = typed_value
        else:
            key = path_parts[0]
            if key not in config:
                config[key] = {}
            self._update_nested_config(config[key], path_parts[1:], value)
    
    def _convert_value_type(self, value: str) -> Any:
        """
        转换配置值类型
        
        Args:
            value: 字符串值
        
        Returns:
            转换后的值
        """
        # 尝试转换为布尔值
        if value.lower() in ['true', 'yes', '1']:
            return True
        if value.lower() in ['false', 'no', '0']:
            return False
        
        # 尝试转换为整数
        try:
            return int(value)
        except ValueError:
            pass
        
        # 尝试转换为浮点数
        try:
            return float(value)
        except ValueError:
            pass
        
        # 返回字符串
        return value
    
    def get_cached_config(self, component_type: str) -> Dict[str, Any]:
        """
        获取缓存的组件配置，如果不存在则创建
        
        Args:
            component_type: 组件类型
        
        Returns:
            组件配置
        """
        if component_type not in self._cached_configs:
            self._cached_configs[component_type] = self.get_component_config(component_type)
        
        return self._cached_configs[component_type] 