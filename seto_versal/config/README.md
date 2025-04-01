# SETO-Versal 配置管理

配置管理模块负责加载、验证和管理系统各组件的配置参数。它提供统一的配置接口和配置验证能力，确保系统各部分使用正确有效的配置。

## 特性

- 支持YAML和JSON格式配置文件
- 基于JSON Schema的配置验证
- 组件级配置管理
- 系统级配置管理
- 配置缓存机制
- 环境变量覆盖配置
- 默认配置生成

## 使用方法

### 基本用法

```python
from seto_versal.config import ConfigManager

# 创建配置管理器，可选择加载配置文件
config_manager = ConfigManager('config.yaml')

# 或者先创建配置管理器，稍后加载配置
config_manager = ConfigManager()
config_manager.load_config('config.yaml')

# 获取系统配置
system_config = config_manager.get_system_config()
print(f"系统名称: {system_config['system_name']}")

# 获取组件配置
risk_config = config_manager.get_component_config('risk_controller')
print(f"最大回撤: {risk_config['max_drawdown_percent']}%")
```

### 更新配置

```python
# 更新系统配置
config_manager.update_system_config({
    'log_level': 'DEBUG',
    'max_threads': 8
})

# 更新组件配置
config_manager.update_component_config('risk_controller', {
    'max_drawdown_percent': 15.0,
    'max_position_percent': 8.0
})

# 保存配置
config_manager.save_config('updated_config.yaml')
```

### 创建默认配置

```python
# 创建默认配置
default_config = config_manager.create_default_config()

# 使用默认配置
config_manager.config = default_config
config_manager._extract_component_configs()

# 保存默认配置
config_manager.save_config('default_config.yaml')
```

### 使用环境变量覆盖配置

环境变量可以覆盖配置，格式为`SETO_[CONFIG_PATH]`，例如：

```bash
# 设置环境变量
export SETO_LOG_LEVEL=DEBUG
export SETO_COMPONENTS_RISK_CONTROLLER_MAX_DRAWDOWN_PERCENT=7.5

# 在代码中应用环境变量覆盖
config_manager.apply_environment_overrides()
```

## 配置文件示例

查看 [example_config.yaml](example_config.yaml) 获取完整的配置示例。

## 配置模式

配置模式定义了各组件的配置结构和验证规则。以下是主要组件的配置模式：

- `risk_controller`: 风险控制器配置
- `evolution_manager`: 策略进化管理器配置
- `strategy_manager`: 策略管理器配置
- `data_manager`: 数据管理器配置
- `backtest_engine`: 回测引擎配置
- `execution_client`: 执行客户端配置

每个组件的详细配置项请参考 [example_config.yaml](example_config.yaml) 文件。 