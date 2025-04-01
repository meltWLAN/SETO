# SETO-Versal 交易系统

SETO-Versal是一个全功能量化交易系统，提供交易策略开发、风险管理、策略优化和执行功能。

## 系统组件

系统由以下主要模块组成：

### 风险控制 (Risk)

风险控制模块负责评估和管理交易策略的风险，确保交易符合预设的风险参数。

- `RiskController`: 验证交易、跟踪风险，并根据市场条件调整风险级别
- `RiskLevel`: 风险级别的枚举类型（LOW, MEDIUM, HIGH, CRITICAL）

### 策略进化 (Evolution)

策略进化模块使用遗传算法优化交易策略参数。

- `EvolutionManager`: 管理策略群体的进化过程
- `EvolutionMetric`: 衡量进化性能的指标

### 策略管理 (Strategy)

策略管理模块提供了交易策略的定义和执行框架。

- `Strategy`: 所有交易策略的基类
- `StrategyManager`: 加载、管理和执行交易策略

### 数据管理 (Data)

数据管理模块处理市场数据的获取、存储和处理。

- `DataManager`: 管理数据源和数据处理
- `TimeFrame`: 时间框架的枚举类型（如1分钟、5分钟、日线等）

### 回测引擎 (Backtest)

回测引擎模块允许在历史数据上测试交易策略。

- `BacktestEngine`: 提供策略回测功能
- `BacktestResult`: 表示回测结果的数据类

### 执行模块 (Execution)

执行模块提供了与交易所和模拟交易系统的接口。

- `ExecutionClient`: 所有执行客户端的基类
- `SimulatedExecutionClient`: 模拟交易所执行客户端
- `BinanceExecutionClient`: 币安交易所的实现
- `Order`: 表示交易订单的数据类
- `OrderManager`: 管理订单的创建、更新和状态跟踪

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/seto-versal.git
cd seto-versal

# 安装依赖
pip install -r requirements.txt
```

### 使用示例

```python
from seto_versal.risk import RiskController
from seto_versal.strategy import StrategyManager
from seto_versal.execution import SimulatedExecutionClient

# 初始化风险控制器
risk_config = {
    "max_drawdown_percent": 5.0,
    "max_position_percent": 10.0,
    "initial_risk_level": "MEDIUM"
}
risk_controller = RiskController(name="main_risk", config=risk_config)

# 初始化执行客户端
exec_config = {
    "initial_balance": 100000.0,
    "commission_rate": 0.001
}
execution_client = SimulatedExecutionClient(name="sim_exchange", config=exec_config)

# 初始化策略管理器
strategy_manager = StrategyManager(
    name="main_strategy_manager",
    risk_controller=risk_controller,
    execution_client=execution_client
)

# 加载策略
strategy_manager.load_strategies("strategies")

# 运行策略
strategy_manager.run_strategies()
``` 