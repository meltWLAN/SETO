# SETO-Versal (Self-Evolving Traderverse Intelligence)

A modular, intelligent trading system designed for the Chinese A-share market, combining multiple agent architectures with self-evolution capabilities.

## Core Architecture

- **Multi-Agent Framework**: Five specialized trading agents with different strategies and objectives
- **Self-Evolution**: Continuous improvement through feedback and parameter optimization
- **T+1 Compatibility**: Fully compliant with Chinese market rules
- **Philosophy-Driven Risk Control**: Embedded risk management principles
- **Behavioral Attribution**: Closed-loop learning from trading activities

## Key Components

1. **Agent Cluster**: Specialized intelligent agents (Fast Profit, Trend Following, Reversal, Sector Rotation, Defensive)
2. **Coordinator**: Decision integration and conflict resolution
3. **Strategy Library**: Modular, parameterized trading strategies
4. **Market Perception**: Real-time data and derived market indicators
5. **Trade Executor**: Order management with T+1 restrictions
6. **Risk Control**: Philosophy-based system constraints
7. **Feedback System**: Performance analysis and attribution
8. **Evolution System**: Continuous optimization and strategy refinement
9. **Narrator**: Human-readable system consciousness output

## Latest Version (v0.7.0)

The latest version includes comprehensive fixes for datetime module issues:
- Fixed incorrect imports of datetime and timedelta
- Added system-wide patch for datetime.now() method
- Fixed type annotations in various files
- Ensured proper operation of PyQt6 GUI

See FIXES.md for more details on system improvements.

## Getting Started

```bash
pip install -r requirements.txt

# Run the PyQt6 GUI with datetime fixes
python run_qt_gui.py

# Or run the standard system
python -m seto_versal.main
```

## Configuration

System parameters and philosophy principles are defined in `config.yaml`.

## Project Status

Under active development. Current stable version is v0.7.0.

See CHANGELOG.md for previous updates. 