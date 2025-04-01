# SETO-Versal Data Quality System

## Overview

The Data Quality System in SETO-Versal provides a comprehensive framework for validating, monitoring, and ensuring the integrity of market data used within the trading system. It consists of several components:

1. **Data Quality Checker**: Core validation algorithms for checking data properties
2. **Data Quality Monitor**: Continuous monitoring system for alerting on quality issues
3. **Data Manager Integration**: Extensions to the DataManager for seamless quality checking

## Key Features

- **Comprehensive Validation**: Multiple validation checks including missing values, stale data, price jumps, and data consistency
- **Anomaly Detection**: Statistical algorithms to detect pricing and volume anomalies
- **Continuous Monitoring**: Background thread monitors data quality at specified intervals
- **Flexible Alert System**: Customizable alert handlers (email, Slack, logging)
- **Reporting**: Historical quality metrics and issue tracking

## Getting Started

### Basic Quality Checks

```python
from seto_versal.data.quality import DataQualityChecker

# Initialize the quality checker
quality_checker = DataQualityChecker()

# Run validation on your market data
result = quality_checker.validate_market_data(market_data)

if result['valid']:
    print("Data quality check passed")
else:
    print("Data quality check failed")
    # Access detailed issues
    failed_checks = [check for check, check_result in result['details'].items() 
                     if not check_result['valid']]
    print(f"Failed checks: {failed_checks}")
```

### Anomaly Detection

```python
from seto_versal.data.quality import detect_data_anomalies

# Detect anomalies in a dataframe
anomaly_df = detect_data_anomalies(symbol_data, price_column='close', volume_column='volume')

# Check if anomalies were found
has_anomalies = anomaly_df['is_anomaly'].any()
if has_anomalies:
    # Get rows with anomalies
    anomalies = anomaly_df[anomaly_df['is_anomaly']]
    print(f"Found {len(anomalies)} anomalies")
```

### Setting Up Monitoring

```python
from seto_versal.data.quality_monitor import DataQualityMonitor, log_alert_handler

# Configuration for the monitor
config = {
    'monitoring_interval': 3600,  # Check every hour
    'alert_threshold': 3,         # Alert after 3 consecutive failures
    'threshold_missing_pct': 0.05 # Threshold for missing values
}

# Initialize the monitor with data manager
monitor = DataQualityMonitor(config, data_manager)

# Register alert handlers
monitor.register_alert_handler(log_alert_handler)

# Start monitoring
monitor.start_monitoring()

# Later, stop monitoring
monitor.stop_monitoring()

# Generate quality report
report = monitor.get_report(days=7)
```

## Data Quality Checks

The system performs these validation checks:

### 1. Missing Values Check
Identifies columns with excessive missing values, configurable threshold (default 5%).

### 2. Stale Data Check
Verifies data freshness by checking the most recent date against current date, with configurable staleness threshold (default 3 days).

### 3. Price Jump Check
Detects abnormal price movements that could indicate data errors or market anomalies, with configurable threshold (default 10%).

### 4. Zero Volume Check
Checks for suspicious zero volume entries, with configurable threshold (default 20%).

### 5. Data Consistency Check
Ensures consistent data across symbols and dates, checking for duplicates and missing dates.

## Using with DataManager

The DataManager class has been extended to integrate with the quality system:

```python
from seto_versal.data.manager import DataManager

# Initialize with quality checks enabled
config = {
    'enable_quality_checks': True,
    'enable_quality_alerts': True,
    'auto_start_monitoring': True
}
data_manager = DataManager(config)

# Get data with automatic quality validation
data = data_manager.get_historical_data('AAPL', validate_quality=True)

# Run anomaly detection
anomalies = data_manager.detect_data_anomalies('AAPL', days=30)

# Get quality report
report = data_manager.get_quality_report(days=7)
```

## Configuration Options

The data quality system can be configured through these parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `threshold_missing_pct` | Maximum allowed percentage of missing values | 0.05 (5%) |
| `threshold_stale_days` | Maximum allowed days since last update | 3 days |
| `threshold_price_jump` | Maximum allowed price jump | 0.1 (10%) |
| `threshold_volume_zero_pct` | Maximum allowed percentage of zero volume entries | 0.2 (20%) |
| `monitoring_interval` | Interval between quality checks in seconds | 3600 (1 hour) |
| `alert_threshold` | Number of consecutive failures before alert | 3 |
| `results_dir` | Directory to store quality check results | 'data_quality_reports' |
| `enable_quality_checks` | Enable quality validation on data retrieval | True |
| `enable_quality_alerts` | Enable alert system | False |
| `auto_start_monitoring` | Start monitoring automatically | False |

## Extending the System

### Custom Alert Handlers

You can create custom alert handlers by defining a function that takes an alert_data parameter:

```python
def custom_alert_handler(alert_data):
    """Custom alert handler that sends to a webhook."""
    import requests
    
    webhook_url = "https://example.com/webhook"
    requests.post(webhook_url, json=alert_data)
    
# Register your custom handler
monitor.register_alert_handler(custom_alert_handler)
```

### Additional Quality Checks

To add custom quality checks, extend the DataQualityChecker class:

```python
from seto_versal.data.quality import DataQualityChecker

class EnhancedQualityChecker(DataQualityChecker):
    """Enhanced quality checker with additional checks."""
    
    def check_trading_hours(self, df, time_column='time'):
        """Check if data points fall within trading hours."""
        if time_column not in df.columns:
            return {'valid': False, 'error': f'Time column {time_column} not found'}
            
        # Convert to datetime
        times = pd.to_datetime(df[time_column]).dt.time
        
        # Define trading hours
        market_open = datetime.time(9, 30)
        market_close = datetime.time(16, 0)
        
        # Check if times are within trading hours
        outside_hours = ((times < market_open) | (times > market_close)).sum()
        outside_pct = outside_hours / len(df)
        
        return {
            'valid': outside_pct < 0.01,  # Less than 1% outside trading hours
            'outside_hours_count': outside_hours,
            'outside_hours_pct': outside_pct
        }
```

## Example Usage Script

See the `examples/data_quality_demo.py` script for a complete demonstration of the data quality system in action.

## Best Practices

1. **Enable automatic quality checks** during data retrieval for critical systems.
2. **Set up appropriate alert handlers** for real-time notification of quality issues.
3. **Tune thresholds** based on your specific data characteristics and requirements.
4. **Review quality reports regularly** to identify emerging issues before they impact trading.
5. **Run anomaly detection** before making trading decisions based on unusual price movements. 