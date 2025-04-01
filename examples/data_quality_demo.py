#!/usr/bin/env python
"""
Data Quality System Demo

This script demonstrates how to use the data quality checking system in the SETO-Versal framework.
It shows how to initialize the quality checker, validate data, and set up monitoring.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
import json

# Add the parent directory to the path so we can import the SETO-Versal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seto_versal.data.quality import DataQualityChecker, detect_data_anomalies
from seto_versal.data.quality_monitor import DataQualityMonitor, log_alert_handler

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data(num_symbols=5, days=90, missing_pct=0.02, anomaly_pct=0.05):
    """Create sample market data for testing.
    
    Args:
        num_symbols: Number of symbols to generate
        days: Number of days of data
        missing_pct: Percentage of missing values
        anomaly_pct: Percentage of anomalies
        
    Returns:
        DataFrame with sample market data
    """
    logger.info(f"Generating sample data for {num_symbols} symbols over {days} days")
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create symbols
    symbols = [f'STOCK{i:03d}' for i in range(1, num_symbols + 1)]
    
    # Create empty DataFrame to hold all data
    all_data = []
    
    # Generate data for each symbol
    for symbol in symbols:
        # Start with a random price between 10 and 100
        start_price = np.random.uniform(10, 100)
        
        # Generate random walks for prices
        prices = np.zeros(len(dates))
        prices[0] = start_price
        
        # Random daily returns with drift
        daily_returns = np.random.normal(0.0005, 0.02, len(dates))
        
        # Add in some anomalies
        anomaly_indices = np.random.choice(
            range(1, len(dates)), 
            size=int(len(dates) * anomaly_pct),
            replace=False
        )
        for idx in anomaly_indices:
            # Add a large price jump (positive or negative)
            daily_returns[idx] = np.random.choice([-0.15, 0.15]) + np.random.normal(0, 0.02)
        
        # Calculate prices from returns
        for i in range(1, len(dates)):
            prices[i] = prices[i-1] * (1 + daily_returns[i])
        
        # Generate volumes - roughly correlated with price changes
        volumes = np.abs(daily_returns) * np.random.uniform(500000, 2000000, len(dates))
        volumes = volumes.astype(int)
        
        # Create DataFrame for this symbol
        symbol_data = pd.DataFrame({
            'date': dates,
            'open': prices * np.random.uniform(0.99, 1.0, len(dates)),
            'high': prices * np.random.uniform(1.0, 1.03, len(dates)),
            'low': prices * np.random.uniform(0.97, 1.0, len(dates)),
            'close': prices,
            'volume': volumes,
            'symbol': symbol
        })
        
        # Introduce missing values
        mask = np.random.random(size=symbol_data.shape) < missing_pct
        symbol_data.mask(mask, inplace=True)
        
        all_data.append(symbol_data)
    
    # Combine all symbol data
    market_data = pd.concat(all_data, ignore_index=True)
    
    logger.info(f"Generated {len(market_data)} rows of sample data")
    return market_data

def demo_basic_quality_checks():
    """Demonstrate basic data quality checks."""
    logger.info("=== Demonstrating Basic Data Quality Checks ===")
    
    # Create sample data
    data = create_sample_data(missing_pct=0.05)
    
    # Initialize quality checker with default settings
    quality_checker = DataQualityChecker()
    
    # Check missing values
    missing_result = quality_checker.check_missing_values(data)
    logger.info(f"Missing values check - Valid: {missing_result['valid']}")
    if not missing_result['valid']:
        logger.info(f"Problems in columns: {missing_result['missing_columns']}")
    
    # Check stale data
    stale_result = quality_checker.check_stale_data(data, date_column='date')
    logger.info(f"Stale data check - Valid: {stale_result['valid']}")
    logger.info(f"Most recent date: {stale_result['most_recent_date']}, "
               f"Days since update: {stale_result['days_since_update']}")
    
    # Check data consistency
    consistency_result = quality_checker.check_data_consistency(data, date_column='date', symbol_column='symbol')
    logger.info(f"Data consistency check - Valid: {consistency_result['valid']}")
    if not consistency_result['valid']:
        if consistency_result.get('duplicates', 0) > 0:
            logger.info(f"Found {consistency_result['duplicates']} duplicate entries")
        if consistency_result.get('symbols_with_missing_dates', 0) > 0:
            logger.info(f"Found {consistency_result['symbols_with_missing_dates']} symbols with missing dates")
    
    # Full validation
    validation_result = quality_checker.validate_market_data(data)
    logger.info(f"Overall validation - Valid: {validation_result['valid']}")
    
    # Save validation result to file for inspection
    with open('data_quality_validation.json', 'w') as f:
        json.dump(validation_result, f, indent=2, default=str)
    logger.info("Saved validation results to data_quality_validation.json")

def demo_anomaly_detection():
    """Demonstrate anomaly detection in market data."""
    logger.info("=== Demonstrating Anomaly Detection ===")
    
    # Create sample data with higher anomaly percentage
    data = create_sample_data(num_symbols=1, anomaly_pct=0.1)
    
    # Filter to one symbol
    symbol_data = data[data['symbol'] == 'STOCK001'].copy()
    
    # Detect anomalies
    anomaly_df = detect_data_anomalies(symbol_data, price_column='close', volume_column='volume')
    
    # Count anomalies
    price_anomalies = anomaly_df['price_anomaly'].sum()
    volume_anomalies = anomaly_df['volume_anomaly'].sum()
    total_anomalies = anomaly_df['is_anomaly'].sum()
    
    logger.info(f"Detected {price_anomalies} price anomalies")
    logger.info(f"Detected {volume_anomalies} volume anomalies")
    logger.info(f"Total of {total_anomalies} combined anomalies")
    
    # Plot the data with anomalies highlighted
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot price
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(anomaly_df['date'], anomaly_df['close'], label='Close Price')
        
        # Highlight price anomalies
        price_anomaly_dates = anomaly_df.loc[anomaly_df['price_anomaly'], 'date']
        price_anomaly_values = anomaly_df.loc[anomaly_df['price_anomaly'], 'close']
        ax1.scatter(price_anomaly_dates, price_anomaly_values, color='red', 
                   label='Price Anomalies', zorder=5)
        
        ax1.set_title('Price with Anomalies')
        ax1.set_ylabel('Price')
        ax1.legend()
        
        # Plot volume
        ax2 = plt.subplot(2, 1, 2)
        ax2.bar(anomaly_df['date'], anomaly_df['volume'], label='Volume', alpha=0.5)
        
        # Highlight volume anomalies
        volume_anomaly_dates = anomaly_df.loc[anomaly_df['volume_anomaly'], 'date']
        volume_anomaly_values = anomaly_df.loc[anomaly_df['volume_anomaly'], 'volume']
        ax2.scatter(volume_anomaly_dates, volume_anomaly_values, color='red', 
                   label='Volume Anomalies', zorder=5)
        
        ax2.set_title('Volume with Anomalies')
        ax2.set_ylabel('Volume')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('anomaly_detection.png')
        logger.info("Saved anomaly plot to anomaly_detection.png")
        plt.close()
    except Exception as e:
        logger.error(f"Error creating anomaly plot: {str(e)}")

class MockDataManager:
    """Mock data manager class for demonstration purposes."""
    
    def __init__(self):
        """Initialize the mock data manager."""
        self.data = create_sample_data()
        self.symbols = self.data['symbol'].unique().tolist()
    
    def get_recent_market_data(self):
        """Get mock market data."""
        return self.data
    
    def get_historical_data(self, symbol, start_date=None, end_date=None):
        """Get mock historical data for a symbol."""
        symbol_data = self.data[self.data['symbol'] == symbol].copy()
        
        if start_date:
            symbol_data = symbol_data[symbol_data['date'] >= pd.Timestamp(start_date)]
        
        if end_date:
            symbol_data = symbol_data[symbol_data['date'] <= pd.Timestamp(end_date)]
            
        return symbol_data
    
    def get_symbols(self):
        """Get available symbols."""
        return self.symbols

def demo_monitoring_system():
    """Demonstrate the data quality monitoring system."""
    logger.info("=== Demonstrating Data Quality Monitoring System ===")
    
    # Create mock data manager
    data_manager = MockDataManager()
    
    # Initialize quality monitor with custom settings
    config = {
        'monitoring_interval': 5,  # Check every 5 seconds for demonstration
        'alert_threshold': 1,      # Alert after one failure for demonstration
        'threshold_missing_pct': 0.01,  # Lower threshold to trigger alerts
        'results_dir': 'quality_monitor_demo'
    }
    
    monitor = DataQualityMonitor(config, data_manager)
    
    # Register alert handler
    monitor.register_alert_handler(log_alert_handler)
    
    # Run a single check
    logger.info("Running a single market data quality check...")
    market_check = monitor.check_market_data()
    logger.info(f"Market data check - Valid: {market_check['valid']}")
    
    # Detect anomalies for a single symbol
    logger.info("Detecting anomalies for a single symbol...")
    anomaly_check = monitor.detect_anomalies(['STOCK001'], days=90)
    logger.info(f"Anomaly check - Valid: {anomaly_check['valid']}")
    if not anomaly_check['valid'] and 'symbols_with_anomalies' in anomaly_check:
        logger.info(f"Found {anomaly_check['symbols_with_anomalies']} symbols with anomalies")
    elif not anomaly_check['valid'] and 'error' in anomaly_check:
        logger.info(f"Anomaly check error: {anomaly_check['error']}")
    
    # Start monitoring in a separate thread
    logger.info("Starting continuous monitoring for 30 seconds...")
    monitor.start_monitoring()
    
    # Let it run for 30 seconds
    try:
        for i in range(6):  # 6 iterations * 5 seconds = 30 seconds
            logger.info(f"Monitoring active... ({i+1}/6)")
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        logger.info("Monitoring stopped")
    
    # Generate a report
    report = monitor.get_report()
    logger.info(f"Generated report with {report.get('total_checks', 0)} checks")
    
    # Save report to file
    with open('quality_monitor_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved monitoring report to quality_monitor_report.json")

if __name__ == "__main__":
    logger.info("Starting Data Quality System Demo")
    
    # Run demonstrations
    demo_basic_quality_checks()
    demo_anomaly_detection()
    demo_monitoring_system()
    
    logger.info("Demo completed") 