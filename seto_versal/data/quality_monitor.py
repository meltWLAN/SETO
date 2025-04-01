import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import os
import json

from seto_versal.data.quality import DataQualityChecker, validate_signals, detect_data_anomalies

logger = logging.getLogger(__name__)

class DataQualityMonitor:
    """Monitor for continuously checking data quality and alerting on issues.
    
    This class provides a system for monitoring data quality in real time,
    scheduling regular checks, and generating alerts when quality issues
    are detected.
    """
    
    def __init__(self, config=None, data_manager=None):
        """Initialize the data quality monitor.
        
        Args:
            config: Configuration dictionary for quality thresholds and monitoring settings
            data_manager: Reference to the data manager to pull data for monitoring
        """
        self.config = config or {}
        self.data_manager = data_manager
        
        # Initialize the quality checker
        self.quality_checker = DataQualityChecker(config)
        
        # Monitoring settings
        self.monitoring_interval = self.config.get('monitoring_interval', 3600)  # Default: hourly
        self.alert_threshold = self.config.get('alert_threshold', 3)  # Alert after this many consecutive failures
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Monitoring state
        self.last_check_time = None
        self.consecutive_failures = 0
        self.alert_handlers = []
        
        # Results storage
        self.results_dir = self.config.get('results_dir', 'data_quality_reports')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def register_alert_handler(self, handler_func):
        """Register a function to be called when an alert is triggered.
        
        Args:
            handler_func: Function that takes alert_data dict as parameter
        """
        if callable(handler_func):
            self.alert_handlers.append(handler_func)
            logger.info(f"Registered alert handler: {handler_func.__name__}")
        else:
            logger.error(f"Cannot register non-callable alert handler: {handler_func}")
    
    def trigger_alert(self, alert_data):
        """Trigger all registered alert handlers with the alert data.
        
        Args:
            alert_data: Dictionary containing alert information
        """
        for handler in self.alert_handlers:
            try:
                handler(alert_data)
            except Exception as e:
                logger.error(f"Error in alert handler {handler.__name__}: {str(e)}")
        
        # Log the alert
        logger.warning(f"Data quality alert: {alert_data['message']}")
        
        # Save alert to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alert_file = os.path.join(self.results_dir, f"alert_{timestamp}.json")
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2, default=str)
    
    def check_market_data(self):
        """Check the quality of market data from the data manager.
        
        Returns:
            dict: Results of the market data quality check
        """
        if self.data_manager is None:
            logger.error("Cannot check market data: No data manager provided")
            return {"valid": False, "error": "No data manager provided"}
        
        try:
            # Get recent market data
            market_data = self.data_manager.get_recent_market_data()
            
            if market_data is None or market_data.empty:
                logger.warning("Empty market data returned from data manager")
                return {"valid": False, "error": "No market data available"}
            
            # Run quality checks
            results = self.quality_checker.validate_market_data(market_data)
            
            # Update monitoring state
            self.last_check_time = datetime.now()
            
            if results['valid']:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                
                # Check if we need to trigger an alert
                if self.consecutive_failures >= self.alert_threshold:
                    alert_data = {
                        "timestamp": datetime.now(),
                        "type": "market_data_quality",
                        "message": f"Market data quality check failed {self.consecutive_failures} times consecutively",
                        "details": results['details']
                    }
                    self.trigger_alert(alert_data)
            
            return results
        
        except Exception as e:
            logger.error(f"Error during market data quality check: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    def check_signal_data(self, signal_sources=None):
        """Check the quality of signal data from various agent sources.
        
        Args:
            signal_sources: List of agent names to check signals for, or None for all
            
        Returns:
            dict: Results of the signal data quality check by source
        """
        if self.data_manager is None:
            logger.error("Cannot check signal data: No data manager provided")
            return {"valid": False, "error": "No data manager provided"}
        
        try:
            # Get signal data from each source
            results = {}
            
            if signal_sources is None and hasattr(self.data_manager, 'get_signal_sources'):
                signal_sources = self.data_manager.get_signal_sources()
            
            if not signal_sources:
                logger.warning("No signal sources specified or available")
                return {"valid": False, "error": "No signal sources available"}
            
            for source in signal_sources:
                # Get signals for this source
                if hasattr(self.data_manager, 'get_agent_signals'):
                    signals_df = self.data_manager.get_agent_signals(source)
                else:
                    logger.warning(f"Data manager has no get_agent_signals method, skipping source: {source}")
                    continue
                
                # Validate signals
                source_result = validate_signals(signals_df)
                results[source] = source_result
            
            # Overall validity
            all_valid = all(result.get('valid', False) for result in results.values())
            
            # Update state and potentially trigger alert
            if not all_valid:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.alert_threshold:
                    failed_sources = [source for source, result in results.items() 
                                    if not result.get('valid', False)]
                    alert_data = {
                        "timestamp": datetime.now(),
                        "type": "signal_data_quality",
                        "message": f"Signal quality check failed for sources: {failed_sources}",
                        "details": {k: v for k, v in results.items() if not v.get('valid', False)}
                    }
                    self.trigger_alert(alert_data)
            else:
                self.consecutive_failures = 0
            
            return {
                "valid": all_valid,
                "timestamp": datetime.now(),
                "details": results
            }
            
        except Exception as e:
            logger.error(f"Error during signal data quality check: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    def detect_anomalies(self, symbols=None, days=5):
        """Detect anomalies in recent market data for specified symbols.
        
        Args:
            symbols: List of symbols to check, or None for all available
            days: Number of days of historical data to analyze
            
        Returns:
            dict: Anomaly detection results by symbol
        """
        if self.data_manager is None:
            logger.error("Cannot detect anomalies: No data manager provided")
            return {"valid": False, "error": "No data manager provided"}
        
        try:
            results = {}
            
            # Get symbols if not provided
            if symbols is None and hasattr(self.data_manager, 'get_symbols'):
                symbols = self.data_manager.get_symbols()
            
            if not symbols:
                logger.warning("No symbols specified or available")
                return {"valid": False, "error": "No symbols available"}
            
            # Get historical data for each symbol
            for symbol in symbols:
                try:
                    # Get data for this symbol
                    if hasattr(self.data_manager, 'get_historical_data'):
                        start_date = datetime.now() - timedelta(days=days)
                        end_date = datetime.now()
                        data = self.data_manager.get_historical_data(symbol, start_date, end_date)
                    else:
                        logger.warning(f"Data manager has no get_historical_data method, skipping symbol: {symbol}")
                        continue
                    
                    if data is None or data.empty:
                        results[symbol] = {"valid": False, "error": "No data available"}
                        continue
                    
                    # Detect anomalies
                    anomaly_df = detect_data_anomalies(data)
                    
                    # Check if anomalies were found
                    has_anomalies = anomaly_df.get('is_anomaly', pd.Series()).any()
                    
                    if has_anomalies:
                        # Get anomaly dates
                        anomaly_dates = anomaly_df[anomaly_df['is_anomaly']].index.tolist()
                        results[symbol] = {
                            "valid": False,
                            "anomalies_detected": True,
                            "anomaly_count": len(anomaly_dates),
                            "anomaly_dates": [str(d) for d in anomaly_dates[:5]]  # First 5 anomalies
                        }
                    else:
                        results[symbol] = {"valid": True, "anomalies_detected": False}
                except Exception as e:
                    logger.error(f"Error detecting anomalies for symbol {symbol}: {str(e)}")
                    results[symbol] = {"valid": False, "error": str(e)}
            
            # Check if any symbols have anomalies
            symbols_with_anomalies = [s for s, r in results.items() 
                                    if r.get('anomalies_detected', False)]
            
            if symbols_with_anomalies:
                alert_data = {
                    "timestamp": datetime.now(),
                    "type": "market_data_anomalies",
                    "message": f"Anomalies detected in {len(symbols_with_anomalies)} symbols",
                    "symbols": symbols_with_anomalies
                }
                self.trigger_alert(alert_data)
            
            return {
                "valid": len(symbols_with_anomalies) == 0,
                "timestamp": datetime.now(),
                "symbols_with_anomalies": len(symbols_with_anomalies),
                "details": results
            }
            
        except Exception as e:
            logger.error(f"Error during anomaly detection: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    def monitoring_loop(self):
        """Run the continuous monitoring loop in a separate thread."""
        logger.info("Starting data quality monitoring loop")
        
        while self.monitoring_active:
            try:
                # Run market data quality check
                market_results = self.check_market_data()
                logger.debug(f"Market data quality check: valid={market_results.get('valid', False)}")
                
                # Run signal data quality check if market data is valid
                if market_results.get('valid', False):
                    signal_results = self.check_signal_data()
                    logger.debug(f"Signal data quality check: valid={signal_results.get('valid', False)}")
                
                # Detect anomalies (periodically, not every run)
                current_hour = datetime.now().hour
                if current_hour in [9, 13, 15]:  # Run at specific hours
                    anomaly_results = self.detect_anomalies()
                    logger.debug(f"Anomaly detection: valid={anomaly_results.get('valid', False)}")
                
                # Save results periodically
                self.save_results()
                
                # Sleep until next check
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Shorter sleep on error
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self.monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("Data quality monitoring started")
        else:
            logger.warning("Monitoring is already active")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=30)
            logger.info("Data quality monitoring stopped")
        else:
            logger.warning("Monitoring is not active")
    
    def save_results(self):
        """Save the latest monitoring results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"quality_check_{timestamp}.json")
        
        # Get the latest results
        results = {
            "timestamp": datetime.now(),
            "market_data": self.quality_checker.validation_results.get(self.last_check_time, {}),
            "consecutive_failures": self.consecutive_failures
        }
        
        # Save to file
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def get_report(self, days=7):
        """Generate a summary report of data quality over time.
        
        Args:
            days: Number of days to include in the report
            
        Returns:
            dict: Summary report of data quality metrics
        """
        # Get validation history
        history = self.quality_checker.get_validation_history(days)
        
        if not history:
            return {"error": "No validation history available"}
        
        # Calculate success rate
        total_checks = len(history)
        successful_checks = sum(1 for result in history.values() if result.get('valid', False))
        success_rate = successful_checks / total_checks if total_checks > 0 else 0
        
        # Find common issues
        common_issues = {}
        for timestamp, result in history.items():
            details = result.get('details', {})
            for check_name, check_result in details.items():
                if not check_result.get('valid', True):
                    if check_name not in common_issues:
                        common_issues[check_name] = 0
                    common_issues[check_name] += 1
        
        # Sort issues by frequency
        common_issues = {k: v for k, v in sorted(common_issues.items(), 
                                               key=lambda item: item[1], reverse=True)}
        
        return {
            "report_date": datetime.now(),
            "period_days": days,
            "total_checks": total_checks,
            "successful_checks": successful_checks,
            "success_rate": success_rate,
            "common_issues": common_issues,
            "most_recent_check": max(history.keys()) if history else None,
            "consecutive_failures": self.consecutive_failures
        }


# Example alert handlers that can be registered with the monitor
def email_alert_handler(alert_data):
    """Send an email alert for critical data quality issues.
    
    This is a placeholder implementation. In a real system, this would
    connect to an email service.
    
    Args:
        alert_data: Dictionary containing alert information
    """
    logger.info(f"[EMAIL ALERT] Would send email: {alert_data['message']}")
    # In a real implementation, this would connect to an SMTP server
    # and send the email to appropriate recipients


def slack_alert_handler(alert_data):
    """Send a Slack alert for critical data quality issues.
    
    This is a placeholder implementation. In a real system, this would
    connect to the Slack API.
    
    Args:
        alert_data: Dictionary containing alert information
    """
    logger.info(f"[SLACK ALERT] Would send to channel: {alert_data['message']}")
    # In a real implementation, this would use the Slack API to post
    # a message to the appropriate channel


def log_alert_handler(alert_data):
    """Log alerts to a file for record-keeping.
    
    Args:
        alert_data: Dictionary containing alert information
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_message = f"{timestamp} - {alert_data['type']}: {alert_data['message']}"
    
    # Log to a specific alerts log file
    with open('data_quality_alerts.log', 'a') as f:
        f.write(f"{alert_message}\n")
    
    # Also log through the standard logging system
    logger.warning(alert_message) 