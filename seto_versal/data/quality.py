import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataQualityChecker:
    """A class to check data quality for market data and trading signals."""
    
    def __init__(self, config=None):
        """Initialize the data quality checker with configuration.
        
        Args:
            config: Configuration dict containing quality thresholds
        """
        self.config = config or {}
        self.threshold_missing_pct = self.config.get('threshold_missing_pct', 0.05)
        self.threshold_stale_days = self.config.get('threshold_stale_days', 3)
        self.threshold_price_jump = self.config.get('threshold_price_jump', 0.1)
        self.threshold_volume_zero_pct = self.config.get('threshold_volume_zero_pct', 0.2)
        
        # Store validation history
        self.validation_results = {}
        
    def check_missing_values(self, df, columns=None):
        """Check for missing values in the dataframe.
        
        Args:
            df: pandas DataFrame to check
            columns: specific columns to check, if None check all
            
        Returns:
            dict: Results of the missing value check with percentage and affected columns
        """
        if df is None or df.empty:
            return {'valid': False, 'error': 'Empty dataframe'}
            
        columns = columns or df.columns
        missing_counts = df[columns].isna().sum()
        missing_pct = missing_counts / len(df)
        
        problematic_columns = missing_pct[missing_pct > self.threshold_missing_pct].index.tolist()
        
        result = {
            'valid': len(problematic_columns) == 0,
            'missing_columns': problematic_columns,
            'missing_pct': missing_pct.to_dict(),
            'threshold': self.threshold_missing_pct
        }
        
        return result
    
    def check_stale_data(self, df, date_column='date'):
        """Check if the data is stale based on the most recent date.
        
        Args:
            df: pandas DataFrame to check
            date_column: name of the date column
            
        Returns:
            dict: Results of the staleness check
        """
        if df is None or df.empty:
            return {'valid': False, 'error': 'Empty dataframe'}
            
        if date_column not in df.columns:
            return {'valid': False, 'error': f'Date column {date_column} not found'}
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
            
        most_recent_date = df[date_column].max()
        current_date = pd.Timestamp(datetime.now().date())
        days_difference = (current_date - most_recent_date).days
        
        result = {
            'valid': days_difference <= self.threshold_stale_days,
            'most_recent_date': most_recent_date,
            'days_since_update': days_difference,
            'threshold': self.threshold_stale_days
        }
        
        return result
    
    def check_price_jumps(self, df, price_column='close', return_column=None):
        """Check for abnormal price jumps in the data.
        
        Args:
            df: pandas DataFrame to check
            price_column: name of the price column
            return_column: name of the return column, if None calculate from price
            
        Returns:
            dict: Results of the price jump check
        """
        if df is None or df.empty:
            return {'valid': False, 'error': 'Empty dataframe'}
            
        if price_column not in df.columns:
            return {'valid': False, 'error': f'Price column {price_column} not found'}
        
        # Calculate returns if return column not provided
        if return_column is None:
            returns = df[price_column].pct_change()
        else:
            if return_column not in df.columns:
                return {'valid': False, 'error': f'Return column {return_column} not found'}
            returns = df[return_column]
        
        # Find extreme price jumps
        abs_returns = returns.abs()
        extreme_jumps = abs_returns > self.threshold_price_jump
        jump_indices = df.index[extreme_jumps].tolist()
        
        result = {
            'valid': len(jump_indices) == 0,
            'extreme_jumps': len(jump_indices),
            'jump_indices': jump_indices,
            'max_jump': abs_returns.max(),
            'threshold': self.threshold_price_jump
        }
        
        return result
    
    def check_zero_volume(self, df, volume_column='volume'):
        """Check for zero volume entries which could indicate data issues.
        
        Args:
            df: pandas DataFrame to check
            volume_column: name of the volume column
            
        Returns:
            dict: Results of the zero volume check
        """
        if df is None or df.empty:
            return {'valid': False, 'error': 'Empty dataframe'}
            
        if volume_column not in df.columns:
            return {'valid': False, 'error': f'Volume column {volume_column} not found'}
        
        zero_volume = (df[volume_column] == 0).sum()
        zero_volume_pct = zero_volume / len(df)
        
        result = {
            'valid': zero_volume_pct <= self.threshold_volume_zero_pct,
            'zero_volume_count': zero_volume,
            'zero_volume_pct': zero_volume_pct,
            'threshold': self.threshold_volume_zero_pct
        }
        
        return result
    
    def check_data_consistency(self, df, date_column='date', symbol_column='symbol'):
        """Check data consistency across dates and symbols.
        
        Args:
            df: pandas DataFrame to check
            date_column: name of the date column
            symbol_column: name of the symbol column
            
        Returns:
            dict: Results of the consistency check
        """
        if df is None or df.empty:
            return {'valid': False, 'error': 'Empty dataframe'}
            
        required_columns = [date_column, symbol_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return {'valid': False, 'error': f'Missing columns: {missing_columns}'}
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Check for duplicate entries
        duplicates = df.duplicated(subset=[date_column, symbol_column]).sum()
        
        # Check for consistency in date coverage per symbol
        symbols = df[symbol_column].unique()
        dates_per_symbol = {symbol: set(df[df[symbol_column] == symbol][date_column]) 
                           for symbol in symbols}
        
        all_dates = set()
        for dates in dates_per_symbol.values():
            all_dates.update(dates)
        
        missing_dates_per_symbol = {
            symbol: len(all_dates - dates_per_symbol[symbol])
            for symbol in symbols
        }
        
        symbols_with_missing = [symbol for symbol, count in missing_dates_per_symbol.items() 
                               if count > 0]
        
        result = {
            'valid': duplicates == 0 and len(symbols_with_missing) == 0,
            'duplicates': duplicates,
            'symbols_with_missing_dates': len(symbols_with_missing),
            'symbol_examples': symbols_with_missing[:5] if symbols_with_missing else []
        }
        
        return result
    
    def validate_market_data(self, df, date_column='date', symbol_column='symbol', 
                           price_column='close', volume_column='volume'):
        """Run all data quality checks on market data.
        
        Args:
            df: pandas DataFrame to check
            date_column: name of the date column
            symbol_column: name of the symbol column
            price_column: name of the price column
            volume_column: name of the volume column
            
        Returns:
            dict: Consolidated results of all checks
        """
        results = {}
        
        # Run all checks
        results['missing_values'] = self.check_missing_values(df)
        results['stale_data'] = self.check_stale_data(df, date_column)
        results['price_jumps'] = self.check_price_jumps(df, price_column)
        results['zero_volume'] = self.check_zero_volume(df, volume_column)
        results['consistency'] = self.check_data_consistency(df, date_column, symbol_column)
        
        # Overall validity
        all_valid = all(check['valid'] for check in results.values())
        
        # Record validation timestamp
        timestamp = datetime.now()
        self.validation_results[timestamp] = {
            'valid': all_valid,
            'details': results
        }
        
        # Log results
        if all_valid:
            logger.info(f"Data validation passed at {timestamp}")
        else:
            failed_checks = [check for check, result in results.items() if not result['valid']]
            logger.warning(f"Data validation failed for checks: {failed_checks} at {timestamp}")
        
        return {
            'valid': all_valid,
            'timestamp': timestamp,
            'details': results
        }
    
    def get_validation_history(self, days=7):
        """Get validation history for the specified number of days.
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            dict: Validation history with timestamps as keys
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_validations = {
            timestamp: results
            for timestamp, results in self.validation_results.items()
            if timestamp >= cutoff_date
        }
        
        return recent_validations


def validate_signals(signals_df, min_confidence=0.5, required_fields=None):
    """Validate trading signals dataframe.
    
    Args:
        signals_df: DataFrame containing trading signals
        min_confidence: Minimum confidence threshold for signals
        required_fields: List of required fields in the signals dataframe
        
    Returns:
        dict: Validation results
    """
    if signals_df is None or signals_df.empty:
        return {'valid': False, 'error': 'Empty signals dataframe'}
    
    # Default required fields
    required_fields = required_fields or ['symbol', 'timestamp', 'signal', 'confidence']
    
    # Check for required fields
    missing_fields = [field for field in required_fields if field not in signals_df.columns]
    if missing_fields:
        return {
            'valid': False, 
            'error': f'Missing required fields: {missing_fields}'
        }
    
    # Check confidence values
    if 'confidence' in signals_df.columns:
        low_confidence = (signals_df['confidence'] < min_confidence).sum()
        if low_confidence > 0:
            return {
                'valid': False,
                'error': f'Found {low_confidence} signals with confidence below threshold {min_confidence}'
            }
    
    # Check signal values
    if 'signal' in signals_df.columns:
        unique_signals = signals_df['signal'].unique()
        valid_signals = [-1, 0, 1]  # Typically -1 (sell), 0 (hold), 1 (buy)
        invalid_signals = [sig for sig in unique_signals if sig not in valid_signals]
        
        if invalid_signals:
            return {
                'valid': False,
                'error': f'Found invalid signal values: {invalid_signals}'
            }
    
    # All checks passed
    return {'valid': True}


def detect_data_anomalies(df, price_column='close', volume_column='volume', window=20):
    """Detect anomalies in price and volume data.
    
    Args:
        df: DataFrame containing price and volume data
        price_column: Name of the price column
        volume_column: Name of the volume column
        window: Window size for rolling statistics
        
    Returns:
        DataFrame: Original dataframe with anomaly indicators
    """
    if df is None or df.empty:
        logger.warning("Empty dataframe provided for anomaly detection")
        return df
    
    result_df = df.copy()
    
    # Price anomalies (using rolling z-score)
    if price_column in df.columns:
        # Calculate rolling mean and std
        rolling_mean = df[price_column].rolling(window=window).mean()
        rolling_std = df[price_column].rolling(window=window).std()
        
        # Calculate z-score
        z_scores = (df[price_column] - rolling_mean) / rolling_std
        
        # Flag anomalies (z-score > 3 standard deviations)
        result_df['price_anomaly'] = (z_scores.abs() > 3)
    
    # Volume anomalies (sudden spikes)
    if volume_column in df.columns:
        # Calculate rolling median and median absolute deviation (more robust than mean/std)
        rolling_median = df[volume_column].rolling(window=window).median()
        rolling_mad = (df[volume_column] - rolling_median).abs().rolling(window=window).median()
        
        # Calculate modified z-score
        modified_z_scores = 0.6745 * (df[volume_column] - rolling_median) / rolling_mad
        
        # Flag anomalies (modified z-score > 3.5, common threshold for outliers)
        result_df['volume_anomaly'] = (modified_z_scores.abs() > 3.5)
    
    # Combined anomaly flag
    if 'price_anomaly' in result_df.columns and 'volume_anomaly' in result_df.columns:
        result_df['is_anomaly'] = result_df['price_anomaly'] | result_df['volume_anomaly']
    
    return result_df 