#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration utilities for SETO-Versal
"""

import os
import yaml
import logging

logger = logging.getLogger(__name__)

def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the configuration file is not valid YAML
    """
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # Add default values for any missing sections
        if 'feedback' not in config:
            config['feedback'] = {
                'record_trades': True,
                'attribution_analysis': True,
                'performance_metrics': ['sharpe', 'sortino', 'win_rate', 'profit_factor']
            }
            
        return config
    
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {config_path}")
        raise e
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in configuration file: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise e 