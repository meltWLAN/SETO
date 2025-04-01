#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strategy templates module for SETO-Versal
Defines templates for different types of trading strategies
"""

import logging

logger = logging.getLogger(__name__)

class StrategyTemplate:
    """Base class for strategy templates"""
    
    def __init__(self, name, description, category):
        """
        Initialize the strategy template
        
        Args:
            name (str): Template name
            description (str): Template description
            category (str): Strategy category (e.g., 'momentum', 'breakout', 'mean_reversion')
        """
        self.name = name
        self.description = description
        self.category = category
        self.parameter_schema = {}
        self.default_parameters = {}
        
    def generate_parameters(self):
        """
        Generate default parameters for this strategy template
        
        Returns:
            dict: Default parameters
        """
        return self.default_parameters.copy()
    
    def validate_parameters(self, parameters):
        """
        Validate parameters against the schema
        
        Args:
            parameters (dict): Parameters to validate
            
        Returns:
            tuple: (is_valid, error_message)
        """
        is_valid = True
        error_message = ""
        
        # Check that all required parameters are present
        for param_name, param_schema in self.parameter_schema.items():
            if param_schema.get('required', False) and param_name not in parameters:
                is_valid = False
                error_message += f"Missing required parameter: {param_name}. "
                continue
            
            # Check parameter type if present
            if param_name in parameters:
                param_value = parameters[param_name]
                param_type = param_schema.get('type')
                
                if param_type == 'integer' and not isinstance(param_value, int):
                    is_valid = False
                    error_message += f"Parameter {param_name} should be an integer. "
                
                elif param_type == 'number' and not isinstance(param_value, (int, float)):
                    is_valid = False
                    error_message += f"Parameter {param_name} should be a number. "
                
                elif param_type == 'boolean' and not isinstance(param_value, bool):
                    is_valid = False
                    error_message += f"Parameter {param_name} should be a boolean. "
                
                elif param_type == 'string' and not isinstance(param_value, str):
                    is_valid = False
                    error_message += f"Parameter {param_name} should be a string. "
                
                elif param_type == 'array' and not isinstance(param_value, list):
                    is_valid = False
                    error_message += f"Parameter {param_name} should be an array. "
                
                # Check range constraints
                if param_type in ['integer', 'number'] and 'range' in param_schema:
                    min_val = param_schema['range'].get('min')
                    max_val = param_schema['range'].get('max')
                    
                    if min_val is not None and param_value < min_val:
                        is_valid = False
                        error_message += f"Parameter {param_name} should be >= {min_val}. "
                    
                    if max_val is not None and param_value > max_val:
                        is_valid = False
                        error_message += f"Parameter {param_name} should be <= {max_val}. "
        
        return (is_valid, error_message)


class MovingAverageCrossTemplate(StrategyTemplate):
    """Template for Moving Average Crossover strategies"""
    
    def __init__(self):
        """Initialize the MA crossover template"""
        super().__init__(
            name="moving_average_cross",
            description="Moving Average Crossover strategy",
            category="trend_following"
        )
        
        # Define parameter schema
        self.parameter_schema = {
            'fast_period': {
                'type': 'integer',
                'description': 'Period for the fast moving average',
                'required': True,
                'range': {'min': 3, 'max': 50}
            },
            'slow_period': {
                'type': 'integer',
                'description': 'Period for the slow moving average',
                'required': True,
                'range': {'min': 10, 'max': 200}
            },
            'ma_type': {
                'type': 'string',
                'description': 'Type of moving average',
                'required': True,
                'enum': ['SMA', 'EMA', 'WMA']
            },
            'price_field': {
                'type': 'string',
                'description': 'Price field to use for calculation',
                'required': False,
                'enum': ['close', 'open', 'high', 'low', 'typical']
            },
            'signal_threshold': {
                'type': 'number',
                'description': 'Minimum difference between MAs to generate a signal (percentage)',
                'required': False,
                'range': {'min': 0, 'max': 5.0}
            },
            'use_stop_loss': {
                'type': 'boolean',
                'description': 'Whether to use a stop loss',
                'required': False
            },
            'stop_loss_pct': {
                'type': 'number',
                'description': 'Stop loss percentage',
                'required': False,
                'range': {'min': 0.5, 'max': 10.0}
            }
        }
        
        # Set default parameters
        self.default_parameters = {
            'fast_period': 5,
            'slow_period': 20,
            'ma_type': 'EMA',
            'price_field': 'close',
            'signal_threshold': 0.5,
            'use_stop_loss': True,
            'stop_loss_pct': 2.0
        }


class RSITemplate(StrategyTemplate):
    """Template for RSI-based strategies"""
    
    def __init__(self):
        """Initialize the RSI template"""
        super().__init__(
            name="rsi_strategy",
            description="Relative Strength Index strategy",
            category="mean_reversion"
        )
        
        # Define parameter schema
        self.parameter_schema = {
            'rsi_period': {
                'type': 'integer',
                'description': 'Period for RSI calculation',
                'required': True,
                'range': {'min': 2, 'max': 30}
            },
            'overbought_threshold': {
                'type': 'number',
                'description': 'Overbought threshold for RSI',
                'required': True,
                'range': {'min': 60, 'max': 90}
            },
            'oversold_threshold': {
                'type': 'number',
                'description': 'Oversold threshold for RSI',
                'required': True,
                'range': {'min': 10, 'max': 40}
            },
            'signal_period': {
                'type': 'integer',
                'description': 'Number of periods to confirm signal',
                'required': False,
                'range': {'min': 1, 'max': 5}
            },
            'use_ema_filter': {
                'type': 'boolean',
                'description': 'Whether to use EMA as trend filter',
                'required': False
            },
            'ema_period': {
                'type': 'integer',
                'description': 'EMA period for trend filter',
                'required': False,
                'range': {'min': 10, 'max': 50}
            },
            'exit_rsi': {
                'type': 'number',
                'description': 'RSI level to exit positions',
                'required': False,
                'range': {'min': 40, 'max': 60}
            }
        }
        
        # Set default parameters
        self.default_parameters = {
            'rsi_period': 14,
            'overbought_threshold': 70,
            'oversold_threshold': 30,
            'signal_period': 1,
            'use_ema_filter': True,
            'ema_period': 20,
            'exit_rsi': 50
        }


class BreakoutTemplate(StrategyTemplate):
    """Template for breakout strategies"""
    
    def __init__(self):
        """Initialize the breakout template"""
        super().__init__(
            name="breakout_strategy",
            description="Price breakout strategy with volume confirmation",
            category="breakout"
        )
        
        # Define parameter schema
        self.parameter_schema = {
            'breakout_period': {
                'type': 'integer',
                'description': 'Lookback period for breakout calculation',
                'required': True,
                'range': {'min': 5, 'max': 60}
            },
            'volume_factor': {
                'type': 'number',
                'description': 'Minimum volume factor for confirmation',
                'required': True,
                'range': {'min': 1.0, 'max': 5.0}
            },
            'price_factor': {
                'type': 'number',
                'description': 'Minimum price movement factor for breakout',
                'required': False,
                'range': {'min': 0.1, 'max': 3.0}
            },
            'use_atr': {
                'type': 'boolean',
                'description': 'Whether to use ATR for breakout calculation',
                'required': False
            },
            'atr_period': {
                'type': 'integer',
                'description': 'Period for ATR calculation',
                'required': False,
                'range': {'min': 5, 'max': 30}
            },
            'atr_multiplier': {
                'type': 'number',
                'description': 'Multiplier for ATR breakout',
                'required': False,
                'range': {'min': 0.5, 'max': 5.0}
            },
            'stop_loss_atr': {
                'type': 'number',
                'description': 'Stop loss in ATR units',
                'required': False,
                'range': {'min': 0.5, 'max': 3.0}
            }
        }
        
        # Set default parameters
        self.default_parameters = {
            'breakout_period': 20,
            'volume_factor': 1.5,
            'price_factor': 0.5,
            'use_atr': True,
            'atr_period': 14,
            'atr_multiplier': 1.0,
            'stop_loss_atr': 1.5
        }


class MACDTemplate(StrategyTemplate):
    """Template for MACD-based strategies"""
    
    def __init__(self):
        """Initialize the MACD template"""
        super().__init__(
            name="macd_strategy",
            description="Moving Average Convergence Divergence strategy",
            category="momentum"
        )
        
        # Define parameter schema
        self.parameter_schema = {
            'fast_period': {
                'type': 'integer',
                'description': 'Fast period for MACD calculation',
                'required': True,
                'range': {'min': 8, 'max': 24}
            },
            'slow_period': {
                'type': 'integer',
                'description': 'Slow period for MACD calculation',
                'required': True,
                'range': {'min': 15, 'max': 52}
            },
            'signal_period': {
                'type': 'integer',
                'description': 'Signal line period',
                'required': True,
                'range': {'min': 5, 'max': 15}
            },
            'use_histogram': {
                'type': 'boolean',
                'description': 'Whether to use histogram for signals',
                'required': False
            },
            'histogram_threshold': {
                'type': 'number',
                'description': 'Threshold for histogram signals',
                'required': False,
                'range': {'min': 0, 'max': 1.0}
            },
            'use_zero_cross': {
                'type': 'boolean',
                'description': 'Whether to use zero line crossovers',
                'required': False
            },
            'use_divergence': {
                'type': 'boolean',
                'description': 'Whether to use price/MACD divergence signals',
                'required': False
            }
        }
        
        # Set default parameters
        self.default_parameters = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'use_histogram': True,
            'histogram_threshold': 0.0,
            'use_zero_cross': False,
            'use_divergence': False
        }


class BollingerBandsTemplate(StrategyTemplate):
    """Template for Bollinger Bands strategies"""
    
    def __init__(self):
        """Initialize the Bollinger Bands template"""
        super().__init__(
            name="bollinger_bands_strategy",
            description="Bollinger Bands mean reversion strategy",
            category="mean_reversion"
        )
        
        # Define parameter schema
        self.parameter_schema = {
            'bb_period': {
                'type': 'integer',
                'description': 'Period for Bollinger Bands calculation',
                'required': True,
                'range': {'min': 10, 'max': 50}
            },
            'bb_std': {
                'type': 'number',
                'description': 'Standard deviation multiplier',
                'required': True,
                'range': {'min': 1.0, 'max': 4.0}
            },
            'entry_threshold': {
                'type': 'number',
                'description': 'Threshold for entry (% of band width)',
                'required': False,
                'range': {'min': 0, 'max': 100.0}
            },
            'exit_threshold': {
                'type': 'number',
                'description': 'Threshold for exit (% to mean)',
                'required': False,
                'range': {'min': 0, 'max': 100.0}
            },
            'use_rsi_filter': {
                'type': 'boolean',
                'description': 'Whether to use RSI as a filter',
                'required': False
            },
            'rsi_period': {
                'type': 'integer',
                'description': 'RSI period for filter',
                'required': False,
                'range': {'min': 2, 'max': 30}
            },
            'rsi_threshold': {
                'type': 'number',
                'description': 'RSI threshold for filter',
                'required': False,
                'range': {'min': 10, 'max': 40}
            }
        }
        
        # Set default parameters
        self.default_parameters = {
            'bb_period': 20,
            'bb_std': 2.0,
            'entry_threshold': 95.0,
            'exit_threshold': 50.0,
            'use_rsi_filter': True,
            'rsi_period': 14,
            'rsi_threshold': 30
        }


class StrategyTemplateFactory:
    """Factory for creating strategy templates"""
    
    def __init__(self):
        """Initialize the template factory"""
        self.templates = {}
        self._register_default_templates()
    
    def _register_default_templates(self):
        """Register the default templates"""
        self.register_template(MovingAverageCrossTemplate())
        self.register_template(RSITemplate())
        self.register_template(BreakoutTemplate())
        self.register_template(MACDTemplate())
        self.register_template(BollingerBandsTemplate())
        
        logger.info(f"Registered {len(self.templates)} strategy templates")
    
    def register_template(self, template):
        """
        Register a new strategy template
        
        Args:
            template (StrategyTemplate): Strategy template to register
        """
        if not isinstance(template, StrategyTemplate):
            raise TypeError("Template must be an instance of StrategyTemplate")
        
        self.templates[template.name] = template
        logger.debug(f"Registered template: {template.name}")
    
    def get_template(self, template_name):
        """
        Get a strategy template by name
        
        Args:
            template_name (str): Name of the template
            
        Returns:
            StrategyTemplate: Strategy template, or None if not found
        """
        return self.templates.get(template_name)
    
    def get_all_templates(self):
        """
        Get all registered templates
        
        Returns:
            dict: Dictionary of all templates
        """
        return self.templates
    
    def get_templates_by_category(self, category):
        """
        Get templates by category
        
        Args:
            category (str): Strategy category
            
        Returns:
            list: List of templates in the category
        """
        return [t for t in self.templates.values() if t.category == category]
    
    def create_strategy_parameters(self, template_name, custom_params=None):
        """
        Create strategy parameters from a template
        
        Args:
            template_name (str): Name of the template
            custom_params (dict, optional): Custom parameters to override defaults
            
        Returns:
            tuple: (parameters, is_valid, error_message)
        """
        template = self.get_template(template_name)
        if not template:
            return None, False, f"Template not found: {template_name}"
        
        # Start with default parameters
        parameters = template.generate_parameters()
        
        # Override with custom parameters if provided
        if custom_params:
            for key, value in custom_params.items():
                parameters[key] = value
        
        # Validate parameters
        is_valid, error_message = template.validate_parameters(parameters)
        
        return parameters, is_valid, error_message 