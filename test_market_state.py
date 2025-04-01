#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script to verify the MarketState class functionality
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('market_state_test')

# Import the MarketState class
from seto_versal.market.state import MarketState

def test_market_state():
    """Test the MarketState class"""
    logger.info("Testing MarketState class...")
    
    # Initialize the MarketState with test mode
    market_state = MarketState(mode='test', data_dir='data/market', universe='test')
    
    # Test basic attributes
    logger.info(f"Mode: {market_state.mode}")
    logger.info(f"Data directory: {market_state.data_dir}")
    logger.info(f"Symbols count: {len(market_state.symbols)}")
    logger.info(f"Symbols: {market_state.symbols}")
    
    # Test methods
    logger.info("Testing get_tradable_symbols method...")
    tradable_symbols = market_state.get_tradable_symbols()
    logger.info(f"Tradable symbols count: {len(tradable_symbols)}")
    logger.info(f"Tradable symbols: {tradable_symbols}")
    
    # Test market status
    logger.info(f"Is market open: {market_state.is_market_open()}")
    
    # Test get_market_summary method
    logger.info("Testing get_market_summary method...")
    market_summary = market_state.get_market_summary()
    logger.info(f"Market summary: {market_summary}")
    
    logger.info("MarketState test completed successfully!")

if __name__ == '__main__':
    test_market_state() 