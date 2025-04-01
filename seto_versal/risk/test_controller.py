#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test cases for RiskController
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import yaml
from datetime import datetime, timedelta

from seto_versal.risk.controller import RiskController, RiskLevel


class TestRiskController(unittest.TestCase):
    """Test cases for the RiskController class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.rules_file = os.path.join(self.temp_dir, 'test_rules.yaml')
        
        # Create a temporary rules file
        rules = {
            'max_drawdown_percent': 2.0,
            'max_position_percent': 20.0,
            'max_sector_percent': 30.0,
            'min_cash_percent': 15.0,
            'consecutive_loss_limit': 2,
            'rules_by_risk_level': {
                'low': {'max_position_percent': 10.0},
                'medium': {'max_position_percent': 20.0},
                'high': {'max_position_percent': 30.0},
                'critical': {'max_position_percent': 5.0},
            }
        }
        
        with open(self.rules_file, 'w') as f:
            yaml.dump(rules, f)
        
        # Create configuration
        self.config = {
            'name': 'test_controller',
            'risk_rules_file': self.rules_file,
            'initial_risk_level': 'medium',
            'max_drawdown_percent': 2.0
        }
        
        # Create controller
        self.controller = RiskController(self.config)
        
        # Create mock objects
        self.mock_portfolio = MagicMock()
        self.mock_portfolio.get_total_value.return_value = 100000.0
        self.mock_portfolio.get_cash.return_value = 50000.0
        self.mock_portfolio.get_current_drawdown.return_value = 1.0
        
        self.mock_market = MagicMock()
        self.mock_market.get_market_regime.return_value = 'neutral'
        
    def tearDown(self):
        """Clean up test fixtures"""
        os.remove(self.rules_file)
        os.rmdir(self.temp_dir)
    
    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.controller.name, 'test_controller')
        self.assertEqual(self.controller.current_risk_level, RiskLevel.MEDIUM)
        self.assertEqual(self.controller.max_drawdown_percent, 2.0)
        self.assertEqual(self.controller.max_position_percent, 20.0)
    
    def test_load_risk_rules(self):
        """Test loading risk rules"""
        rules = self.controller._load_risk_rules()
        self.assertEqual(rules['max_drawdown_percent'], 2.0)
        self.assertEqual(rules['max_position_percent'], 20.0)
    
    def test_validate_trade_valid(self):
        """Test validating a valid trade"""
        # Mock position
        mock_position = {'symbol': 'AAPL', 'quantity': 0, 'value': 0.0, 'avg_price': 0.0}
        self.mock_portfolio.get_position.return_value = mock_position
        
        # Create valid trade
        trade = {
            'symbol': 'AAPL',
            'direction': 'buy',
            'quantity': 10,
            'stop_loss': 90.0
        }
        
        # Patch internal methods
        with patch.object(self.controller, '_get_price', return_value=100.0):
            with patch.object(self.controller, '_get_sector', return_value='Technology'):
                with patch.object(self.controller, '_get_sector_exposure', return_value=5.0):
                    valid, reason = self.controller.validate_trade(trade, self.mock_portfolio, self.mock_market)
                    
                    self.assertTrue(valid)
                    self.assertIsNone(reason)
    
    def test_validate_trade_exceeds_position_limit(self):
        """Test validating a trade that exceeds position limits"""
        # Mock position
        mock_position = {'symbol': 'AAPL', 'quantity': 0, 'value': 0.0, 'avg_price': 0.0}
        self.mock_portfolio.get_position.return_value = mock_position
        
        # Create trade that would exceed position limit (20% of portfolio)
        trade = {
            'symbol': 'AAPL',
            'direction': 'buy',
            'quantity': 250,  # 250 shares at $100 = $25,000 = 25% of portfolio
            'stop_loss': 90.0
        }
        
        # Patch internal methods
        with patch.object(self.controller, '_get_price', return_value=100.0):
            with patch.object(self.controller, '_get_sector', return_value='Technology'):
                with patch.object(self.controller, '_get_sector_exposure', return_value=5.0):
                    valid, reason = self.controller.validate_trade(trade, self.mock_portfolio, self.mock_market)
                    
                    self.assertFalse(valid)
                    self.assertIn("Position size", reason)
    
    def test_validate_trade_exceeds_sector_limit(self):
        """Test validating a trade that exceeds sector limits"""
        # Mock position
        mock_position = {'symbol': 'AAPL', 'quantity': 0, 'value': 0.0, 'avg_price': 0.0}
        self.mock_portfolio.get_position.return_value = mock_position
        
        # Create valid trade size
        trade = {
            'symbol': 'AAPL',
            'direction': 'buy',
            'quantity': 100,  # $10,000 = 10% of portfolio
            'stop_loss': 90.0
        }
        
        # Patch internal methods
        with patch.object(self.controller, '_get_price', return_value=100.0):
            with patch.object(self.controller, '_get_sector', return_value='Technology'):
                # Technology sector already at 25%, would exceed 30% limit
                with patch.object(self.controller, '_get_sector_exposure', return_value=25.0):
                    valid, reason = self.controller.validate_trade(trade, self.mock_portfolio, self.mock_market)
                    
                    self.assertFalse(valid)
                    self.assertIn("Sector Technology exposure", reason)
    
    def test_validate_trade_missing_stop_loss(self):
        """Test validating a trade missing stop loss"""
        # Mock position
        mock_position = {'symbol': 'AAPL', 'quantity': 0, 'value': 0.0, 'avg_price': 0.0}
        self.mock_portfolio.get_position.return_value = mock_position
        
        # Create trade without stop loss
        trade = {
            'symbol': 'AAPL',
            'direction': 'buy',
            'quantity': 10
        }
        
        # Patch internal methods
        with patch.object(self.controller, '_get_price', return_value=100.0):
            with patch.object(self.controller, '_get_sector', return_value='Technology'):
                with patch.object(self.controller, '_get_sector_exposure', return_value=5.0):
                    valid, reason = self.controller.validate_trade(trade, self.mock_portfolio, self.mock_market)
                    
                    self.assertFalse(valid)
                    self.assertIn("stop loss", reason.lower())
    
    def test_risk_level_update(self):
        """Test updating risk level based on drawdown"""
        # Set drawdown to 80% of max (1.6% of 2.0%)
        self.mock_portfolio.get_current_drawdown.return_value = 1.6
        
        self.controller.update_risk_level(self.mock_portfolio, self.mock_market)
        
        # Should elevate to CRITICAL risk level
        self.assertEqual(self.controller.current_risk_level, RiskLevel.CRITICAL)
        
        # Reset and test lower drawdown
        self.controller.current_risk_level = RiskLevel.MEDIUM
        self.mock_portfolio.get_current_drawdown.return_value = 0.5
        
        self.controller.update_risk_level(self.mock_portfolio, self.mock_market)
        
        # Should lower to LOW risk level
        self.assertEqual(self.controller.current_risk_level, RiskLevel.LOW)
    
    def test_record_trade_result(self):
        """Test recording trade results and consecutive losses"""
        # Start with no consecutive losses
        self.assertEqual(self.controller.current_consecutive_losses, 0)
        
        # Record a losing trade
        losing_trade = {'symbol': 'AAPL', 'pnl': -500.0}
        self.controller.record_trade_result(losing_trade)
        
        # Should have 1 consecutive loss
        self.assertEqual(self.controller.current_consecutive_losses, 1)
        
        # Record another losing trade
        self.controller.record_trade_result(losing_trade)
        
        # Should have 2 consecutive losses and be in cooling period
        self.assertEqual(self.controller.current_consecutive_losses, 2)
        self.assertIsNotNone(self.controller.cooling_until)
        
        # Record a winning trade
        winning_trade = {'symbol': 'AAPL', 'pnl': 500.0}
        self.controller.record_trade_result(winning_trade)
        
        # Should reset consecutive losses
        self.assertEqual(self.controller.current_consecutive_losses, 0)
    
    def test_cooling_period(self):
        """Test cooling period enforcement"""
        # Set cooling period
        self.controller.cooling_until = datetime.now() + timedelta(hours=1)
        
        # Try to validate a trade during cooling period
        trade = {
            'symbol': 'AAPL',
            'direction': 'buy',
            'quantity': 10,
            'stop_loss': 90.0
        }
        
        valid, reason = self.controller.validate_trade(trade, self.mock_portfolio, self.mock_market)
        
        # Should reject the trade
        self.assertFalse(valid)
        self.assertIn("cooling period", reason.lower())
    
    def test_generate_risk_report(self):
        """Test generating risk report"""
        # Add some history data
        self.controller.drawdown_history.append({
            'timestamp': datetime.now().isoformat(),
            'drawdown': 1.0
        })
        
        rule_violation = {
            'timestamp': datetime.now().isoformat(),
            'rule': 'max_position_percent',
            'symbol': 'AAPL',
            'reason': 'Position too large'
        }
        self.controller.rule_violations.append(rule_violation)
        
        # Generate report
        report = self.controller.generate_risk_report()
        
        # Verify report structure
        self.assertIn('timestamp', report)
        self.assertIn('risk_level', report)
        self.assertIn('consecutive_losses', report)
        self.assertIn('recent_drawdowns', report)
        self.assertIn('rule_violations', report)
        
        # Check values
        self.assertEqual(report['risk_level'], 'medium')
        self.assertEqual(report['consecutive_losses'], 0)
        self.assertEqual(len(report['recent_drawdowns']), 1)
        self.assertEqual(report['rule_violations']['max_position_percent'], 1)


if __name__ == '__main__':
    unittest.main() 