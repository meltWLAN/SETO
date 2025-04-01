#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal: Self-Evolving Traderverse Intelligence
Main entry point for the system
"""

import os
import time
import yaml
import logging
import argparse
from datetime import datetime

from seto_versal.market.state import MarketState
from seto_versal.agents.factory import AgentFactory
from seto_versal.coordinator.coordinator import TradeCoordinator
from seto_versal.executor.executor import TradeExecutor
from seto_versal.philosophy.risk_control import RiskManager
from seto_versal.feedback.analyzer import FeedbackAnalyzer
from seto_versal.evolution.engine import EvolutionEngine
from seto_versal.narrator.storyteller import Narrator
from seto_versal.utils.config import load_config


class SetoVersal:
    """Main class that orchestrates the entire trading system"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize the SETO-Versal system"""
        self.config = load_config(config_path)
        self.setup_logging()
        
        self.logger.info(f"Initializing SETO-Versal v{self.config['system']['version']}")
        
        # Initialize core components
        try:
            # Ensure required sections exist in config
            if 'system' not in self.config:
                self.config['system'] = {'version': '0.1.0', 'mode': 'backtest'}
            
            if 'market' not in self.config:
                self.config['market'] = {'universe': 'default', 'update_interval': 5}
            
            if 'logging' not in self.config:
                self.config['logging'] = {'level': 'INFO'}
            
            if 'agents' not in self.config:
                self.config['agents'] = {'enabled': True}
            
            if 'philosophy' not in self.config:
                self.config['philosophy'] = {'risk_level': 'medium'}
            
            if 'evolution' not in self.config:
                self.config['evolution'] = {'enabled': True}
            
            if 'feedback' not in self.config:
                self.config['feedback'] = {}
            
            # Initialize market state with proper mode setting
            mode = self.config['system'].get('mode', 'backtest')
            self.logger.info(f"Setting system mode to: {mode}")
            
            # Configure market state with mode
            market_config = self.config['market'].copy()
            market_config['mode'] = mode
            
            # Default universe if not specified
            if 'universe' not in market_config:
                market_config['universe'] = 'default'  # Will load any available stocks
            
            # Ensure update_interval exists
            if 'update_interval' not in market_config:
                market_config['update_interval'] = 5
            
            # Ensure market data directory exists
            data_dir = market_config.get('data_dir', os.path.join(os.path.dirname(__file__), 'data', 'market'))
            os.makedirs(data_dir, exist_ok=True)
            market_config['data_dir'] = data_dir
            
            # Initialize core components
            self.logger.info("Initializing market state...")
            self.market_state = MarketState(market_config)
            
            self.logger.info("Initializing agent factory...")
            self.agent_factory = AgentFactory(self.config['agents'])
            
            self.logger.info("Creating agents...")
            self.agents = self.agent_factory.create_agents()
            
            self.logger.info("Initializing trade coordinator...")
            self.coordinator = TradeCoordinator(self.config.get('coordinator', {}))
            
            self.logger.info("Initializing trade executor...")
            self.executor = TradeExecutor(self.config)
            
            self.logger.info("Initializing risk manager...")
            self.risk_manager = RiskManager(self.config.get('philosophy', {'risk_level': 'medium'}))
            
            self.logger.info("Initializing feedback analyzer...")
            self.feedback_analyzer = FeedbackAnalyzer(self.config.get('feedback', {}))
            
            self.logger.info("Initializing evolution engine...")
            self.evolution_engine = EvolutionEngine(self.config.get('evolution', {'enabled': True}))
            
            self.logger.info("Initializing narrator...")
            self.narrator = Narrator(self.config.get('logging', {'style': 'balanced'}))
            
            self.running = False
            self.logger.info("SETO-Versal initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing SETO-Versal: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, os.environ.get('LOG_LEVEL', log_config.get('level', 'INFO')))
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers
        root_logger.handlers = []
        
        # Console handler
        if log_config.get('console_output', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if log_config.get('file_output', True):
            log_dir = log_config.get('log_dir', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d')}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        self.logger = logging.getLogger('seto_versal')
    
    def run_once(self):
        """Execute a single trading cycle"""
        try:
            self.logger.info("Starting trading cycle")
            
            # 1. Update market state
            update_success = self.market_state.update()
            if not update_success:
                self.logger.warning("Market state update failed, using previous state")
            
            # 2. Each agent generates trading intentions
            intentions = []
            for agent in self.agents:
                try:
                    agent_intentions = agent.generate_intentions(self.market_state)
                    intentions.extend(agent_intentions)
                    self.logger.debug(f"Agent {agent.name} generated {len(agent_intentions)} intentions")
                except Exception as e:
                    self.logger.error(f"Error generating intentions for agent {agent.name}: {e}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
            
            # 3. Coordinator integrates signals and makes decisions
            try:
                decisions = self.coordinator.coordinate(intentions, self.market_state)
            except Exception as e:
                self.logger.error(f"Error in trade coordination: {e}")
                decisions = []
            
            # 4. Apply risk management
            try:
                filtered_decisions = self.risk_manager.filter_decisions(decisions, self.executor.positions)
            except Exception as e:
                self.logger.error(f"Error in risk management: {e}")
                filtered_decisions = decisions  # Fall back to unfiltered decisions
            
            # 5. Execute trades
            try:
                results = self.executor.execute_decisions(filtered_decisions, self.market_state)
            except Exception as e:
                self.logger.error(f"Error executing trades: {e}")
                results = []
            
            # 6. Record feedback
            try:
                self.feedback_analyzer.record_cycle(
                    intentions=intentions,
                    decisions=decisions,
                    filtered_decisions=filtered_decisions,
                    results=results,
                    market_state=self.market_state
                )
            except Exception as e:
                self.logger.error(f"Error recording feedback: {e}")
            
            # Generate narrative
            try:
                narrative = self.narrator.generate_cycle_narrative(
                    market_state=self.market_state,
                    decisions=filtered_decisions,
                    results=results
                )
                return narrative
            except Exception as e:
                self.logger.error(f"Error generating narrative: {e}")
                return f"Trading cycle completed with {len(results)} trades executed."
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return "Error occurred during trading cycle."
    
    def start(self):
        """Start the continuous trading loop"""
        self.running = True
        self.logger.info(f"Starting SETO-Versal trading system in {self.config['system'].get('mode', 'backtest')} mode")
        
        # 记录启动时间
        start_time = datetime.now()
        self.logger.info(f"System started at {start_time}")
        
        cycle_count = 0
        
        try:
            while self.running:
                # 检查市场是否开放
                is_open = False
                try:
                    is_open = self.market_state.is_market_open()
                except Exception as e:
                    self.logger.error(f"Error checking if market is open: {e}")
                    # 出错时默认为开放状态，确保系统继续运行
                    is_open = True
                
                if is_open:
                    cycle_count += 1
                    self.logger.info(f"Starting trading cycle #{cycle_count}")
                    
                    # 运行一个交易周期
                    try:
                        narrative = self.run_once()
                        self.logger.info(narrative)
                    except Exception as e:
                        self.logger.error(f"Error in trading cycle: {e}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                    
                    # 检查是否应该运行进化引擎
                    try:
                        if self.market_state.should_evolve():
                            self.logger.info("Running evolution cycle")
                            self.evolution_engine.evolve(
                                self.agents,
                                {'feedback_analyzer': self.feedback_analyzer}
                            )
                    except Exception as e:
                        self.logger.error(f"Error in evolution cycle: {e}")
                else:
                    self.logger.debug("Market is closed, waiting for next cycle")
                
                # 等待下一个周期
                interval = self.config['market'].get('update_interval', 5)
                self.logger.debug(f"Waiting for {interval} seconds until next cycle")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.exception(f"Error in main loop: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            # 记录系统运行时间
            run_time = datetime.now() - start_time
            self.logger.info(f"System ran for {run_time} with {cycle_count} trading cycles")
            
            # 停止系统
            self.stop()
    
    def stop(self):
        """Stop the trading system gracefully"""
        self.running = False
        self.logger.info("Shutting down SETO-Versal")
        
        # Generate end of day report
        report = self.feedback_analyzer.generate_daily_report(
            market_state=self.market_state,
            agents=self.agents,
            positions=self.executor.positions
        )
        
        self.logger.info(f"Daily report: {report}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SETO-Versal Trading System')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['backtest', 'paper', 'live'],
                        help='Override trading mode')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Initialize and start the system
    seto = SetoVersal(args.config)
    
    # Override mode if specified
    if args.mode:
        seto.config['system']['mode'] = args.mode
        
    seto.start() 