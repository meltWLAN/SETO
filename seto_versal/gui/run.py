#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal GUI launcher
"""

import sys
import logging
from PyQt6.QtWidgets import QApplication
from .main_window import MainWindow

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main entry point for the GUI"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Create application
        app = QApplication(sys.argv)
        
        # Create and show main window
        window = MainWindow()
        window.show()
        
        # Start event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Error starting GUI: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 