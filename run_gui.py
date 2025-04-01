#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal 交易系统启动脚本
"""

import os
import sys
import time
import logging
import importlib.util
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'{time.strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('seto_gui')

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def is_package_installed(package_name):
    """检查Python包是否已安装"""
    return importlib.util.find_spec(package_name) is not None

def install_dependencies():
    """安装依赖包"""
    try:
        import pip
        pip.main(['install', '-r', 'requirements.txt'])
        return True
    except Exception as e:
        logger.error(f"安装依赖失败: {e}")
        return False

def run_data_preprocessing():
    """运行数据预处理"""
    print("正在准备交易系统所需数据...")
    try:
        # 导入数据预处理模块
        from seto_versal.data.preprocess import DataPreprocessor
        
        # 初始化数据预处理器
        processor = DataPreprocessor()
        
        # 运行数据创建
        processor.test_data_creation()
        
        # 验证数据
        result = processor.verify_data()
        
        # 输出摘要
        print(processor.generate_summary())
        
        if result.get('price_data_exists') and result.get('fundamental_data_exists'):
            print("数据准备完成，开始启动交易系统...")
            return True
        else:
            print("警告：部分数据准备失败，系统可能无法正常运行！")
            return False
    except Exception as e:
        logger.error(f"数据预处理失败: {e}")
        print(f"数据准备失败: {e}")
        return False

def main():
    """主函数"""
    # 确保日志目录存在
    os.makedirs('logs', exist_ok=True)
    
    print("正在启动 SETO-Versal 交易系统界面...")
    
    # 检查依赖
    required_packages = ['pandas', 'numpy', 'matplotlib', 'pyyaml']
    missing_packages = [pkg for pkg in required_packages if not is_package_installed(pkg)]
    
    if missing_packages:
        print(f"缺少必要的依赖包: {', '.join(missing_packages)}")
        print("正在安装依赖...")
        if not install_dependencies():
            print("安装依赖失败，无法启动系统")
            return
    
    # 运行数据预处理
    run_data_preprocessing()
    
    # 确保配置正确
    config = {
        'system': {
            'name': 'SETO-Versal Trading System',
            'version': '0.9.0'
        },
        'market': {
            'data_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seto_versal', 'data', 'market'),
        },
        'logging': {
            'log_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'),
            'level': 'INFO'
        }
    }
    
    # 确保数据目录存在
    os.makedirs(config['market']['data_dir'], exist_ok=True)
    
    # 如果使用PyQt, 需要先创建QApplication
    try:
        from PyQt6.QtWidgets import QApplication
        app = QApplication(sys.argv)
    except ImportError:
        try:
            from PyQt5.QtWidgets import QApplication
            app = QApplication(sys.argv)
        except ImportError:
            # 不使用PyQt, 继续使用tkinter
            pass
    
    # 启动GUI（优先使用visualization中的GUI，如果不存在则使用ui中的GUI）
    try:
        # 尝试从visualization导入
        from seto_versal.visualization.gui_dashboard import SetoGUI
        print("使用原版GUI界面...")
        
        # 启动GUI
        logger.info("Starting SETO-Versal Trading System")
        try:
            root = tk.Tk()
            app = SetoGUI(root)
            # 预先加载配置
            app.config = config
            logger.info("配置已加载: config.yaml")
            app.update_status()
            root.mainloop()
        except Exception as e:
            logger.error(f"GUI运行失败: {e}")
            print(f"GUI运行失败: {e}")
        
    except ImportError:
        # 备用：从ui导入
        try:
            from seto_versal.ui.gui_dashboard import SetoGUI
            print("使用UI目录中的界面...")
            
            # 启动GUI
            logger.info("Starting SETO-Versal Trading System")
            try:
                root = tk.Tk()
                app = SetoGUI(root)
                # 预先加载配置
                app.config = config
                logger.info("配置已加载: config.yaml")
                app.update_status()
                root.mainloop()
            except Exception as e:
                logger.error(f"GUI运行失败: {e}")
                print(f"GUI运行失败: {e}")
        except ImportError:
            print("错误：无法导入GUI组件")
            return

if __name__ == "__main__":
    main() 