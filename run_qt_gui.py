#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal 交易系统启动脚本 - PyQt6 版本
"""

import os
import sys
import time
import logging
import importlib.util
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'{time.strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('seto_gui')

# 确保logs目录存在
os.makedirs('logs', exist_ok=True)

# 将项目根目录添加到Python路径
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
    
    print("正在启动 SETO-Versal 交易系统界面 (PyQt6版本)...")
    
    # 检查依赖
    required_packages = ['pandas', 'numpy', 'matplotlib', 'pyyaml', 'PyQt6']
    missing_packages = [pkg for pkg in required_packages if not is_package_installed(pkg)]
    
    if missing_packages:
        print(f"缺少必要的依赖包: {', '.join(missing_packages)}")
        print("正在安装依赖...")
        if not install_dependencies():
            print("安装依赖失败，无法启动系统")
            return
    
    # 运行数据预处理
    run_data_preprocessing()
    
    # 启动PyQt6 GUI
    try:
        from seto_versal.gui.qt_main_window import start_gui
        # 启动GUI
        logger.info("Starting SETO-Versal Trading System (PyQt6)")
        sys.exit(start_gui())
    except Exception as e:
        logger.error(f"GUI运行失败: {e}")
        print(f"GUI运行失败: {e}")

if __name__ == "__main__":
    main() 