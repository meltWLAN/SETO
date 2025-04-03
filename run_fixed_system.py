#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal 优化修复后的启动脚本
这个脚本会启动已修复datetime导入问题的交易系统
"""

import os
import sys
import logging
import time
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f"seto_{time.strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("seto_launcher")
logger.info("正在启动优化后的SETO-Versal交易系统...")

def fix_cache_file():
    """修复cache.py文件中的datetime类型注解问题"""
    cache_file = 'seto_versal/data/cache.py'
    
    try:
        # 读取文件内容
        with open(cache_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 使用正则表达式查找并替换所有datetime类型注解
        modified = re.sub(
            r'(start_time|end_time): Optional\[datetime\]', 
            r"\1: Optional['datetime.datetime']", 
            content
        )
        
        # 替换不带Optional的注解
        modified = re.sub(
            r'(start_time|end_time): datetime', 
            r"\1: 'datetime.datetime'", 
            modified
        )
        
        # 如果有修改，写回文件
        if modified != content:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(modified)
            logger.info(f"修复了cache.py中的datetime类型注解")
    except Exception as e:
        logger.error(f"修复cache.py时出错: {e}")

def fix_remaining_datetime_typings():
    """修复系统中的其他datetime类型注解问题"""
    
    # 需要修复的文件列表
    files_to_fix = [
        'seto_versal/data/sources/yahoo_finance.py',
        'seto_versal/data/enhanced_manager.py',
        'seto_versal/data/setup.py',
        'seto_versal/backtest/engine.py'
    ]
    
    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            continue
            
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 替换 'start_time: "datetime.datetime"' 为 'start_time: "datetime.datetime"'
            modified = content.replace('start_time: "datetime.datetime"', 'start_time: "datetime.datetime"')
            modified = modified.replace('end_time: "datetime.datetime"', 'end_time: "datetime.datetime"')
            
            # 如果有修改，写回文件
            if modified != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified)
                logger.info(f"修复了类型注解: {file_path}")
        except Exception as e:
            logger.error(f"修复 {file_path} 时出错: {e}")

def fix_imports():
    """修复所有import datetime, timedelta语句"""
    for root, dirs, files in os.walk('seto_versal'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                try:
                    # 读取文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 替换import语句
                    if 'import datetime, timedelta' in content:
                        modified = content.replace(
                            'import datetime, timedelta',
                            'import datetime\nfrom datetime import timedelta'
                        )
                        
                        # 写回文件
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(modified)
                        logger.info(f"修复了import语句: {file_path}")
                except Exception as e:
                    logger.error(f"修复import语句时出错 {file_path}: {e}")

def main():
    """主函数"""
    # 确保日志目录存在
    os.makedirs('logs', exist_ok=True)
    
    try:
        # 修复导入语句
        logger.info("修复datetime导入问题...")
        fix_imports()
        
        # 修复cache.py文件
        logger.info("修复cache.py文件的类型注解问题...")
        fix_cache_file()
        
        # 先修复剩余的类型问题
        logger.info("修复剩余的datetime类型注解问题...")
        fix_remaining_datetime_typings()
        
        # 运行数据预处理
        logger.info("正在运行数据预处理...")
        from seto_versal.data.preprocess import DataPreprocessor
        
        # 初始化数据预处理器
        processor = DataPreprocessor()
        logger.info("数据预处理器初始化完成")
        
        # 运行数据创建
        processor.test_data_creation()
        logger.info("测试数据创建完成")
        
        # 验证数据
        result = processor.verify_data()
        if result.get('price_data_exists') and result.get('fundamental_data_exists'):
            logger.info("数据验证通过，准备启动交易系统...")
        else:
            logger.warning("数据验证未完全通过，但仍将尝试启动系统")
        
        # 启动GUI
        logger.info("正在启动GUI界面...")
        
        # 从run_gui.py导入主函数
        try:
            from run_gui import main as gui_main
            gui_main()
        except ImportError:
            logger.info("尝试直接启动GUI...")
            # 如果无法导入，则尝试直接启动GUI
            try:
                from seto_versal.gui.qt_main_window import start_gui
                start_gui()
            except Exception as e:
                logger.error(f"启动GUI失败: {e}")
                raise
                
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 