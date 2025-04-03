#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal 交易系统启动脚本 - PyQt6 版本
集成了datetime修复模块，确保所有导入和注解问题都得到解决
"""

import os
import sys
import time
import logging
import re
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
        'seto_versal/backtest/engine.py',
        'seto_versal/gui/qt_main_window.py',
        'seto_versal/gui/main_window.py'
    ]
    
    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            continue
            
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 替换 Optional[datetime]
            modified = re.sub(
                r'(start_time|end_time|timestamp|dt|date):\s*Optional\[datetime\]',
                r'\1: Optional["datetime.datetime"]',
                content
            )
            
            # 替换 datetime (不带Optional)
            modified = re.sub(
                r'(start_time|end_time|timestamp|dt|date):\s*datetime(?![.])',
                r'\1: "datetime.datetime"',
                modified
            )
            
            # 如果有修改，写回文件
            if modified != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified)
                logger.info(f"修复了类型注解: {file_path}")
        except Exception as e:
            logger.error(f"修复 {file_path} 时出错: {e}")

def fix_datetime_imports():
    """修复所有datetime和timedelta导入问题"""
    
    for root, dirs, files in os.walk('seto_versal'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                try:
                    # 读取文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    modified = content
                    
                    # 修复 'import datetime, timedelta' 情况
                    if 'import datetime, timedelta' in modified:
                        modified = modified.replace(
                            'import datetime, timedelta',
                            'import datetime\nfrom datetime import timedelta'
                        )
                    
                    # 修复 'from datetime import datetime, timedelta' 情况
                    if re.search(r'from\s+datetime\s+import\s+datetime\s*,\s*timedelta', modified):
                        modified = re.sub(
                            r'from\s+datetime\s+import\s+datetime\s*,\s*timedelta',
                            'import datetime\nfrom datetime import timedelta',
                            modified
                        )
                    
                    # 修复单独导入timedelta但没有正确导入
                    if 'import timedelta' in modified and 'from datetime import timedelta' not in modified:
                        modified = modified.replace(
                            'import timedelta',
                            'from datetime import timedelta'
                        )
                    
                    # 如果有修改，写回文件
                    if modified != content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(modified)
                        logger.info(f"修复了导入语句: {file_path}")
                except Exception as e:
                    logger.error(f"修复导入语句时出错 {file_path}: {e}")

def fix_datetime_now_calls():
    """修复datetime.datetime.now()调用问题"""
    for root, dirs, files in os.walk('seto_versal'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                try:
                    # 读取文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查导入方式
                    has_datetime_import = re.search(r'import\s+datetime', content) and not re.search(r'from\s+datetime\s+import\s+datetime', content)
                    
                    if has_datetime_import:
                        # 修复datetime.datetime.now()调用
                        modified = re.sub(
                            r'(?<!\.|")datetime\.now\(\)',
                            'datetime.datetime.now()',
                            content
                        )
                        
                        if modified != content:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(modified)
                            logger.info(f"修复了datetime.datetime.now()调用: {file_path}")
                except Exception as e:
                    logger.error(f"处理文件失败: {file_path}, 错误: {e}")

def run_data_preprocessing():
    """运行数据预处理"""
    print("正在准备交易系统所需数据...")
    try:
        # 导入数据预处理模块
        from seto_versal.data.preprocess import DataPreprocessor, run_test
        
        # 运行数据预处理测试
        success = run_test()
        
        if success:
            print("数据预处理完成: 成功")
            return True
        else:
            print("数据预处理失败: 请检查日志文件")
            return False
            
    except Exception as e:
        logger.error(f"数据预处理失败: {e}")
        print(f"数据预处理出错: {e}")
        return False

def fix_direct_modules():
    """直接修复特定模块中的导入问题"""
    # 以下是可能存在导入问题的关键模块
    critical_modules = [
        'seto_versal/gui/main_window.py',
        'seto_versal/gui/qt_main_window.py',
        'seto_versal/market/state.py',
        'seto_versal/market/market_state.py',
        'seto_versal/data/cache.py',
        'seto_versal/data/manager.py',
        'seto_versal/gui/__init__.py'
    ]
    
    for module_path in critical_modules:
        if not os.path.exists(module_path):
            logger.warning(f"模块不存在: {module_path}")
            continue
            
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 检查并修复多种错误的导入方式
            modified = content
            
            # 检查模式1: import timedelta
            if re.search(r'import\s+timedelta\s*$', modified) or re.search(r'import\s+timedelta\s*;', modified):
                modified = re.sub(
                    r'import\s+timedelta\s*$', 
                    'from datetime import timedelta', 
                    modified
                )
                modified = re.sub(
                    r'import\s+timedelta\s*;', 
                    'from datetime import timedelta;', 
                    modified
                )
                
            # 检查模式2: import datetime, timedelta
            if 'import datetime, timedelta' in modified:
                modified = modified.replace(
                    'import datetime, timedelta',
                    'import datetime\nfrom datetime import timedelta'
                )
                
            # 检查模式3: from datetime import datetime, timedelta
            if re.search(r'from\s+datetime\s+import\s+datetime\s*,\s*timedelta', modified):
                modified = re.sub(
                    r'from\s+datetime\s+import\s+datetime\s*,\s*timedelta',
                    'import datetime\nfrom datetime import timedelta',
                    modified
                )
            
            # 如果有修改，写回文件
            if modified != content:
                with open(module_path, 'w', encoding='utf-8') as f:
                    f.write(modified)
                logger.info(f"直接修复了模块导入: {module_path}")
        except Exception as e:
            logger.error(f"修复模块 {module_path} 时出错: {e}")

def main():
    """主函数"""
    # 确保日志目录存在
    os.makedirs('logs', exist_ok=True)
    
    print("正在启动 SETO-Versal 交易系统界面 (PyQt6版本)...")
    
    # 添加全面的datetime补丁，确保所有模块都能正确导入和使用datetime
    logger.info("添加全面的datetime模块补丁...")
    import datetime
    from datetime import timedelta
    
    # 添加datetime.now函数，确保导入的datetime模块具有now方法
    if not hasattr(datetime, 'now'):
        logger.info("修复datetime.now方法...")
        # 从原始datetime模块中获取now方法并添加到datetime模块
        datetime.now = datetime.datetime.now
    
    # 将timedelta注入到sys.modules中
    sys.modules['timedelta'] = timedelta
    
    # 直接修复关键模块
    logger.info("直接修复关键模块中的导入问题...")
    fix_direct_modules()
    
    # 修复datetime相关问题
    logger.info("修复datetime导入问题...")
    fix_datetime_imports()
    
    logger.info("修复cache.py文件的类型注解问题...")
    fix_cache_file()
    
    logger.info("修复其他datetime类型注解问题...")
    fix_remaining_datetime_typings()
    
    logger.info("修复datetime.now()调用问题...")
    fix_datetime_now_calls()
    
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
        logger.info("正在启动SETO-Versal交易系统GUI...")
        sys.exit(start_gui())
    except Exception as e:
        logger.error(f"GUI运行失败: {e}")
        print(f"GUI运行失败: {e}")
        
        # 尝试直接启动备用GUI
        try:
            logger.info("尝试使用备用GUI启动方式...")
            from seto_versal.gui.main_window import MainWindow
            from PyQt6.QtWidgets import QApplication
            
            app = QApplication(sys.argv)
            window = MainWindow()
            window.show()
            sys.exit(app.exec())
        except Exception as e2:
            logger.error(f"备用GUI也启动失败: {e2}")
            print(f"备用GUI启动失败: {e2}")

if __name__ == "__main__":
    main() 