#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
全面修复SETO-Versal系统中的datetime导入和类型注释问题

这个脚本会扫描整个项目代码库，修复以下问题：
1. 将 `import datetime
from datetime import timedelta` 修改为正确的导入方式
2. 将类型注释中的 `datetime` 修改为 `"datetime.datetime"`
3. 将 `datetime.datetime.now()` 修改为 `datetime.datetime.now()`
"""

import os
import re
import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_script")

def fix_datetime_imports(file_path):
    """修复datetime和timedelta的导入问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'import datetime, timedelta' in content:
            modified = content.replace(
                'import datetime, timedelta',
                'import datetime\nfrom datetime import timedelta'
            )
            
            if modified != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified)
                logger.info(f"修复了import语句: {file_path}")
                return True
    except Exception as e:
        logger.error(f"处理文件失败: {file_path}, 错误: {e}")
    
    return False

def fix_datetime_typings(file_path):
    """修复datetime类型注释问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修复 Optional[datetime]
        modified = re.sub(
            r'(start_time|end_time|timestamp|dt|date):\s*Optional\[datetime\]',
            r'\1: Optional["datetime.datetime"]',
            content
        )
        
        # 修复 datetime (不带Optional)
        modified = re.sub(
            r'(start_time|end_time|timestamp|dt|date):\s*datetime(?![.])',
            r'\1: "datetime.datetime"',
            modified
        )
        
        if modified != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified)
            logger.info(f"修复了类型注释: {file_path}")
            return True
    except Exception as e:
        logger.error(f"处理文件失败: {file_path}, 错误: {e}")
    
    return False

def fix_datetime_now_calls(file_path):
    """修复datetime.datetime.now()调用问题"""
    try:
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
                return True
    except Exception as e:
        logger.error(f"处理文件失败: {file_path}, 错误: {e}")
    
    return False

def process_directory(directory_path):
    """处理指定目录中的所有Python文件"""
    fixed_imports = 0
    fixed_typings = 0
    fixed_now_calls = 0
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # 修复导入问题
                if fix_datetime_imports(file_path):
                    fixed_imports += 1
                
                # 修复类型注释问题
                if fix_datetime_typings(file_path):
                    fixed_typings += 1
                
                # 修复now()调用问题
                if fix_datetime_now_calls(file_path):
                    fixed_now_calls += 1
    
    return fixed_imports, fixed_typings, fixed_now_calls

def main():
    """主函数"""
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.getcwd()
    
    logger.info(f"开始修复 {directory} 中的datetime相关问题...")
    
    fixed_imports, fixed_typings, fixed_now_calls = process_directory(directory)
    
    logger.info("修复完成!")
    logger.info(f"- 修复的导入问题: {fixed_imports}个文件")
    logger.info(f"- 修复的类型注释问题: {fixed_typings}个文件")
    logger.info(f"- 修复的datetime.datetime.now()调用问题: {fixed_now_calls}个文件")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 