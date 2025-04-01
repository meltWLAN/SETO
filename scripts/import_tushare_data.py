#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TuShare数据导入脚本

用于从TuShare导入行业分类数据和各类指数成分股数据到SETO系统
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from seto_versal.data.tushare_provider import TuShareProvider

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/tushare_import.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TuShare数据导入工具')
    
    parser.add_argument('--token', type=str, help='TuShare API令牌')
    parser.add_argument('--cache-days', type=int, default=7, help='缓存天数')
    parser.add_argument('--output-dir', type=str, default='data/market', help='输出目录')
    parser.add_argument('--industry', action='store_true', help='导入行业分类数据')
    parser.add_argument('--index', type=str, nargs='+', 
                        help='导入指定指数成分股，例如000300.SH表示沪深300')
    parser.add_argument('--all-indices', action='store_true', help='导入所有主要指数成分股')
    parser.add_argument('--convert-format', action='store_true', help='转换为SETO格式')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 确保日志目录存在
    os.makedirs('logs', exist_ok=True)
    
    # 获取TuShare令牌
    token = args.token or os.environ.get('TUSHARE_TOKEN')
    if not token:
        logger.error("未提供TuShare令牌，请通过--token参数或设置TUSHARE_TOKEN环境变量提供")
        return 1
    
    try:
        # 初始化数据提供者
        provider = TuShareProvider(token=token, cache_days=args.cache_days, 
                                  cache_dir=os.path.join(args.output_dir, 'cache/tushare'))
        
        # 导入行业分类数据
        if args.industry:
            logger.info("开始导入行业分类数据...")
            success = provider.save_industry_data(
                output_dir=os.path.join(args.output_dir, 'sectors')
            )
            if success:
                logger.info("行业分类数据导入成功")
            else:
                logger.error("行业分类数据导入失败")
        
        # 导入指数成分股
        if args.index:
            for index_code in args.index:
                logger.info(f"开始导入指数 {index_code} 成分股...")
                stocks = provider.get_index_stocks(index_code)
                
                if not stocks:
                    logger.warning(f"未找到指数 {index_code} 的成分股数据")
                    continue
                
                # 保存成分股数据
                output_dir = os.path.join(args.output_dir, 'indices')
                os.makedirs(output_dir, exist_ok=True)
                
                import json
                with open(os.path.join(output_dir, f"{index_code}.json"), 'w') as f:
                    json.dump({
                        "index_code": index_code,
                        "update_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "stocks": stocks
                    }, f, indent=2)
                
                logger.info(f"指数 {index_code} 成分股数据已保存，共 {len(stocks)} 只股票")
        
        # 导入所有主要指数成分股
        if args.all_indices:
            indices = ['000001.SH', '000300.SH', '000905.SH', '000852.SH', '399006.SZ']
            index_names = {
                '000001.SH': '上证指数',
                '000300.SH': '沪深300',
                '000905.SH': '中证500',
                '000852.SH': '中证1000',
                '399006.SZ': '创业板指'
            }
            
            for index_code in indices:
                logger.info(f"开始导入指数 {index_names.get(index_code, index_code)} 成分股...")
                stocks = provider.get_index_stocks(index_code)
                
                if not stocks:
                    logger.warning(f"未找到指数 {index_code} 的成分股数据")
                    continue
                
                # 保存成分股数据
                output_dir = os.path.join(args.output_dir, 'indices')
                os.makedirs(output_dir, exist_ok=True)
                
                import json
                with open(os.path.join(output_dir, f"{index_code}.json"), 'w') as f:
                    json.dump({
                        "index_code": index_code,
                        "index_name": index_names.get(index_code, index_code),
                        "update_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "stocks": stocks
                    }, f, indent=2)
                
                logger.info(f"指数 {index_names.get(index_code, index_code)} 成分股数据已保存，共 {len(stocks)} 只股票")
        
        # 转换为SETO格式
        if args.convert_format:
            logger.info("开始转换为SETO格式...")
            success = provider.save_seto_industry_format(
                output_file=os.path.join(args.output_dir, 'sectors/seto_format.json')
            )
            if success:
                logger.info("SETO格式转换成功")
            else:
                logger.error("SETO格式转换失败")
        
        logger.info("数据导入任务完成")
        return 0
        
    except Exception as e:
        logger.exception("数据导入过程中发生错误")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 