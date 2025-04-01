#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TuShare数据提供模块

用于从TuShare获取各类市场数据，包括行业分类、成分股等
"""

import os
import time
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union

# 避免未安装tushare时报错，使用try-except
try:
    import tushare as ts
except ImportError:
    ts = None

logger = logging.getLogger(__name__)

class TuShareProvider:
    """TuShare数据提供类，管理与TuShare的数据交互"""
    
    def __init__(self, token: Optional[str] = None, 
                 cache_dir: str = 'data/cache/tushare',
                 cache_days: int = 1):
        """
        初始化TuShare数据提供者
        
        Args:
            token: TuShare API token，如果不提供则尝试从环境变量TUSHARE_TOKEN获取
            cache_dir: 缓存目录
            cache_days: 缓存天数
        """
        # 检查tushare是否已安装
        if ts is None:
            raise ImportError("请先安装tushare: pip install tushare")
        
        # 获取token
        self.token = token or os.environ.get('TUSHARE_TOKEN')
        if not self.token:
            raise ValueError("需要提供TuShare API token或设置TUSHARE_TOKEN环境变量")
        
        # 设置tushare
        ts.set_token(self.token)
        self.pro = ts.pro_api()
        
        # 缓存设置
        self.cache_dir = cache_dir
        self.cache_days = cache_days
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 行业分类缓存
        self.industry_cache = {}
        
        logger.info("TuShare数据提供模块初始化完成")
    
    def get_index_stocks(self, index_code: str) -> List[str]:
        """
        获取指数成分股
        
        Args:
            index_code: 指数代码，如'000300.SH'表示沪深300
            
        Returns:
            成分股代码列表
        """
        cache_file = os.path.join(self.cache_dir, f"index_{index_code}.json")
        
        # 检查缓存
        if os.path.exists(cache_file):
            cache_time = os.path.getmtime(cache_file)
            if (time.time() - cache_time) / 86400 < self.cache_days:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"读取指数成分股缓存失败: {e}")
        
        try:
            # 获取指数成分
            df = self.pro.index_weight(index_code=index_code, 
                                       trade_date=datetime.now().strftime('%Y%m%d'))
            
            if df.empty:
                # 如果当天数据不可用，尝试获取最近的数据
                dates = [(datetime.now() - timedelta(days=i)).strftime('%Y%m%d') 
                         for i in range(1, 31)]
                for date in dates:
                    df = self.pro.index_weight(index_code=index_code, trade_date=date)
                    if not df.empty:
                        break
            
            if df.empty:
                logger.warning(f"无法获取指数 {index_code} 的成分股")
                return []
            
            # 提取成分股代码
            stocks = df['con_code'].tolist()
            
            # 缓存结果
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(stocks, f)
            
            return stocks
            
        except Exception as e:
            logger.error(f"获取指数 {index_code} 成分股失败: {e}")
            return []
    
    def get_industry_classification(self, level: str = 'L1') -> Dict[str, List[str]]:
        """
        获取行业分类数据
        
        Args:
            level: 行业级别，L1为一级行业，L2为二级行业，L3为三级行业
            
        Returns:
            行业分类字典，key为行业名称，value为该行业的股票代码列表
        """
        cache_file = os.path.join(self.cache_dir, f"industry_{level}.json")
        
        # 检查缓存
        if os.path.exists(cache_file):
            cache_time = os.path.getmtime(cache_file)
            if (time.time() - cache_time) / 86400 < self.cache_days:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"读取行业分类缓存失败: {e}")
        
        try:
            # 获取股票列表
            stock_basic = self.pro.stock_basic(exchange='', list_status='L', 
                                              fields='ts_code,name,industry,market')
            
            # 按行业分组
            industry_dict = {}
            for _, row in stock_basic.iterrows():
                industry = row['industry']
                if not industry:
                    continue
                
                if industry not in industry_dict:
                    industry_dict[industry] = []
                
                industry_dict[industry].append(row['ts_code'])
            
            # 如果需要二级或三级行业，则需要调用其他接口获取更详细的分类
            if level in ['L2', 'L3']:
                # 这里需要根据实际TuShare接口扩展
                pass
            
            # 缓存结果
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(industry_dict, f)
            
            return industry_dict
            
        except Exception as e:
            logger.error(f"获取行业分类数据失败: {e}")
            return {}
    
    def get_stock_daily(self, ts_code: str, 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None,
                       adjust: str = 'qfq') -> pd.DataFrame:
        """
        获取股票日线数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期，格式YYYYMMDD，默认为过去一年
            end_date: 结束日期，格式YYYYMMDD，默认为今天
            adjust: 复权类型，qfq为前复权，hfq为后复权，None为不复权
            
        Returns:
            股票日线数据DataFrame
        """
        # 设置默认日期
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        cache_file = os.path.join(self.cache_dir, 
                                 f"daily_{ts_code}_{start_date}_{end_date}_{adjust}.pkl")
        
        # 检查缓存
        if os.path.exists(cache_file):
            cache_time = os.path.getmtime(cache_file)
            if (time.time() - cache_time) / 86400 < self.cache_days:
                try:
                    return pd.read_pickle(cache_file)
                except Exception as e:
                    logger.warning(f"读取股票日线数据缓存失败: {e}")
        
        try:
            # 获取日线数据
            df = ts.pro_bar(ts_code=ts_code, start_date=start_date, end_date=end_date,
                           adj=adjust, freq='D')
            
            if df is None or df.empty:
                logger.warning(f"无法获取股票 {ts_code} 的日线数据")
                return pd.DataFrame()
            
            # 按日期排序
            df.sort_values('trade_date', inplace=True)
            
            # 缓存结果
            df.to_pickle(cache_file)
            
            return df
            
        except Exception as e:
            logger.error(f"获取股票 {ts_code} 日线数据失败: {e}")
            return pd.DataFrame()
    
    def get_all_industry_stock_data(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有行业股票数据，用于行业分析
        
        Returns:
            包含行业分类和股票数据的字典
        """
        try:
            # 获取行业分类
            industry_dict = self.get_industry_classification()
            
            # 统计各行业股票数量
            industry_stats = {industry: len(stocks) for industry, stocks in industry_dict.items()}
            
            # 获取每个行业的代表性股票
            industry_samples = {}
            for industry, stocks in industry_dict.items():
                # 每个行业选取前10支股票
                samples = stocks[:min(10, len(stocks))]
                industry_samples[industry] = samples
            
            # 返回结果
            return {
                "industry_dict": industry_dict,
                "industry_stats": industry_stats,
                "industry_samples": industry_samples
            }
            
        except Exception as e:
            logger.error(f"获取所有行业股票数据失败: {e}")
            return {}
    
    def save_industry_data(self, output_dir: str = 'data/market/sectors') -> bool:
        """
        保存行业数据到文件
        
        Args:
            output_dir: 输出目录
            
        Returns:
            是否成功
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取行业数据
            industry_data = self.get_all_industry_stock_data()
            
            # 保存行业分类数据
            with open(os.path.join(output_dir, 'classification.json'), 'w', encoding='utf-8') as f:
                json.dump(industry_data["industry_dict"], f, ensure_ascii=False, indent=2)
            
            # 保存行业统计数据
            with open(os.path.join(output_dir, 'statistics.json'), 'w', encoding='utf-8') as f:
                json.dump(industry_data["industry_stats"], f, ensure_ascii=False, indent=2)
            
            # 保存行业样本数据
            with open(os.path.join(output_dir, 'samples.json'), 'w', encoding='utf-8') as f:
                json.dump(industry_data["industry_samples"], f, ensure_ascii=False, indent=2)
            
            logger.info(f"行业数据已保存到 {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"保存行业数据失败: {e}")
            return False
    
    def get_index_component_change(self, index_code: str) -> pd.DataFrame:
        """
        获取指数成分股变化情况
        
        Args:
            index_code: 指数代码
            
        Returns:
            成分股变化DataFrame
        """
        try:
            # 获取指数成分股变更记录
            df = self.pro.index_dailybasic(ts_code=index_code)
            
            if df.empty:
                logger.warning(f"无法获取指数 {index_code} 的成分股变化情况")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            logger.error(f"获取指数 {index_code} 成分股变化情况失败: {e}")
            return pd.DataFrame()
    
    def convert_industry_format(self) -> Dict[str, Dict[str, List[str]]]:
        """
        转换行业格式为SETO-Versal使用的格式
        
        Returns:
            转换后的行业格式
        """
        try:
            # 获取行业分类
            industry_dict = self.get_industry_classification()
            
            # 定义主要行业映射
            major_industries = {
                "金融": ["银行", "证券", "保险", "多元金融"],
                "科技": ["计算机", "通信", "电子", "互联网"],
                "消费": ["食品饮料", "家用电器", "纺织服装", "商业贸易", "农林牧渔"],
                "医药": ["医药生物", "医疗器械"],
                "能源": ["石油", "煤炭", "电力", "燃气", "采掘"],
                "工业": ["机械设备", "化工", "建筑", "建材", "钢铁", "有色金属"]
            }
            
            # 初始化结果
            seto_format = {major: {"stocks": []} for major in major_industries}
            
            # 为每个股票分配主要行业
            for industry, stocks in industry_dict.items():
                assigned = False
                for major, sub_industries in major_industries.items():
                    for sub in sub_industries:
                        if sub in industry:
                            seto_format[major]["stocks"].extend(stocks)
                            assigned = True
                            break
                    if assigned:
                        break
                
                # 如果没有分配，放入其他类别
                if not assigned and "其他" not in seto_format:
                    seto_format["其他"] = {"stocks": stocks}
                elif not assigned:
                    seto_format["其他"]["stocks"].extend(stocks)
            
            # 添加行业描述
            descriptions = {
                "金融": "包括银行、证券、保险等金融服务业",
                "科技": "包括计算机、通信、电子等技术密集型产业",
                "消费": "包括食品饮料、家用电器等消费品产业",
                "医药": "包括医药生物、医疗器械等医疗健康产业",
                "能源": "包括石油、煤炭、电力等能源产业",
                "工业": "包括机械设备、化工、建筑等工业制造业"
            }
            
            for major, desc in descriptions.items():
                if major in seto_format:
                    seto_format[major]["description"] = desc
            
            return seto_format
            
        except Exception as e:
            logger.error(f"转换行业格式失败: {e}")
            return {}
    
    def save_seto_industry_format(self, output_file: str = 'data/market/sectors/seto_format.json') -> bool:
        """
        保存SETO-Versal格式的行业数据
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            是否成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 获取SETO格式行业数据
            seto_format = self.convert_industry_format()
            
            # 保存数据
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(seto_format, f, ensure_ascii=False, indent=2)
            
            logger.info(f"SETO-Versal格式行业数据已保存到 {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存SETO-Versal格式行业数据失败: {e}")
            return False


# 示例使用代码
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 检查环境变量
    token = os.environ.get('TUSHARE_TOKEN')
    if not token:
        print("请设置环境变量TUSHARE_TOKEN")
        exit(1)
    
    # 初始化数据提供者
    provider = TuShareProvider(token=token, cache_days=7)
    
    # 获取沪深300成分股
    hs300_stocks = provider.get_index_stocks('000300.SH')
    print(f"沪深300成分股数量: {len(hs300_stocks)}")
    
    # 获取行业分类
    industry_dict = provider.get_industry_classification()
    print(f"行业数量: {len(industry_dict)}")
    
    # 保存行业数据
    provider.save_industry_data()
    
    # 保存SETO格式行业数据
    provider.save_seto_industry_format()
    
    print("数据保存完成") 