#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal 数据初始化脚本

在系统运行前获取、整理和保存所需的数据，确保数据的一致性和有效性。
支持从Tushare和AKShare获取数据，并按照统一格式进行标准化。
"""

import os
import logging
import time
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import tqdm

# 导入数据管理组件
from seto_versal.data.enhanced_manager import EnhancedDataManager, TimeFrame
from seto_versal.data.enhanced_manager import TuShareDataSource, AKShareDataSource

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_setup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataSetup:
    """
    数据初始化工具，用于预处理和保存交易系统所需数据
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        初始化数据设置工具
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.data_manager = None
        self.output_dir = self.config.get('output_dir', 'data/market')
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置多线程执行器
        self.max_workers = self.config.get('max_workers', 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # 初始化数据管理器
        self._init_data_manager()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        import yaml
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('data_setup', {})
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}
    
    def _init_data_manager(self) -> None:
        """初始化数据管理器"""
        try:
            data_config = self.config.get('data_manager', {})
            self.data_manager = EnhancedDataManager(data_config)
            logger.info("数据管理器初始化完成")
        except Exception as e:
            logger.error(f"初始化数据管理器失败: {e}")
            raise
    
    def get_stock_list(self) -> List[str]:
        """
        获取股票列表
        
        Returns:
            股票代码列表
        """
        # 先检查是否有预定义的股票列表
        predefined_symbols = self.config.get('symbols', [])
        if predefined_symbols:
            logger.info(f"使用预定义的股票列表: {len(predefined_symbols)} 只股票")
            return predefined_symbols
        
        # 获取指数成分股
        index_code = self.config.get('index', 'hs300')
        symbols = self._get_index_components(index_code)
        
        if not symbols:
            # 如果无法获取指数成分股，尝试获取所有A股
            logger.warning(f"无法获取指数 {index_code} 成分股，尝试获取所有A股")
            symbols = self.data_manager.get_available_symbols()
        
        if self.config.get('limit_symbols', 0) > 0:
            # 限制股票数量（用于测试）
            limit = min(self.config.get('limit_symbols'), len(symbols))
            symbols = symbols[:limit]
        
        logger.info(f"获取股票列表完成: {len(symbols)} 只股票")
        return symbols
    
    def _get_index_components(self, index_code: str) -> List[str]:
        """
        获取指数成分股
        
        Args:
            index_code: 指数代码，如'hs300'
            
        Returns:
            成分股代码列表
        """
        try:
            # 尝试从TuShare获取
            tushare_source = self.data_manager.data_manager.get_data_source('tushare')
            if tushare_source and hasattr(tushare_source, 'api'):
                if index_code == 'hs300':
                    data = tushare_source.api.index_weight(index_code='000300.SH', 
                                                   start_date=datetime.now().strftime('%Y%m%d'))
                    return data['con_code'].tolist() if data is not None and not data.empty else []
                elif index_code == 'zz500':
                    data = tushare_source.api.index_weight(index_code='000905.SH', 
                                                   start_date=datetime.now().strftime('%Y%m%d'))
                    return data['con_code'].tolist() if data is not None and not data.empty else []
                elif index_code == 'sz50':
                    data = tushare_source.api.index_weight(index_code='000016.SH', 
                                                   start_date=datetime.now().strftime('%Y%m%d'))
                    return data['con_code'].tolist() if data is not None and not data.empty else []
            
            # 如果TuShare不可用，尝试使用AKShare
            import akshare as ak
            if index_code == 'hs300':
                data = ak.index_stock_cons_weight_csindex(symbol="000300")
            elif index_code == 'zz500':
                data = ak.index_stock_cons_weight_csindex(symbol="000905")
            elif index_code == 'sz50':
                data = ak.index_stock_cons_weight_csindex(symbol="000016")
            else:
                return []
            
            if data is not None and not data.empty:
                # 处理股票代码格式
                symbols = []
                for _, row in data.iterrows():
                    code = row['成分券代码']
                    if code.startswith('6'):
                        symbols.append(f"{code}.SH")
                    else:
                        symbols.append(f"{code}.SZ")
                return symbols
            
            return []
        except Exception as e:
            logger.error(f"获取指数成分股失败: {e}")
            return []
    
    def get_historical_data(self, 
                          symbols: List[str], 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None,
                          timeframes: Optional[List[str]] = None) -> None:
        """
        获取并保存历史行情数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期，格式：YYYY-MM-DD，默认为一年前
            end_date: 结束日期，格式：YYYY-MM-DD，默认为今天
            timeframes: 时间周期列表，默认为['1d']
        """
        # 设置默认值
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        start_time = datetime.strptime(start_date, '%Y-%m-%d')
        end_time = datetime.strptime(end_date, '%Y-%m-%d')
        
        if timeframes is None:
            timeframes = ['1d']
        
        # 转换时间周期字符串为枚举
        tf_enums = []
        for tf in timeframes:
            try:
                tf_enums.append(TimeFrame.from_string(tf))
            except ValueError:
                logger.warning(f"不支持的时间周期: {tf}")
        
        if not tf_enums:
            logger.error("没有有效的时间周期")
            return
        
        logger.info(f"开始获取 {len(symbols)} 只股票的历史数据，从 {start_date} 到 {end_date}")
        
        # 对每个股票和时间周期获取数据
        for tf in tf_enums:
            # 创建目录
            tf_dir = os.path.join(self.output_dir, str(tf))
            os.makedirs(tf_dir, exist_ok=True)
            
            logger.info(f"获取 {tf} 周期数据")
            
            futures = []
            for symbol in symbols:
                future = self.executor.submit(
                    self._get_and_save_data,
                    symbol, tf, start_time, end_time, tf_dir
                )
                futures.append((symbol, future))
            
            # 显示进度条
            total = len(symbols)
            success = 0
            errors = 0
            
            with tqdm.tqdm(total=total, desc=f"{tf} 数据获取") as pbar:
                for symbol, future in futures:
                    try:
                        result = future.result()
                        if result:
                            success += 1
                        else:
                            errors += 1
                    except Exception as e:
                        logger.error(f"获取 {symbol} {tf} 数据失败: {e}")
                        errors += 1
                    finally:
                        pbar.update(1)
            
            logger.info(f"{tf} 数据获取完成。成功: {success}, 失败: {errors}, 总计: {total}")
        
        logger.info("所有历史数据获取完成")
    
    def _get_and_save_data(self, 
                          symbol: str, 
                          timeframe: TimeFrame, 
                          start_time: datetime, 
                          end_time: datetime,
                          output_dir: str) -> bool:
        """
        获取并保存单只股票的数据
        
        Args:
            symbol: 股票代码
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            output_dir: 输出目录
            
        Returns:
            是否成功
        """
        try:
            # 获取数据
            data = self.data_manager.get_historical_data(
                symbol=symbol, 
                timeframe=timeframe, 
                start_time=start_time, 
                end_time=end_time
            )
            
            if data.empty:
                logger.warning(f"无数据: {symbol} {timeframe}")
                return False
            
            # 保存为CSV
            filename = f"{symbol}.csv".replace('.', '_')
            filepath = os.path.join(output_dir, filename)
            
            # 重置索引保存
            data_to_save = data.copy()
            data_to_save.reset_index(inplace=True)
            data_to_save.to_csv(filepath, index=False)
            
            return True
        except Exception as e:
            logger.error(f"获取/保存 {symbol} {timeframe} 数据失败: {e}")
            return False
    
    def get_fundamental_data(self, symbols: List[str]) -> None:
        """
        获取并保存基本面数据
        
        Args:
            symbols: 股票代码列表
        """
        logger.info(f"开始获取 {len(symbols)} 只股票的基本面数据")
        
        # 创建目录
        fund_dir = os.path.join(self.output_dir, 'fundamentals')
        os.makedirs(fund_dir, exist_ok=True)
        
        futures = []
        for symbol in symbols:
            future = self.executor.submit(
                self._get_and_save_fundamental,
                symbol, fund_dir
            )
            futures.append((symbol, future))
        
        # 显示进度条
        total = len(symbols)
        success = 0
        errors = 0
        
        with tqdm.tqdm(total=total, desc="基本面数据获取") as pbar:
            for symbol, future in futures:
                try:
                    result = future.result()
                    if result:
                        success += 1
                    else:
                        errors += 1
                except Exception as e:
                    logger.error(f"获取 {symbol} 基本面数据失败: {e}")
                    errors += 1
                finally:
                    pbar.update(1)
        
        logger.info(f"基本面数据获取完成。成功: {success}, 失败: {errors}, 总计: {total}")
    
    def _get_and_save_fundamental(self, symbol: str, output_dir: str) -> bool:
        """
        获取并保存单只股票的基本面数据
        
        Args:
            symbol: 股票代码
            output_dir: 输出目录
            
        Returns:
            是否成功
        """
        try:
            # 首先尝试从TuShare获取
            tushare_source = self.data_manager.data_manager.get_data_source('tushare')
            fundamental_data = {}
            
            if tushare_source and hasattr(tushare_source, 'api'):
                # 股票基本信息
                basic_info = tushare_source.api.stock_basic(ts_code=symbol, fields='ts_code,name,area,industry,market,list_date')
                if not basic_info.empty:
                    fundamental_data['basic_info'] = basic_info.to_dict('records')[0]
                
                # 最新财务指标
                fin_indicator = tushare_source.api.fina_indicator(ts_code=symbol, period='20200331')
                if not fin_indicator.empty:
                    fundamental_data['financial_indicator'] = fin_indicator.to_dict('records')[0]
                
                # 最近的利润表
                income = tushare_source.api.income(ts_code=symbol, period='20200331')
                if not income.empty:
                    fundamental_data['income'] = income.to_dict('records')[0]
                
                # 最近的资产负债表
                balancesheet = tushare_source.api.balancesheet(ts_code=symbol, period='20200331')
                if not balancesheet.empty:
                    fundamental_data['balance_sheet'] = balancesheet.to_dict('records')[0]
                
                # 主要指标
                daily_basic = tushare_source.api.daily_basic(ts_code=symbol, trade_date=datetime.now().strftime('%Y%m%d'))
                if not daily_basic.empty:
                    fundamental_data['daily_basic'] = daily_basic.to_dict('records')[0]
            
            # 如果没有从TuShare获得数据，尝试从AKShare获取
            if not fundamental_data:
                # TODO: 从AKShare获取基本面数据
                pass
            
            # 如果仍然没有数据，生成模拟数据用于测试
            if not fundamental_data:
                logger.warning(f"无法获取 {symbol} 的基本面数据，生成模拟数据")
                fundamental_data = self._generate_mock_fundamental(symbol)
            
            # 保存为JSON
            filename = f"{symbol}.json".replace('.', '_')
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(fundamental_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"获取/保存 {symbol} 基本面数据失败: {e}")
            return False
    
    def _generate_mock_fundamental(self, symbol: str) -> Dict[str, Any]:
        """
        生成模拟基本面数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            模拟基本面数据字典
        """
        # 生成一些随机但合理的基本面指标
        pe = round(10 + np.random.random() * 20, 2)  # PE 10-30
        pb = round(1 + np.random.random() * 3, 2)    # PB 1-4
        roe = round(5 + np.random.random() * 20, 2)  # ROE 5%-25%
        debt_to_asset = round(20 + np.random.random() * 50, 2)  # 资产负债率 20%-70%
        profit_margin = round(5 + np.random.random() * 25, 2)  # 净利率 5%-30%
        
        return {
            'basic_info': {
                'ts_code': symbol,
                'name': f"模拟股票{symbol.split('.')[0]}",
                'industry': '模拟行业',
                'area': '模拟地区',
                'market': 'MOCK',
                'list_date': '20000101'
            },
            'indicators': {
                'pe': pe,
                'pb': pb,
                'roe': roe,
                'debt_to_asset': debt_to_asset,
                'profit_margin': profit_margin,
                'dividend_yield': round(np.random.random() * 3, 2),
                'market_cap': round(np.random.random() * 1000, 2),
                'revenue_yoy': round(-10 + np.random.random() * 50, 2),
                'profit_yoy': round(-20 + np.random.random() * 60, 2)
            },
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_mock': True
        }
    
    def get_industry_data(self) -> None:
        """获取行业数据"""
        logger.info("开始获取行业分类数据")
        
        # 创建目录
        industry_dir = os.path.join(self.output_dir, 'industry')
        os.makedirs(industry_dir, exist_ok=True)
        
        try:
            # 尝试从TuShare获取行业分类
            tushare_source = self.data_manager.data_manager.get_data_source('tushare')
            if tushare_source and hasattr(tushare_source, 'api'):
                # 中信行业分类
                industry_citics = tushare_source.api.citics_industry()
                if industry_citics is not None and not industry_citics.empty:
                    industry_citics.to_csv(os.path.join(industry_dir, 'citics_industry.csv'), index=False)
                    logger.info("已保存中信行业分类数据")
                
                # 申万行业分类
                industry_sw = tushare_source.api.sw_ind()
                if industry_sw is not None and not industry_sw.empty:
                    industry_sw.to_csv(os.path.join(industry_dir, 'sw_industry.csv'), index=False)
                    logger.info("已保存申万行业分类数据")
                
                # 股票行业分类
                stock_industry = tushare_source.api.stock_basic(fields='ts_code,name,industry')
                if stock_industry is not None and not stock_industry.empty:
                    stock_industry.to_csv(os.path.join(industry_dir, 'stock_industry.csv'), index=False)
                    logger.info("已保存股票行业分类数据")
            
            # 如果没有数据，尝试从AKShare获取
            # TODO: 从AKShare获取行业数据
            
            logger.info("行业数据获取完成")
            return True
        except Exception as e:
            logger.error(f"获取行业数据失败: {e}")
            return False
    
    def run_full_setup(self):
        """
        运行完整的数据初始化流程
        """
        start_time = time.time()
        logger.info("开始数据初始化流程")
        
        try:
            # 1. 获取股票列表
            symbols = self.get_stock_list()
            if not symbols:
                logger.error("未能获取有效的股票列表，初始化中止")
                return
            
            # 2. 获取历史行情数据
            timeframes = self.config.get('timeframes', ['1d'])
            start_date = self.config.get('start_date')
            end_date = self.config.get('end_date')
            self.get_historical_data(symbols, start_date, end_date, timeframes)
            
            # 3. 获取基本面数据
            self.get_fundamental_data(symbols)
            
            # 4. 获取行业数据
            self.get_industry_data()
            
            # 5. 索引和优化数据（可选）
            if self.config.get('optimize_data', False):
                self._optimize_data()
            
            elapsed = time.time() - start_time
            logger.info(f"数据初始化完成，用时: {elapsed:.2f} 秒")
            
        except Exception as e:
            logger.error(f"数据初始化过程中发生错误: {e}", exc_info=True)
    
    def _optimize_data(self):
        """优化和索引数据"""
        # TODO: 实现数据优化，例如创建索引、合并文件等
        pass

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SETO-Versal 数据初始化工具')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['full', 'market', 'fundamental', 'industry'],
                        help='初始化模式：full=全部, market=仅行情, fundamental=仅基本面, industry=仅行业')
    args = parser.parse_args()
    
    setup = DataSetup(args.config)
    
    if args.mode == 'full':
        setup.run_full_setup()
    elif args.mode == 'market':
        symbols = setup.get_stock_list()
        timeframes = setup.config.get('timeframes', ['1d'])
        start_date = setup.config.get('start_date')
        end_date = setup.config.get('end_date')
        setup.get_historical_data(symbols, start_date, end_date, timeframes)
    elif args.mode == 'fundamental':
        symbols = setup.get_stock_list()
        setup.get_fundamental_data(symbols)
    elif args.mode == 'industry':
        setup.get_industry_data()

if __name__ == '__main__':
    main() 