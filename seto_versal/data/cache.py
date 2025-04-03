#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SETO-Versal 数据缓存模块

提供高效的多级缓存系统，支持内存和磁盘缓存
具有自动过期和清理功能
"""

import os
import logging
import pandas as pd
import numpy as np
import json
import pickle
import hashlib
import threading
import time
import datetime
from datetime import timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from enum import Enum

from seto_versal.data.manager import TimeFrame

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """缓存级别枚举"""
    MEMORY = "memory"  # 内存缓存，速度最快但不持久
    DISK = "disk"      # 磁盘缓存，速度适中且持久
    REDIS = "redis"    # Redis缓存，支持分布式访问

class CachePolicy(Enum):
    """缓存策略枚举"""
    LRU = "lru"        # 最近最少使用策略
    TTL = "ttl"        # 基于时间的过期策略
    FIFO = "fifo"      # 先进先出策略

class DataCache:
    """
    数据缓存器，支持多级缓存
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化缓存系统
        
        Args:
            config: 缓存配置参数
        """
        self.config = config or {}
        
        # 缓存策略
        policy_str = self.config.get('policy', 'lru')
        self.policy = CachePolicy(policy_str.lower())
        
        # 缓存参数
        self.max_memory_items = self.config.get('max_memory_items', 1000)
        self.max_disk_size_mb = self.config.get('max_disk_size_mb', 1024)
        self.default_ttl = self.config.get('default_ttl', 3600)  # 默认1小时
        
        # 缓存目录
        self.base_dir = self.config.get('cache_dir', 'data/cache')
        os.makedirs(self.base_dir, exist_ok=True)
        
        # 内存缓存
        self.memory_cache: Dict[str, Dict] = {}
        
        # 访问时间记录
        self.access_times: Dict[str, float] = {}
        
        # 过期时间记录
        self.expiry_times: Dict[str, float] = {}
        
        # 统计信息
        self.stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'writes': 0,
            'evictions': 0
        }
        
        # 缓存锁，用于线程安全
        self._lock = threading.RLock()
        
        logger.info(f"数据缓存初始化完成，策略: {self.policy.value}, "
                   f"内存容量: {self.max_memory_items}项, "
                   f"磁盘容量: {self.max_disk_size_mb}MB")
    
    def _generate_key(self, symbol: str, timeframe=None, 
                     start_time: Optional['datetime.datetime'] = None, 
                     end_time: Optional['datetime.datetime'] = None) -> str:
        """
        生成缓存键
        
        Args:
            symbol: 交易品种代码或键名
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            缓存键字符串
        """
        # 构建基本键
        key_parts = [symbol]
        
        # 如果指定了时间周期，添加到键中
        if timeframe is not None:
            try:
                key_parts.append(timeframe.value)
            except AttributeError:
                # 如果timeframe不是枚举，直接使用字符串表示
                key_parts.append(str(timeframe))
        
        # 如果指定了时间范围，添加到键中
        if start_time:
            key_parts.append(start_time.strftime('%Y%m%d%H%M'))
        if end_time:
            key_parts.append(end_time.strftime('%Y%m%d%H%M'))
            
        # 组合并哈希化
        key = "_".join(key_parts)
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_disk_path(self, key: str) -> str:
        """获取磁盘缓存路径"""
        return os.path.join(self.base_dir, f"{key}.pkl")
    
    def get(self, symbol: str, timeframe=None, 
           start_time: Optional['datetime.datetime'] = None, 
           end_time: Optional['datetime.datetime'] = None) -> Optional[Any]:
        """
        从缓存获取数据
        
        Args:
            symbol: 交易品种代码或键名
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            DataFrame数据或缓存的对象，如果缓存未命中则返回None
        """
        key = self._generate_key(symbol, timeframe, start_time, end_time)
        
        with self._lock:
            # 尝试从内存缓存获取
            if key in self.memory_cache:
                # 更新访问时间
                self.access_times[key] = time.time()
                
                # 检查是否已过期
                if key in self.expiry_times and time.time() > self.expiry_times[key]:
                    logger.debug(f"缓存过期: {symbol} {timeframe}")
                    self._remove_from_memory(key)
                else:
                    self.stats['memory_hits'] += 1
                    logger.debug(f"内存缓存命中: {symbol} {timeframe}")
                    return self.memory_cache[key]['data']
            
            # 尝试从磁盘缓存获取
            disk_path = self._get_disk_path(key)
            if os.path.exists(disk_path):
                try:
                    with open(disk_path, 'rb') as f:
                        cache_item = pickle.load(f)
                        
                    # 检查是否已过期
                    if 'expiry' in cache_item and time.time() > cache_item['expiry']:
                        logger.debug(f"磁盘缓存过期: {symbol} {timeframe}")
                        os.remove(disk_path)
                    else:
                        # 加载到内存缓存
                        self._add_to_memory(key, cache_item)
                        
                        self.stats['disk_hits'] += 1
                        logger.debug(f"磁盘缓存命中: {symbol} {timeframe}")
                        return cache_item['data']
                except Exception as e:
                    logger.warning(f"读取磁盘缓存失败: {e}")
            
            # 缓存未命中
            self.stats['misses'] += 1
            logger.debug(f"缓存未命中: {symbol} {timeframe}")
            return None
    
    def set(self, symbol: str, timeframe=None, data: Any=None,
           start_time: Optional['datetime.datetime'] = None, 
           end_time: Optional['datetime.datetime'] = None,
           ttl: Optional[int] = None,
           cache_levels: Set[CacheLevel] = None) -> bool:
        """
        将数据写入缓存
        
        Args:
            symbol: 交易品种代码或键名
            timeframe: 时间周期
            data: 要缓存的数据
            start_time: 开始时间
            end_time: 结束时间
            ttl: 缓存生存时间（秒），如果为None则使用默认值
            cache_levels: 要写入的缓存级别集合
            
        Returns:
            是否写入成功
        """
        if data is None:
            logger.warning(f"尝试缓存空数据: {symbol}")
            return False
            
        # 使用默认缓存级别
        if cache_levels is None:
            cache_levels = {CacheLevel.MEMORY, CacheLevel.DISK}
        
        key = self._generate_key(symbol, timeframe, start_time, end_time)
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        
        # 准备元数据
        metadata = {}
        try:
            # 如果数据是DataFrame，添加行数和时间范围信息
            if hasattr(data, 'index') and hasattr(data, '__len__'):
                metadata = {
                    'rows': len(data),
                    'start': data.index[0].strftime('%Y-%m-%d %H:%M:%S') if len(data) > 0 else None,
                    'end': data.index[-1].strftime('%Y-%m-%d %H:%M:%S') if len(data) > 0 else None
                }
            # 对于其他类型的数据，添加基本信息
            elif isinstance(data, dict):
                metadata = {'items': len(data)}
            elif isinstance(data, list) or isinstance(data, set):
                metadata = {'items': len(data)}
            else:
                metadata = {'type': type(data).__name__}
        except Exception as e:
            logger.warning(f"生成缓存元数据失败: {e}")
            metadata = {'error': '元数据生成失败'}
        
        cache_item = {
            'data': data,
            'symbol': symbol,
            'timeframe': timeframe,
            'created': time.time(),
            'expiry': expiry,
            'metadata': metadata
        }
        
        with self._lock:
            # 写入内存缓存
            if CacheLevel.MEMORY in cache_levels:
                self._add_to_memory(key, cache_item)
            
            # 写入磁盘缓存
            if CacheLevel.DISK in cache_levels:
                try:
                    disk_path = self._get_disk_path(key)
                    with open(disk_path, 'wb') as f:
                        pickle.dump(cache_item, f)
                    logger.debug(f"数据已写入磁盘缓存: {symbol} {timeframe}")
                except Exception as e:
                    logger.error(f"写入磁盘缓存失败: {e}")
                    return False
            
            self.stats['writes'] += 1
            return True
    
    def _add_to_memory(self, key: str, cache_item: Dict) -> None:
        """将数据添加到内存缓存"""
        # 检查是否需要清理缓存
        if len(self.memory_cache) >= self.max_memory_items:
            self._evict_from_memory()
        
        # 添加到缓存
        self.memory_cache[key] = cache_item
        self.access_times[key] = time.time()
        if 'expiry' in cache_item:
            self.expiry_times[key] = cache_item['expiry']
    
    def _remove_from_memory(self, key: str) -> None:
        """从内存缓存中移除数据"""
        if key in self.memory_cache:
            del self.memory_cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.expiry_times:
            del self.expiry_times[key]
    
    def _evict_from_memory(self) -> None:
        """根据缓存策略淘汰内存缓存项"""
        if not self.memory_cache:
            return
        
        if self.policy == CachePolicy.LRU:
            # 淘汰最久未访问的项
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            self._remove_from_memory(oldest_key)
        elif self.policy == CachePolicy.FIFO:
            # 淘汰最早创建的项
            oldest_key = min(self.memory_cache.items(), 
                           key=lambda x: x[1]['created'])[0]
            self._remove_from_memory(oldest_key)
        elif self.policy == CachePolicy.TTL:
            # 淘汰已过期的项，如果没有则淘汰最接近过期的
            now = time.time()
            expired = [k for k, v in self.expiry_times.items() if v <= now]
            
            if expired:
                self._remove_from_memory(expired[0])
            else:
                # 淘汰最快要过期的
                closest_to_expire = min(self.expiry_times.items(), key=lambda x: x[1])[0]
                self._remove_from_memory(closest_to_expire)
        
        self.stats['evictions'] += 1
    
    def invalidate(self, symbol: str = None, timeframe = None) -> int:
        """
        使缓存失效
        
        Args:
            symbol: 如果指定，则只使该交易品种的缓存失效
            timeframe: 如果指定，则只使该时间周期的缓存失效
            
        Returns:
            失效的缓存项数量
        """
        count = 0
        
        with self._lock:
            # 移除内存缓存
            keys_to_remove = []
            for key, item in self.memory_cache.items():
                if ((symbol is None or item['symbol'] == symbol) and 
                    (timeframe is None or item['timeframe'] == timeframe)):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_from_memory(key)
                count += 1
            
            # 移除磁盘缓存
            if symbol is None and timeframe is None:
                # 清空所有磁盘缓存
                for f in os.listdir(self.base_dir):
                    if f.endswith('.pkl'):
                        os.remove(os.path.join(self.base_dir, f))
                        count += 1
            else:
                # 选择性清除磁盘缓存
                for f in os.listdir(self.base_dir):
                    if f.endswith('.pkl'):
                        try:
                            with open(os.path.join(self.base_dir, f), 'rb') as file:
                                item = pickle.load(file)
                                if ((symbol is None or item['symbol'] == symbol) and 
                                    (timeframe is None or item['timeframe'] == timeframe)):
                                    os.remove(os.path.join(self.base_dir, f))
                                    count += 1
                        except Exception as e:
                            logger.warning(f"读取缓存文件失败: {e}")
        
        logger.info(f"缓存失效: {count}项")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            stats = self.stats.copy()
            stats['memory_items'] = len(self.memory_cache)
            
            # 计算磁盘缓存大小
            disk_size = 0
            for f in os.listdir(self.base_dir):
                if f.endswith('.pkl'):
                    disk_size += os.path.getsize(os.path.join(self.base_dir, f))
            
            stats['disk_size_mb'] = disk_size / (1024 * 1024)
            stats['memory_utilization'] = len(self.memory_cache) / self.max_memory_items
            stats['disk_utilization'] = stats['disk_size_mb'] / self.max_disk_size_mb
            
            if stats['memory_hits'] + stats['disk_hits'] + stats['misses'] > 0:
                hit_rate = (stats['memory_hits'] + stats['disk_hits']) / (
                    stats['memory_hits'] + stats['disk_hits'] + stats['misses']
                )
                stats['hit_rate'] = hit_rate
            else:
                stats['hit_rate'] = 0
            
            return stats
    
    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            # 清空内存缓存
            self.memory_cache.clear()
            self.access_times.clear()
            self.expiry_times.clear()
            
            # 清空磁盘缓存
            for f in os.listdir(self.base_dir):
                if f.endswith('.pkl'):
                    os.remove(os.path.join(self.base_dir, f))
            
            logger.info("缓存已清空")
    
    def preload(self, symbol: str, timeframe,
              start_time: "datetime.datetime", end_time: "datetime.datetime",
              data: pd.DataFrame) -> bool:
        """
        预加载数据到缓存
        
        Args:
            symbol: 交易品种代码
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            data: 要预加载的数据
            
        Returns:
            是否预加载成功
        """
        try:
            timeframe_str = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
            logger.info(f"预加载数据到缓存: {symbol} {timeframe_str}, "
                       f"{start_time.strftime('%Y-%m-%d')} 至 "
                       f"{end_time.strftime('%Y-%m-%d')}")
        except Exception as e:
            logger.info(f"预加载数据到缓存: {symbol}, {start_time.strftime('%Y-%m-%d')} 至 {end_time.strftime('%Y-%m-%d')}")
                   
        return self.set(symbol, timeframe, data, start_time, end_time) 