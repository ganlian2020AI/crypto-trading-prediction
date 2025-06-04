#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import ccxt
from urllib.parse import urljoin, urlparse

# 添加项目根目录到sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from common.utils import TradingLogger, load_config
from common.database import DatabaseManager


class DataDownloader:
    """数据下载器，支持多种数据源和代理设置"""
    
    def __init__(self, config: Dict[str, Any], db_manager: Optional[DatabaseManager] = None):
        """
        初始化数据下载器
        
        参数:
            config: 配置信息
            db_manager: 数据库管理器实例（可选）
        """
        self.config = config
        self.logger = TradingLogger('data_downloader', log_level=config.get('general', {}).get('log_level', 'INFO'))
        self.data_dir = os.path.join(ROOT_DIR, 'data')
        
        # 确保数据目录存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # 获取下载配置
        self.download_config = config.get('data_download', {})
        
        # 代理设置
        self.proxy = self.download_config.get('proxy', {})
        self.proxy_url = self.proxy.get('url', '')
        self.use_proxy = self.proxy.get('enabled', False)
        
        # API代理前缀URL
        self.api_proxy_prefix = self.download_config.get('api_proxy_prefix', '')
        self.use_api_proxy = bool(self.api_proxy_prefix)
        if self.use_api_proxy:
            self.logger.info(f"使用API代理前缀: {self.api_proxy_prefix}")
        
        # 数据源配置
        self.sources = self.download_config.get('sources', {})
        
        # 白名单和黑名单
        self.whitelist = self.download_config.get('whitelist', [])
        self.blacklist = self.download_config.get('blacklist', [])
        
        # 数据库管理器
        self.db_manager = db_manager
        
        # 初始化API客户端
        self._init_clients()
    
    def _init_clients(self):
        """初始化各数据源的API客户端"""
        self.clients = {}
        
        # 初始化币安客户端
        if self.sources.get('binance', {}).get('enabled', False):
            try:
                binance_config = {}
                
                # 设置HTTP代理
                if self.use_proxy and self.proxy_url:
                    binance_config['proxies'] = {
                        'http': self.proxy_url,
                        'https': self.proxy_url
                    }
                
                # 设置API代理前缀
                if self.use_api_proxy:
                    # 创建自定义的币安客户端
                    self.clients['binance'] = BinanceProxyClient(
                        api_proxy_prefix=self.api_proxy_prefix,
                        config=binance_config
                    )
                else:
                    # 创建标准币安客户端
                    self.clients['binance'] = ccxt.binance(binance_config)
                
                self.logger.info("币安API客户端初始化成功")
            except Exception as e:
                self.logger.error(f"初始化币安API客户端失败: {str(e)}")
        
        # 其他客户端可以在这里添加
    
    def _is_symbol_allowed(self, symbol: str) -> bool:
        """
        检查交易对是否允许下载
        
        参数:
            symbol: 交易对符号
            
        返回:
            是否允许下载
        """
        # 如果白名单不为空，只允许白名单中的交易对
        if self.whitelist:
            return symbol in self.whitelist
        
        # 如果黑名单不为空，不允许黑名单中的交易对
        if self.blacklist:
            return symbol not in self.blacklist
        
        # 默认允许所有交易对
        return True
    
    def _apply_api_proxy_to_url(self, url: str) -> str:
        """
        将API代理前缀应用到URL
        
        参数:
            url: 原始URL
            
        返回:
            应用了API代理前缀的URL
        """
        if not self.use_api_proxy:
            return url
        
        # 确保API代理前缀以/结尾
        prefix = self.api_proxy_prefix
        if not prefix.endswith('/'):
            prefix += '/'
        
        # 解析原始URL
        parsed_url = urlparse(url)
        
        # 构建新URL
        if parsed_url.netloc:
            # 完整URL
            new_url = f"{prefix.rstrip('/')}{url}"
        else:
            # 相对URL
            new_url = urljoin(prefix, url.lstrip('/'))
        
        return new_url
    
    def download_alphavantage_data(self, symbol: str, interval: str = 'daily', 
                                  output_size: str = 'full') -> pd.DataFrame:
        """
        从Alpha Vantage下载数据
        
        参数:
            symbol: 交易对符号
            interval: 时间间隔 (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            output_size: 输出大小 (compact, full)
            
        返回:
            数据DataFrame
        """
        if not self._is_symbol_allowed(symbol):
            self.logger.warning(f"交易对 {symbol} 不在允许下载的列表中")
            return pd.DataFrame()
        
        try:
            alphavantage_config = self.sources.get('alphavantage', {})
            api_key = alphavantage_config.get('api_key', '')
            
            if not api_key:
                self.logger.error("未配置Alpha Vantage API密钥")
                return pd.DataFrame()
            
            # 构建请求URL
            base_url = 'https://www.alphavantage.co/query'
            
            # 如果启用了API代理前缀，应用到URL
            if self.use_api_proxy:
                base_url = self._apply_api_proxy_to_url(base_url)
            
            # 根据时间间隔选择函数
            if interval == 'daily':
                function = 'TIME_SERIES_DAILY'
            elif interval == 'weekly':
                function = 'TIME_SERIES_WEEKLY'
            elif interval == 'monthly':
                function = 'TIME_SERIES_MONTHLY'
            else:
                function = 'TIME_SERIES_INTRADAY'
            
            # 构建参数
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': api_key,
                'outputsize': output_size,
                'datatype': 'json'
            }
            
            # 如果是分钟级数据，添加interval参数
            if function == 'TIME_SERIES_INTRADAY':
                params['interval'] = interval
            
            # 设置代理
            proxies = None
            if self.use_proxy and self.proxy_url:
                proxies = {
                    'http': self.proxy_url,
                    'https': self.proxy_url
                }
            
            # 发送请求
            self.logger.info(f"请求Alpha Vantage API: {base_url}")
            response = requests.get(base_url, params=params, proxies=proxies)
            data = response.json()
            
            # 检查是否有错误
            if 'Error Message' in data:
                self.logger.error(f"Alpha Vantage API错误: {data['Error Message']}")
                return pd.DataFrame()
            
            # 提取时间序列数据
            if function == 'TIME_SERIES_DAILY':
                time_series_key = 'Time Series (Daily)'
            elif function == 'TIME_SERIES_WEEKLY':
                time_series_key = 'Weekly Time Series'
            elif function == 'TIME_SERIES_MONTHLY':
                time_series_key = 'Monthly Time Series'
            else:
                time_series_key = f"Time Series ({interval})"
            
            if time_series_key not in data:
                self.logger.error(f"Alpha Vantage API返回数据中没有找到 {time_series_key}")
                return pd.DataFrame()
            
            # 转换为DataFrame
            time_series = data[time_series_key]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # 重命名列
            df = df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            })
            
            # 转换类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # 添加timestamp列并设置为索引
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df['timestamp'] = df.index
            
            self.logger.info(f"从Alpha Vantage下载了 {len(df)} 条 {symbol} 数据")
            
            return df
            
        except Exception as e:
            self.logger.error(f"从Alpha Vantage下载 {symbol} 数据失败: {str(e)}")
            return pd.DataFrame()
    
    def download_binance_data(self, symbol: str, interval: str = '1d', 
                             start_time: Optional[str] = None, 
                             end_time: Optional[str] = None) -> pd.DataFrame:
        """
        从币安下载数据
        
        参数:
            symbol: 交易对符号 (例如 'BTC/USDT')
            interval: 时间间隔 (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            start_time: 开始时间 (YYYY-MM-DD 格式)
            end_time: 结束时间 (YYYY-MM-DD 格式)
            
        返回:
            数据DataFrame
        """
        # 检查交易对是否允许下载
        if not self._is_symbol_allowed(symbol):
            self.logger.warning(f"交易对 {symbol} 不在允许下载的列表中")
            return pd.DataFrame()
        
        try:
            # 检查币安客户端是否初始化
            if 'binance' not in self.clients:
                self.logger.error("币安API客户端未初始化")
                return pd.DataFrame()
            
            # 币安客户端
            binance = self.clients['binance']
            
            # 处理交易对格式
            if '/' not in symbol:
                # 如果是BTC，转换为BTC/USDT格式
                symbol = f"{symbol}/USDT"
            
            # 转换时间格式
            since = None
            if start_time:
                since = int(datetime.strptime(start_time, '%Y-%m-%d').timestamp() * 1000)
            
            until = None
            if end_time:
                until = int(datetime.strptime(end_time, '%Y-%m-%d').timestamp() * 1000)
            
            # 下载数据
            self.logger.info(f"开始从币安下载 {symbol} {interval} 数据...")
            
            # 初始化空列表存储所有数据
            all_ohlcv = []
            
            # 如果没有指定开始时间，默认获取最近1000条数据
            if not since:
                ohlcv = binance.fetch_ohlcv(symbol, interval)
                all_ohlcv.extend(ohlcv)
            else:
                # 分批下载数据
                current_since = since
                while True:
                    # 获取一批数据
                    ohlcv = binance.fetch_ohlcv(symbol, interval, since=current_since, limit=1000)
                    
                    # 如果没有数据，退出循环
                    if not ohlcv:
                        break
                    
                    # 添加到结果中
                    all_ohlcv.extend(ohlcv)
                    
                    # 更新since时间为最后一条数据的时间 + 1
                    current_since = ohlcv[-1][0] + 1
                    
                    # 如果已经到达结束时间，退出循环
                    if until and current_since > until:
                        break
                    
                    # 避免请求过于频繁
                    time.sleep(1)
            
            # 转换为DataFrame
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换timestamp为日期时间
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 设置timestamp为索引
            df = df.set_index('timestamp')
            
            # 排序
            df = df.sort_index()
            
            # 重置索引，保留timestamp列
            df = df.reset_index()
            
            self.logger.info(f"从币安下载了 {len(df)} 条 {symbol} 数据")
            
            return df
            
        except Exception as e:
            self.logger.error(f"从币安下载 {symbol} 数据失败: {str(e)}")
            return pd.DataFrame()
    
    def save_data_to_csv(self, df: pd.DataFrame, symbol: str) -> str:
        """
        将数据保存到CSV文件
        
        参数:
            df: 数据DataFrame
            symbol: 交易对符号
            
        返回:
            保存的文件路径
        """
        try:
            # 清理符号名称，去掉/等特殊字符
            clean_symbol = symbol.replace('/', '_').replace('-', '_')
            
            # 构建文件路径
            file_path = os.path.join(self.data_dir, f"{clean_symbol}.csv")
            
            # 保存到CSV
            df.to_csv(file_path, index=False)
            
            self.logger.info(f"{symbol} 数据已保存到 {file_path}")
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"保存 {symbol} 数据到CSV失败: {str(e)}")
            return ""
    
    def save_data_to_database(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        将数据保存到数据库
        
        参数:
            df: 数据DataFrame
            symbol: 交易对符号
            
        返回:
            是否保存成功
        """
        # 检查数据库管理器是否初始化
        if not self.db_manager:
            self.logger.error("数据库管理器未初始化，无法保存数据到数据库")
            return False
        
        try:
            # 清理符号名称，去掉/等特殊字符
            clean_symbol = symbol.replace('/', '_').replace('-', '_')
            
            # 确保数据库连接
            if not self.db_manager.conn:
                self.db_manager.connect()
            
            # 确保数据表存在
            self.db_manager.create_tables()
            
            # 保存数据到数据库
            success = self.db_manager.save_market_data(clean_symbol, df)
            
            if success:
                self.logger.info(f"{symbol} 数据已保存到数据库")
            else:
                self.logger.error(f"保存 {symbol} 数据到数据库失败")
            
            return success
            
        except Exception as e:
            self.logger.error(f"保存 {symbol} 数据到数据库失败: {str(e)}")
            return False
    
    def download_data(self, symbol: str, source: str = 'binance', 
                     interval: str = '1d', save_to_csv: bool = True, 
                     save_to_db: bool = True, **kwargs) -> pd.DataFrame:
        """
        下载数据并保存
        
        参数:
            symbol: 交易对符号
            source: 数据源 (binance, alphavantage)
            interval: 时间间隔
            save_to_csv: 是否保存到CSV
            save_to_db: 是否保存到数据库
            **kwargs: 其他参数
            
        返回:
            数据DataFrame
        """
        df = pd.DataFrame()
        
        # 根据数据源选择下载方法
        if source == 'binance':
            df = self.download_binance_data(symbol, interval, **kwargs)
        elif source == 'alphavantage':
            df = self.download_alphavantage_data(symbol, interval, **kwargs)
        else:
            self.logger.error(f"不支持的数据源: {source}")
            return df
        
        # 如果数据为空，返回
        if df.empty:
            self.logger.warning(f"下载的 {symbol} 数据为空")
            return df
        
        # 保存到CSV
        if save_to_csv:
            self.save_data_to_csv(df, symbol)
        
        # 保存到数据库
        if save_to_db and self.db_manager:
            self.save_data_to_database(df, symbol)
        
        return df
    
    def download_multiple_symbols(self, symbols: List[str], source: str = 'binance', 
                                 interval: str = '1d', save_to_csv: bool = True, 
                                 save_to_db: bool = True, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        下载多个交易对的数据
        
        参数:
            symbols: 交易对符号列表
            source: 数据源 (binance, alphavantage)
            interval: 时间间隔
            save_to_csv: 是否保存到CSV
            save_to_db: 是否保存到数据库
            **kwargs: 其他参数
            
        返回:
            {symbol: DataFrame} 格式的字典
        """
        results = {}
        
        for symbol in symbols:
            # 检查交易对是否允许下载
            if not self._is_symbol_allowed(symbol):
                self.logger.warning(f"交易对 {symbol} 不在允许下载的列表中，跳过")
                continue
            
            self.logger.info(f"开始下载 {symbol} 数据...")
            
            df = self.download_data(symbol, source, interval, save_to_csv, save_to_db, **kwargs)
            
            if not df.empty:
                results[symbol] = df
            
            # 避免请求过于频繁
            time.sleep(1)
        
        return results
    
    def download_symbols_from_config(self) -> Dict[str, pd.DataFrame]:
        """
        根据配置文件下载交易对数据
        
        返回:
            {symbol: DataFrame} 格式的字典
        """
        symbols_config = self.config.get('symbols', [])
        symbols = [s['name'] for s in symbols_config if s.get('enabled', True)]
        
        # 获取下载配置
        source = self.download_config.get('default_source', 'binance')
        interval = self.download_config.get('default_interval', '1d')
        save_to_csv = self.download_config.get('save_to_csv', True)
        save_to_db = self.download_config.get('save_to_db', True)
        
        # 其他参数
        kwargs = {}
        if 'start_date' in self.download_config:
            kwargs['start_time'] = self.download_config['start_date']
        if 'end_date' in self.download_config:
            kwargs['end_time'] = self.download_config['end_date']
        
        return self.download_multiple_symbols(symbols, source, interval, save_to_csv, save_to_db, **kwargs)


class BinanceProxyClient:
    """
    自定义的币安客户端，支持API代理前缀
    """
    
    def __init__(self, api_proxy_prefix: str, config: Dict = None):
        """
        初始化币安代理客户端
        
        参数:
            api_proxy_prefix: API代理前缀URL
            config: 其他配置选项
        """
        self.api_proxy_prefix = api_proxy_prefix
        self.config = config or {}
        self.logger = TradingLogger('binance_proxy_client')
        
        # 确保API代理前缀以/结尾
        if not self.api_proxy_prefix.endswith('/'):
            self.api_proxy_prefix += '/'
        
        # 标准币安API端点
        self.base_url = 'https://api.binance.com'
        
        # 代理设置
        self.proxies = self.config.get('proxies')
    
    def _get_proxied_url(self, endpoint: str) -> str:
        """
        获取应用了API代理前缀的URL
        
        参数:
            endpoint: API端点
            
        返回:
            完整URL
        """
        # 构建完整的币安URL
        full_url = f"{self.base_url}{endpoint}"
        
        # 应用API代理前缀
        proxied_url = f"{self.api_proxy_prefix.rstrip('/')}{full_url}"
        
        return proxied_url
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1d', since: int = None, limit: int = None, params: Dict = None) -> List:
        """
        获取K线数据
        
        参数:
            symbol: 交易对符号
            timeframe: 时间间隔
            since: 开始时间戳（毫秒）
            limit: 返回数据的最大数量
            params: 其他参数
            
        返回:
            K线数据列表
        """
        # 处理参数
        params = params or {}
        
        # 处理交易对格式
        if '/' in symbol:
            symbol = symbol.replace('/', '')
        
        # 构建请求参数
        request_params = {
            'symbol': symbol,
            'interval': timeframe,
        }
        
        if since:
            request_params['startTime'] = since
        
        if limit:
            request_params['limit'] = limit
        
        # 合并其他参数
        request_params.update(params)
        
        # 构建API端点
        endpoint = '/api/v3/klines'
        
        # 获取代理URL
        url = self._get_proxied_url(endpoint)
        
        self.logger.info(f"请求币安K线数据: {url}")
        
        # 发送请求
        response = requests.get(url, params=request_params, proxies=self.proxies)
        
        # 检查响应状态
        if response.status_code != 200:
            self.logger.error(f"币安API请求失败: {response.status_code} - {response.text}")
            return []
        
        # 解析响应
        data = response.json()
        
        # 转换为OHLCV格式
        ohlcv = []
        for item in data:
            ohlcv.append([
                int(item[0]),  # timestamp
                float(item[1]),  # open
                float(item[2]),  # high
                float(item[3]),  # low
                float(item[4]),  # close
                float(item[5]),  # volume
            ])
        
        return ohlcv


def main():
    """主函数，用于命令行调用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据下载工具')
    parser.add_argument('--config', type=str, default='config/trading_config.yaml', help='配置文件路径')
    parser.add_argument('--symbol', type=str, help='交易对符号')
    parser.add_argument('--source', type=str, default='binance', choices=['binance', 'alphavantage'], help='数据源')
    parser.add_argument('--interval', type=str, default='1d', help='时间间隔')
    parser.add_argument('--start', type=str, help='开始时间 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='结束时间 (YYYY-MM-DD)')
    parser.add_argument('--csv', action='store_true', help='保存到CSV')
    parser.add_argument('--db', action='store_true', help='保存到数据库')
    parser.add_argument('--all', action='store_true', help='下载配置文件中的所有交易对')
    parser.add_argument('--proxy-prefix', type=str, help='API代理前缀URL')
    
    args = parser.parse_args()
    
    # 加载配置文件
    config_path = os.path.join(ROOT_DIR, args.config)
    config = load_config(config_path)
    
    # 如果命令行指定了代理前缀，覆盖配置文件中的设置
    if args.proxy_prefix:
        if 'data_download' not in config:
            config['data_download'] = {}
        config['data_download']['api_proxy_prefix'] = args.proxy_prefix
    
    # 创建数据库管理器
    from common.database import DatabaseManager
    db_config = config.get('database', {})
    db_manager = DatabaseManager(db_config)
    db_manager.connect()
    
    # 创建数据下载器
    downloader = DataDownloader(config, db_manager)
    
    try:
        if args.all:
            # 下载配置文件中的所有交易对
            downloader.download_symbols_from_config()
        elif args.symbol:
            # 下载单个交易对
            kwargs = {}
            if args.start:
                kwargs['start_time'] = args.start
            if args.end:
                kwargs['end_time'] = args.end
                
            downloader.download_data(args.symbol, args.source, args.interval, args.csv, args.db, **kwargs)
        else:
            print("请指定要下载的交易对或使用--all下载所有交易对")
    finally:
        # 关闭数据库连接
        if db_manager:
            db_manager.disconnect()


if __name__ == "__main__":
    main() 