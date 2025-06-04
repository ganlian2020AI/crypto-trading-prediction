 #!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode, urljoin, urlparse
from typing import Dict, List, Optional, Union, Any

# 添加项目根目录到sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from common.utils import TradingLogger


class BinanceTradingClient:
    """
    币安交易客户端，支持API代理前缀
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, 
                 api_proxy_prefix: str = '', proxy_url: str = ''):
        """
        初始化币安交易客户端
        
        参数:
            api_key: API密钥
            api_secret: API密钥
            testnet: 是否使用测试网络
            api_proxy_prefix: API代理前缀URL
            proxy_url: HTTP代理URL
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.api_proxy_prefix = api_proxy_prefix
        self.proxy_url = proxy_url
        
        # 初始化日志
        self.logger = TradingLogger('binance_trading_client')
        
        # 设置API基础URL
        if testnet:
            self.base_url = 'https://testnet.binance.vision'
        else:
            self.base_url = 'https://api.binance.com'
        
        # 设置HTTP代理
        self.proxies = None
        if proxy_url:
            self.proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
            self.logger.info(f"使用HTTP代理: {proxy_url}")
        
        # 设置API代理前缀
        self.use_api_proxy = bool(api_proxy_prefix)
        if self.use_api_proxy:
            self.logger.info(f"使用API代理前缀: {api_proxy_prefix}")
            # 确保API代理前缀以/结尾
            if not self.api_proxy_prefix.endswith('/'):
                self.api_proxy_prefix += '/'
    
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
        
        # 如果使用API代理前缀，应用前缀
        if self.use_api_proxy:
            return f"{self.api_proxy_prefix.rstrip('/')}{full_url}"
        
        return full_url
    
    def _generate_signature(self, params: Dict) -> str:
        """
        生成API签名
        
        参数:
            params: 请求参数
            
        返回:
            签名字符串
        """
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _send_request(self, method: str, endpoint: str, params: Dict = None, 
                     signed: bool = False, headers: Dict = None) -> Dict:
        """
        发送API请求
        
        参数:
            method: HTTP方法 (GET, POST, DELETE, PUT)
            endpoint: API端点
            params: 请求参数
            signed: 是否需要签名
            headers: 自定义请求头
            
        返回:
            API响应
        """
        # 初始化参数
        params = params or {}
        headers = headers or {}
        
        # 添加API密钥到请求头
        headers['X-MBX-APIKEY'] = self.api_key
        
        # 如果需要签名，添加时间戳和签名
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        # 获取完整URL
        url = self._get_proxied_url(endpoint)
        
        # 发送请求
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, proxies=self.proxies)
            elif method == 'POST':
                response = requests.post(url, data=params, headers=headers, proxies=self.proxies)
            elif method == 'DELETE':
                response = requests.delete(url, params=params, headers=headers, proxies=self.proxies)
            elif method == 'PUT':
                response = requests.put(url, data=params, headers=headers, proxies=self.proxies)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            # 检查响应状态
            if response.status_code != 200:
                self.logger.error(f"API请求失败: {response.status_code} - {response.text}")
                return {'error': True, 'message': response.text, 'code': response.status_code}
            
            # 解析JSON响应
            return response.json()
            
        except Exception as e:
            self.logger.error(f"API请求异常: {str(e)}")
            return {'error': True, 'message': str(e)}
    
    # 账户信息相关方法
    def get_account_info(self) -> Dict:
        """获取账户信息"""
        return self._send_request('GET', '/api/v3/account', signed=True)
    
    def get_account_balance(self) -> List:
        """获取账户余额"""
        account_info = self.get_account_info()
        if 'error' in account_info:
            return []
        return account_info.get('balances', [])
    
    # 交易相关方法
    def create_order(self, symbol: str, side: str, order_type: str, 
                    quantity: float = None, price: float = None, 
                    time_in_force: str = 'GTC', **kwargs) -> Dict:
        """
        创建订单
        
        参数:
            symbol: 交易对符号
            side: 买卖方向 (BUY, SELL)
            order_type: 订单类型 (LIMIT, MARKET, STOP_LOSS, STOP_LOSS_LIMIT, TAKE_PROFIT, TAKE_PROFIT_LIMIT)
            quantity: 数量
            price: 价格 (LIMIT订单必须)
            time_in_force: 有效期 (GTC, IOC, FOK)
            **kwargs: 其他参数
            
        返回:
            订单信息
        """
        # 构建参数
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
        }
        
        # 添加数量
        if quantity:
            params['quantity'] = quantity
        
        # 添加价格 (LIMIT订单必须)
        if price and order_type == 'LIMIT':
            params['price'] = price
            params['timeInForce'] = time_in_force
        
        # 添加其他参数
        params.update(kwargs)
        
        # 发送请求
        return self._send_request('POST', '/api/v3/order', params=params, signed=True)
    
    def cancel_order(self, symbol: str, order_id: int = None, client_order_id: str = None) -> Dict:
        """
        取消订单
        
        参数:
            symbol: 交易对符号
            order_id: 订单ID
            client_order_id: 客户端订单ID
            
        返回:
            取消结果
        """
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['origClientOrderId'] = client_order_id
        else:
            return {'error': True, 'message': '必须提供order_id或client_order_id'}
        
        return self._send_request('DELETE', '/api/v3/order', params=params, signed=True)
    
    def get_order(self, symbol: str, order_id: int = None, client_order_id: str = None) -> Dict:
        """
        查询订单
        
        参数:
            symbol: 交易对符号
            order_id: 订单ID
            client_order_id: 客户端订单ID
            
        返回:
            订单信息
        """
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['origClientOrderId'] = client_order_id
        else:
            return {'error': True, 'message': '必须提供order_id或client_order_id'}
        
        return self._send_request('GET', '/api/v3/order', params=params, signed=True)
    
    def get_open_orders(self, symbol: str = None) -> List:
        """
        查询当前挂单
        
        参数:
            symbol: 交易对符号 (可选)
            
        返回:
            订单列表
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return self._send_request('GET', '/api/v3/openOrders', params=params, signed=True)
    
    # 市场数据相关方法
    def get_exchange_info(self) -> Dict:
        """获取交易规则和交易对信息"""
        return self._send_request('GET', '/api/v3/exchangeInfo')
    
    def get_ticker_price(self, symbol: str = None) -> Union[Dict, List]:
        """
        获取最新价格
        
        参数:
            symbol: 交易对符号 (可选)
            
        返回:
            价格信息
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return self._send_request('GET', '/api/v3/ticker/price', params=params)
    
    def get_ticker_24hr(self, symbol: str = None) -> Union[Dict, List]:
        """
        获取24小时价格变动情况
        
        参数:
            symbol: 交易对符号 (可选)
            
        返回:
            24小时价格变动信息
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return self._send_request('GET', '/api/v3/ticker/24hr', params=params)
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500, 
                  start_time: int = None, end_time: int = None) -> List:
        """
        获取K线数据
        
        参数:
            symbol: 交易对符号
            interval: 时间间隔 (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: 返回数量 (默认500，最大1000)
            start_time: 开始时间戳 (毫秒)
            end_time: 结束时间戳 (毫秒)
            
        返回:
            K线数据列表
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        
        if end_time:
            params['endTime'] = end_time
        
        return self._send_request('GET', '/api/v3/klines', params=params)