import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import requests
import json
import ccxt
from typing import Dict, List, Tuple, Union, Optional, Any
import schedule

# 添加项目根目录到sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from common.utils import TradingLogger, AlertManager, DataHandler, load_config
from live.binance_trading_client import BinanceTradingClient


class LiveTrader:
    """实盘交易类"""
    
    def __init__(self, config_path: str = None, config: dict = None):
        """
        初始化实盘交易
        
        参数:
            config_path: 配置文件路径
            config: 配置字典，如果提供则优先使用
        """
        # 加载配置
        if config:
            self.config = config
        elif config_path:
            self.config = load_config(config_path)
        else:
            config_path = os.path.join(ROOT_DIR, 'config', 'trading_config.yaml')
            self.config = load_config(config_path)
        
        # 初始化日志
        log_level = self.config.get('general', {}).get('log_level', 'INFO')
        log_file = os.path.join(ROOT_DIR, 'logs', 'live_trader.log')
        self.logger = TradingLogger('live_trader', log_level, log_file)
        
        # 初始化警报管理器
        self.alert_manager = AlertManager(self.config, self.logger)
        
        # 初始化数据处理器
        self.data_handler = DataHandler(self.config, self.logger)
        
        # 交易参数
        self.live_config = self.config.get('live_trading', {})
        self.exchange_id = self.live_config.get('exchange', 'binance')
        self.api_key = self.live_config.get('api_key', '')
        self.api_secret = self.live_config.get('api_secret', '')
        self.testnet = self.live_config.get('testnet', True)
        
        # API代理前缀和代理设置
        self.api_proxy_prefix = self.live_config.get('api_proxy_prefix', '')
        self.proxy_config = self.live_config.get('proxy', {})
        self.proxy_url = self.proxy_config.get('url', '') if self.proxy_config.get('enabled', False) else ''
        
        # 交易品种配置
        self.symbols_config = self.config.get('symbols', [])
        self.symbols = [s['name'] for s in self.symbols_config if s.get('enabled', True)]
        
        # 风险管理参数
        self.risk_config = self.config.get('risk_management', {})
        self.max_open_trades = self.risk_config.get('max_open_trades', 5)
        self.max_daily_trades = self.risk_config.get('max_daily_trades', 10)
        self.max_drawdown_pct = self.risk_config.get('max_drawdown_pct', 0.1)
        
        # 交易状态
        self.positions = {}  # 当前持仓
        self.trade_history = []  # 交易历史
        self.daily_trades = 0  # 今日交易次数
        self.initial_balance = None  # 初始余额
        self.current_balance = None  # 当前余额
        self.running = False  # 是否正在运行
        self.last_update_time = None  # 上次更新时间
        
        # 交易所API客户端
        self.exchange = None
        
        self.logger.info(f"实盘交易初始化完成，交易品种: {', '.join(self.symbols)}")
    
    def connect_exchange(self):
        """连接交易所API"""
        try:
            # 检查是否提供了API密钥
            if not self.api_key or not self.api_secret:
                self.logger.error("未提供API密钥，无法连接交易所")
                return False
            
            # 初始化交易所API客户端
            if self.exchange_id == 'binance':
                # 使用我们自定义的币安交易客户端
                self.exchange = BinanceTradingClient(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=self.testnet,
                    api_proxy_prefix=self.api_proxy_prefix,
                    proxy_url=self.proxy_url
                )
                
                # 测试API连接
                account_info = self.exchange.get_account_info()
                if 'error' in account_info:
                    self.logger.error(f"连接币安交易所失败: {account_info.get('message', '未知错误')}")
                    return False
                
                self.logger.info(f"成功连接到币安交易所")
                return True
            else:
                # 对于其他交易所，使用ccxt库
                exchange_class = getattr(ccxt, self.exchange_id)
                
                # 配置交易所客户端
                exchange_config = {
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot'  # 默认现货交易
                    }
                }
                
                # 添加代理设置
                if self.proxy_url:
                    exchange_config['proxies'] = {
                        'http': self.proxy_url,
                        'https': self.proxy_url
                    }
                
                self.exchange = exchange_class(exchange_config)
                
                # 设置测试网络
                if self.testnet:
                    if hasattr(self.exchange, 'set_sandbox_mode'):
                        self.exchange.set_sandbox_mode(True)
                        self.logger.info(f"已启用{self.exchange_id}测试网络模式")
                    else:
                        self.logger.warning(f"{self.exchange_id}不支持测试网络模式")
                
                # 测试API连接
                self.exchange.fetch_balance()
                
                self.logger.info(f"成功连接到{self.exchange_id}交易所")
                return True
            
        except Exception as e:
            self.logger.error(f"连接交易所失败: {str(e)}")
            return False
    
    def load_model(self, model_path: str):
        """
        加载预测模型
        
        参数:
            model_path: 模型路径
        """
        try:
            # 这里需要导入模型模块，根据项目实际情况调整
            try:
                # 尝试导入主项目的模型
                sys.path.append(os.path.dirname(ROOT_DIR))
                from model import CryptoLSTMModel
                
                # 加载模型
                self.model = CryptoLSTMModel(os.path.join(ROOT_DIR, 'config', 'trading_config.yaml'))
                self.model.load(model_path)
                
                self.logger.info(f"已加载模型: {model_path}")
                return True
                
            except ImportError:
                self.logger.error("无法导入模型模块，请确保模型文件可用")
                return False
                
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            return False
    
    def fetch_latest_data(self):
        """获取最新市场数据"""
        try:
            data = {}
            
            # 获取每个品种的最新数据
            for symbol in self.symbols:
                # 构建交易对格式
                market_symbol = f"{symbol}/USDT"
                
                # 获取K线数据
                # 这里假设我们需要获取最近100根30分钟K线
                timeframe = '30m'
                limit = 100
                
                if isinstance(self.exchange, BinanceTradingClient):
                    # 使用我们自定义的币安客户端
                    ohlcv = self.exchange.get_klines(
                        symbol=f"{symbol}USDT",
                        interval=timeframe,
                        limit=limit
                    )
                    
                    # 转换为DataFrame格式
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                                                     'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                                                     'taker_buy_quote_asset_volume', 'ignore'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # 只保留OHLCV列
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    
                    # 转换为数值类型
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col])
                else:
                    # 使用ccxt库
                    ohlcv = self.exchange.fetch_ohlcv(market_symbol, timeframe, limit=limit)
                    
                    # 转换为DataFrame
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                
                # 预处理数据
                df = self.data_handler.preprocess_data(df)
                
                data[symbol] = df
            
            # 对齐多品种数据
            if len(data) > 1:
                self.market_data = self.data_handler.align_multi_symbol_data(data)
            else:
                # 单品种情况，不需要对齐
                symbol = list(data.keys())[0]
                self.market_data = data[symbol].copy()
            
            self.last_update_time = datetime.now()
            self.logger.info(f"已获取最新市场数据，形状: {self.market_data.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"获取最新市场数据失败: {str(e)}")
            return False
    
    def generate_signals(self):
        """生成交易信号"""
        try:
            if not hasattr(self, 'model') or not hasattr(self, 'market_data'):
                self.logger.error("无法生成信号，请先加载模型和市场数据")
                return None
            
            # 使用模型生成预测
            # 这里需要根据实际模型的预测方法进行调整
            predictions, interpreted_predictions = self.model.predict(self.market_data)
            
            # 获取最新的预测信号
            latest_signals = {}
            for symbol in self.symbols:
                if f"{symbol}_signal" in interpreted_predictions.columns:
                    # 获取最新信号
                    latest_signal = interpreted_predictions[f"{symbol}_signal"].iloc[-1]
                    latest_signals[symbol] = latest_signal
                    
                    self.logger.info(f"{symbol}最新信号: {latest_signal}")
            
            return latest_signals
            
        except Exception as e:
            self.logger.error(f"生成交易信号失败: {str(e)}")
            return None
    
    def execute_trades(self, signals: Dict[str, int]):
        """
        执行交易
        
        参数:
            signals: 交易信号字典，{symbol: signal}
        """
        if not signals:
            self.logger.info("没有交易信号，不执行交易")
            return
        
        if not self.exchange:
            self.logger.error("交易所API未连接，无法执行交易")
            return
        
        try:
            # 获取当前账户余额
            if isinstance(self.exchange, BinanceTradingClient):
                balances = self.exchange.get_account_balance()
                usdt_balance = next((b for b in balances if b['asset'] == 'USDT'), {'free': 0})
                self.current_balance = float(usdt_balance.get('free', 0))
            else:
                balance = self.exchange.fetch_balance()
                self.current_balance = balance['total']['USDT'] if 'USDT' in balance['total'] else 0
            
            # 记录初始余额
            if self.initial_balance is None:
                self.initial_balance = self.current_balance
            
            # 检查是否超过最大回撤限制
            if self.initial_balance > 0:
                current_drawdown = (self.initial_balance - self.current_balance) / self.initial_balance
                if current_drawdown > self.max_drawdown_pct:
                    self.logger.warning(f"当前回撤({current_drawdown:.2%})超过最大回撤限制({self.max_drawdown_pct:.2%})，暂停交易")
                    self.alert_manager.send_alert(
                        f"当前回撤({current_drawdown:.2%})超过最大回撤限制({self.max_drawdown_pct:.2%})，暂停交易",
                        level="warning"
                    )
                    return
            
            # 检查是否超过每日最大交易次数
            if self.daily_trades >= self.max_daily_trades:
                self.logger.warning(f"已达到每日最大交易次数({self.max_daily_trades})，暂停交易")
                return
            
            # 获取当前持仓
            self.update_positions()
            
            # 处理每个品种的信号
            for symbol, signal in signals.items():
                # 获取品种配置
                symbol_config = next((s for s in self.symbols_config if s['name'] == symbol), None)
                if not symbol_config:
                    continue
                
                # 获取交易大小
                trade_size_usd = symbol_config.get('trade_size_usd', 100)
                max_position = symbol_config.get('max_position', 3)
                
                # 构建交易对格式
                market_symbol = f"{symbol}/USDT"
                symbol_pair = f"{symbol}USDT"  # 币安API格式
                
                # 获取当前市场价格
                if isinstance(self.exchange, BinanceTradingClient):
                    ticker = self.exchange.get_ticker_price(symbol=symbol_pair)
                    current_price = float(ticker['price'])
                else:
                    ticker = self.exchange.fetch_ticker(market_symbol)
                    current_price = ticker['last']
                
                # 计算可交易数量
                quantity = trade_size_usd / current_price
                
                # 处理买入信号
                if signal > 0:
                    # 检查是否已达到最大持仓数量
                    if symbol in self.positions and len(self.positions[symbol]) >= max_position:
                        self.logger.info(f"{symbol}已达到最大持仓数量({max_position})，不执行买入")
                        continue
                    
                    # 检查是否有足够的资金
                    if self.current_balance < trade_size_usd:
                        self.logger.warning(f"余额不足，无法买入{symbol}")
                        continue
                    
                    try:
                        # 执行买入
                        if isinstance(self.exchange, BinanceTradingClient):
                            order = self.exchange.create_order(
                                symbol=symbol_pair,
                                side='BUY',
                                order_type='MARKET',
                                quantity=quantity
                            )
                            order_id = order.get('orderId', '')
                        else:
                            order = self.exchange.create_market_buy_order(market_symbol, quantity)
                            order_id = order['id']
                        
                        # 记录交易
                        trade_record = {
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'action': 'buy',
                            'price': current_price,
                            'quantity': quantity,
                            'order_id': order_id
                        }
                        
                        self.trade_history.append(trade_record)
                        self.daily_trades += 1
                        
                        # 更新持仓
                        if symbol not in self.positions:
                            self.positions[symbol] = []
                        
                        self.positions[symbol].append({
                            'entry_time': datetime.now(),
                            'entry_price': current_price,
                            'quantity': quantity,
                            'order_id': order_id
                        })
                        
                        # 发送警报
                        self.alert_manager.send_alert(
                            f"买入{symbol}: 价格={current_price}, 数量={quantity}, 订单ID={order_id}",
                            level="info",
                            subject="交易执行通知"
                        )
                        
                        self.logger.info(f"买入{symbol}: 价格={current_price}, 数量={quantity}, 订单ID={order_id}")
                        
                    except Exception as e:
                        self.logger.error(f"买入{symbol}失败: {str(e)}")
                        self.alert_manager.send_alert(
                            f"买入{symbol}失败: {str(e)}",
                            level="error",
                            subject="交易执行错误"
                        )
                
                # 处理卖出信号
                elif signal < 0:
                    # 检查是否有持仓可卖
                    if symbol not in self.positions or not self.positions[symbol]:
                        self.logger.info(f"没有{symbol}的持仓，不执行卖出")
                        continue
                    
                    try:
                        # 获取最早的持仓
                        position = self.positions[symbol].pop(0)
                        
                        # 执行卖出
                        if isinstance(self.exchange, BinanceTradingClient):
                            order = self.exchange.create_order(
                                symbol=symbol_pair,
                                side='SELL',
                                order_type='MARKET',
                                quantity=position['quantity']
                            )
                            order_id = order.get('orderId', '')
                        else:
                            order = self.exchange.create_market_sell_order(market_symbol, position['quantity'])
                            order_id = order['id']
                        
                        # 计算盈亏
                        profit_loss = (current_price - position['entry_price']) * position['quantity']
                        profit_loss_pct = (current_price / position['entry_price'] - 1) * 100
                        
                        # 记录交易
                        trade_record = {
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'action': 'sell',
                            'price': current_price,
                            'quantity': position['quantity'],
                            'order_id': order_id,
                            'profit_loss': profit_loss,
                            'profit_loss_pct': profit_loss_pct,
                            'holding_period': (datetime.now() - position['entry_time']).total_seconds() / 86400  # 持仓天数
                        }
                        
                        self.trade_history.append(trade_record)
                        self.daily_trades += 1
                        
                        # 发送警报
                        self.alert_manager.send_alert(
                            f"卖出{symbol}: 价格={current_price}, 数量={position['quantity']}, 盈亏={profit_loss:.2f}({profit_loss_pct:.2f}%), 订单ID={order_id}",
                            level="info",
                            subject="交易执行通知"
                        )
                        
                        self.logger.info(f"卖出{symbol}: 价格={current_price}, 数量={position['quantity']}, 盈亏={profit_loss:.2f}({profit_loss_pct:.2f}%), 订单ID={order_id}")
                        
                    except Exception as e:
                        self.logger.error(f"卖出{symbol}失败: {str(e)}")
                        self.alert_manager.send_alert(
                            f"卖出{symbol}失败: {str(e)}",
                            level="error",
                            subject="交易执行错误"
                        )
            
        except Exception as e:
            self.logger.error(f"执行交易失败: {str(e)}")
            self.alert_manager.send_alert(
                f"执行交易失败: {str(e)}",
                level="error",
                subject="交易执行错误"
            )
    
    def update_positions(self):
        """更新当前持仓信息"""
        try:
            # 获取当前持仓
            positions = {}
            
            # 获取当前持仓
            for symbol in self.symbols:
                # 构建交易对格式
                market_symbol = f"{symbol}/USDT"
                symbol_pair = f"{symbol}USDT"  # 币安API格式
                
                # 获取当前持仓
                if isinstance(self.exchange, BinanceTradingClient):
                    balances = self.exchange.get_account_balance()
                    asset_balance = next((b for b in balances if b['asset'] == symbol), {'free': '0'})
                    quantity = float(asset_balance.get('free', '0'))
                else:
                    balance = self.exchange.fetch_balance()
                    quantity = balance['free'][symbol] if symbol in balance['free'] else 0
                
                if quantity > 0:
                    # 获取当前市场价格
                    if isinstance(self.exchange, BinanceTradingClient):
                        ticker = self.exchange.get_ticker_price(symbol=symbol_pair)
                        current_price = float(ticker['price'])
                    else:
                        ticker = self.exchange.fetch_ticker(market_symbol)
                        current_price = ticker['last']
                    
                    # 记录持仓
                    positions[symbol] = [{
                        'entry_time': datetime.now(),  # 这里无法获取实际的入场时间，使用当前时间代替
                        'entry_price': 0,  # 这里无法获取实际的入场价格，使用0代替
                        'quantity': quantity,
                        'current_price': current_price,
                        'market_value': quantity * current_price
                    }]
            
            self.positions = positions
            self.logger.info(f"已更新持仓信息: {positions}")
            
        except Exception as e:
            self.logger.error(f"更新持仓信息失败: {str(e)}")
    
    def reset_daily_counters(self):
        """重置每日计数器"""
        self.daily_trades = 0
        self.logger.info("已重置每日交易计数器")
    
    def save_trade_history(self):
        """保存交易历史"""
        try:
            if not self.trade_history:
                return
            
            # 转换为DataFrame
            df = pd.DataFrame(self.trade_history)
            
            # 保存到CSV文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(ROOT_DIR, 'reports', f"trade_history_{timestamp}.csv")
            df.to_csv(file_path, index=False)
            
            self.logger.info(f"交易历史已保存到: {file_path}")
            
        except Exception as e:
            self.logger.error(f"保存交易历史失败: {str(e)}")
    
    def trading_cycle(self):
        """交易周期"""
        try:
            self.logger.info("开始交易周期")
            
            # 获取最新市场数据
            if not self.fetch_latest_data():
                return
            
            # 生成交易信号
            signals = self.generate_signals()
            if not signals:
                return
            
            # 执行交易
            self.execute_trades(signals)
            
            self.logger.info("交易周期完成")
            
        except Exception as e:
            self.logger.error(f"交易周期执行失败: {str(e)}")
            self.alert_manager.send_alert(
                f"交易周期执行失败: {str(e)}",
                level="error",
                subject="交易系统错误"
            )
    
    def start_trading(self, model_path: str, check_interval_minutes: int = 30):
        """
        启动交易服务
        
        参数:
            model_path: 模型路径
            check_interval_minutes: 检查间隔（分钟）
        """
        if self.running:
            self.logger.warning("交易服务已在运行中")
            return
        
        try:
            # 连接交易所
            if not self.connect_exchange():
                return
            
            # 加载模型
            if not self.load_model(model_path):
                return
            
            # 设置检查间隔
            check_interval_minutes = max(1, check_interval_minutes)  # 至少1分钟
            
            # 设置定时任务
            schedule.every(check_interval_minutes).minutes.do(self.trading_cycle)
            schedule.every().day.at("00:00").do(self.reset_daily_counters)
            
            # 初始运行一次
            self.trading_cycle()
            
            # 标记为运行中
            self.running = True
            
            # 发送启动通知
            self.alert_manager.send_alert(
                f"交易系统已启动，检查间隔: {check_interval_minutes}分钟，交易品种: {', '.join(self.symbols)}",
                level="info",
                subject="交易系统启动"
            )
            
            self.logger.info(f"交易服务已启动，检查间隔: {check_interval_minutes}分钟")
            
            # 主循环
            while self.running:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("收到中断信号，正在停止交易服务...")
            self.stop_trading()
            
        except Exception as e:
            self.logger.error(f"交易服务运行失败: {str(e)}")
            self.alert_manager.send_alert(
                f"交易服务运行失败: {str(e)}",
                level="error",
                subject="交易系统错误"
            )
            self.stop_trading()
    
    def stop_trading(self):
        """停止交易服务"""
        if not self.running:
            self.logger.warning("交易服务未在运行")
            return
        
        try:
            # 标记为停止
            self.running = False
            
            # 保存交易历史
            self.save_trade_history()
            
            # 发送停止通知
            self.alert_manager.send_alert(
                "交易系统已停止",
                level="info",
                subject="交易系统停止"
            )
            
            self.logger.info("交易服务已停止")
            
        except Exception as e:
            self.logger.error(f"停止交易服务失败: {str(e)}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='加密货币交易系统实盘交易工具')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--interval', type=int, default=30, help='检查间隔（分钟）')
    parser.add_argument('--symbols', type=str, default=None, help='交易品种，用逗号分隔')
    
    args = parser.parse_args()
    
    # 创建实盘交易实例
    trader = LiveTrader(args.config)
    
    # 设置交易品种
    if args.symbols:
        trader.symbols = args.symbols.split(',')
    
    # 启动交易服务
    trader.start_trading(args.model, args.interval)


if __name__ == "__main__":
    main()