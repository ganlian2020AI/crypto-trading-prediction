#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from common.utils import load_config, TradingLogger
from live.live_trader import LiveTrader


def start_trading(args):
    """启动实盘交易"""
    # 创建实盘交易实例
    trader = LiveTrader(args.config)
    
    # 设置交易品种
    if args.symbols:
        trader.symbols = args.symbols.split(',')
    
    # 启动交易服务
    trader.start_trading(args.model, args.interval)


def stop_trading(args):
    """停止实盘交易"""
    # 创建实盘交易实例
    trader = LiveTrader(args.config)
    
    # 停止交易服务
    trader.stop_trading()


def show_status(args):
    """显示交易状态"""
    # 加载配置
    config = load_config(args.config)
    
    # 创建日志
    logger = TradingLogger('live_status', log_level=config.get('general', {}).get('log_level', 'INFO'))
    
    # 创建实盘交易实例
    trader = LiveTrader(args.config)
    
    # 检查交易所连接
    if trader.connect_exchange():
        logger.info("交易所连接成功")
        
        # 获取账户余额
        if hasattr(trader.exchange, 'get_account_balance'):
            balances = trader.exchange.get_account_balance()
            logger.info("账户余额:")
            for balance in balances:
                if float(balance.get('free', 0)) > 0:
                    logger.info(f"  {balance['asset']}: {balance['free']}")
        elif hasattr(trader.exchange, 'fetch_balance'):
            balance = trader.exchange.fetch_balance()
            logger.info("账户余额:")
            for asset, amount in balance['free'].items():
                if amount > 0:
                    logger.info(f"  {asset}: {amount}")
        
        # 获取当前持仓
        trader.update_positions()
        logger.info("当前持仓:")
        for symbol, positions in trader.positions.items():
            for pos in positions:
                logger.info(f"  {symbol}: 数量={pos['quantity']}, 当前价格={pos['current_price']}, 市值={pos['market_value']}")
        
        # 获取交易历史
        logger.info("交易历史:")
        for i, trade in enumerate(trader.trade_history[-5:]):
            logger.info(f"  {i+1}. {trade['timestamp']}: {trade['action']} {trade['symbol']}, 价格={trade['price']}, 数量={trade['quantity']}")
    else:
        logger.error("交易所连接失败")


def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description='加密货币交易系统 - 实盘交易工具')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 启动命令
    start_parser = subparsers.add_parser('start', help='启动实盘交易')
    start_parser.add_argument('--config', type=str, default='config/trading_config.yaml', help='配置文件路径')
    start_parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    start_parser.add_argument('--interval', type=int, default=30, help='检查间隔（分钟）')
    start_parser.add_argument('--symbols', type=str, default=None, help='交易品种，用逗号分隔')
    
    # 停止命令
    stop_parser = subparsers.add_parser('stop', help='停止实盘交易')
    stop_parser.add_argument('--config', type=str, default='config/trading_config.yaml', help='配置文件路径')
    
    # 状态命令
    status_parser = subparsers.add_parser('status', help='显示交易状态')
    status_parser.add_argument('--config', type=str, default='config/trading_config.yaml', help='配置文件路径')
    
    args = parser.parse_args()
    
    if args.command == 'start':
        start_trading(args)
    elif args.command == 'stop':
        stop_trading(args)
    elif args.command == 'status':
        show_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 