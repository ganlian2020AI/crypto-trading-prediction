#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据库管理器使用示例
展示如何连接数据库、创建表、插入和查询数据
"""

import os
import sys
import yaml
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# 添加项目根目录到sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from common.database import DatabaseManager
from common.utils import TradingLogger

def load_config():
    """加载配置文件"""
    config_path = os.path.join(ROOT_DIR, 'config', 'trading_config.yaml')
    
    if not os.path.exists(config_path):
        # 如果配置文件不存在，使用模板创建
        template_path = os.path.join(ROOT_DIR, 'config', 'trading_config.yaml.template')
        with open(template_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # 创建配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"已创建配置文件: {config_path}")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def generate_sample_data(symbol='BTCUSDT', days=30, interval_hours=1):
    """生成示例价格数据"""
    # 创建日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # 生成时间戳
    timestamps = []
    current_date = start_date
    while current_date <= end_date:
        timestamps.append(current_date)
        current_date += timedelta(hours=interval_hours)
    
    # 生成价格数据
    np.random.seed(42)  # 设置随机种子，确保结果可重现
    
    # 初始价格
    initial_price = 50000.0
    
    # 生成价格序列
    prices = [initial_price]
    for i in range(1, len(timestamps)):
        # 随机价格变动，模拟真实市场
        change_pct = np.random.normal(0, 0.02)  # 均值0，标准差0.02
        new_price = prices[-1] * (1 + change_pct)
        prices.append(new_price)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'volume': [np.random.uniform(100, 1000) for _ in prices]
    })
    
    return df

def main():
    """主函数"""
    # 设置日志
    logger = TradingLogger('database_example', log_level='INFO')
    logger.info("开始数据库示例程序")
    
    # 加载配置
    config = load_config()
    
    # 创建数据库管理器
    db_manager = DatabaseManager(config)
    
    try:
        # 创建数据表
        logger.info("创建数据表...")
        db_manager.create_tables()
        
        # 生成示例数据
        logger.info("生成示例价格数据...")
        btc_data = generate_sample_data(symbol='BTCUSDT', days=30, interval_hours=1)
        eth_data = generate_sample_data(symbol='ETHUSDT', days=30, interval_hours=1)
        
        # 保存数据到数据库
        logger.info("保存BTC价格数据到数据库...")
        db_manager.save_price_data(btc_data, symbol='BTCUSDT', source='example')
        
        logger.info("保存ETH价格数据到数据库...")
        db_manager.save_price_data(eth_data, symbol='ETHUSDT', source='example')
        
        # 从数据库加载数据
        logger.info("从数据库加载BTC价格数据...")
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        btc_loaded = db_manager.load_price_data('BTCUSDT', start_date=start_date)
        
        logger.info(f"加载到 {len(btc_loaded)} 条BTC价格记录")
        if not btc_loaded.empty:
            logger.info(f"最新价格: {btc_loaded.iloc[-1].name}, 收盘价: {btc_loaded.iloc[-1]['close']}")
        
        # 保存交易记录示例
        logger.info("保存交易记录示例...")
        trade_data = {
            'symbol': 'BTCUSDT',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'order_type': 'MARKET',
            'side': 'BUY',
            'price': 50000.0,
            'quantity': 0.1,
            'fee': 5.0,
            'total_value': 5000.0,
            'status': 'FILLED',
            'trade_id': 'example_trade_001',
            'strategy': 'example_strategy',
            'notes': '示例交易记录'
        }
        db_manager.save_trade(trade_data)
        
        # 保存交易信号示例
        logger.info("保存交易信号示例...")
        signal_data = {
            'symbol': 'BTCUSDT',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'signal_type': 1,  # 1表示买入信号
            'confidence': 0.85,
            'price': 50000.0,
            'executed': False,
            'notes': '示例交易信号'
        }
        db_manager.save_signal(signal_data)
        
        # 查询示例
        logger.info("执行自定义查询示例...")
        trades = db_manager.query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 5")
        logger.info(f"最近5笔交易: {trades}")
        
        signals = db_manager.query("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 5")
        logger.info(f"最近5个信号: {signals}")
        
        # 统计查询示例
        logger.info("执行统计查询示例...")
        stats = db_manager.query("""
        SELECT 
            symbol,
            COUNT(*) as record_count,
            MIN(timestamp) as earliest_date,
            MAX(timestamp) as latest_date,
            AVG(close) as avg_close
        FROM price_data
        GROUP BY symbol
        """)
        logger.info(f"价格数据统计: {stats}")
        
    except Exception as e:
        logger.error(f"示例执行出错: {str(e)}")
    finally:
        # 关闭数据库连接
        db_manager.close()
        logger.info("数据库连接已关闭")
        logger.info("示例程序结束")

if __name__ == "__main__":
    main() 