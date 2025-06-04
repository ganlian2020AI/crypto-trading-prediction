#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AlertManager使用示例
展示如何使用修复后的AlertManager类发送不同类型的警报
"""

import os
import sys
import yaml
from datetime import datetime

# 添加项目根目录到sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from common.utils import TradingLogger, AlertManager, load_config

def main():
    """主函数"""
    # 设置日志
    log_file = os.path.join(ROOT_DIR, 'logs', 'alert_example.log')
    logger = TradingLogger('alert_example', log_level='INFO', log_file=log_file)
    logger.info("开始警报管理器示例程序")
    
    # 加载配置
    config = load_config(os.path.join(ROOT_DIR, 'config', 'trading_config.yaml'))
    
    # 创建警报管理器
    alert_manager = AlertManager(config, logger)
    
    # 发送不同级别的警报
    logger.info("发送信息级别警报...")
    alert_manager.send_alert(
        alert_type="价格提醒",
        message="BTC价格已突破50000美元",
        level="INFO",
        data={
            "symbol": "BTCUSDT",
            "price": 50000,
            "time": datetime.now().isoformat()
        }
    )
    
    logger.info("发送警告级别警报...")
    alert_manager.send_alert(
        alert_type="波动提醒",
        message="ETH价格波动超过5%",
        level="WARNING",
        data={
            "symbol": "ETHUSDT",
            "price": 3500,
            "change_percent": 5.2,
            "time": datetime.now().isoformat()
        }
    )
    
    logger.info("发送错误级别警报...")
    alert_manager.send_alert(
        alert_type="交易失败",
        message="BTC买入订单执行失败",
        level="ERROR",
        data={
            "symbol": "BTCUSDT",
            "order_type": "BUY",
            "price": 49800,
            "quantity": 0.1,
            "error": "余额不足",
            "time": datetime.now().isoformat()
        }
    )
    
    logger.info("发送严重错误级别警报...")
    alert_manager.send_alert(
        alert_type="系统错误",
        message="交易系统连接中断",
        level="CRITICAL",
        data={
            "error_code": "CONNECTION_LOST",
            "details": "与交易所API的连接意外断开",
            "time": datetime.now().isoformat()
        }
    )
    
    # 查看警报文件
    alert_dir = os.path.join(ROOT_DIR, 'alerts')
    logger.info(f"警报已保存到目录: {alert_dir}")
    
    # 列出生成的警报文件
    alert_files = os.listdir(alert_dir)
    logger.info(f"生成的警报文件: {alert_files}")
    
    logger.info("示例程序结束")

if __name__ == "__main__":
    main() 