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
from backtest.backtest_engine import BacktestEngine


def run_backtest(args):
    """运行回测"""
    # 加载配置
    config = load_config(args.config)
    
    # 创建日志
    logger = TradingLogger('backtest', log_level=config.get('general', {}).get('log_level', 'INFO'))
    
    # 创建回测引擎
    engine = BacktestEngine(args.config)
    
    # 解析交易对
    symbols = args.symbols.split(',') if args.symbols else None
    
    # 加载数据
    logger.info(f"加载数据: 交易对={symbols}, 开始日期={args.start_date}, 结束日期={args.end_date}")
    engine.load_data(symbols, args.start_date, args.end_date)
    
    # 加载模型预测
    if args.model:
        logger.info(f"加载模型预测: {args.model}")
        engine.load_model_predictions(args.model)
    
    # 运行回测
    logger.info(f"开始回测: 初始资金={args.capital}")
    results = engine.run_backtest(args.capital, symbols, args.start_date, args.end_date)
    
    if results:
        # 打印性能指标
        logger.info("===== 回测结果 =====")
        logger.info(f"初始资金: {results['performance_metrics']['initial_equity']:.2f}")
        logger.info(f"最终权益: {results['performance_metrics']['final_equity']:.2f}")
        logger.info(f"总收益率: {results['performance_metrics']['total_return_pct']:.2f}%")
        logger.info(f"年化收益率: {results['performance_metrics']['annual_return_pct']:.2f}%")
        logger.info(f"最大回撤: {results['performance_metrics']['max_drawdown_pct']:.2f}%")
        logger.info(f"夏普比率: {results['performance_metrics']['sharpe_ratio']:.2f}")
        logger.info(f"胜率: {results['performance_metrics']['win_rate']:.2f}")
        logger.info(f"盈亏比: {results['performance_metrics']['profit_loss_ratio']:.2f}")
        logger.info(f"总交易次数: {results['performance_metrics']['total_trades']}")
        logger.info("=====================")
        
        # 生成报告
        if args.report:
            report_path = engine.generate_report()
            logger.info(f"报告已生成: {report_path}")
        
        # 生成权益曲线图
        if args.plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(ROOT_DIR, 'reports', f"equity_curve_{timestamp}.png")
            engine.plot_equity_curve(show=False, save_path=plot_path)
            logger.info(f"权益曲线已保存到: {plot_path}")
            
            if args.show:
                engine.plot_equity_curve(show=True)
    else:
        logger.error("回测失败")


def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description='加密货币交易系统 - 回测工具')
    
    parser.add_argument('--config', type=str, default='config/trading_config.yaml', help='配置文件路径')
    parser.add_argument('--symbols', type=str, default=None, help='交易对符号，用逗号分隔')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='auto', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000, help='初始资金')
    parser.add_argument('--model', type=str, default=None, help='模型文件路径')
    parser.add_argument('--report', action='store_true', help='生成详细报告')
    parser.add_argument('--plot', action='store_true', help='生成权益曲线图')
    parser.add_argument('--show', action='store_true', help='显示权益曲线图')
    
    args = parser.parse_args()
    
    run_backtest(args)


if __name__ == "__main__":
    main() 