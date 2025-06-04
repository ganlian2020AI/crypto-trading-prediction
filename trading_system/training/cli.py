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


def train_model(args):
    """训练模型"""
    from trading_system.training.model_trainer import ModelTrainer
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建日志
    logger = TradingLogger('model_trainer', log_level=config.get('general', {}).get('log_level', 'INFO'))
    
    # 创建训练器
    trainer = ModelTrainer(config, logger)
    
    # 训练模型
    trainer.train(
        symbols=args.symbols.split(',') if args.symbols else None,
        start_date=args.start_date,
        end_date=args.end_date,
        model_type=args.model_type,
        output_dir=args.output
    )
    
    logger.info("模型训练完成")


def evaluate_model(args):
    """评估模型"""
    from trading_system.training.model_evaluator import ModelEvaluator
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建日志
    logger = TradingLogger('model_evaluator', log_level=config.get('general', {}).get('log_level', 'INFO'))
    
    # 创建评估器
    evaluator = ModelEvaluator(config, logger)
    
    # 评估模型
    results = evaluator.evaluate(
        model_path=args.model,
        symbols=args.symbols.split(',') if args.symbols else None,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # 打印评估结果
    for symbol, metrics in results.items():
        logger.info(f"{symbol} 评估结果:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value}")
    
    logger.info("模型评估完成")


def optimize_model(args):
    """优化模型超参数"""
    from trading_system.training.model_optimizer import ModelOptimizer
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建日志
    logger = TradingLogger('model_optimizer', log_level=config.get('general', {}).get('log_level', 'INFO'))
    
    # 创建优化器
    optimizer = ModelOptimizer(config, logger)
    
    # 优化模型
    best_params = optimizer.optimize(
        symbols=args.symbols.split(',') if args.symbols else None,
        start_date=args.start_date,
        end_date=args.end_date,
        model_type=args.model_type,
        n_trials=args.trials
    )
    
    # 打印最佳参数
    logger.info("最佳超参数:")
    for param_name, param_value in best_params.items():
        logger.info(f"  {param_name}: {param_value}")
    
    logger.info("模型优化完成")


def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description='加密货币交易系统 - 模型训练工具')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', type=str, default='config/trading_config.yaml', help='配置文件路径')
    train_parser.add_argument('--symbols', type=str, default=None, help='交易对符号，用逗号分隔')
    train_parser.add_argument('--start-date', type=str, default=None, help='开始日期 (YYYY-MM-DD)')
    train_parser.add_argument('--end-date', type=str, default=None, help='结束日期 (YYYY-MM-DD)')
    train_parser.add_argument('--model-type', type=str, default='lstm', choices=['lstm', 'gru', 'cnn', 'transformer'], help='模型类型')
    train_parser.add_argument('--output', type=str, default='models', help='模型输出目录')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--config', type=str, default='config/trading_config.yaml', help='配置文件路径')
    eval_parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    eval_parser.add_argument('--symbols', type=str, default=None, help='交易对符号，用逗号分隔')
    eval_parser.add_argument('--start-date', type=str, default=None, help='开始日期 (YYYY-MM-DD)')
    eval_parser.add_argument('--end-date', type=str, default=None, help='结束日期 (YYYY-MM-DD)')
    
    # 优化命令
    optimize_parser = subparsers.add_parser('optimize', help='优化模型超参数')
    optimize_parser.add_argument('--config', type=str, default='config/trading_config.yaml', help='配置文件路径')
    optimize_parser.add_argument('--symbols', type=str, default=None, help='交易对符号，用逗号分隔')
    optimize_parser.add_argument('--start-date', type=str, default=None, help='开始日期 (YYYY-MM-DD)')
    optimize_parser.add_argument('--end-date', type=str, default=None, help='结束日期 (YYYY-MM-DD)')
    optimize_parser.add_argument('--model-type', type=str, default='lstm', choices=['lstm', 'gru', 'cnn', 'transformer'], help='模型类型')
    optimize_parser.add_argument('--trials', type=int, default=100, help='优化试验次数')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'optimize':
        optimize_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 