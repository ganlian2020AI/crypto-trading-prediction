import os
import argparse
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime
import json
import tensorflow as tf
import psutil
import multiprocessing

def setup_system_resources(config_path: str = 'config.yaml'):
    """
    初始化系统资源配置
    
    参数:
        config_path: 配置文件路径
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # 获取系统资源配置
    system_config = config.get('system_resources', {})
    
    # CPU配置
    cpu_config = system_config.get('cpu', {})
    num_threads = cpu_config.get('num_threads', 'auto')
    if num_threads == 'auto':
        num_threads = multiprocessing.cpu_count()
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    
    # GPU配置
    gpu_config = system_config.get('gpu', {})
    if gpu_config.get('enabled', True):
        # 检查是否有可用的GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # 设置GPU内存增长
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # 设置可见的GPU设备
                device_ids = gpu_config.get('device_ids', [0])
                visible_gpus = [gpus[i] for i in device_ids if i < len(gpus)]
                tf.config.set_visible_devices(visible_gpus, 'GPU')
                
                # 设置GPU内存限制
                memory_limit = gpu_config.get('memory_limit', 0.8)
                for gpu in visible_gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=int(memory_limit * 1024)
                        )]
                    )
                
                # 启用混合精度训练
                if gpu_config.get('mixed_precision', True):
                    tf.keras.mixed_precision.set_global_policy('mixed_float16')
                
                # 启用XLA加速
                if gpu_config.get('xla_acceleration', True):
                    tf.config.optimizer.set_jit(True)
                
                logging.info(f"已配置GPU: 设备={device_ids}, 内存限制={memory_limit*100}%, "
                           f"混合精度={gpu_config.get('mixed_precision', True)}, "
                           f"XLA加速={gpu_config.get('xla_acceleration', True)}")
            except RuntimeError as e:
                logging.error(f"GPU配置失败: {str(e)}")
        else:
            logging.warning("未检测到可用的GPU，将使用CPU进行计算")
    
    # 内存优化配置
    memory_config = system_config.get('memory', {})
    batch_size = memory_config.get('data_loading_batch', 10000)
    prefetch_buffer = memory_config.get('prefetch_buffer_size', 5)
    
    # 设置TensorFlow数据加载优化
    tf.data.experimental.enable_debug_mode()
    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    
    logging.info(f"系统资源配置完成: CPU线程数={num_threads}, 数据加载批次={batch_size}, "
                f"预取缓冲区大小={prefetch_buffer}")
    
    return {
        'batch_size': batch_size,
        'prefetch_buffer': prefetch_buffer,
        'num_threads': num_threads
    }

from data_processor import DataProcessor
from feature_engineering import FeatureEngineer
from model import CryptoLSTMModel
from evaluator import ModelEvaluator
from trader import CryptoTrader
from fibonacci_position_manager import FibonacciPositionManager


def setup_logging(config_path: str = 'config.yaml'):
    """
    设置日志
    
    参数:
        config_path: 配置文件路径
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # 提取日志配置
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_file = log_config.get('file', 'logs/trading_system.log')
    
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('main')


def train_model(config_path: str = 'config.yaml', resume: bool = True):
    """
    训练模型
    
    参数:
        config_path: 配置文件路径
        resume: 是否从上次中断处恢复训练
    """
    logger = logging.getLogger('train_model')
    logger.info("开始训练模型")
    
    # 初始化系统资源
    logger.info("初始化系统资源配置...")
    system_resources = setup_system_resources(config_path)
    logger.info(f"系统资源配置: {system_resources}")
    
    # 创建模型保存目录
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # 创建检查点目录
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 检查是否有可恢复的训练
    model_checkpoint = None
    X_train_file = os.path.join(checkpoint_dir, 'X_train.npy')
    y_train_file = os.path.join(checkpoint_dir, 'y_train.npy')
    X_val_file = os.path.join(checkpoint_dir, 'X_val.npy')
    y_val_file = os.path.join(checkpoint_dir, 'y_val.npy')
    model_state_file = os.path.join(checkpoint_dir, 'model_state.keras')
    
    can_resume = (resume and 
                 os.path.exists(X_train_file) and 
                 os.path.exists(y_train_file) and 
                 os.path.exists(X_val_file) and 
                 os.path.exists(y_val_file) and
                 os.path.exists(model_state_file))
    
    if can_resume:
        logger.info("发现可恢复的训练检查点，尝试恢复训练...")
        try:
            # 加载模型配置
            with open(os.path.join(checkpoint_dir, 'model_config.json'), 'r') as f:
                model_config = json.load(f)
            
            # 创建模型
            model = CryptoLSTMModel(config_path)
            
            # 加载训练数据
            logger.info("加载训练数据...")
            X_train = np.load(X_train_file)
            y_train = np.load(y_train_file)
            X_val = np.load(X_val_file)
            y_val = np.load(y_val_file)
            
            logger.info(f"恢复的训练数据形状: X_train={X_train.shape}, y_train={y_train.shape}")
            logger.info(f"恢复的验证数据形状: X_val={X_val.shape}, y_val={y_val.shape}")
            
            # 构建模型
            input_shape = (X_train.shape[1], X_train.shape[2])
            output_dim = y_train.shape[1]
            logger.info(f"构建模型: 输入形状={input_shape}, 输出维度={output_dim}")
            model.build_model(input_shape, output_dim)
            
            # 加载模型权重
            logger.info("加载模型权重...")
            model.model.load_weights(model_state_file)
            
            # 恢复训练
            logger.info("恢复训练...")
            history = model.train(X_train, y_train, X_val, y_val, resume=True)
            
            logger.info("模型训练完成")
            return True
        except Exception as e:
            logger.error(f"恢复训练失败: {str(e)}")
            logger.info("将从头开始训练")
    
    try:
        # 加载数据
        logger.info("加载和处理数据")
        data_processor = DataProcessor(config_path)
        
        # 尝试直接加载对齐后的数据检查点
        if data_processor._load_checkpoint('aligned_data'):
            logger.info("成功从检查点加载对齐后的数据，跳过原始数据加载和预处理步骤")
        else:
            # 如果没有对齐数据检查点，则正常加载和处理数据
            logger.info("未找到对齐数据检查点，开始正常数据处理流程")
            
            # 加载数据
            logger.info("正在加载原始数据...")
            raw_data = data_processor.load_data()
            if not raw_data:
                logger.error("未能加载任何数据，请检查数据文件和路径")
                return False
            logger.info(f"成功加载了{len(raw_data)}个品种的数据")
                
            # 检查是否有有效数据
            valid_data = [df for df in raw_data.values() if len(df) > 0]
            if not valid_data:
                logger.error("所有加载的数据都为空，请检查时间戳解析和日期过滤")
                return False
            logger.info(f"有效数据品种数量: {len(valid_data)}")
                
            # 处理数据
            logger.info("正在预处理数据并计算特征...")
            processed_data = data_processor.preprocess_data()
            if not processed_data:
                logger.error("数据预处理失败")
                return False
            logger.info("数据预处理和特征计算完成")
            
            # 打印特征列，帮助调试
            for symbol, df in processed_data.items():
                logger.info(f"{symbol}数据包含以下特征列: {df.columns.tolist()[:10]}... (共{len(df.columns)}列)")
                
            # 对齐数据
            logger.info("正在对齐多品种数据...")
            aligned_data = data_processor.align_data()
            if aligned_data.empty:
                logger.error("数据对齐后为空，无法继续训练")
                return False
            logger.info("数据对齐完成")
        
        # 确保aligned_data已加载
        if data_processor.aligned_data is None or data_processor.aligned_data.empty:
            logger.error("对齐后的数据为空或未加载，无法继续训练")
            return False
            
        # 打印对齐后的数据信息
        logger.info(f"对齐后的数据形状: {data_processor.aligned_data.shape}")
        logger.info(f"对齐后的数据列: {data_processor.aligned_data.columns.tolist()[:20]}... (共{len(data_processor.aligned_data.columns)}列)")
        
        # 准备模型数据
        logger.info("正在准备模型数据...")
        try:
            # 使用特征选择来减少内存使用
            max_features_per_symbol = 10  # 每个品种最多使用10个特征
            logger.info(f"为了减少内存使用，每个品种最多使用{max_features_per_symbol}个特征")
            
            X_train, y_train, X_val, y_val = data_processor.prepare_model_data(max_features=max_features_per_symbol)
            logger.info(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
            logger.info(f"验证集形状: X={X_val.shape}, y={y_val.shape}")
            
            # 保存训练数据，以便在中断后恢复
            np.save(X_train_file, X_train)
            np.save(y_train_file, y_train)
            np.save(X_val_file, X_val)
            np.save(y_val_file, y_val)
            logger.info("已保存训练数据检查点")
            
        except Exception as e:
            logger.error(f"准备模型数据失败: {str(e)}")
            return False
        
        # 训练模型
        logger.info("正在构建LSTM模型...")
        model = CryptoLSTMModel(config_path)
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_dim = y_train.shape[1]  # 输出维度直接使用标签维度
        
        # 保存模型配置
        model_config = {
            'input_shape': input_shape,
            'output_dim': output_dim,
            'lstm_units': model.lstm_units,
            'dropout': model.dropout,
            'learning_rate': model.learning_rate
        }
        with open(os.path.join(checkpoint_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f)
        
        logger.info(f"模型输入形状: {input_shape}, 输出维度: {output_dim}")
        model.build_model(input_shape, output_dim)
        
        logger.info("开始训练模型...")
        logger.info("这可能需要较长时间，请耐心等待...")
        history = model.train(X_train, y_train, X_val, y_val, checkpoint_dir=checkpoint_dir)
        logger.info("模型训练完成")
        
        # 评估模型
        logger.info("正在评估模型性能...")
        evaluator = ModelEvaluator(config_path)
        
        # 评估分类性能
        logger.info("生成验证集预测...")
        y_pred, y_pred_interpreted = model.predict(X_val)
        # 将解释后的预测结果与原始标签进行比较
        y_val_original = np.zeros((y_val.shape[0], y_val.shape[1] // 3))
        for i in range(y_val.shape[0]):
            for j in range(y_val.shape[1] // 3):
                # 从one-hot中找到为1的位置
                class_idx = np.argmax(y_val[i, j*3:(j+1)*3])
                # 转换回原始标签（-1,0,1）
                y_val_original[i, j] = class_idx - 1

        logger.info("计算分类性能指标...")
        # 使用解释后的预测结果和原始标签计算指标
        y_pred_flat = y_pred_interpreted.flatten()
        y_val_flat = y_val_original.flatten()
        classification_metrics = evaluator.evaluate_classification(y_val_flat, y_pred_flat)
        for metric, value in classification_metrics.items():
            logger.info(f"{metric}: {value}")
        
        # 生成报告
        logger.info("正在生成评估报告...")
        
        # 创建模拟回测结果（实际应用中需要真实数据）
        dummy_backtest_metrics = {
            'total_return': 0.5,
            'annual_return': 0.3,
            'max_drawdown': -0.2,
            'sharpe_ratio': 1.5,
            'sortino_ratio': 2.0,
            'win_rate': 0.6,
            'profit_loss_ratio': 1.8,
            'total_trades': 100
        }
        
        evaluator.generate_evaluation_report(classification_metrics, dummy_backtest_metrics)
        
        logger.info("模型训练和评估完成")
        
        # 清理临时检查点文件
        try:
            for file in [X_train_file, y_train_file, X_val_file, y_val_file, model_state_file]:
                if os.path.exists(file):
                    os.remove(file)
            logger.info("已清理临时检查点文件")
        except Exception as e:
            logger.warning(f"清理临时文件时出错: {str(e)}")
        
        return True
    
    except Exception as e:
        logger.error(f"模型训练失败: {str(e)}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False


def backtest(config_path: str = 'config.yaml', model_path: str = None, use_simple_backtest: bool = True):
    """
    回测模型
    
    参数:
        config_path: 配置文件路径
        model_path: 模型路径，如果为None则使用最新的模型
        use_simple_backtest: 是否使用简单回测策略
    """
    logger = logging.getLogger('backtest')
    logger.info("开始回测")
    
    try:
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # 确定模型路径
        if model_path is None:
            model_dir = 'models'
            model_files = [f for f in os.listdir(model_dir) if f.endswith('_final.keras')]
            if not model_files:
                logger.error("找不到模型文件")
                return False
            model_path = os.path.join(model_dir, sorted(model_files)[-1])
        
        logger.info(f"使用模型: {model_path}")
        
        # 加载数据
        logger.info("加载和处理数据")
        data_processor = DataProcessor(config_path)
        
        # 尝试直接加载对齐后的数据检查点
        if data_processor._load_checkpoint('aligned_data'):
            logger.info("成功从检查点加载对齐后的数据，跳过原始数据加载和预处理步骤")
        else:
            # 如果没有对齐数据检查点，则正常加载和处理数据
            data_processor.load_data()
            data_processor.preprocess_data()
            data_processor.align_data()
        
        # 获取回测数据
        backtest_data = data_processor.aligned_data
        
        # 加载模型
        logger.info("加载模型")
        model = CryptoLSTMModel(config_path)
        model.load(model_path)
        
        # 准备回测输入
        logger.info("准备回测数据")
        # 使用特征选择来减少内存使用
        max_features_per_symbol = 10  # 每个品种最多使用10个特征
        logger.info(f"为了减少内存使用，每个品种最多使用{max_features_per_symbol}个特征")
        X, _, _, _ = data_processor.prepare_model_data(max_features=max_features_per_symbol)
        
        # 进行预测
        logger.info("生成预测")
        predictions, interpreted_predictions = model.predict(X)
        
        # 进行回测
        logger.info("执行回测")
        evaluator = ModelEvaluator(config_path)
        symbols = config['data']['symbols']
        
        if use_simple_backtest:
            logger.info("使用简单回测策略")
            backtest_results = evaluator.simple_backtest(
                backtest_data,
                interpreted_predictions,
                symbols
            )
            
            # 保存回测结果
            backtest_dir = 'backtest_results'
            os.makedirs(backtest_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存总体回测结果
            result_summary = {
                'initial_capital': backtest_results['initial_total_capital'],
                'final_equity': backtest_results['final_total_equity'],
                'total_return_pct': backtest_results['total_return_pct'],
                'max_drawdown_pct': backtest_results['max_drawdown_pct']
            }
            
            # 保存每个品种的回测结果
            symbol_results = {}
            for symbol, result in backtest_results['symbol_results'].items():
                symbol_results[symbol] = {
                    'initial_capital': result['initial_capital'],
                    'final_equity': result['final_equity'],
                    'total_return_pct': result['total_return_pct'],
                    'max_drawdown_pct': result['max_drawdown_pct'],
                    'total_trades': result['total_trades'],
                    'win_rate': result['win_rate']
                }
            
            # 保存结果为CSV
            result_df = pd.DataFrame(symbol_results).T
            result_df.to_csv(os.path.join(backtest_dir, f'simple_backtest_results_{timestamp}.csv'))
            
            # 保存总体权益曲线
            if 'equity_data' in backtest_results:
                backtest_results['equity_data'].to_csv(os.path.join(backtest_dir, f'simple_backtest_equity_{timestamp}.csv'))
            
            logger.info(f"简单回测完成，结果已保存到 {backtest_dir} 目录")
        else:
            logger.info("使用标准回测策略")
            backtest_results = evaluator.backtest(
                backtest_data,
                interpreted_predictions,
                symbols
            )
            
            # 输出回测指标
            metrics = backtest_results.copy()
            if 'backtest_df' in metrics:
                del metrics['backtest_df']
            
            logger.info(f"回测指标: {metrics}")
            
            # 保存回测结果
            backtest_dir = 'backtest_results'
            os.makedirs(backtest_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if 'backtest_df' in backtest_results:
                backtest_results['backtest_df'].to_csv(os.path.join(backtest_dir, f'backtest_{timestamp}.csv'))
        
        logger.info("回测完成")
        return True
    
    except Exception as e:
        logger.error(f"回测失败: {str(e)}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False


def predict(config_path: str = 'config.yaml', model_path: str = None):
    """
    使用模型进行预测
    
    参数:
        config_path: 配置文件路径
        model_path: 模型路径，如果为None则使用最新的模型
    """
    logger = logging.getLogger('predict')
    logger.info("开始预测")
    
    try:
        # 确定模型路径
        if model_path is None:
            model_dir = 'models'
            model_files = [f for f in os.listdir(model_dir) if f.endswith('_final.keras')]
            if not model_files:
                logger.error("找不到模型文件")
                return False
            model_path = os.path.join(model_dir, sorted(model_files)[-1])
        
        logger.info(f"使用模型: {model_path}")
        
        # 加载数据
        logger.info("加载和处理数据")
        data_processor = DataProcessor(config_path)
        data_processor.load_data()
        data_processor.preprocess_data()
        data_processor.align_data()
        
        # 加载模型
        logger.info("加载模型")
        model = CryptoLSTMModel(config_path)
        model.load(model_path)
        
        # 准备预测数据
        logger.info("准备预测数据")
        # 使用特征选择来减少内存使用
        max_features_per_symbol = 10  # 每个品种最多使用10个特征
        logger.info(f"为了减少内存使用，每个品种最多使用{max_features_per_symbol}个特征")
        X, _, _, _ = data_processor.prepare_model_data(max_features=max_features_per_symbol)
        
        # 进行预测
        logger.info("生成预测")
        predictions, interpreted_predictions = model.predict(X)
        
        # 创建交易模块来生成信号
        logger.info("生成交易信号")
        trader = CryptoTrader(config_path)
        trader.model = model
        trader.model_loaded = True
        
        # 将预测结果同步到交易模块
        trader.data_processor = data_processor
        trader.predictions = interpreted_predictions  # 使用解释后的预测结果
        
        # 生成信号
        signals = trader.generate_signals()
        
        # 保存信号
        trader.save_signals(signals)
        
        logger.info(f"预测完成，生成了{len(signals)}个信号")
        return True
    
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        return False


def deploy(config_path: str = 'config.yaml', model_path: str = None, interval_minutes: int = 30):
    """
    部署交易服务
    
    参数:
        config_path: 配置文件路径
        model_path: 模型路径，如果为None则使用最新的模型
        interval_minutes: 交易周期间隔（分钟）
    """
    logger = logging.getLogger('deploy')
    logger.info("开始部署交易服务")
    
    try:
        # 确定模型路径
        if model_path is None:
            model_dir = 'models'
            model_files = [f for f in os.listdir(model_dir) if f.endswith('_final.keras')]
            if not model_files:
                logger.error("找不到模型文件")
                return False
            model_path = os.path.join(model_dir, sorted(model_files)[-1])
        
        logger.info(f"使用模型: {model_path}")
        
        # 创建交易模块
        trader = CryptoTrader(config_path)
        
        # 启动交易服务
        logger.info(f"启动交易服务，间隔: {interval_minutes}分钟")
        trader.start_trading_service(model_path, interval_minutes)
        
        return True
    
    except Exception as e:
        logger.error(f"部署失败: {str(e)}")
        return False


def main():
    """主函数"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='加密货币交易系统')
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='选择命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    
    # 回测命令
    backtest_parser = subparsers.add_parser('backtest', help='回测模型')
    backtest_parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    backtest_parser.add_argument('--model', default=None, help='模型路径')
    backtest_parser.add_argument('--simple', action='store_true', help='使用简单回测策略')
    backtest_parser.add_argument('--standard', action='store_true', help='使用标准回测策略')
    
    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='使用模型进行预测')
    predict_parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    predict_parser.add_argument('--model', default=None, help='模型路径')
    
    # 部署命令
    deploy_parser = subparsers.add_parser('deploy', help='部署交易服务')
    deploy_parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    deploy_parser.add_argument('--model', default=None, help='模型路径')
    deploy_parser.add_argument('--interval', type=int, default=30, help='交易周期间隔（分钟）')
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.config if hasattr(args, 'config') else 'config.yaml')
    
    # 根据命令执行相应的功能
    if args.command == 'train':
        logger.info("执行训练命令")
        train_model(args.config)
    elif args.command == 'backtest':
        logger.info("执行回测命令")
        use_simple_backtest = not args.standard  # 如果没有指定--standard，则使用简单回测
        backtest(args.config, args.model, use_simple_backtest)
    elif args.command == 'predict':
        logger.info("执行预测命令")
        predict(args.config, args.model)
    elif args.command == 'deploy':
        logger.info("执行部署命令")
        deploy(args.config, args.model, args.interval)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()