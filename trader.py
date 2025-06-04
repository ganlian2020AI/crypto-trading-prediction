import os
import pandas as pd
import numpy as np
import yaml
import json
import time
import requests
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import tensorflow as tf
from model import CryptoLSTMModel
from data_processor import DataProcessor
from fibonacci_position_manager import FibonacciPositionManager


class CryptoTrader:
    """
    加密货币交易模块：负责信号生成和发送
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        初始化交易模块
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # 提取交易配置
        self.trading_config = self.config['trading']
        
        # 设置日志
        self.setup_logging()
        
        # 初始化数据处理器
        self.data_processor = DataProcessor(config_path)
        
        # 初始化资金管理器
        self.position_manager = FibonacciPositionManager(
            base_capital=self.config['capital_management'].get('base_capital', 10000.0)
        )
        
        # 初始化模型
        self.model = None
        self.model_loaded = False
        
        # 当前仓位
        self.current_positions = {}
    
    def setup_logging(self):
        """设置日志"""
        log_config = self.config.get('logging', {})
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
        
        self.logger = logging.getLogger('crypto_trader')
    
    def load_model(self, model_path: str):
        """
        加载预训练模型
        
        参数:
            model_path: 模型路径
        """
        try:
            self.model = CryptoLSTMModel(config_path='config.yaml')
            self.model.load(model_path)
            self.model_loaded = True
            self.logger.info(f"模型已加载: {model_path}")
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            self.model_loaded = False
    
    def update_data(self, symbols: List[str] = None):
        """
        更新数据
        
        参数:
            symbols: 交易品种列表，如果为None则使用配置中的所有品种
        """
        if symbols is None:
            symbols = self.config['data']['symbols']
        
        try:
            # 加载数据
            self.data_processor.load_data(symbols)
            
            # 预处理数据
            self.data_processor.preprocess_data()
            
            # 对齐多品种数据
            self.data_processor.align_data()
            
            self.logger.info(f"数据已更新: {symbols}")
        except Exception as e:
            self.logger.error(f"更新数据失败: {str(e)}")
    
    def generate_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        生成交易信号
        
        返回:
            交易信号字典 {symbol: signal_data}
        """
        if not self.model_loaded:
            self.logger.error("模型未加载，无法生成信号")
            return {}
        
        try:
            # 准备模型输入数据
            X, _, _, _ = self.data_processor.prepare_model_data()
            
            # 使用模型进行预测
            predictions = self.model.predict(X)
            
            # 创建信号字典
            signals = {}
            
            # 为每个品种生成信号
            for i, symbol in enumerate(self.config['data']['symbols']):
                # 获取最新数据
                latest_data = self.data_processor.aligned_data.iloc[-1]
                
                # 获取最新预测
                latest_prediction = predictions[-1, i]
                
                # 获取价格数据
                price = latest_data.get(f"{symbol}_close", 0)
                
                # 获取斐波那契水平
                fib_level = latest_data.get(f"{symbol}_highbi", "6U")
                
                # 获取当前仓位
                current_position = self.current_positions.get(symbol, 0.0)
                
                # 生成信号（-1/0/1）
                signal = np.argmax(latest_prediction) - 1  # 转换为 -1, 0, 1
                
                # 计算置信度
                confidence = latest_prediction[np.argmax(latest_prediction)]
                
                # 获取仓位建议
                position_info = self.position_manager.adjust_position_on_signal(
                    current_position=current_position,
                    fib_level=fib_level,
                    price=price,
                    signal=signal
                )
                
                # 计算止损止盈价格
                stop_loss, take_profit = self.position_manager.get_stop_loss_take_profit(
                    fib_level=fib_level,
                    entry_price=price,
                    is_long=(signal > 0)
                )
                
                # 创建信号数据
                signal_data = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'price': price,
                    'signal': signal,  # -1, 0, 1
                    'confidence': float(confidence),
                    'position_size': float(position_info['adjustment']),
                    'current_position': float(current_position),
                    'target_position': float(position_info['target_position']),
                    'market_state': position_info['market_state'],
                    'stop_loss': float(stop_loss),
                    'take_profit': float(take_profit),
                    'recommendation': position_info['recommendation']
                }
                
                signals[symbol] = signal_data
                
                # 更新当前仓位
                self.current_positions[symbol] = position_info['target_position']
            
            self.logger.info(f"已生成{len(signals)}个交易信号")
            return signals
        
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            return {}
    
    def send_signals(self, signals: Dict[str, Dict[str, Any]]) -> bool:
        """
        发送交易信号
        
        参数:
            signals: 交易信号字典
            
        返回:
            是否成功发送
        """
        if not signals:
            self.logger.warning("没有信号可发送")
            return False
        
        # 获取webhook URL
        webhook_url = self.trading_config.get('webhook_url')
        if not webhook_url:
            self.logger.error("未配置webhook URL")
            return False
        
        try:
            # 准备负载
            payload = {
                'signals': signals,
                'timestamp': datetime.now().isoformat(),
                'source': 'crypto_trading_system'
            }
            
            # 发送请求
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            # 检查响应
            if response.status_code == 200:
                self.logger.info(f"信号发送成功: {len(signals)}个品种")
                return True
            else:
                self.logger.error(f"信号发送失败: HTTP {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            self.logger.error(f"发送信号时出错: {str(e)}")
            return False
    
    def save_signals(self, signals: Dict[str, Dict[str, Any]], directory: str = 'signals'):
        """
        保存交易信号到文件
        
        参数:
            signals: 交易信号字典
            directory: 保存目录
        """
        if not signals:
            return
        
        # 创建目录
        os.makedirs(directory, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(directory, f'signals_{timestamp}.json')
        
        # 保存信号
        try:
            with open(file_path, 'w') as f:
                json.dump(signals, f, indent=2)
            
            self.logger.info(f"信号已保存到: {file_path}")
        except Exception as e:
            self.logger.error(f"保存信号失败: {str(e)}")
    
    def run_trading_cycle(self, model_path: str):
        """
        运行一个完整的交易周期
        
        参数:
            model_path: 模型路径
        """
        # 加载模型
        if not self.model_loaded:
            self.load_model(model_path)
        
        # 更新数据
        self.update_data()
        
        # 生成信号
        signals = self.generate_signals()
        
        # 保存信号
        self.save_signals(signals)
        
        # 发送信号
        if signals:
            success = self.send_signals(signals)
            if success:
                self.logger.info("交易周期完成")
            else:
                self.logger.warning("交易周期完成，但信号发送失败")
        else:
            self.logger.warning("交易周期完成，但没有生成信号")
    
    def start_trading_service(self, model_path: str, interval_minutes: int = 30):
        """
        启动交易服务
        
        参数:
            model_path: 模型路径
            interval_minutes: 交易周期间隔（分钟）
        """
        self.logger.info(f"启动交易服务，间隔: {interval_minutes}分钟")
        
        try:
            while True:
                # 运行一个交易周期
                self.run_trading_cycle(model_path)
                
                # 等待下一个周期
                self.logger.info(f"等待{interval_minutes}分钟后运行下一个周期")
                time.sleep(interval_minutes * 60)
        
        except KeyboardInterrupt:
            self.logger.info("交易服务已手动停止")
        except Exception as e:
            self.logger.error(f"交易服务异常: {str(e)}")
            raise


# 使用示例
if __name__ == "__main__":
    # 创建交易模块
    trader = CryptoTrader('config.yaml')
    
    # 加载模型（假设已训练好的模型路径）
    model_path = 'models/lstm_model_final.h5'
    
    # 启动交易服务（每30分钟一个周期）
    # trader.start_trading_service(model_path, interval_minutes=30)
    
    # 或者运行单个交易周期
    trader.run_trading_cycle(model_path)