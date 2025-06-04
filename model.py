import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, SparseCategoricalAccuracy
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, List, Tuple, Optional, Union, Any
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
import psutil
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

# 设置TensorFlow日志级别
tf.get_logger().setLevel('WARNING')

# 设置内存增长
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU内存增长设置失败: {e}")

# 资源监控类
class ResourceMonitor:
    def __init__(self, interval=1):
        self.interval = interval
        self.running = False
        self.stats = []
        
    def start(self):
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()
        
    def stop(self):
        self.running = False
        self.monitor_thread.join()
        
    def _monitor(self):
        while self.running:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            try:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                gpu_memory_used = gpu_memory['current'] / (1024 ** 3)  # 转换为GB
            except:
                gpu_memory_used = 0
                
            self.stats.append({
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'gpu_memory_used': gpu_memory_used
            })
            time.sleep(self.interval)
    
    def get_stats(self):
        return self.stats


class CryptoLSTMModel:
    """
    加密货币预测LSTM模型
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        初始化模型
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # 提取模型配置
        self.model_config = self.config['model']
        
        # 获取系统资源配置
        self.system_config = self.config.get('system_resources', {})
        
        # GPU配置
        self.gpu_config = self.system_config.get('gpu', {})
        if self.gpu_config.get('enabled', True):
            # 设置GPU内存增长
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # 设置可见的GPU设备
                    device_ids = self.gpu_config.get('device_ids', [0])
                    visible_gpus = [gpus[i] for i in device_ids if i < len(gpus)]
                    tf.config.set_visible_devices(visible_gpus, 'GPU')
                    
                    # 启用混合精度训练
                    if self.gpu_config.get('mixed_precision', True):
                        tf.keras.mixed_precision.set_global_policy('mixed_float16')
                    
                    # 启用XLA加速
                    if self.gpu_config.get('xla_acceleration', True):
                        tf.config.optimizer.set_jit(True)
                except RuntimeError as e:
                    print(f"GPU配置失败: {str(e)}")
        
        # 批处理配置
        self.batch_config = self.system_config.get('batch_processing', {})
        if self.batch_config.get('auto_tune_batch_size', True):
            self.batch_size = self.batch_config.get('max_batch_size', 128)
        else:
            self.batch_size = self.model_config.get('batch_size', 64)
        
        # 其他模型参数
        self.lstm_units = self.model_config.get('lstm_units', [256, 128])
        self.dropout = self.model_config.get('dropout', 0.3)
        self.recurrent_dropout = self.model_config.get('recurrent_dropout', 0.2)
        self.epochs = self.model_config.get('epochs', 150)
        self.early_stopping_patience = self.model_config.get('early_stopping_patience', 15)
        self.learning_rate = self.model_config.get('learning_rate', 0.001)
        self.optimizer = self.model_config.get('optimizer', 'adam')
        
        # 数据加载配置
        self.memory_config = self.system_config.get('memory', {})
        self.data_loading_batch = self.memory_config.get('data_loading_batch', 10000)
        self.prefetch_buffer_size = self.memory_config.get('prefetch_buffer_size', 5)
        
        # 模型存储
        self.model = None
        self.history = None
    
    def build_model(self, input_shape: Tuple[int, int], output_dim: int) -> Sequential:
        """
        构建LSTM模型
        
        参数:
            input_shape: 输入形状 (时间步, 特征数)
            output_dim: 输出维度
            
        返回:
            构建好的模型
        """
        # 打印输入和输出维度，帮助调试
        print(f"构建模型 - 输入形状: {input_shape}, 输出维度: {output_dim}")
        
        model = Sequential()
        
        # 添加输入批归一化层
        model.add(BatchNormalization(input_shape=input_shape))
        
        # 添加第一个LSTM层 - 使用更少的单元和更强的正则化
        model.add(LSTM(
            64,  # 减少单元数量
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.3,
            kernel_constraint=tf.keras.constraints.MaxNorm(3),
            recurrent_constraint=tf.keras.constraints.MaxNorm(3),
            bias_constraint=tf.keras.constraints.MaxNorm(3),
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            activity_regularizer=tf.keras.regularizers.l1(0.001),  # 添加L1活动正则化
            kernel_initializer='glorot_uniform',  # 使用Xavier初始化
            recurrent_initializer='orthogonal'  # 使用正交初始化提高稳定性
        ))
        
        # 添加批量归一化
        model.add(BatchNormalization())
        
        # 添加第二个LSTM层
        model.add(LSTM(
            32,  # 减少单元数量
            return_sequences=False,
            dropout=0.3,
            recurrent_dropout=0.3,
            kernel_constraint=tf.keras.constraints.MaxNorm(3),
            recurrent_constraint=tf.keras.constraints.MaxNorm(3),
            bias_constraint=tf.keras.constraints.MaxNorm(3),
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            activity_regularizer=tf.keras.regularizers.l1(0.001),  # 添加L1活动正则化
            kernel_initializer='glorot_uniform',  # 使用Xavier初始化
            recurrent_initializer='orthogonal'  # 使用正交初始化提高稳定性
        ))
        
        # 添加批量归一化
        model.add(BatchNormalization())
        
        # 添加Dropout
        model.add(Dropout(0.4))
        
        # 添加一个更小的Dense层
        model.add(Dense(32, 
                       activation='relu',
                       kernel_constraint=tf.keras.constraints.MaxNorm(3),
                       kernel_regularizer=tf.keras.regularizers.l2(0.001),
                       kernel_initializer='he_uniform'))  # He初始化适合ReLU
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        
        # 添加输出层
        model.add(Dense(output_dim, 
                       activation='softmax',
                       kernel_constraint=tf.keras.constraints.MaxNorm(3),
                       kernel_initializer='glorot_uniform'))
        
        # 编译模型
        # 使用更小的学习率和更强的梯度裁剪
        optimizer = Adam(
            learning_rate=0.0005,  # 降低学习率
            clipnorm=1.0,          # 更强的梯度裁剪
            clipvalue=0.5,         # 更强的值裁剪
            epsilon=1e-7,          # 增加数值稳定性
            beta_1=0.9,            # 默认值
            beta_2=0.999           # 默认值
        )
        
        # 使用分类交叉熵损失函数
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 打印模型摘要
        model.summary()
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              checkpoint_dir: str = 'checkpoints',
              resume: bool = False) -> Dict[str, List[float]]:
        """
        训练模型
        
        参数:
            X_train: 训练数据特征
            y_train: 训练数据标签
            X_val: 验证数据特征
            y_val: 验证数据标签
            checkpoint_dir: 检查点保存目录
            resume: 是否从上次中断处恢复训练
            
        返回:
            训练历史记录
        """
        # 创建数据加载管道
        def create_dataset(X, y, is_training=True):
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            if is_training:
                dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(self.prefetch_buffer_size)
            return dataset
        
        # 自动调整批处理大小
        if self.batch_config.get('auto_tune_batch_size', True):
            print("启用自动批处理大小调整...")
            try:
                # 尝试不同的批处理大小
                test_batch_sizes = []
                batch_size = self.batch_config.get('min_batch_size', 16)
                max_batch_size = self.batch_config.get('max_batch_size', 128)
                while batch_size <= max_batch_size:
                    test_batch_sizes.append(batch_size)
                    batch_size *= 2
                
                # 测试每个批处理大小
                optimal_batch_size = self.batch_size
                min_time = float('inf')
                
                for batch_size in test_batch_sizes:
                    print(f"\n测试批处理大小: {batch_size}")
                    # 创建小数据集进行测试
                    test_size = min(1000, len(X_train))
                    test_dataset = create_dataset(X_train[:test_size], y_train[:test_size])
                    
                    # 测量训练时间
                    start_time = time.time()
                    for _ in range(5):  # 运行几个批次来测试性能
                        for batch in test_dataset:
                            self.model.train_on_batch(batch[0], batch[1])
                    end_time = time.time()
                    
                    batch_time = (end_time - start_time) / 5
                    print(f"每批次平均时间: {batch_time:.4f}秒")
                    
                    if batch_time < min_time:
                        min_time = batch_time
                        optimal_batch_size = batch_size
                
                print(f"\n选择最优批处理大小: {optimal_batch_size}")
                self.batch_size = optimal_batch_size
            except Exception as e:
                print(f"自动调整批处理大小失败: {str(e)}")
                print("使用默认批处理大小继续")
        
        # 创建训练和验证数据集
        train_dataset = create_dataset(X_train, y_train, True)
        if X_val is not None and y_val is not None:
            val_dataset = create_dataset(X_val, y_val, False)
        else:
            val_dataset = None
        if self.model is None:
            raise ValueError("模型未构建，请先调用build_model方法")
        
        # 创建模型保存目录
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 检查点文件路径
        model_state_file = os.path.join(checkpoint_dir, 'model_state.keras')
        training_state_file = os.path.join(checkpoint_dir, 'training_state.json')
        
        # 初始化训练状态
        initial_epoch = 0
        
        # 如果需要恢复训练
        if resume and os.path.exists(training_state_file):
            try:
                # 加载训练状态
                with open(training_state_file, 'r') as f:
                    training_state = json.load(f)
                
                initial_epoch = training_state.get('epoch', 0)
                print(f"从第{initial_epoch}轮恢复训练")
                
                # 如果存在历史记录，加载它
                if 'history' in training_state:
                    self.history = training_state['history']
                    print("已加载训练历史记录")
                
            except Exception as e:
                print(f"加载训练状态失败: {str(e)}")
                print("将从头开始训练")
                initial_epoch = 0
        
        # 检查输入数据是否包含NaN或无穷值
        print("检查训练数据是否包含NaN或无穷值...")
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            print("警告：训练特征中包含NaN或无穷值，正在替换为0...")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(y_train).any() or np.isinf(y_train).any():
            print("警告：训练标签中包含NaN或无穷值，正在替换为0...")
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        if X_val is not None and (np.isnan(X_val).any() or np.isinf(X_val).any()):
            print("警告：验证特征中包含NaN或无穷值，正在替换为0...")
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        
        if y_val is not None and (np.isnan(y_val).any() or np.isinf(y_val).any()):
            print("警告：验证标签中包含NaN或无穷值，正在替换为0...")
            y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 设置早停
        early_stopping = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=25,  # 增加耐心值
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001  # 最小改善阈值
        )
        
        # 设置学习率调度器
        def lr_schedule(epoch, lr):
            # 学习率预热阶段 - 缓慢增加学习率
            if epoch < 10:
                return self.learning_rate * ((epoch + 1) / 10)
            # 学习率阶段性衰减
            elif epoch < 50:
                return self.learning_rate
            elif epoch < 100:
                return self.learning_rate * 0.5
            elif epoch < 150:
                return self.learning_rate * 0.25
            else:
                return self.learning_rate * 0.1
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
        
        # 设置检查点
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(models_dir, f'model_checkpoint_{timestamp}.keras')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            verbose=1
        )
        
        # 添加模型状态检查点 - 每个epoch保存一次
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_state_file,
            save_weights_only=True,
            save_freq='epoch',
            verbose=0
        )
        
        # 添加训练状态保存回调
        class SaveTrainingStateCallback(tf.keras.callbacks.Callback):
            def __init__(self, checkpoint_file, history=None):
                super().__init__()
                self.checkpoint_file = checkpoint_file
                self.history = history or {}
                
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                
                # 更新历史记录
                for key, value in logs.items():
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(float(value))
                
                # 保存训练状态
                training_state = {
                    'epoch': epoch + 1,  # 下一个要训练的epoch
                    'history': self.history,
                    'timestamp': datetime.now().isoformat()
                }
                
                # 保存到文件
                with open(self.checkpoint_file, 'w') as f:
                    json.dump(training_state, f)
        
        # 创建训练状态保存回调
        save_state_callback = SaveTrainingStateCallback(
            training_state_file, 
            history=self.history
        )
        
        # 添加TensorBoard回调
        log_dir = os.path.join('logs', f'run_{timestamp}')
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        
        # 添加NaN检测回调
        class TerminateOnNaN(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.nan_count = 0
                self.max_nan_count = 10  # 允许更多次数的NaN
            
            def on_batch_end(self, batch, logs=None):
                logs = logs or {}
                loss = logs.get('loss')
                if loss is not None and (np.isnan(loss) or np.isinf(loss)):
                    self.nan_count += 1
                    print(f'批次 {batch}: 检测到无效的损失值 ({self.nan_count}/{self.max_nan_count})')
                    print(f'最后一批次的日志: {logs}')
                    
                    # 尝试检查权重
                    try:
                        for layer in self.model.layers:
                            weights = layer.get_weights()
                            for w in weights:
                                if np.isnan(w).any() or np.isinf(w).any():
                                    print(f"层 {layer.name} 包含NaN或Inf权重")
                    except Exception as e:
                        print(f"检查权重时出错: {str(e)}")
                    
                    if self.nan_count >= self.max_nan_count:
                        print(f'连续{self.max_nan_count}次检测到NaN，终止训练。')
                        self.model.stop_training = True
                else:
                    # 重置计数器
                    self.nan_count = 0
        
        nan_callback = TerminateOnNaN()
        
        # 添加梯度值范围检查回调
        class CheckGradients(tf.keras.callbacks.Callback):
            def on_batch_begin(self, batch, logs=None):
                # 在每个批次开始时重置梯度
                self.model.optimizer.iterations.assign_add(0)
        
        gradient_callback = CheckGradients()
        
        # 添加提前停止训练的回调，如果验证损失为NaN
        class StopOnNaNValidationLoss(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                val_loss = logs.get('val_loss')
                if val_loss is not None and (np.isnan(val_loss) or np.isinf(val_loss)):
                    print(f'验证损失为NaN，停止训练。')
                    self.model.stop_training = True
                    
                    # 尝试恢复到上一个有效状态
                    try:
                        print("尝试恢复到上一个有效状态...")
                        last_checkpoint = checkpoint_path
                        if os.path.exists(last_checkpoint):
                            self.model.load_weights(last_checkpoint)
                            print(f"已恢复到上一个有效检查点: {last_checkpoint}")
                    except Exception as e:
                        print(f"恢复失败: {str(e)}")
        
        val_nan_callback = StopOnNaNValidationLoss()
        
        # 添加学习率减少回调
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
        
        callbacks = [
            early_stopping, 
            lr_scheduler, 
            checkpoint, 
            tensorboard_callback, 
            nan_callback, 
            gradient_callback, 
            val_nan_callback,
            reduce_lr,
            model_checkpoint,
            save_state_callback
        ]
        
        # 打印训练信息
        print(f"\n训练模型：")
        print(f"- 训练集形状: X={X_train.shape}, y={y_train.shape}")
        if X_val is not None:
            print(f"- 验证集形状: X={X_val.shape}, y={y_val.shape}")
        print(f"- 批处理大小: {self.batch_size}")
        print(f"- 训练轮数: {self.epochs}")
        print(f"- 学习率: {self.learning_rate} (带预热和调度)")
        print(f"- 早停耐心值: 25")
        print(f"- 起始轮次: {initial_epoch}")
        print("\n开始训练...")
        
        # 使用进度条
        try:
            # 创建类来捕获TensorFlow进度条输出并格式化
            class TqdmCallback(tf.keras.callbacks.Callback):
                def __init__(self):
                    super().__init__()
                    self.epochs = 0
                    self.progbar = None
                    self.epoch_progbar = None
                    
                def on_train_begin(self, logs=None):
                    try:
                        from tqdm.auto import tqdm
                        self.progbar = tqdm(total=self.params['epochs'], desc="总进度", position=0)
                        self.epoch_progbar = tqdm(total=self.params['steps'], desc="本轮进度", position=1, leave=False)
                    except ImportError:
                        print("要显示进度条，请安装tqdm库: pip install tqdm")
                        self.progbar = None
                        self.epoch_progbar = None
                    
                def on_epoch_begin(self, epoch, logs=None):
                    if self.epoch_progbar is not None:
                        self.epoch_progbar.reset()
                        self.epoch_progbar.set_description(f"第{epoch+1}轮")
                    
                def on_epoch_end(self, epoch, logs=None):
                    if self.progbar is not None:
                        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                        self.progbar.set_postfix_str(metrics_str)
                        self.progbar.update(1)
                    
                def on_train_batch_end(self, batch, logs=None):
                    if self.epoch_progbar is not None:
                        self.epoch_progbar.update(1)
                        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                        self.epoch_progbar.set_postfix_str(metrics_str)
                    
                def on_train_end(self, logs=None):
                    if self.progbar is not None:
                        self.progbar.close()
                    if self.epoch_progbar is not None:
                        self.epoch_progbar.close()
            
            tqdm_callback = TqdmCallback()
            callbacks.append(tqdm_callback)
        except Exception as e:
            print(f"设置进度条时出错: {str(e)}")
        
        try:
            # 训练模型
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                callbacks=callbacks,
                verbose=1,
                shuffle=True,  # 确保每个epoch都打乱数据
                use_multiprocessing=True,  # 使用多进程加速
                workers=4,  # 工作进程数
                initial_epoch=initial_epoch  # 从指定的epoch开始训练
            )
            
            # 保存最终模型
            final_model_path = os.path.join(models_dir, f'model_{timestamp}_final.keras')
            self.model.save(final_model_path)
            print(f"\n模型训练完成，已保存到: {final_model_path}")
            
            self.history = history.history
            
            # 打印最终性能
            if X_val is not None and y_val is not None:
                val_loss = min(history.history['val_loss'])
                print(f"最佳验证损失: {val_loss:.4f}")
                
                if 'val_accuracy' in history.history:
                    val_acc = max(history.history['val_accuracy'])
                    print(f"最佳验证准确率: {val_acc:.4f}")
            
            # 绘制训练历史
            self._plot_history()
            
            # 清理临时检查点文件
            try:
                if os.path.exists(model_state_file):
                    os.remove(model_state_file)
                if os.path.exists(training_state_file):
                    os.remove(training_state_file)
            except Exception as e:
                print(f"清理临时检查点文件时出错: {str(e)}")
            
            return self.history
            
        except KeyboardInterrupt:
            print("\n训练被用户中断！")
            print("模型状态和训练进度已保存，可以稍后恢复训练。")
            
            # 保存当前模型
            interrupted_model_path = os.path.join(models_dir, f'model_{timestamp}_interrupted.keras')
            self.model.save(interrupted_model_path)
            print(f"已保存中断时的模型到: {interrupted_model_path}")
            
            return self.history if hasattr(self, 'history') else {}
            
        except Exception as e:
            print(f"训练过程中出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # 尝试保存当前模型
            try:
                error_model_path = os.path.join(models_dir, f'model_{timestamp}_error.keras')
                self.model.save(error_model_path)
                print(f"已保存错误时的模型到: {error_model_path}")
            except:
                print("无法保存错误时的模型")
            
            return self.history if hasattr(self, 'history') else {}
    
    def _compute_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        计算类权重以处理不平衡数据
        
        参数:
            y_train: 训练标签数据
            
        返回:
            类权重字典
        """
        # 对于多标签问题，我们为每个标签计算类权重
        class_weights = {}
        
        # 将预测转换为单一标签序列
        y_flat = y_train.flatten()
        
        # 计算唯一的类
        unique_classes = np.unique(y_flat)
        
        # 计算类权重
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_flat
        )
        
        # 创建类权重字典
        for i, cls in enumerate(unique_classes):
            class_weights[int(cls)] = weights[i]
        
        return class_weights
    
    def _plot_history(self):
        """绘制训练历史"""
        if self.history is None:
            print("没有训练历史可绘制")
            return
        
        # 创建图表目录
        plot_dir = 'plots'
        os.makedirs(plot_dir, exist_ok=True)
        
        # 获取时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 绘制损失
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='训练损失')
        if 'val_loss' in self.history:
            plt.plot(self.history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.ylabel('损失')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        
        # 绘制准确率
        plt.subplot(1, 2, 2)
        plt.plot(self.history['accuracy'], label='训练准确率')
        if 'val_accuracy' in self.history:
            plt.plot(self.history['val_accuracy'], label='验证准确率')
        plt.title('模型准确率')
        plt.ylabel('准确率')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'training_history_{timestamp}.png'))
        plt.close()
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用模型进行预测
        
        参数:
            X: 输入特征，形状(样本数, 时间步, 特征数)
            
        返回:
            预测结果元组: (原始预测, 解释后的预测)
        """
        if self.model is None:
            raise ValueError("模型未加载，请先加载模型")
        
        # 打印预测信息
        print(f"进行预测，输入形状: {X.shape}")
        
        # 进行预测
        try:
            predictions = self.model.predict(X, verbose=1)
            print(f"预测完成，输出形状: {predictions.shape}")
            
            # 解释多分类输出
            # 对于每个品种，从one-hot编码中提取标签
            n_samples = predictions.shape[0]
            n_symbols = predictions.shape[1] // 3
            
            # 创建更容易解释的预测结果
            interpreted_predictions = np.zeros((n_samples, n_symbols))
            
            # 创建预测概率数组，用于存储每个类别的概率
            prediction_probabilities = np.zeros((n_samples, n_symbols, 3))
            
            for i in range(n_samples):
                for j in range(n_symbols):
                    # 获取当前品种的3个类别概率
                    class_probs = predictions[i, j*3:(j+1)*3]
                    
                    # 存储预测概率
                    prediction_probabilities[i, j] = class_probs
                    
                    # 获取最高概率的类别索引
                    max_class = np.argmax(class_probs)
                    
                    # 转换回-1, 0, 1标签
                    interpreted_predictions[i, j] = max_class - 1
            
            # 计算预测置信度
            confidence = np.max(prediction_probabilities, axis=2)
            avg_confidence = np.mean(confidence)
            
            print("预测结果已解释为-1(跌)/0(平)/1(涨)标签")
            print(f"平均预测置信度: {avg_confidence:.4f}")
            
            # 计算各类别的分布
            class_counts = {
                "-1 (跌)": np.sum(interpreted_predictions == -1),
                "0 (平)": np.sum(interpreted_predictions == 0),
                "1 (涨)": np.sum(interpreted_predictions == 1)
            }
            total = sum(class_counts.values())
            
            print("预测类别分布:")
            for cls, count in class_counts.items():
                percentage = (count / total) * 100
                print(f"  {cls}: {count} ({percentage:.2f}%)")
            
            return predictions, interpreted_predictions
        except Exception as e:
            print(f"预测过程中出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        参数:
            X: 特征数据，形状 (样本数, 时间步, 特征数)
            y: 标签数据，形状 (样本数, 标签数)
            
        返回:
            评估指标字典
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train方法")
        
        # 评估模型
        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        
        # 进行预测
        y_pred = self.model.predict(X)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # 计算每个类别的准确率
        class_accuracies = {}
        for cls in np.unique(y):
            mask = y == cls
            if np.sum(mask) > 0:
                class_accuracies[f'class_{int(cls)}_accuracy'] = np.mean(y_pred_classes[mask] == cls)
        
        # 返回评估指标
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            **class_accuracies
        }
        
        return metrics
    
    def save(self, path: str):
        """
        保存模型
        
        参数:
            path: 保存路径
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train方法")
        
        self.model.save(path)
    
    def load(self, path: str):
        """
        加载模型
        
        参数:
            path: 模型路径
        """
        self.model = load_model(path)


# 使用示例
if __name__ == "__main__":
    # 模拟训练数据
    X_train = np.random.random((1000, 48, 100))  # 1000个样本，48个时间步，100个特征
    y_train = np.random.randint(0, 3, (1000, 10))  # 1000个样本，10个品种，每个品种3个类别
    X_val = np.random.random((200, 48, 100))
    y_val = np.random.randint(0, 3, (200, 10))
    
    # 创建模型
    model = CryptoLSTMModel('config.yaml')
    
    # 构建模型
    model.build_model((48, 100), 30)  # 30 = 10个品种 * 3个类别
    
    # 打印模型摘要
    model.model.summary()
    
    # 训练模型
    history = model.train(X_train, y_train, X_val, y_val)
    
    # 评估模型
    metrics = model.evaluate(X_val, y_val)
    print("评估指标:", metrics)