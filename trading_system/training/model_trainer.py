 #!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 添加项目根目录到sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from common.utils import load_config, TradingLogger


class ModelTrainer:
    """模型训练器类，负责模型的训练和保存"""
    
    def __init__(self, config_path='config/trading_config.yaml'):
        """初始化模型训练器"""
        self.config = load_config(config_path)
        self.logger = TradingLogger('model_trainer', log_level=self.config.get('general', {}).get('log_level', 'INFO'))
        
        # 模型存储目录
        self.models_dir = os.path.join(ROOT_DIR, 'models', 'saved_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 模型元数据文件
        self.metadata_file = os.path.join(self.models_dir, 'models_metadata.json')
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f)
        
        # 训练配置
        training_config = self.config.get('training', {})
        self.batch_size = training_config.get('batch_size', 32)
        self.epochs = training_config.get('epochs', 100)
        self.validation_split = training_config.get('validation_split', 0.2)
        self.early_stopping_patience = training_config.get('early_stopping_patience', 10)
        self.learning_rate = training_config.get('learning_rate', 0.001)
        
        # 数据处理配置
        self.sequence_length = training_config.get('sequence_length', 60)
        self.prediction_horizon = training_config.get('prediction_horizon', 30)  # 预测未来15小时（30个半小时）
        self.threshold = training_config.get('threshold', 0.01)  # 涨跌幅阈值，默认1%
    
    def prepare_data(self, symbols, start_date=None, end_date=None):
        """准备训练数据
        
        Args:
            symbols: 交易对列表
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'
            
        Returns:
            X_train, X_val, y_train, y_val: 训练集和验证集
        """
        self.logger.info(f"准备训练数据: 交易对={symbols}, 开始日期={start_date}, 结束日期={end_date}")
        
        # 加载数据
        data_frames = []
        for symbol in symbols:
            # 从CSV文件加载数据
            csv_path = os.path.join(ROOT_DIR, 'data', f'{symbol}.csv')
            if not os.path.exists(csv_path):
                self.logger.warning(f"数据文件不存在: {csv_path}")
                continue
            
            df = pd.read_csv(csv_path)
            
            # 确保时间戳列是日期时间格式
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            # 筛选日期范围
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # 添加交易对标识
            df['symbol'] = symbol
            
            # 计算技术指标
            df = self._calculate_technical_indicators(df)
            
            # 生成标签
            df = self._generate_labels(df)
            
            # 添加到数据列表
            data_frames.append(df)
        
        # 合并所有数据
        if not data_frames:
            self.logger.error("没有有效的数据")
            return None, None, None, None
        
        all_data = pd.concat(data_frames)
        all_data.sort_index(inplace=True)
        
        # 删除包含NaN的行
        all_data.dropna(inplace=True)
        
        # 准备特征和标签
        features = all_data.drop(['label', 'symbol'], axis=1)
        labels = all_data['label']
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 创建序列数据
        X, y = self._create_sequences(features_scaled, labels.values)
        
        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, shuffle=False
        )
        
        self.logger.info(f"数据准备完成: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        
        return X_train, X_val, y_train, y_val, scaler
    
    def _calculate_technical_indicators(self, df):
        """计算技术指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含技术指标的DataFrame
        """
        # 计算移动平均线
        df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['EMA144'] = df['close'].ewm(span=144, adjust=False).mean()
        df['EMA169'] = df['close'].ewm(span=169, adjust=False).mean()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # 计算价格变化率
        df['price_change'] = df['close'].pct_change()
        df['price_change_1d'] = df['close'].pct_change(periods=48)  # 1天（48个半小时）
        df['price_change_3d'] = df['close'].pct_change(periods=144)  # 3天（144个半小时）
        df['price_change_7d'] = df['close'].pct_change(periods=336)  # 7天（336个半小时）
        
        # 计算成交量变化率
        df['volume_change'] = df['volume'].pct_change()
        df['volume_change_1d'] = df['volume'].pct_change(periods=48)
        
        # 计算波动率
        df['volatility'] = df['close'].rolling(window=48).std() / df['close'].rolling(window=48).mean()
        
        return df
    
    def _generate_labels(self, df):
        """生成标签
        
        Args:
            df: 包含价格数据的DataFrame
            
        Returns:
            包含标签的DataFrame
        """
        # 计算未来价格变化率
        future_returns = df['close'].shift(-self.prediction_horizon) / df['close'] - 1
        
        # 根据阈值生成标签
        df['label'] = 0  # 默认为平
        df.loc[future_returns > self.threshold, 'label'] = 1  # 涨
        df.loc[future_returns < -self.threshold, 'label'] = -1  # 跌
        
        return df
    
    def _create_sequences(self, features, labels):
        """创建序列数据
        
        Args:
            features: 特征数组
            labels: 标签数组
            
        Returns:
            X, y: 序列特征和标签
        """
        X, y = [], []
        for i in range(len(features) - self.sequence_length - self.prediction_horizon + 1):
            X.append(features[i:i+self.sequence_length])
            y.append(labels[i+self.sequence_length+self.prediction_horizon-1])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape, num_classes=3):
        """构建LSTM模型
        
        Args:
            input_shape: 输入形状，如(sequence_length, num_features)
            num_classes: 类别数量，默认为3（涨/平/跌）
            
        Returns:
            构建好的模型
        """
        self.logger.info(f"构建LSTM模型: input_shape={input_shape}, num_classes={num_classes}")
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.logger.info(f"模型构建完成")
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, model_type='lstm'):
        """训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            model_type: 模型类型，默认为'lstm'
            
        Returns:
            训练好的模型和训练历史
        """
        self.logger.info(f"开始训练模型: model_type={model_type}, batch_size={self.batch_size}, epochs={self.epochs}")
        
        # 确定输入形状
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # 构建模型
        if model_type.lower() == 'lstm':
            model = self.build_lstm_model(input_shape)
        else:
            self.logger.error(f"不支持的模型类型: {model_type}")
            return None, None
        
        # 回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.early_stopping_patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, 'temp_model.keras'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # 训练模型
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start_time
        
        self.logger.info(f"模型训练完成: 耗时={training_time:.2f}秒")
        
        return model, history
    
    def evaluate_model(self, model, X_val, y_val):
        """评估模型
        
        Args:
            model: 训练好的模型
            X_val: 验证特征
            y_val: 验证标签
            
        Returns:
            评估指标字典
        """
        self.logger.info("评估模型性能")
        
        # 预测验证集
        y_pred_proba = model.predict(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 将标签转换为0,1,2（原始标签为-1,0,1）
        y_val_transformed = y_val + 1
        
        # 计算评估指标
        accuracy = accuracy_score(y_val_transformed, y_pred)
        precision = precision_score(y_val_transformed, y_pred, average='weighted')
        recall = recall_score(y_val_transformed, y_pred, average='weighted')
        f1 = f1_score(y_val_transformed, y_pred, average='weighted')
        cm = confusion_matrix(y_val_transformed, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
        
        self.logger.info(f"评估指标: accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
        
        return metrics
    
    def save_model(self, model, symbols, model_type='lstm', metrics=None):
        """保存模型
        
        Args:
            model: 训练好的模型
            symbols: 交易对列表
            model_type: 模型类型
            metrics: 评估指标
            
        Returns:
            模型ID和保存路径
        """
        # 生成模型ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbols_str = '_'.join(symbols)
        model_id = f"{model_type}_{symbols_str}_{timestamp}"
        
        # 模型文件名
        model_filename = f"{model_id}.keras"
        model_path = os.path.join(self.models_dir, model_filename)
        
        # 保存模型
        model.save(model_path)
        
        # 计算文件哈希值
        file_hash = self._calculate_file_hash(model_path)
        
        # 更新元数据
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
        
        metadata[model_id] = {
            'filename': model_filename,
            'name': model_id,
            'version': '1.0',
            'type': model_type.upper(),
            'symbols': symbols,
            'created_at': datetime.now().isoformat(),
            'file_hash': file_hash,
            'metrics': metrics
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"模型保存成功: id={model_id}, path={model_path}")
        
        return model_id, model_path
    
    def _calculate_file_hash(self, file_path):
        """计算文件的SHA256哈希值"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


if __name__ == "__main__":
    # 示例用法
    trainer = ModelTrainer()
    
    # 准备数据
    X_train, X_val, y_train, y_val, scaler = trainer.prepare_data(
        symbols=['BTC', 'ETH'],
        start_date='2022-01-01',
        end_date='2022-12-31'
    )
    
    # 训练模型
    model, history = trainer.train_model(X_train, y_train, X_val, y_val)
    
    # 评估模型
    metrics = trainer.evaluate_model(model, X_val, y_val)
    
    # 保存模型
    model_id, model_path = trainer.save_model(model, ['BTC', 'ETH'], metrics=metrics)