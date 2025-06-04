import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import yaml
from scipy import interpolate
from feature_engineering import FeatureEngineer
import concurrent.futures
import time
import pickle
import hashlib
import json
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import logging
from threading import Lock

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建线程锁用于内存监控
memory_lock = Lock()


class DataProcessor:
    """
    数据处理模块：负责加载数据、预处理、标准化和特征工程
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        初始化数据处理器
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # 提取数据配置
        self.data_config = self.config['data']
        self.feature_config = self.config['features']
        self.system_config = self.config.get('system_resources', {})
        
        # 获取系统资源配置
        self.cpu_config = self.system_config.get('cpu', {})
        self.memory_config = self.system_config.get('memory', {})
        self.batch_config = self.system_config.get('batch_processing', {})
        self.parallel_config = self.system_config.get('parallel_processing', {})
        
        # 设置并行处理
        self.num_workers = self.cpu_config.get('num_threads', 'auto')
        if self.num_workers == 'auto':
            self.num_workers = os.cpu_count()
        
        # 设置内存限制
        self.memory_limit = self.memory_config.get('limit_gb', 24.0) * 1024 * 1024 * 1024  # 转换为字节
        self.data_loading_batch = self.memory_config.get('data_loading_batch', 10000)
        self.prefetch_buffer_size = self.memory_config.get('prefetch_buffer_size', 5)
        
        # 设置批处理
        self.batch_size = self.batch_config.get('max_batch_size', 128)
        self.min_batch_size = self.batch_config.get('min_batch_size', 16)
        
        # 数据存储
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.processed_data: Dict[str, pd.DataFrame] = {}
        self.aligned_data: Optional[pd.DataFrame] = None
        
        # 特征标准化参数
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}
        
        # 创建特征工程器
        self.feature_engineer = FeatureEngineer(config_path)
        
        # 检查点配置
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_interval = 1800  # 每30分钟保存一次检查点
        self.last_checkpoint_time = time.time()
        
    def _generate_checkpoint_id(self) -> str:
        """生成检查点ID，基于配置和数据状态"""
        # 使用配置文件的哈希值作为检查点ID的一部分
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # 使用时间戳作为另一部分
        timestamp = datetime.now().strftime("%Y%m%d")
        
        return f"{timestamp}_{config_hash}"
    
    def _save_checkpoint(self, stage: str, data: dict = None) -> str:
        """
        保存检查点
        
        参数:
            stage: 处理阶段名称
            data: 要保存的额外数据
            
        返回:
            检查点文件路径
        """
        checkpoint_id = self._generate_checkpoint_id()
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}_{stage}.pkl")
        
        # 准备要保存的数据
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'config': self.config,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds
        }
        
        # 根据阶段保存不同的数据
        if stage == 'raw_data':
            # 只保存文件名和形状信息，而不是完整数据
            checkpoint_data['raw_data_info'] = {
                symbol: {'shape': df.shape, 'columns': df.columns.tolist()} 
                for symbol, df in self.raw_data.items()
            }
            # 单独保存每个原始数据文件
            for symbol, df in self.raw_data.items():
                symbol_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}_raw_{symbol}.pkl")
                df.to_pickle(symbol_file)
                
        elif stage == 'processed_data':
            # 只保存文件名和形状信息
            checkpoint_data['processed_data_info'] = {
                symbol: {'shape': df.shape, 'columns': df.columns.tolist()} 
                for symbol, df in self.processed_data.items()
            }
            # 单独保存每个处理后的数据文件
            for symbol, df in self.processed_data.items():
                symbol_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}_processed_{symbol}.pkl")
                df.to_pickle(symbol_file)
                
        elif stage == 'aligned_data':
            if self.aligned_data is not None:
                # 保存对齐数据的形状信息
                checkpoint_data['aligned_data_info'] = {
                    'shape': self.aligned_data.shape,
                    'columns': self.aligned_data.columns.tolist()
                }
                # 单独保存对齐数据
                aligned_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}_aligned.pkl")
                self.aligned_data.to_pickle(aligned_file)
        
        # 添加额外数据
        if data:
            checkpoint_data.update(data)
        
        # 保存检查点元数据
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"✓ 保存{stage}阶段检查点: {checkpoint_file}")
        self.last_checkpoint_time = time.time()
        
        return checkpoint_file
    
    def _load_checkpoint(self, stage: str) -> bool:
        """
        加载最新的检查点
        
        参数:
            stage: 处理阶段名称
            
        返回:
            是否成功加载
        """
        # 查找最新的检查点
        checkpoint_id = self._generate_checkpoint_id()
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}_{stage}.pkl")
        
        if not os.path.exists(checkpoint_file):
            print(f"未找到{stage}阶段的检查点")
            return False
        
        try:
            # 加载检查点元数据
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # 恢复配置和标准化参数
            self.feature_means = checkpoint_data['feature_means']
            self.feature_stds = checkpoint_data['feature_stds']
            
            # 根据阶段加载不同的数据
            if stage == 'raw_data':
                self.raw_data = {}
                for symbol in checkpoint_data['raw_data_info']:
                    symbol_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}_raw_{symbol}.pkl")
                    if os.path.exists(symbol_file):
                        self.raw_data[symbol] = pd.read_pickle(symbol_file)
                
            elif stage == 'processed_data':
                self.processed_data = {}
                for symbol in checkpoint_data['processed_data_info']:
                    symbol_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}_processed_{symbol}.pkl")
                    if os.path.exists(symbol_file):
                        self.processed_data[symbol] = pd.read_pickle(symbol_file)
                
            elif stage == 'aligned_data':
                aligned_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}_aligned.pkl")
                if os.path.exists(aligned_file):
                    self.aligned_data = pd.read_pickle(aligned_file)
            
            print(f"✓ 成功加载{stage}阶段检查点")
            return True
            
        except Exception as e:
            print(f"加载{stage}阶段检查点失败: {str(e)}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            return False
    
    def _should_save_checkpoint(self) -> bool:
        """检查是否应该保存检查点"""
        return (time.time() - self.last_checkpoint_time) >= self.checkpoint_interval
    
    def _monitor_memory_usage(self):
        """监控内存使用情况，如果超过限制则触发垃圾回收"""
        with memory_lock:
            current_memory = psutil.Process().memory_info().rss
            memory_percent = current_memory / self.memory_limit
            
            if memory_percent > 0.9:  # 如果内存使用超过90%
                logger.warning(f"内存使用率高: {memory_percent:.2%}, 触发垃圾回收")
                gc.collect()
                return True
            return False
    
    def load_data(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        加载CSV数据文件，使用并行处理和内存优化
        
        参数:
            symbols: 需要加载的交易品种列表，如果为None则使用配置中的所有品种
            
        返回:
            加载的原始数据字典 {symbol: dataframe}
        """
        def load_symbol_data(symbol: str) -> Tuple[str, Optional[pd.DataFrame]]:
            """加载单个品种的数据"""
            try:
                file_path = os.path.join(input_dir, f"{symbol}.csv")
                if not os.path.exists(file_path):
                    logger.warning(f"找不到{symbol}的数据文件: {file_path}")
                    return symbol, None

                # 使用分块读取来控制内存使用
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=self.data_loading_batch):
                    chunks.append(chunk)
                    
                    # 检查内存使用
                    with memory_lock:
                        current_memory = psutil.Process().memory_info().rss
                        if current_memory > self.memory_limit * 0.9:
                            logger.warning(f"{symbol}数据加载达到内存限制的90%，尝试合并当前块")
                            # 合并当前块并清理内存
                            temp_df = pd.concat(chunks, axis=0)
                            chunks = [temp_df]
                            gc.collect()

                # 合并所有块
                df = pd.concat(chunks, axis=0)
                
                # 标准化列名
                df.columns = [col.lower() for col in df.columns]
                
                # 确保必要的列存在
                required_cols = ['timestamp', 'name', 'high', 'low', 'open', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(f"{symbol}数据缺少必要列: {missing_cols}")
                    return symbol, None
                
                # 转换时间戳
                if 'timestamp' in df.columns:
                    df = self._process_timestamp(df, symbol)
                    if df is None or len(df) == 0:
                        return symbol, None
                
                # 设置时间戳为索引并排序
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # 过滤日期范围
                if 'start_date' in self.data_config and 'end_date' in self.data_config:
                    df = self._filter_date_range(df, symbol)
                
                return symbol, df
                
            except Exception as e:
                logger.error(f"加载{symbol}数据时出错: {str(e)}")
                return symbol, None
        # 尝试加载检查点
        if self._load_checkpoint('raw_data'):
            print("已从检查点恢复原始数据")
            return self.raw_data
        
        if symbols is None:
            symbols = self.data_config['symbols']
        
        input_dir = self.data_config['input_dir']
        
        # 检查目录是否存在
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"数据目录不存在: {input_dir}")
        
        start_time = time.time()
        print(f"开始加载{len(symbols)}个品种的数据...")
        
        # 加载每个品种的CSV文件
        for symbol in symbols:
            file_path = os.path.join(input_dir, f"{symbol}.csv")
            if not os.path.exists(file_path):
                print(f"警告: 找不到{symbol}的数据文件: {file_path}")
                continue
            
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 标准化列名
                df.columns = [col.lower() for col in df.columns]
                
                # 确保必要的列存在
                required_cols = ['timestamp', 'name', 'high', 'low', 'open', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"警告: {symbol}数据缺少必要列: {missing_cols}")
                    continue
                
                # 转换时间戳为datetime
                if 'timestamp' in df.columns:
                    try:
                        # 首先尝试移除列值中的双引号
                        df['timestamp'] = df['timestamp'].str.replace('"', '')
                        
                        # 尝试多种常见格式解析时间戳
                        formats_to_try = [
                            '%m/%d/%Y %H:%M:%S',  # 1/1/2024 08:00:00
                            '%d/%m/%Y %H:%M:%S',  # 日/月/年
                            '%Y-%m-%d %H:%M:%S',  # ISO格式
                            '%Y/%m/%d %H:%M:%S',  # 年/月/日
                            '%m-%d-%Y %H:%M:%S',  # 月-日-年
                            '%d-%m-%Y %H:%M:%S',  # 日-月-年
                        ]
                        
                        success = False
                        for fmt in formats_to_try:
                            try:
                                df['timestamp'] = pd.to_datetime(df['timestamp'], format=fmt, errors='raise')
                                print(f"成功使用格式 '{fmt}' 解析 {symbol} 的时间戳")
                                success = True
                                break
                            except Exception:
                                continue
                        
                        # 如果所有格式都失败，尝试让pandas自动推断
                        if not success:
                            print(f"尝试让pandas自动推断 {symbol} 的时间戳格式")
                            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                        
                        # 删除仍然是NaT的行
                        na_count = df['timestamp'].isna().sum()
                        if na_count > 0:
                            print(f"警告: {symbol}数据中有{na_count}行时间戳无法解析，将被删除")
                            df = df.dropna(subset=['timestamp'])
                        
                        # 检查是否还有数据
                        if len(df) == 0:
                            print(f"警告: {symbol}的所有时间戳都无法解析")
                            continue
                        
                    except Exception as e:
                        print(f"警告: 无法解析{symbol}的时间戳: {str(e)}")
                        print(f"时间戳样例: {df['timestamp'].iloc[:5].tolist()}")
                        continue
                
                # 设置时间戳为索引
                df.set_index('timestamp', inplace=True)
                
                # 按时间排序
                df.sort_index(inplace=True)
                
                # 过滤日期范围
                if 'start_date' in self.data_config and 'end_date' in self.data_config:
                    try:
                        start_date = pd.to_datetime(self.data_config['start_date'])
                        end_date = pd.to_datetime(self.data_config['end_date'])
                        # 确保df.index是datetime类型
                        if not pd.api.types.is_datetime64_any_dtype(df.index):
                            print(f"警告: {symbol}的索引不是日期时间类型，跳过日期过滤")
                        else:
                            df = df[(df.index >= start_date) & (df.index <= end_date)]
                            print(f"日期过滤后{symbol}数据: {len(df)}行")
                    except Exception as e:
                        print(f"日期过滤{symbol}数据时出错: {str(e)}")
                
                # 存储到原始数据字典
                self.raw_data[symbol] = df
                print(f"成功加载{symbol}数据: {len(df)}行")
                
                # 定期保存检查点
                if self._should_save_checkpoint():
                    self._save_checkpoint('raw_data')
                
            except Exception as e:
                print(f"加载{symbol}数据时出错: {str(e)}")
        
        end_time = time.time()
        print(f"所有品种数据加载完成，总耗时: {end_time - start_time:.2f}秒")
        
        # 保存最终检查点
        self._save_checkpoint('raw_data')
        
        return self.raw_data
    
    def preprocess_data(self) -> Dict[str, pd.DataFrame]:
        """
        对加载的数据进行预处理：处理缺失值、异常值等
        
        返回:
            处理后的数据字典 {symbol: dataframe}
        """
        # 尝试加载检查点
        if self._load_checkpoint('processed_data'):
            print("已从检查点恢复处理后的数据")
            return self.processed_data
            
        if not self.raw_data:
            # 尝试加载原始数据检查点
            if not self._load_checkpoint('raw_data'):
                raise ValueError("未加载数据，请先调用load_data方法")
        
        # 获取预处理配置
        missing_value_method = self.feature_config.get('missing_value_method', 'interpolation')
        clip_outliers = self.feature_config.get('clip_outliers', True)
        outlier_threshold = self.feature_config.get('outlier_threshold', 3.0)
        
        start_time = time.time()
        print(f"开始预处理{len(self.raw_data)}个品种的数据...")
        
        # 定义单个品种的预处理函数
        def process_single_symbol(symbol_data):
            symbol, df = symbol_data
            try:
                print(f"开始处理{symbol}数据...")
                # 创建副本
                processed_df = df.copy()
                
                # 检查并处理缺失值
                missing_values = processed_df.isnull().sum()
                if missing_values.sum() > 0:
                    print(f"{symbol}数据中有缺失值: {missing_values[missing_values > 0].to_dict()}")
                    
                    # 根据配置选择缺失值处理方法
                    if missing_value_method == 'interpolation':
                        # 插值法
                        processed_df = processed_df.interpolate(method='time')
                    elif missing_value_method == 'ffill':
                        # 前向填充
                        processed_df = processed_df.fillna(method='ffill')
                    elif missing_value_method == 'bfill':
                        # 后向填充
                        processed_df = processed_df.fillna(method='bfill')
                    elif missing_value_method == 'median':
                        # 使用中位数填充数值列
                        for col in processed_df.columns:
                            if np.issubdtype(processed_df[col].dtype, np.number):
                                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                            else:
                                processed_df[col] = processed_df[col].ffill().bfill()
                    
                    # 处理开头和结尾可能的缺失值
                    processed_df = processed_df.ffill().bfill()
                
                # 检查并处理异常值
                if clip_outliers:
                    for col in ['high', 'low', 'open', 'close', 'volume']:
                        if col in processed_df.columns:
                            # 使用稳健的分位数方法处理异常值
                            q_low = processed_df[col].quantile(0.01)
                            q_high = processed_df[col].quantile(0.99)
                            
                            # 统计异常值数量
                            outliers = ((processed_df[col] < q_low) | (processed_df[col] > q_high)).sum()
                            if outliers > 0:
                                print(f"{symbol}的{col}列中有{outliers}个异常值")
                                
                                # 裁剪异常值
                                processed_df[col] = processed_df[col].clip(q_low, q_high)
                
                # 使用特征工程器计算技术指标和自定义指标
                try:
                    print(f"为{symbol}计算技术指标和自定义指标...")
                    symbol_start = time.time()
                    processed_df = self.feature_engineer.calculate_all_features(processed_df)
                    symbol_end = time.time()
                    print(f"{symbol}特征计算完成，添加了{len(processed_df.columns) - len(df.columns)}个特征，耗时: {symbol_end - symbol_start:.2f}秒")
                    return (symbol, processed_df)
                except Exception as e:
                    print(f"为{symbol}计算特征时出错: {str(e)}")
                    import traceback
                    print(f"详细错误信息: {traceback.format_exc()}")
                    return (symbol, df)  # 返回原始数据
            except Exception as e:
                print(f"处理{symbol}时出错: {str(e)}")
                return (symbol, df)  # 返回原始数据
        
        # 使用线程池并行处理多个品种
        processed_results = {}
        max_workers = min(8, len(self.raw_data))  # 最多8个线程，或者品种数量
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_symbol = {executor.submit(process_single_symbol, (symbol, df)): symbol 
                                   for symbol, df in self.raw_data.items()}
                
                # 获取结果
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result_symbol, result_df = future.result()
                        processed_results[result_symbol] = result_df
                        completed_count += 1
                        print(f"✓ {result_symbol}处理完成 ({completed_count}/{len(self.raw_data)})")
                        
                        # 定期保存检查点
                        if self._should_save_checkpoint():
                            # 临时保存当前进度
                            temp_processed_data = self.processed_data.copy()
                            temp_processed_data.update(processed_results)
                            self.processed_data = temp_processed_data
                            self._save_checkpoint('processed_data')
                            
                    except Exception as e:
                        print(f"× 获取{symbol}处理结果时出错: {str(e)}")
                        # 如果处理失败，使用原始数据
                        processed_results[symbol] = self.raw_data[symbol]
        except Exception as e:
            print(f"多线程处理失败，回退到单线程: {str(e)}")
            # 回退到单线程处理
            for symbol, df in self.raw_data.items():
                symbol, processed_df = process_single_symbol((symbol, df))
                processed_results[symbol] = processed_df
                
                # 定期保存检查点
                if self._should_save_checkpoint():
                    # 临时保存当前进度
                    temp_processed_data = self.processed_data.copy()
                    temp_processed_data.update({symbol: processed_df})
                    self.processed_data = temp_processed_data
                    self._save_checkpoint('processed_data')
        
        # 保存处理后的数据
        self.processed_data = processed_results
        
        end_time = time.time()
        print(f"所有品种数据预处理完成，总耗时: {end_time - start_time:.2f}秒")
        
        # 保存最终检查点
        self._save_checkpoint('processed_data')
        
        return self.processed_data
    
    def align_data(self) -> pd.DataFrame:
        """
        将不同品种的数据对齐到相同的时间戳
        
        返回:
            对齐后的多品种合并数据框
        """
        # 尝试加载检查点
        if self._load_checkpoint('aligned_data'):
            print("已从检查点恢复对齐后的数据")
            return self.aligned_data
            
        if not self.processed_data:
            # 尝试加载处理后数据检查点
            if not self._load_checkpoint('processed_data'):
                raise ValueError("未处理数据，请先调用preprocess_data方法")
        
        start_time = time.time()
        print("开始对齐多品种数据...")
        
        # 检查是否至少有一个非空的数据帧
        if all(len(df) == 0 for df in self.processed_data.values()):
            print("警告: 所有品种的数据都为空，无法进行对齐")
            self.aligned_data = pd.DataFrame()
            return self.aligned_data
        
        # 获取所有品种的所有时间戳
        all_timestamps = set()
        for df in self.processed_data.values():
            if len(df) > 0:  # 只处理非空数据
                # 确保索引是datetime类型
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    print(f"警告: 发现非日期时间类型的索引，尝试转换为datetime")
                    try:
                        # 尝试转换索引为datetime
                        df.index = pd.to_datetime(df.index)
                    except Exception as e:
                        print(f"转换索引失败: {str(e)}")
                        continue
                
                # 添加到时间戳集合
                all_timestamps.update(df.index)
        
        # 检查是否有时间戳
        if not all_timestamps:
            print("警告: 没有有效的时间戳，无法进行对齐")
            self.aligned_data = pd.DataFrame()
            return self.aligned_data
        
        # 检查时间戳类型是否一致
        timestamp_types = set(type(ts) for ts in all_timestamps)
        if len(timestamp_types) > 1:
            print(f"警告: 发现多种时间戳类型: {timestamp_types}，尝试统一转换为pandas Timestamp")
            # 统一转换为pandas Timestamp
            converted_timestamps = set()
            for ts in all_timestamps:
                try:
                    if not isinstance(ts, pd.Timestamp):
                        converted_timestamps.add(pd.Timestamp(ts))
                    else:
                        converted_timestamps.add(ts)
                except Exception as e:
                    print(f"转换时间戳 {ts} (类型: {type(ts)}) 失败: {str(e)}")
            all_timestamps = converted_timestamps
        
        # 对齐到相同的时间戳
        aligned_dfs = []
        for symbol, df in self.processed_data.items():
            # 为所有时间戳创建空的DataFrame
            try:
                aligned_df = pd.DataFrame(index=sorted(all_timestamps))
                
                # 对每列进行重采样和插值
                for col in df.columns:
                    if col != 'name':  # 跳过非数值列
                        # 合并原始数据
                        aligned_df[f"{symbol}_{col}"] = df[col]
                        
                        # 使用时间插值填充缺失值
                        if aligned_df[f"{symbol}_{col}"].isnull().any():
                            aligned_df[f"{symbol}_{col}"] = aligned_df[f"{symbol}_{col}"].interpolate(method='time')
                
                aligned_dfs.append(aligned_df)
                print(f"✓ {symbol}数据对齐完成，形状: {aligned_df.shape}")
            except Exception as e:
                print(f"× {symbol}数据对齐失败: {str(e)}")
                import traceback
                print(f"详细错误信息: {traceback.format_exc()}")
        
        if not aligned_dfs:
            print("警告: 所有品种的数据对齐都失败了")
            self.aligned_data = pd.DataFrame()
            return self.aligned_data
        
        # 合并所有对齐后的数据框
        try:
            self.aligned_data = pd.concat(aligned_dfs, axis=1)
            
            # 处理任何剩余的缺失值
            self.aligned_data = self.aligned_data.ffill().bfill()
            
            print(f"所有数据对齐完成，最终形状: {self.aligned_data.shape}")
            
            # 保存检查点
            self._save_checkpoint('aligned_data')
            
            end_time = time.time()
            print(f"数据对齐完成，耗时: {end_time - start_time:.2f}秒")
            
            return self.aligned_data
        except Exception as e:
            print(f"合并对齐数据失败: {str(e)}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            self.aligned_data = pd.DataFrame()
            return self.aligned_data
    
    def normalize_features(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        特征标准化/归一化
        
        参数:
            data: 要标准化的数据，如果为None则使用aligned_data
            
        返回:
            标准化后的数据
        """
        if data is None:
            if self.aligned_data is None:
                raise ValueError("未对齐数据，请先调用align_data方法")
            data = self.aligned_data
        
        # 获取标准化方法
        norm_method = self.feature_config.get('normalization', 'z-score')
        
        # 创建副本避免修改原始数据
        normalized_data = data.copy()
        
        # 首先检查并处理NaN和无穷值
        print("检查数据中的NaN和无穷值...")
        has_nan = normalized_data.isna().any().any()
        has_inf = np.isinf(normalized_data.select_dtypes(include=[np.number])).any().any()
        
        if has_nan:
            print("警告: 数据中存在NaN值，使用更稳健的方法填充")
            # 首先使用中位数填充NaN（比均值更稳健）
            for col in normalized_data.columns:
                if normalized_data[col].isna().any():
                    # 使用中位数填充NaN（数值列）
                    if np.issubdtype(normalized_data[col].dtype, np.number):
                        median_val = normalized_data[col].median()
                        if pd.isna(median_val):  # 如果中位数也是NaN
                            # 尝试使用非NaN值的中位数
                            non_nan_vals = normalized_data[col].dropna()
                            if len(non_nan_vals) > 0:
                                median_val = non_nan_vals.median()
                            else:
                                median_val = 0  # 如果全是NaN，使用0
                        print(f"列 {col} 使用中位数 {median_val} 填充NaN")
                        normalized_data[col] = normalized_data[col].fillna(median_val)
                    else:
                        # 非数值列使用前向填充
                        normalized_data[col] = normalized_data[col].fillna(method='ffill').fillna(method='bfill')
        
        if has_inf:
            print("警告: 数据中存在无穷值，使用更稳健的方法替换")
            # 处理无穷值
            for col in normalized_data.select_dtypes(include=[np.number]).columns:
                # 获取列的有限值
                finite_vals = normalized_data[col][np.isfinite(normalized_data[col])]
                if len(finite_vals) > 0:
                    # 使用分位数替换无穷值（比最大/最小值更稳健）
                    q_low = finite_vals.quantile(0.01)  # 1%分位数
                    q_high = finite_vals.quantile(0.99)  # 99%分位数
                    
                    # 替换正无穷为高分位数
                    normalized_data[col] = normalized_data[col].replace(np.inf, q_high)
                    # 替换负无穷为低分位数
                    normalized_data[col] = normalized_data[col].replace(-np.inf, q_low)
                    
                    print(f"列 {col} 的无穷值已替换为分位数: 正无穷 -> {q_high}, 负无穷 -> {q_low}")
                else:
                    # 如果没有有限值，替换为0
                    normalized_data[col] = normalized_data[col].replace([np.inf, -np.inf], 0)
                    print(f"列 {col} 的无穷值已替换为0（无有限值）")
        
        # 检查是否还有NaN或无穷值
        if normalized_data.isna().any().any() or np.isinf(normalized_data.select_dtypes(include=[np.number])).any().any():
            print("警告: 仍有NaN或无穷值无法处理，替换为0")
            normalized_data = normalized_data.fillna(0).replace([np.inf, -np.inf], 0)
        
        # 对每一列进行标准化
        print("开始标准化特征...")
        for col in normalized_data.columns:
            # 跳过非数值列
            if not np.issubdtype(normalized_data[col].dtype, np.number):
                continue
            
            # 使用稳健的标准化方法
            if norm_method == 'z-score':
                # 使用中位数和四分位距进行稳健标准化
                median = normalized_data[col].median()
                q1 = normalized_data[col].quantile(0.25)
                q3 = normalized_data[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    normalized_data[col] = (normalized_data[col] - median) / iqr
                    # 存储标准化参数
                    self.feature_means[col] = median
                    self.feature_stds[col] = iqr
                    print(f"列 {col} 使用稳健Z-score标准化: 中位数={median:.4f}, IQR={iqr:.4f}")
                else:
                    # 如果IQR为0，使用MAD（中位数绝对偏差）
                    mad = (normalized_data[col] - median).abs().median() * 1.4826  # 常数使MAD与标准差一致
                    if mad > 0:
                        normalized_data[col] = (normalized_data[col] - median) / mad
                        # 存储标准化参数
                        self.feature_means[col] = median
                        self.feature_stds[col] = mad
                        print(f"列 {col} 使用MAD标准化: 中位数={median:.4f}, MAD={mad:.4f}")
                    else:
                        # 如果MAD也为0，不进行标准化
                        self.feature_means[col] = 0
                        self.feature_stds[col] = 1
                        print(f"列 {col} 无变异性，不进行标准化")
            
            elif norm_method == 'min-max':
                # 使用稳健的分位数进行归一化
                min_val = normalized_data[col].quantile(0.01)  # 使用1%分位数
                max_val = normalized_data[col].quantile(0.99)  # 使用99%分位数
                if max_val > min_val:
                    normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
                    # 裁剪到[0,1]范围
                    normalized_data[col] = normalized_data[col].clip(0, 1)
                    # 存储归一化参数
                    self.feature_means[col] = min_val
                    self.feature_stds[col] = max_val - min_val
                    print(f"列 {col} 使用稳健Min-Max归一化: Min(1%)={min_val:.4f}, Max(99%)={max_val:.4f}")
                else:
                    # 如果范围为0，不进行归一化
                    self.feature_means[col] = 0
                    self.feature_stds[col] = 1
                    print(f"列 {col} 无有效范围，不进行归一化")
        
        # 最后检查并裁剪极端值
        print("裁剪标准化后的极端值...")
        for col in normalized_data.columns:
            if np.issubdtype(normalized_data[col].dtype, np.number):
                # 计算异常值数量
                extreme_count = ((normalized_data[col] < -5) | (normalized_data[col] > 5)).sum()
                if extreme_count > 0:
                    print(f"列 {col} 有 {extreme_count} 个极端值被裁剪")
                
                # 裁剪到[-5,5]范围，避免极端值
                normalized_data[col] = normalized_data[col].clip(-5, 5)
        
        return normalized_data
    
    def inverse_normalize(self, data: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        反向标准化，将标准化后的数据转换回原始尺度
        
        参数:
            data: 要反向标准化的数据
            columns: 要反向标准化的列，如果为None则处理所有列
            
        返回:
            反向标准化后的数据
        """
        # 获取标准化方法
        norm_method = self.feature_config.get('normalization', 'z-score')
        
        # 创建副本避免修改原始数据
        inverse_data = data.copy()
        
        # 确定要处理的列
        if columns is None:
            columns = [col for col in inverse_data.columns if col in self.feature_means]
        
        # 对每列进行反向标准化
        for col in columns:
            if col in self.feature_means and col in self.feature_stds:
                if norm_method == 'z-score':
                    # 反向Z-Score: x * std + mean
                    inverse_data[col] = inverse_data[col] * self.feature_stds[col] + self.feature_means[col]
                
                elif norm_method == 'min-max':
                    # 反向Min-Max: x * (max - min) + min
                    inverse_data[col] = inverse_data[col] * self.feature_stds[col] + self.feature_means[col]
        
        return inverse_data
    
    def generate_labels(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        生成标签：根据未来价格变化生成多标签分类
        
        参数:
            data: 要添加标签的数据，如果为None则使用aligned_data
            
        返回:
            添加了标签的数据
        """
        if data is None:
            if self.aligned_data is None:
                raise ValueError("未对齐数据，请先调用align_data方法")
            data = self.aligned_data
        
        # 获取标签配置
        forward_period = self.config['labels'].get('forward_period', 15)
        label_type = self.config['labels'].get('label_type', 'dynamic_atr')
        atr_multiple = self.config['labels'].get('atr_multiple', 0.8)
        fixed_threshold = self.config['labels'].get('fixed_threshold', 0.005)
        
        # 创建副本避免修改原始数据
        labeled_data = data.copy()
        
        # 为每个品种生成标签
        for symbol in self.data_config['symbols']:
            if f"{symbol}_close" in labeled_data.columns:
                # 计算未来价格变化率
                labeled_data[f"{symbol}_future_return"] = (
                    labeled_data[f"{symbol}_close"].shift(-forward_period) / 
                    labeled_data[f"{symbol}_close"] - 1
                )
                
                # 根据标签类型选择阈值
                if label_type == 'dynamic_atr':
                    # 计算ATR
                    if all(col in labeled_data.columns for col in [f"{symbol}_high", f"{symbol}_low", f"{symbol}_close"]):
                        labeled_data[f"{symbol}_atr"] = self._calculate_atr(
                            labeled_data[f"{symbol}_high"],
                            labeled_data[f"{symbol}_low"],
                            labeled_data[f"{symbol}_close"],
                            period=self.feature_config.get('atr_period', 14)
                        )
                        
                        # 动态阈值 = ATR / 收盘价 * 倍数
                        labeled_data[f"{symbol}_threshold"] = (
                            labeled_data[f"{symbol}_atr"] / 
                            labeled_data[f"{symbol}_close"] * 
                            atr_multiple
                        )
                    else:
                        print(f"警告：缺少计算{symbol} ATR所需的列，使用固定阈值")
                        labeled_data[f"{symbol}_threshold"] = fixed_threshold
                else:
                    # 使用固定阈值
                    labeled_data[f"{symbol}_threshold"] = fixed_threshold
                
                # 生成标签：1(涨)/0(平)/-1(跌)
                labeled_data[f"{symbol}_label"] = 0  # 默认为0（平）
                labeled_data.loc[
                    labeled_data[f"{symbol}_future_return"] > labeled_data[f"{symbol}_threshold"], 
                    f"{symbol}_label"
                ] = 1  # 涨
                labeled_data.loc[
                    labeled_data[f"{symbol}_future_return"] < -labeled_data[f"{symbol}_threshold"], 
                    f"{symbol}_label"
                ] = -1  # 跌
        
        return labeled_data
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        计算ATR（平均真实波幅）
        
        参数:
            high: 高价序列
            low: 低价序列
            close: 收盘价序列
            period: ATR周期
            
        返回:
            ATR序列
        """
        # 计算三种价差
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        # 取最大值得到真实波幅（TR）
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算ATR
        atr = tr.rolling(window=period).mean()
        
        # 处理开头的NaN值
        atr = atr.fillna(tr.mean())
        
        return atr
    
    def prepare_model_data(self, lookback_period: int = None, max_features: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备模型输入数据：X_train, y_train, X_test, y_test
        
        参数:
            lookback_period: 回溯期，如果为None则使用配置中的值
            max_features: 每个品种最大使用的特征数量，如果为None则使用所有特征
            
        返回:
            (X_train, y_train, X_test, y_test)元组
        """
        if self.aligned_data is None:
            raise ValueError("未对齐数据，请先调用align_data方法")
        
        # 检查数据是否为空
        if len(self.aligned_data) == 0:
            raise ValueError("对齐后的数据为空，无法进行训练")
        
        # 检查数据列数
        if len(self.aligned_data.columns) == 0:
            raise ValueError("对齐后的数据没有列，无法进行训练")
        
        # 打印数据信息，帮助调试
        print(f"对齐后的数据形状: {self.aligned_data.shape}")
        print(f"对齐后的数据列: {self.aligned_data.columns.tolist()}")
        
        # 生成标签
        print("正在生成标签...")
        labeled_data = self.generate_labels()
        print(f"标签生成完成，形状: {labeled_data.shape}")
        
        # 获取模型配置
        if lookback_period is None:
            lookback_period = self.config['model'].get('lookback_period', 48)
        print(f"使用回溯期: {lookback_period}")
        
        # 检查数据长度是否足够
        if len(labeled_data) <= lookback_period:
            raise ValueError(f"数据长度({len(labeled_data)})不足以创建时间窗口序列(需要>{lookback_period})")
        
        train_test_split = self.data_config.get('train_test_split', 0.8)
        print(f"训练集比例: {train_test_split}")
        
        # 准备特征列和标签列
        feature_cols = []
        label_cols = []
        
        # 遍历所有品种
        print("准备特征列和标签列...")
        symbols = self.data_config.get('symbols', [])
        for symbol in symbols:
            # 添加该品种的特征列
            symbol_features = [col for col in labeled_data.columns if col.startswith(f"{symbol}_") and not col.endswith("_label")]
            
            # 如果指定了最大特征数，则只使用部分特征
            if max_features is not None and len(symbol_features) > max_features:
                print(f"为{symbol}选择{max_features}个特征（原有{len(symbol_features)}个）")
                # 保留基本价格特征
                basic_features = [col for col in symbol_features if any(x in col for x in ['_open', '_high', '_low', '_close', '_volume'])]
                # 其他技术指标特征
                other_features = [col for col in symbol_features if col not in basic_features]
                # 如果基本特征已经超过最大特征数，则只保留基本特征的一部分
                if len(basic_features) >= max_features:
                    selected_features = basic_features[:max_features]
                else:
                    # 否则保留所有基本特征，并添加部分技术指标特征
                    remaining_slots = max_features - len(basic_features)
                    selected_features = basic_features + other_features[:remaining_slots]
                
                symbol_features = selected_features
                print(f"为{symbol}选择了以下特征: {symbol_features}")
            
            feature_cols.extend(symbol_features)
            
            # 添加该品种的标签列
            symbol_labels = [col for col in labeled_data.columns if col.startswith(f"{symbol}_") and col.endswith("_label")]
            label_cols.extend(symbol_labels)
        
        print(f"特征列数量: {len(feature_cols)}")
        print(f"标签列数量: {len(label_cols)}")
        
        if not feature_cols or not label_cols:
            raise ValueError("未找到有效的特征列或标签列")
            
        # 打印特征列名，帮助调试
        print("特征列包括:")
        for i, col in enumerate(feature_cols[:20]):  # 只打印前20个特征列
            print(f"  {i}: {col}")
        if len(feature_cols) > 20:
            print(f"  ... 以及其他 {len(feature_cols) - 20} 个特征")
        
        # 标准化特征
        print("正在标准化特征...")
        normalized_data = self.normalize_features(labeled_data)
        print("特征标准化完成")
        
        # 分割训练集和测试集
        print("正在分割训练集和测试集...")
        split_idx = int(len(normalized_data) * train_test_split)
        train_data = normalized_data.iloc[:split_idx]
        test_data = normalized_data.iloc[split_idx:]
        print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")
        
        # 创建时间窗口序列
        print("正在创建时间窗口序列...")
        try:
            # 导入tqdm以显示进度条
            try:
                from tqdm import tqdm
                has_tqdm = True
            except ImportError:
                has_tqdm = False
            
            # 优化序列创建，减少内存使用
            # 使用批处理方式创建序列，避免一次性加载所有数据到内存
            batch_size = 1000  # 每批处理的样本数
            total_samples = len(train_data) - lookback_period
            num_batches = (total_samples + batch_size - 1) // batch_size  # 向上取整
            
            X_train_batches = []
            y_train_batches = []
            
            for batch in range(num_batches):
                if has_tqdm:
                    print(f"处理训练批次 {batch+1}/{num_batches}")
                
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, total_samples)
                
                X_batch = []
                y_batch = []
                
                for i in range(start_idx, end_idx):
                    # 提取时间窗口内的特征
                    X_batch.append(train_data[feature_cols].iloc[i:i+lookback_period].values)
                    
                    # 提取目标时间点的标签
                    y_batch.append(train_data[label_cols].iloc[i+lookback_period].values)
                    
                    # 每处理100个样本清理一次内存
                    if (i - start_idx) % 100 == 0 and i > start_idx:
                        import gc
                        gc.collect()
                
                # 转换为numpy数组并添加到批次列表
                X_train_batches.append(np.array(X_batch))
                y_train_batches.append(np.array(y_batch))
                
                # 清理临时变量
                del X_batch, y_batch
                import gc
                gc.collect()
            
            # 合并所有批次
            X_train = np.concatenate(X_train_batches, axis=0)
            y_train = np.concatenate(y_train_batches, axis=0)
            
            # 清理批次数据
            del X_train_batches, y_train_batches
            import gc
            gc.collect()
            
            print(f"训练序列创建完成: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
            
            print("正在创建测试集时间窗口序列...")
            # 对测试集也使用批处理方式
            total_test_samples = len(test_data) - lookback_period
            num_test_batches = (total_test_samples + batch_size - 1) // batch_size
            
            X_test_batches = []
            y_test_batches = []
            
            for batch in range(num_test_batches):
                if has_tqdm:
                    print(f"处理测试批次 {batch+1}/{num_test_batches}")
                
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, total_test_samples)
                
                X_batch = []
                y_batch = []
                
                for i in range(start_idx, end_idx):
                    X_batch.append(test_data[feature_cols].iloc[i:i+lookback_period].values)
                    y_batch.append(test_data[label_cols].iloc[i+lookback_period].values)
                    
                    # 每处理100个样本清理一次内存
                    if (i - start_idx) % 100 == 0 and i > start_idx:
                        import gc
                        gc.collect()
                
                X_test_batches.append(np.array(X_batch))
                y_test_batches.append(np.array(y_batch))
                
                # 清理临时变量
                del X_batch, y_batch
                import gc
                gc.collect()
            
            # 合并所有批次
            X_test = np.concatenate(X_test_batches, axis=0)
            y_test = np.concatenate(y_test_batches, axis=0)
            
            # 清理批次数据
            del X_test_batches, y_test_batches
            import gc
            gc.collect()
            
            print(f"测试序列创建完成: X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")
        
        except Exception as e:
            print(f"创建时间窗口序列时出错: {str(e)}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            raise
        
        # 将标签转换为one-hot编码
        print("将标签转换为适合模型的格式...")
        # 对于每个品种的标签（-1，0，1），转换为(0,1,2)然后再转为one-hot编码
        # 例如：-1 -> 0, 0 -> 1, 1 -> 2，然后再one-hot编码
        
        # 获取品种数量
        n_symbols = y_train.shape[1]
        # 转换为模型可以使用的格式
        y_train_processed = np.zeros((y_train.shape[0], n_symbols * 3))
        
        # 处理训练集标签
        for i in range(n_symbols):
            for j in range(y_train.shape[0]):
                label = int(y_train[j, i]) + 1  # 将-1,0,1转换为0,1,2
                y_train_processed[j, i*3 + label] = 1
        
        # 处理测试集标签
        y_test_processed = np.zeros((y_test.shape[0], n_symbols * 3))
        for i in range(n_symbols):
            for j in range(y_test.shape[0]):
                label = int(y_test[j, i]) + 1  # 将-1,0,1转换为0,1,2
                y_test_processed[j, i*3 + label] = 1
        
        print("标签转换完成")
        print(f"处理后的训练标签形状: {y_train_processed.shape}")
        print(f"处理后的测试标签形状: {y_test_processed.shape}")
        
        print("模型数据准备完成")
        return X_train, y_train_processed, X_test, y_test_processed
    
    def save_processed_data(self, output_dir: str = 'processed_data'):
        """
        保存处理后的数据到文件
        
        参数:
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存各品种处理后的数据
        for symbol, df in self.processed_data.items():
            df.to_csv(os.path.join(output_dir, f"{symbol}_processed.csv"))
        
        # 保存对齐后的数据
        if self.aligned_data is not None:
            self.aligned_data.to_csv(os.path.join(output_dir, "aligned_data.csv"))
        
        # 保存标准化参数
        pd.Series(self.feature_means).to_csv(os.path.join(output_dir, "feature_means.csv"))
        pd.Series(self.feature_stds).to_csv(os.path.join(output_dir, "feature_stds.csv"))
        
        print(f"已保存处理后的数据到目录: {output_dir}")


# 使用示例
if __name__ == "__main__":
    # 创建数据处理器
    processor = DataProcessor('config.yaml')
    
    # 加载数据
    raw_data = processor.load_data()
    
    # 预处理数据
    processed_data = processor.preprocess_data()
    
    # 对齐多品种数据
    aligned_data = processor.align_data()
    
    # 标准化特征
    normalized_data = processor.normalize_features()
    
    # 生成标签
    labeled_data = processor.generate_labels()
    
    # 准备模型数据
    X_train, y_train, X_test, y_test = processor.prepare_model_data()
    
    print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")
    
    # 保存处理后的数据
    processor.save_processed_data() 