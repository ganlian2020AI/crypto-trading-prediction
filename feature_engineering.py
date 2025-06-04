import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import yaml
import concurrent.futures
import time
from functools import partial

class FeatureEngineer:
    """
    特征工程模块：负责计算各种技术指标和自定义指标
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        初始化特征工程器
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # 提取特征配置
        self.feature_config = self.config['features']
        
        # 常量定义
        self.changdu = self.feature_config.get('custom_indicator1_length', 576)  # 自定义指标1长度
        
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        计算指数移动平均线(EMA)
        
        参数:
            prices: 价格序列
            period: EMA周期
            
        返回:
            EMA序列
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_all_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有配置的EMA
        
        参数:
            df: 包含价格数据的DataFrame
            
        返回:
            添加了EMA列的DataFrame
        """
        result_df = df.copy()
        
        # 获取EMA周期配置
        ema_periods = self.feature_config.get('ema_periods', [21, 144, 169])
        
        # 计算每个周期的EMA
        for period in ema_periods:
            result_df[f'ema{period}'] = self.calculate_ema(result_df['close'], period)
            
        return result_df
    
    def calculate_hema(self, prices: pd.Series) -> pd.Series:
        """
        计算Holt指数平滑EMA (HEMA)
        
        参数:
            prices: 价格序列
            
        返回:
            HEMA序列
        """
        # 获取HEMA参数
        alpha_length = self.feature_config.get('hema_alpha_length', 20)
        gamma_length = self.feature_config.get('hema_gamma_length', 20)
        
        # 计算alpha和gamma参数
        alpha = 2.0 / (alpha_length + 1)
        gamma = 2.0 / (gamma_length + 1)
        
        # 初始化结果序列
        hema = np.zeros(len(prices))
        level = np.zeros(len(prices))
        trend = np.zeros(len(prices))
        
        # 初始值
        level[0] = prices.iloc[0]
        trend[0] = 0
        hema[0] = level[0]
        
        # 计算HEMA
        for i in range(1, len(prices)):
            # 更新level和trend
            level[i] = alpha * prices.iloc[i] + (1 - alpha) * (level[i-1] + trend[i-1])
            trend[i] = gamma * (level[i] - level[i-1]) + (1 - gamma) * trend[i-1]
            
            # 计算HEMA值
            hema[i] = level[i] + trend[i]
        
        return pd.Series(hema, index=prices.index)
    
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """
        计算相对强弱指标(RSI)
        
        参数:
            prices: 价格序列
            period: RSI周期，如果为None则使用配置
            
        返回:
            RSI序列
        """
        if period is None:
            period = self.feature_config.get('rsi_period', 14)
        
        # 计算价格变化
        delta = prices.diff()
        
        # 分离上涨和下跌
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # 计算平均上涨和下跌
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # 处理开始的NaN值
        avg_gain.iloc[:period] = avg_gain.iloc[period]
        avg_loss.iloc[:period] = avg_loss.iloc[period]
        
        # 计算相对强度
        rs = avg_gain / avg_loss
        
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = None) -> pd.Series:
        """
        计算平均真实波幅(ATR)
        
        参数:
            high: 高价序列
            low: 低价序列
            close: 收盘价序列
            period: ATR周期，如果为None则使用配置
            
        返回:
            ATR序列
        """
        if period is None:
            period = self.feature_config.get('atr_period', 14)
        
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
    
    def compute_predicted_price(self, df: pd.DataFrame) -> pd.Series:
        """
        计算自定义指标1：预测价格
        
        参数:
            df: 包含必要列的DataFrame (shortnum, rsiValue)
            
        返回:
            预测价格序列
        """
        # 初始化结果序列
        predicted_prices = np.zeros(len(df))
        
        # 计算每个点的预测价格
        for idx in range(len(df)):
            # 从当前位置向前查找changdu(576)根K线
            sumWeights = 0.0
            weightedSum = 0.0
            
            for offset in range(0, min(self.changdu + 1, idx + 1)):
                j = idx - offset
                if j < 0:
                    break
                
                # 获取shortnum和rsiValue
                # 注意：这里假设df中已经有这些列，如果没有，需要先计算
                if 'shortnum' not in df.columns or 'rsiValue' not in df.columns:
                    # 如果没有必要的列，先计算
                    if 'shortnum' not in df.columns:
                        # 这里简化处理，实际应根据具体算法计算shortnum
                        sn = df['close'].iloc[j]
                    else:
                        sn = df['shortnum'].iloc[j]
                    
                    if 'rsiValue' not in df.columns:
                        # 如果没有rsiValue，使用计算的RSI
                        if 'rsi' not in df.columns:
                            rsi_val = 50  # 默认值
                        else:
                            rsi_val = df['rsi'].iloc[j]
                    else:
                        rsi_val = df['rsiValue'].iloc[j]
                else:
                    sn = df['shortnum'].iloc[j]
                    rsi_val = df['rsiValue'].iloc[j]
                
                # 检查值是否有效
                if pd.isna(sn) or sn == 0 or pd.isna(rsi_val):
                    continue
                
                # 计算权重
                weight = offset / sn if sn != 0 else 0
                adjustedWeight = weight * (1 + rsi_val / 100.0)
                sumWeights += adjustedWeight
                weightedSum += adjustedWeight * sn
            
            # 计算预测价格
            predicted_prices[idx] = weightedSum / sumWeights if sumWeights > 0 else np.nan
        
        # 处理NaN值
        predicted_prices = pd.Series(predicted_prices, index=df.index).ffill().bfill()
        
        # 使用12周期移动平均进行平滑处理
        predicted_prices = predicted_prices.rolling(window=12).mean().fillna(predicted_prices)
        
        return predicted_prices
    
    def calculate_support_resistance(self, df: pd.DataFrame, predicted_prices: pd.Series) -> pd.DataFrame:
        """
        计算扩展自定义指标1：支撑阻力位
        
        参数:
            df: 包含价格数据的DataFrame
            predicted_prices: 预测价格序列
            
        返回:
            添加了支撑阻力位的DataFrame
        """
        result_df = df.copy()
        
        # 添加预测价格
        result_df['predicted_price'] = predicted_prices
        
        # 计算价格与预测价格的偏差率
        result_df['hightopredicted'] = (result_df['high'] / result_df['predicted_price'] - 1) * 100
        result_df['lowtopredicted'] = (result_df['low'] / result_df['predicted_price'] - 1) * 100
        
        # 滚动window计算最高最低价
        result_df['highprice'] = result_df['high'].rolling(window=self.changdu+1).max()
        result_df['lowprice'] = result_df['low'].rolling(window=self.changdu+1).min()
        result_df['highbai'] = result_df['hightopredicted'].rolling(window=self.changdu+1).max()
        result_df['lowbai'] = result_df['lowtopredicted'].rolling(window=self.changdu+1).min()
        
        # 计算支撑阻力位
        result_df['highxian'] = result_df['highprice'] / (1 + result_df['highbai'] / 100.0)
        result_df['lowxian'] = result_df['lowprice'] / (1 + result_df['lowbai'] / 100.0)
        
        return result_df
    
    def process_prices(self, highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
        """
        自定义算法2：价格对处理
        
        参数:
            highs: 高价数组
            lows: 低价数组
            
        返回:
            处理后的特征数组
        """
        # 简化实现，实际应根据具体算法处理
        n = len(highs)
        result = np.zeros(n)
        
        # 构造上升和下降价格对
        for i in range(1, n):
            # 计算价格变化率
            high_change = highs[i] / highs[i-1] - 1 if highs[i-1] > 0 else 0
            low_change = lows[i] / lows[i-1] - 1 if lows[i-1] > 0 else 0
            
            # 简单处理：取变化率的平均值，并映射到0-10范围
            avg_change = (high_change + low_change) / 2
            result[i] = 5 + avg_change * 50  # 简单映射到0-10范围
        
        # 裁剪到0-10范围
        result = np.clip(result, 0, 10)
        
        return result
    
    def process_highs_and_lows(self, highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
        """
        自定义算法3：高低点处理
        
        参数:
            highs: 高价数组
            lows: 低价数组
            
        返回:
            处理后的特征数组
        """
        # 简化实现，实际应根据具体算法处理
        n = len(highs)
        result = np.zeros(n)
        
        # 找到局部高低点
        for i in range(2, n-2):
            # 简单判断局部高点
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                # 局部高点强度
                strength = (highs[i] / np.mean([highs[i-2], highs[i-1], highs[i+1], highs[i+2]]) - 1) * 5
                result[i] = 5 + strength
            
            # 简单判断局部低点
            elif lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                # 局部低点强度
                strength = (1 - lows[i] / np.mean([lows[i-2], lows[i-1], lows[i+1], lows[i+2]])) * 5
                result[i] = 5 - strength
        
        # 裁剪到0-10范围
        result = np.clip(result, 0, 10)
        
        return result
    
    def apply_fibonacci_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用斐波那契水平进行价格分类
        
        参数:
            df: 包含支撑阻力位的DataFrame
            
        返回:
            添加了斐波那契水平分类的DataFrame
        """
        result_df = df.copy()
        
        # 斐波那契水平定义
        high_thresholds = [4.236, 3.618, 2.618, 2, 1.786, 1.618, 1.5, 1.382, 1.236, 1, 0.786, 0.618, 0.5]
        low_thresholds = [4.236, 3.618, 2.618, 2, 1.786, 1.618, 1.5, 1.382, 1.236, 1, 0.786, 0.618, 0.5, 0.382, 0.236]
        
        # 初始化分类列
        result_df['highbi_level'] = ""
        result_df['lowbi_level'] = ""
        
        # 计算当前价格与支撑阻力位的比例
        for i in range(len(result_df)):
            if pd.isna(result_df['highxian'].iloc[i]) or pd.isna(result_df['close'].iloc[i]):
                continue
                
            # 计算高位比例
            high_ratio = result_df['close'].iloc[i] / result_df['highxian'].iloc[i]
            
            # 分类高位
            for j, threshold in enumerate(high_thresholds):
                if high_ratio >= threshold:
                    if j == 0:
                        result_df.at[i, 'highbi_level'] = "100%"
                    elif j == 1:
                        result_df.at[i, 'highbi_level'] = "50%"
                    elif j == 2:
                        result_df.at[i, 'highbi_level'] = "44%"
                    elif j == 3:
                        result_df.at[i, 'highbi_level'] = "38.2%"
                    elif j == 4:
                        result_df.at[i, 'highbi_level'] = "25%"
                    elif j == 5:
                        result_df.at[i, 'highbi_level'] = "23.6%"
                    elif j == 6:
                        result_df.at[i, 'highbi_level'] = "14%"
                    elif j == 7:
                        result_df.at[i, 'highbi_level'] = "10%"
                    elif j == 8:
                        result_df.at[i, 'highbi_level'] = "8%"
                    elif j == 9:
                        result_df.at[i, 'highbi_level'] = "6%"
                    elif j == 10:
                        result_df.at[i, 'highbi_level'] = "3%"
                    elif j == 11:
                        result_df.at[i, 'highbi_level'] = "2%"
                    else:
                        result_df.at[i, 'highbi_level'] = "1%"
                    break
            
            # 计算低位比例
            if pd.isna(result_df['lowxian'].iloc[i]):
                continue
                
            low_ratio = result_df['close'].iloc[i] / result_df['lowxian'].iloc[i]
            
            # 分类低位
            for j, threshold in enumerate(low_thresholds):
                if low_ratio <= threshold:
                    if j == 0:
                        result_df.at[i, 'lowbi_level'] = "100%"
                    elif j == 1:
                        result_df.at[i, 'lowbi_level'] = "50%"
                    elif j == 2:
                        result_df.at[i, 'lowbi_level'] = "44%"
                    elif j == 3:
                        result_df.at[i, 'lowbi_level'] = "38.2%"
                    elif j == 4:
                        result_df.at[i, 'lowbi_level'] = "25%"
                    elif j == 5:
                        result_df.at[i, 'lowbi_level'] = "23.6%"
                    elif j == 6:
                        result_df.at[i, 'lowbi_level'] = "14%"
                    elif j == 7:
                        result_df.at[i, 'lowbi_level'] = "10%"
                    elif j == 8:
                        result_df.at[i, 'lowbi_level'] = "8%"
                    elif j == 9:
                        result_df.at[i, 'lowbi_level'] = "6%"
                    elif j == 10:
                        result_df.at[i, 'lowbi_level'] = "3%"
                    elif j == 11:
                        result_df.at[i, 'lowbi_level'] = "2%"
                    elif j == 12:
                        result_df.at[i, 'lowbi_level'] = "1%"
                    else:
                        result_df.at[i, 'lowbi_level'] = "0%"
                    break
        
        return result_df
    
    def preprocess_custom_algorithm_feature(self, feature_array: np.ndarray) -> np.ndarray:
        """
        预处理自定义算法特征
        
        参数:
            feature_array: 特征数组
            
        返回:
            处理后的特征数组
        """
        # 负数归0
        feature_array = np.where(feature_array < 0, 0, feature_array)
        # 超过10归10
        feature_array = np.where(feature_array > 10, 10, feature_array)
        return feature_array
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有特征
        
        参数:
            df: 原始DataFrame
            
        返回:
            添加了所有特征的DataFrame
        """
        start_time = time.time()
        print(f"开始计算特征，数据形状: {df.shape}")
        
        result_df = df.copy()
        
        # 1. 计算EMA
        print("计算EMA...")
        result_df = self.calculate_all_emas(result_df)
        
        # 2. 计算HEMA
        print("计算HEMA...")
        result_df['hema'] = self.calculate_hema(result_df['close'])
        
        # 3. 计算RSI
        print("计算RSI...")
        result_df['rsi'] = self.calculate_rsi(result_df['close'])
        
        # 4. 计算ATR
        print("计算ATR...")
        result_df['atr'] = self.calculate_atr(result_df['high'], result_df['low'], result_df['close'])
        
        # 5. 计算预测价格（自定义指标1）
        # 为了计算预测价格，我们需要先计算shortnum
        # 这里简化处理，使用收盘价作为shortnum
        print("准备计算预测价格...")
        result_df['shortnum'] = result_df['close']
        result_df['rsiValue'] = result_df['rsi']
        
        # 使用多线程计算计算密集型指标
        print("使用多线程计算复杂指标...")
        try:
            # 创建计算任务
            tasks = [
                ("预测价格", lambda: self.compute_predicted_price(result_df)),
                ("价格对处理", lambda: self.process_prices(result_df['high'].values, result_df['low'].values)),
                ("高低点处理", lambda: self.process_highs_and_lows(result_df['high'].values, result_df['low'].values))
            ]
            
            # 使用线程池并行执行任务
            results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # 提交所有任务
                future_to_task = {executor.submit(task_func): task_name for task_name, task_func in tasks}
                
                # 获取结果
                for future in concurrent.futures.as_completed(future_to_task):
                    task_name = future_to_task[future]
                    try:
                        results[task_name] = future.result()
                        print(f"✓ {task_name}计算完成")
                    except Exception as e:
                        print(f"× {task_name}计算失败: {str(e)}")
                        # 如果线程执行失败，回退到单线程计算
                        if task_name == "预测价格":
                            print("尝试单线程计算预测价格...")
                            results[task_name] = self.compute_predicted_price(result_df)
                        elif task_name == "价格对处理":
                            print("尝试单线程计算价格对处理...")
                            results[task_name] = self.process_prices(result_df['high'].values, result_df['low'].values)
                        elif task_name == "高低点处理":
                            print("尝试单线程计算高低点处理...")
                            results[task_name] = self.process_highs_and_lows(result_df['high'].values, result_df['low'].values)
            
            # 将结果添加到DataFrame
            predicted_prices = results.get("预测价格")
            if predicted_prices is not None:
                result_df['predicted_price'] = predicted_prices
                
                # 6. 计算支撑阻力位（扩展自定义指标1）
                print("计算支撑阻力位...")
                result_df = self.calculate_support_resistance(result_df, predicted_prices)
            
            # 7. 价格对处理（自定义算法2）
            custom_algo2 = results.get("价格对处理")
            if custom_algo2 is not None:
                result_df['custom_algo2'] = custom_algo2
            
            # 8. 高低点处理（自定义算法3）
            custom_algo3 = results.get("高低点处理")
            if custom_algo3 is not None:
                result_df['custom_algo3'] = custom_algo3
                
        except Exception as e:
            print(f"多线程计算失败，回退到单线程: {str(e)}")
            # 回退到单线程计算
            predicted_prices = self.compute_predicted_price(result_df)
            result_df['predicted_price'] = predicted_prices
            
            # 6. 计算支撑阻力位（扩展自定义指标1）
            result_df = self.calculate_support_resistance(result_df, predicted_prices)
            
            # 7. 价格对处理（自定义算法2）
            custom_algo2 = self.process_prices(result_df['high'].values, result_df['low'].values)
            result_df['custom_algo2'] = custom_algo2
            
            # 8. 高低点处理（自定义算法3）
            custom_algo3 = self.process_highs_and_lows(result_df['high'].values, result_df['low'].values)
            result_df['custom_algo3'] = custom_algo3
        
        # 9. 应用斐波那契水平
        print("应用斐波那契水平...")
        result_df = self.apply_fibonacci_levels(result_df)
        
        # 10. 预处理自定义算法特征
        if self.feature_config.get('custom_algorithm2_enabled', True):
            result_df['custom_algo2'] = self.preprocess_custom_algorithm_feature(result_df['custom_algo2'].values)
        
        if self.feature_config.get('custom_algorithm3_enabled', True):
            result_df['custom_algo3'] = self.preprocess_custom_algorithm_feature(result_df['custom_algo3'].values)
        
        end_time = time.time()
        print(f"特征计算完成，耗时: {end_time - start_time:.2f}秒")
        return result_df


# 使用示例
if __name__ == "__main__":
    # 加载示例数据
    try:
        print("加载示例数据...")
        df = pd.read_csv('data/BTC.csv')
        
        # 标准化列名
        df.columns = [col.lower() for col in df.columns]
        
        # 转换时间戳为datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # 创建特征工程器
        print("初始化特征工程器...")
        feature_engineer = FeatureEngineer('config.yaml')
        
        # 计算所有特征
        print("开始计算所有特征...")
        start_time = time.time()
        result_df = feature_engineer.calculate_all_features(df)
        end_time = time.time()
        
        # 显示结果
        print(f"特征计算完成，总耗时: {end_time - start_time:.2f}秒")
        print(result_df.head())
        print(f"特征列: {result_df.columns.tolist()}")
        
    except Exception as e:
        print(f"示例运行出错: {str(e)}")