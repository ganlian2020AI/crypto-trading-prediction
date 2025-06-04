import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any


def ris_vol_ema(df: pd.DataFrame, 
               yubool: bool = True,
               alpha_length: int = 20,
               gamma_length: int = 20,
               changdu: int = 576) -> pd.DataFrame:
    """
    实现自定义算法1: 预测价格计算
    
    参数:
        df: 包含OHLCV数据的DataFrame
        yubool: 是否计算预测价格
        alpha_length: alpha长度
        gamma_length: gamma长度
        changdu: 计算长度（默认576）
        
    返回:
        添加了自定义指标的DataFrame
    """
    # 创建副本避免修改原始数据
    result_df = df.copy()
    
    # 计算价格变化率
    result_df['change'] = result_df['close'].pct_change()
    
    # 计算RSI
    rsi_period = 14
    delta = result_df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    result_df['rsiValue'] = 100 - (100 / (1 + rs))
    
    # 计算成交量与价格的比值
    result_df['bodxp'] = result_df['volume'] / result_df['close']
    
    # 计算emay (bodxp的144周期指数移动平均)
    result_df['emay'] = result_df['bodxp'].ewm(span=144, adjust=False).mean()
    
    # 计算volty (bodxp与emay的比值)
    result_df['volty'] = result_df['bodxp'] / result_df['emay']
    
    # 计算shortnum (close / 100简化计算)
    result_df['shortnum'] = result_df['close'] / 100
    
    # 如果需要计算预测价格
    if yubool:
        # 计算预测价格
        result_df['predictedPrice'] = 0.0
        
        # 对每个时间点计算预测价格
        for i in range(changdu, len(result_df)):
            result_df.loc[result_df.index[i], 'predictedPrice'] = compute_predicted_price(result_df, i, changdu)
        
        # 平滑预测价格（12周期移动平均）
        result_df['smoothPredictedPrice'] = result_df['predictedPrice'].rolling(window=12).mean()
        
        # 计算价格与预测价格的偏差率
        result_df['priceToPredicted'] = (result_df['close'] - result_df['predictedPrice']) / result_df['predictedPrice'] * 100
        
        # 计算高低价与预测价格的偏差率
        result_df['hightopredicted'] = (result_df['high'] - result_df['predictedPrice']) / result_df['predictedPrice'] * 100
        result_df['lowtopredicted'] = (result_df['low'] - result_df['predictedPrice']) / result_df['predictedPrice'] * 100
        
        # 计算历史最高最低价
        result_df['highprice'] = result_df['high'].rolling(window=changdu+1).max()
        result_df['lowprice'] = result_df['low'].rolling(window=changdu+1).min()
        
        # 计算历史最大偏差率
        result_df['highbai'] = result_df['hightopredicted'].rolling(window=changdu+1).max()
        result_df['lowbai'] = result_df['lowtopredicted'].rolling(window=changdu+1).min()
        
        # 计算支撑阻力位
        result_df['highxian'] = result_df['highprice'] / (1 + result_df['highbai'] / 100.0)
        result_df['lowxian'] = result_df['lowprice'] / (1 + result_df['lowbai'] / 100.0)
        
        # 计算斐波那契水平分类
        result_df['highbi'] = classify_fib_level(result_df['hightopredicted'], 'high')
        result_df['lowbi'] = classify_fib_level(result_df['lowtopredicted'], 'low')
    
    return result_df


def compute_predicted_price(df: pd.DataFrame, idx: int, changdu: int) -> float:
    """
    计算预测价格
    
    参数:
        df: 包含价格和指标的DataFrame
        idx: 当前位置索引
        changdu: 计算长度
        
    返回:
        预测价格
    """
    # 从当前位置向前查找changdu根K线
    sumWeights = 0.0
    weightedSum = 0.0
    
    for offset in range(0, changdu + 1):
        j = idx - offset
        if j < 0:
            break
        
        sn = df.iloc[j]['shortnum']
        rsi_val = df.iloc[j]['rsiValue']
        
        if pd.isna(sn) or sn == 0 or pd.isna(rsi_val):
            continue
        
        # 计算权重
        weight = offset / sn
        adjustedWeight = weight * (1 + rsi_val / 100.0)
        
        # 累加权重和加权和
        sumWeights += adjustedWeight
        weightedSum += adjustedWeight * sn
    
    # 计算加权平均
    return weightedSum / sumWeights if sumWeights > 0 else np.nan


def classify_fib_level(series: pd.Series, level_type: str) -> pd.Series:
    """
    对偏差率进行斐波那契水平分类
    
    参数:
        series: 偏差率序列
        level_type: 'high'或'low'，指定高位或低位分类
        
    返回:
        分类结果序列
    """
    # 定义斐波那契水平
    high_thresholds = [4.236, 3.618, 2.618, 2, 1.786, 1.618, 1.5, 1.382, 1.236, 1, 0.786, 0.618, 0.5]
    low_thresholds = [4.236, 3.618, 2.618, 2, 1.786, 1.618, 1.5, 1.382, 1.236, 1, 0.786, 0.618, 0.5, 0.382, 0.236]
    
    # 选择阈值
    thresholds = high_thresholds if level_type == 'high' else low_thresholds
    
    # 初始化结果序列
    result = pd.Series(index=series.index, dtype=object)
    
    # 对每个值进行分类
    for idx, value in series.items():
        # 如果是NaN，跳过
        if pd.isna(value):
            result[idx] = None
            continue
        
        # 找到第一个小于等于偏差率的阈值
        found_level = False
        for i, threshold in enumerate(thresholds):
            if abs(value) <= threshold:
                if level_type == 'high':
                    result[idx] = f"{int(threshold * 100)}%"
                else:
                    result[idx] = f"{int(threshold * 100)}%"
                found_level = True
                break
        
        # 如果没有找到，使用最大阈值
        if not found_level:
            if level_type == 'high':
                result[idx] = f"{int(thresholds[0] * 100)}%"
            else:
                result[idx] = f"{int(thresholds[0] * 100)}%"
    
    return result


# 测试代码
if __name__ == "__main__":
    # 创建模拟数据
    dates = pd.date_range(start='2022-01-01', periods=1000, freq='30min')
    df = pd.DataFrame(index=dates)
    
    df['high'] = np.random.random(len(df)) * 1000 + 30000
    df['low'] = df['high'] * (1 - np.random.random(len(df)) * 0.01)
    df['open'] = df['low'] + np.random.random(len(df)) * (df['high'] - df['low'])
    df['close'] = df['low'] + np.random.random(len(df)) * (df['high'] - df['low'])
    df['volume'] = np.random.random(len(df)) * 100
    
    # 计算自定义指标
    result_df = ris_vol_ema(df)
    
    # 显示结果
    print(result_df.columns)
    print(result_df[['close', 'predictedPrice', 'smoothPredictedPrice', 'highxian', 'lowxian', 'highbi', 'lowbi']].tail())