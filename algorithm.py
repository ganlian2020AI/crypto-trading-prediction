import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any


def preprocess_custom_algorithm_feature(feature_array: np.ndarray) -> np.ndarray:
    """
    预处理自定义算法特征，裁剪异常值
    
    参数:
        feature_array: 特征数组
        
    返回:
        处理后的特征数组
    """
    feature_array = np.where(feature_array < 0, 0, feature_array)
    feature_array = np.where(feature_array > 10, 10, feature_array)
    return feature_array


def process_prices(highs: List[float], lows: List[float]) -> Dict[str, List[Tuple[float, float]]]:
    """
    自定义算法2: 价格对处理
    
    参数:
        highs: 高价列表
        lows: 低价列表
        
    返回:
        处理后的价格对字典
    """
    # 检查输入
    if len(highs) != len(lows):
        raise ValueError("高价和低价列表长度必须相同")
    
    if len(highs) < 3:
        return {"upPrices": [], "downPrices": []}
    
    # 初始化上升和下降价格对
    up_prices = []
    down_prices = []
    
    # 第一轮：构造价格对
    for i in range(1, len(highs)):
        # 上升价格对
        if highs[i] > highs[i-1]:
            up_prices.append((highs[i-1], highs[i]))
        
        # 下降价格对
        if lows[i] < lows[i-1]:
            down_prices.append((lows[i-1], lows[i]))
    
    # 第二轮：删除无效价格对
    up_prices = filter_invalid_pairs(up_prices)
    down_prices = filter_invalid_pairs(down_prices)
    
    # 第三轮：合并相似价格对
    up_prices = merge_similar_pairs(up_prices)
    down_prices = merge_similar_pairs(down_prices)

    return {
        "upPrices": up_prices,
        "downPrices": down_prices
    }


def filter_invalid_pairs(pairs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    过滤无效价格对
    
    参数:
        pairs: 价格对列表
        
    返回:
        过滤后的价格对列表
    """
    # 删除变化率小于0.1%的价格对
    filtered_pairs = []
    for pair in pairs:
        if pair[0] > 0 and abs(pair[1] - pair[0]) / pair[0] >= 0.001:
            filtered_pairs.append(pair)
    
    return filtered_pairs


def merge_similar_pairs(pairs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    合并相似价格对
    
    参数:
        pairs: 价格对列表
        
    返回:
        合并后的价格对列表
    """
    if not pairs:
        return []
    
    # 排序
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    
    # 合并相似价格对
    merged_pairs = [sorted_pairs[0]]
    
    for i in range(1, len(sorted_pairs)):
        current = sorted_pairs[i]
        last = merged_pairs[-1]
        
        # 如果当前价格对的起始价与上一个价格对的结束价相似度高，则合并
        if abs(current[0] - last[1]) / last[1] < 0.005:
            merged_pairs[-1] = (last[0], current[1])
        else:
            merged_pairs.append(current)
    
    return merged_pairs


def process_highs_and_lows(highs: List[float], lows: List[float]) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    自定义算法3: 高低点处理
    
    参数:
        highs: 高价列表
        lows: 低价列表
        
    返回:
        四个列表: up_return1, up_return2, down_return1, down_return2
    """
    # 检查输入
    if len(highs) != len(lows):
        raise ValueError("高价和低价列表长度必须相同")
    
    if len(highs) < 3:
        return [], [], [], []
    
    # 找到最大最小值索引
    high_max_idx = np.argmax(highs)
    low_min_idx = np.argmin(lows)
    
    # 截取数据
    if high_max_idx > low_min_idx:
        # 最高点在最低点之后
        highs_segment = highs[low_min_idx:high_max_idx+1]
        lows_segment = lows[low_min_idx:high_max_idx+1]
    else:
        # 最低点在最高点之后
        highs_segment = highs[high_max_idx:low_min_idx+1]
        lows_segment = lows[high_max_idx:low_min_idx+1]
    
    # 排序索引
    sorted_high_idx = np.argsort(highs_segment)[::-1]  # 从大到小
    sorted_low_idx = np.argsort(lows_segment)  # 从小到大
    
    # 构建前20个最高点和最低点
    top_highs_idx = sorted_high_idx[:min(20, len(sorted_high_idx))]
    top_lows_idx = sorted_low_idx[:min(20, len(sorted_low_idx))]
    
    # 过滤掉相邻的点
    filtered_highs_idx = filter_adjacent_points(top_highs_idx)
    filtered_lows_idx = filter_adjacent_points(top_lows_idx)
    
    # 生成上升和下降返回值
    up_return1 = [highs_segment[i] for i in filtered_highs_idx]
    up_return2 = [highs_segment[i] * 1.01 for i in filtered_highs_idx]  # 增加1%
    
    down_return1 = [lows_segment[i] for i in filtered_lows_idx]
    down_return2 = [lows_segment[i] * 0.99 for i in filtered_lows_idx]  # 减少1%
    
    # 根据最小绝对差值进行合并
    up_return1, up_return2 = merge_by_min_abs_diff(up_return1, up_return2)
    down_return1, down_return2 = merge_by_min_abs_diff(down_return1, down_return2)
    
    return up_return1, up_return2, down_return1, down_return2


def filter_adjacent_points(indices: np.ndarray) -> List[int]:
    """
    过滤相邻点
    
    参数:
        indices: 索引数组
        
    返回:
        过滤后的索引列表
    """
    if len(indices) <= 1:
        return indices.tolist()
    
    sorted_indices = sorted(indices)
    filtered = [sorted_indices[0]]
    
    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] - filtered[-1] > 2:  # 要求至少间隔2个点
            filtered.append(sorted_indices[i])
    
    return filtered


def merge_by_min_abs_diff(values1: List[float], values2: List[float]) -> Tuple[List[float], List[float]]:
    """
    根据最小绝对差值合并两个列表
    
    参数:
        values1: 第一个值列表
        values2: 第二个值列表
        
    返回:
        合并后的两个列表
    """
    if len(values1) != len(values2):
        return values1, values2
    
    if len(values1) <= 1:
        return values1, values2
    
    # 计算绝对差值
    diffs = [abs(values1[i] - values2[i]) for i in range(len(values1))]
    
    # 找出最小差值的索引
    min_diff_idx = np.argmin(diffs)
    
    # 合并
    merged_values1 = values1[:min_diff_idx] + values1[min_diff_idx+1:]
    merged_values2 = values2[:min_diff_idx] + values2[min_diff_idx+1:]
    
    return merged_values1, merged_values2


# 测试代码
if __name__ == "__main__":
    # 模拟数据
    n = 100
    highs = np.random.normal(100, 10, n).tolist()
    lows = np.random.normal(90, 10, n).tolist()
    
    # 测试自定义算法2
    result = process_prices(highs, lows)
    print("自定义算法2结果:")
    print(f"上升价格对数量: {len(result['upPrices'])}")
    print(f"下降价格对数量: {len(result['downPrices'])}")
    
    # 测试自定义算法3
    up_return1, up_return2, down_return1, down_return2 = process_highs_and_lows(highs, lows)
    print("\n自定义算法3结果:")
    print(f"up_return1长度: {len(up_return1)}")
    print(f"up_return2长度: {len(up_return2)}")
    print(f"down_return1长度: {len(down_return1)}")
    print(f"down_return2长度: {len(down_return2)}")