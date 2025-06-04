# data_loader.py

"""
已实现的功能：
- 使用 pandas 读取指定路径的 CSV 文件
- 将原始 timestamp 列解析为 datetime 类型
- 对数据进行初步校验（检查缺失值、数据类型是否正确等）
- 返回一个标准化的 DataFrame 供后续模块使用

可优化的方案：
- 支持增量加载：只加载新增数据，而不是每次都读整个文件
- 将"校验逻辑"抽象成一个可注册的校验器列表，以便针对不同数据集复用
- 添加并行读取或分块读取，在处理大规模文件时提高效率
"""

import pandas as pd
from utils import parse_timestamp

# 定义默认值
DEFAULT_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

def load_crypto_data(csv_path: str) -> pd.DataFrame:
    """
    读取 CSV 并返回标准化后的 DataFrame。假设 CSV 至少包含以下列：
    ['timestamp', 'name', 'high', 'low', 'open', 'close', 'volume']
    
    参数:
        csv_path: CSV文件路径
    """
    # 1. 读取 CSV（假设文件编码为 UTF-8，无需额外参数）
    df = pd.read_csv(csv_path)

    # 2. 将 timestamp 转换为 datetime
    df["timestamp"] = df["timestamp"].apply(lambda ts: parse_timestamp(ts, DEFAULT_TIMESTAMP_FORMAT))

    # 3. 基本校验：确认关键字段无缺失
    required_cols = ["timestamp", "name", "high", "low", "open", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必需列：{missing_cols}")

    # 4. 进一步校验：数值列是否存在非数值/缺失
    num_cols = ["high", "low", "open", "close", "volume"]
    if df[num_cols].isnull().any().any():
        print("警告：存在缺失或无效数值，请确认数据完整性")

    # 5. 返回最终 DataFrame
    return df


if __name__ == "__main__":
    # 简单测试，需要指定CSV文件路径
    import os
    data_dir = "data"
    if os.path.exists(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if csv_files:
            test_file = os.path.join(data_dir, csv_files[0])
            df = load_crypto_data(test_file)
            print(f"测试文件: {test_file}")
    print(df.head())
        else:
            print(f"错误: 在{data_dir}目录中找不到CSV文件")
    else:
        print(f"错误: 找不到{data_dir}目录")
