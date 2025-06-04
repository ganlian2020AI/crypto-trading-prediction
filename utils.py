# utils.py

"""
已实现的功能：
- 提供通用辅助函数，包括时间戳解析、数据格式转换等
- parse_timestamp(): 将字符串格式的时间戳转换为 pandas.Timestamp

可优化的方案：
- 为 parse_timestamp 添加多种格式自动识别（传入 format=None 时尝试多种常见格式）
- 如果后续引入更多工具函数，可以将其拆分到 submodule（如：string_utils.py、date_utils.py）
- 增加异常处理：当解析失败时，记录日志并返回 NaT 而不是抛出
"""

import pandas as pd
from datetime import datetime


def parse_timestamp(ts_str: str, fmt: str = None) -> pd.Timestamp:
    """
    将给定格式的时间戳字符串转换为 pandas.Timestamp。
    如果 fmt 为空，则尝试从 ISO 格式解析。
    """
    if fmt:
        try:
            dt = datetime.strptime(ts_str, fmt)
            return pd.Timestamp(dt)
        except Exception as e:
            # 如果解析失败，尝试 pandas 自带解析
            try:
                return pd.to_datetime(ts_str)
            except Exception:
                raise ValueError(f"无法解析时间戳: {ts_str}") from e
    else:
        # 直接使用 pandas 解析
        return pd.to_datetime(ts_str)
