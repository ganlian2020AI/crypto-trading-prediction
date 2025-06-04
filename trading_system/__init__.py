"""
加密货币交易系统

这是一个模块化的加密货币交易系统，支持实盘交易、回测、模型训练和数据下载等功能。
"""

__version__ = "0.1.0"

# 导出主要模块
from . import common
from . import live
from . import backtest
from . import training 