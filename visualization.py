# visualization.py

"""
已实现的功能：
- 提供基本折线图函数，绘制收盘价等随时间变化的趋势图
  - plot_price_trend(): 绘制指定品种某一价格列（如 close）随时间的走势
  - plot_moving_average(): 在同一图中叠加原始价格与移动平均线

可优化的方案：
- 增加子图支持（subplot），一次绘制多个品种或多个指标
- 设置可选参数来自定义图的标签、标题、日期格式、图例位置、输出尺寸等
- 支持将可视化结果输出为多种格式（PNG、PDF、SVG），并提供自动保存路径
"""

import matplotlib.pyplot as plt
import os


def plot_price_trend(df, price_col="close", coin_name="BTC", output_path=None):
    """
    绘制指定 coin_name 的 price_col（默认 close）随时间的折线图。
    如果 output_path 提供，则将图保存到该路径，否则直接显示。
    """
    plt.figure(figsize=(10, 5))
    subset = df[df["name"] == coin_name]
    plt.plot(subset["timestamp"], subset[price_col], label=f"{coin_name} {price_col}")
    plt.xlabel("Date")
    plt.ylabel(price_col.capitalize())
    plt.title(f"{coin_name} {price_col} Trend Over Time")
    plt.legend()
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_moving_average(df, window=5, price_col="close", coin_name="BTC", output_path=None):
    """
    在原始收盘价折线图上，叠加计算好的移动平均线。
    """
    plt.figure(figsize=(10, 5))
    subset = df[df["name"] == coin_name]
    plt.plot(subset["timestamp"], subset[price_col], label=f"{coin_name} {price_col}")
    ma_series = subset[price_col].rolling(window=window).mean()
    plt.plot(subset["timestamp"], ma_series, label=f"{coin_name} MA-{window}")
    plt.xlabel("Date")
    plt.ylabel(price_col.capitalize())
    plt.title(f"{coin_name} {price_col} with MA-{window}")
    plt.legend()
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
