import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_drawdown(equity: pd.Series) -> pd.DataFrame:
    """
    计算给定权益曲线的回撤 (drawdown)。
    返回一个 DataFrame，其中包含：
      - equity: 当日账户价值
      - previous_peak: 截至当前时点的历史最高点
      - drawdown: 当前时点相对于历史最高点的回撤百分比
    """
    previous_peak = equity.cummax()                # 计算当前及之前的最高价值
    drawdown = (equity - previous_peak) / previous_peak
    return pd.DataFrame({
        'equity': equity,
        'previous_peak': previous_peak,
        'drawdown': drawdown
    })

# 1. 构造示例模型收益率（随机波动）
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
# 假设每日收益率 ~ 正态分布
daily_returns = pd.Series(
    np.random.normal(loc=0.0005, scale=0.01, size=len(dates)),
    index=dates
)

# 2. 构造权益曲线（假设初始资金为 1.0）
equity_curve = (1 + daily_returns).cumprod()

# 3. 计算回撤
dd_df = compute_drawdown(equity_curve)

# 4. 找到最大回撤及其发生日期
max_drawdown = dd_df['drawdown'].min()
max_drawdown_date = dd_df['drawdown'].idxmin()

# 5. 输出：最大回撤及日期
print(f"最大回撤: {max_drawdown:.2%}")
print(f"发生日期: {max_drawdown_date.date()}")

# 6. 绘制权益曲线与回撤曲线
plt.figure(figsize=(12, 6))

# 第一张图：权益曲线
plt.subplot(2, 1, 1)
plt.plot(equity_curve.index, equity_curve.values, label='Equity Curve')
plt.title("示例模型权益曲线")
plt.ylabel("账户价值")
plt.legend()
plt.grid(True)

# 第二张图：回撤百分比曲线
plt.subplot(2, 1, 2)
plt.plot(dd_df.index, dd_df['drawdown'].values, color='red', label='Drawdown')
plt.title("示例模型回撤百分比")
plt.ylabel("回撤 (%)")
plt.xlabel("日期")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 7. 显示前十行回撤数据
print("\n前10行回撤数据：")
print(dd_df.head(10))
