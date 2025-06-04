import pandas as pd
import numpy as np

def get_drawdown_periods(equity: pd.Series) -> pd.DataFrame:
    """
    返回所有回撤区间的起止日期及该区间的最大回撤。
    输出 DataFrame，包含列：['peak_date', 'trough_date', 'peak_value', 'trough_value', 'drawdown_pct', 'duration_days']。
    """
    df = compute_drawdown(equity)
    peak_idx = None
    trough_idx = None
    in_drawdown = False
    records = []

    for date, row in df.iterrows():
        if not in_drawdown and row['drawdown'] < 0:
            # 回撤区间开始
            in_drawdown = True
            peak_idx = df.loc[:date, 'equity'].idxmax()
            trough_idx = date
        if in_drawdown:
            # 每次遍历更新当前最深回撤点
            if row['drawdown'] < df.at[trough_idx, 'drawdown']:
                trough_idx = date
            # 如果回撤结束（当前值创出新高）
            if equity.loc[date] >= df.at[peak_idx, 'previous_peak']:
                # 记录该次回撤信息
                peak_val = equity.loc[peak_idx]
                trough_val = equity.loc[trough_idx]
                drawdown_pct = (trough_val - peak_val) / peak_val
                duration = (date - peak_idx).days
                records.append({
                    'peak_date': peak_idx,
                    'trough_date': trough_idx,
                    'peak_value': peak_val,
                    'trough_value': trough_val,
                    'drawdown_pct': drawdown_pct,
                    'duration_days': duration
                })
                in_drawdown = False

    return pd.DataFrame(records)

# 示例：找出所有回撤区间
periods_df = get_drawdown_periods(equity_curve)
periods_df
