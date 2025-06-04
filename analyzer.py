import pandas as pd
import numpy as np
import math
from typing import List, Optional, Any, Tuple

# """
# 已实现的功能：
# - 提供对 DataFrame 的基本统计分析函数
#   - get_summary_stats(): 计算 open/high/low/close/volume 的描述性统计
#   - compute_moving_average(): 根据给定窗口计算移动平均
#   - filter_by_date_range(): 根据开始和结束日期筛选数据
#
# 可优化的方案：
# - 增加更多技术指标函数，例如 RSI、MACD、布林带、成交量加权平均价（VWAP）等
# - 按品种（name 分组）进行批量指标计算，进而支持多品种对比分析
# - 并行化处理：当数据集很大时，加速指标计算
# """


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 open/high/low/close/volume 的基本描述性统计：
    count、mean、std、min、25%、50%、75%、max
    """
    return df[["open", "high", "low", "close", "volume"]].describe()


def compute_moving_average(df: pd.DataFrame, window: int = 5, price_col: str = "close") -> pd.Series:
    """
    计算指定价格列在给定窗口下的简单移动平均（SMA）。
    返回一个与 df 对齐的 Series，前面 window-1 个值为 NaN。
    """
    return df[price_col].rolling(window=window).mean()


def filter_by_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    根据给定的 start_date 和 end_date（格式：'YYYY-MM-DD'），
    筛选 timestamp 在该范围内的行（包含 start_date、end_date）。
    """
    mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
    return df.loc[mask].copy()


# 假设 Ta 类已在 ta.py 中实现，并提供相应方法
from ta import Ta


class LiangJia:
    def __init__(self):
        self.db = CommodityDatabase.get_instance()  # 假设 CommodityDatabase 有 get_instance() 方法
        self.ta = Ta()

    def up_all_js(self) -> None:
        """更新所有计算"""
        table_names = self.db.get_all_table_names()
        for table_name in table_names:
            self.db.reorder_serials(table_name)
            self.calculate_metrics(table_name)

    def calculate_metrics(self, table_name: str) -> None:
        """总调用函数"""
        if self.db.is_serial_300_fields_empty(table_name):
            records = self.db.find_and_delete_records(table_name)
            for serial in records:
                data_range = range(serial - 576, serial + 1)
                result = self.db.fetch_data(table_name, data_range)

                smooth_pred_arr = self.lianjia(
                    highs=result.high_prices,
                    lows=result.low_prices,
                    closes=result.close_prices,
                    volumes=result.volumes,
                )
                if smooth_pred_arr is None:
                    continue
                smooth_pred_last = self.get_last_value(smooth_pred_arr)
                high_last = result.high_prices[-1] if result.high_prices else 0.0
                low_last = result.low_prices[-1] if result.low_prices else 0.0
                close_last = result.close_prices[-1] if result.close_prices else 0.0

                yubai = self.calculate_yubai(
                    smooth_predicted_price=smooth_pred_last,
                    high_price=high_last,
                    low_price=low_last,
                    close_price=close_last,
                )
                hema_last = self.get_last_value(self.hema(prices=result.close_prices))
                ema_vals = self.ema_values(close_prices=result.close_prices) or (0.0, 0.0, 0.0)

                self.db.update_fields(
                    table_name=table_name,
                    serial=serial,
                    ris=hema_last,
                    daoone=ema_vals[0],  # ema169
                    daotwo=ema_vals[1],  # ema144
                    dao=ema_vals[2],     # ema difference
                    yu=smooth_pred_last,
                    yubai=yubai,
                )
        else:
            initial_data = self.db.fetch_data(table_name, range(1, 721))

            for i in range(1, 720):
                sub_range = range(0, i)  # Python 索引从 0 开始，对应 Swift 中 1..<i
                sub_result = self.extract_sub_result(initial_data, sub_range)
                if sub_result is None:
                    continue

                smooth_pred_arr = self.lianjia(
                    highs=sub_result["high_prices"],
                    lows=sub_result["low_prices"],
                    closes=sub_result["close_prices"],
                    volumes=sub_result["volumes"],
                )
                if smooth_pred_arr is None:
                    continue
                smooth_pred_last = self.get_last_value(smooth_pred_arr)
                high_val = initial_data.high_prices[i - 1] if initial_data.high_prices else 0.0
                low_val = initial_data.low_prices[i - 1] if initial_data.low_prices else 0.0
                close_val = initial_data.close_prices[i - 1] if initial_data.close_prices else 0.0

                yubai = self.calculate_yubai(
                    smooth_predicted_price=smooth_pred_last,
                    high_price=high_val,
                    low_price=low_val,
                    close_price=close_val,
                )
                hema_last = self.get_last_value(self.hema(prices=sub_result["close_prices"]))
                ema_vals = self.ema_values(close_prices=sub_result["close_prices"]) or (0.0, 0.0, 0.0)

                self.db.update_fields(
                    table_name=table_name,
                    serial=i,
                    ris=hema_last,
                    daoone=ema_vals[0],
                    daotwo=ema_vals[1],
                    dao=ema_vals[2],
                    yu=smooth_pred_last,
                    yubai=yubai,
                )

    def extract_sub_result(
        self,
        data: Any,
        sub_range: range,
    ) -> Optional[dict]:
        """提取子范围的数据"""
        try:
            high_prices = data.high_prices[sub_range.start : sub_range.stop]
            low_prices = data.low_prices[sub_range.start : sub_range.stop]
            close_prices = data.close_prices[sub_range.start : sub_range.stop]
            volumes = data.volumes[sub_range.start : sub_range.stop]
        except Exception:
            print(f"范围无效：{sub_range}")
            return None
        return {
            "high_prices": high_prices,
            "low_prices": low_prices,
            "close_prices": close_prices,
            "volumes": volumes,
        }

    def calculate_yubai(
        self,
        smooth_predicted_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
    ) -> float:
        """计算 yubai 的值"""
        if smooth_predicted_price < close_price:
            additional_value = (high_price - smooth_predicted_price) / smooth_predicted_price
        else:
            additional_value = (low_price - smooth_predicted_price) / smooth_predicted_price
        return round(additional_value, 2)

    def ema_values(self, close_prices: List[float]) -> Optional[Tuple[float, float, float]]:
        """计算 ema169 和 ema144 的值，并返回它们的差值"""
        ema169 = self.ema(prices=close_prices, length=169)
        ema144 = self.ema(prices=close_prices, length=144)
        if ema169 is None or ema144 is None:
            return None
        ema_difference = (ema144 - ema169) / ema144 if ema144 != 0 else 0.0
        return ema169, ema144, round(ema_difference, 2)

    @staticmethod
    def get_last_value(array: List[float]) -> float:
        """获取数组最后一个值"""
        return array[-1] if array else 0.0

    def lianjia(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
    ) -> Optional[List[float]]:
        """计算平滑预测价格"""
        if not (len(highs) == len(lows) == len(closes) == len(volumes)):
            print("数组长度不一致")
            return None
        if not highs:
            return None

        bodxp: List[float] = []
        for i in range(len(highs)):
            denom = (highs[i] + closes[i] - lows[i] * 2)
            if denom == 0 or lows[i] == 0:
                bodxp.append(0.0)
            else:
                bodxp.append(volumes[i] / denom / lows[i] * 100.0)

        emay = self.ema(prices=bodxp, length=144) or 0.0
        emas = self.ema(prices=closes, length=12) or 0.0

        imhigh = highs.copy()
        imlow = lows.copy()
        imclose = closes.copy()

        for i in range(len(imhigh)):
            if not (emay > bodxp[i] * 2) or not (closes[i] < emas):
                imhigh[i] = 0.0
            if not (emay > bodxp[i] * 2) or not (closes[i] > emas):
                imlow[i] = 0.0
            if not (emay < bodxp[i] * 0.5):
                imclose[i] = 0.0

        shurtnums = [max(imhigh[i], imlow[i]) for i in range(len(imhigh))]
        shurtnums1 = [max(shurtnums[i], imclose[i]) for i in range(len(shurtnums))]
        shurtnums2 = [max(lows[i], shurtnums1[i]) for i in range(len(shurtnums1))]

        shortnum = self.ema(prices=shurtnums2, length=len(bodxp)) or 0.0
        rsivalue = self.rsi(prices=closes, length=14) or 0.0

        if shortnum == 0:
            return [0.0]
        weight = 1.0 / shortnum
        adjusted_weight = weight * (1.0 + rsivalue / 100.0)
        sum_weights = adjusted_weight
        weighted_sum = adjusted_weight * shortnum

        predicted_price = weighted_sum / sum_weights if sum_weights != 0 else 0.0
        return [predicted_price]

    def hema(self, prices: List[float], alpha_length: int = 20, gamma_length: int = 20) -> List[float]:
        """计算 HEMA 序列"""
        if len(prices) < alpha_length:
            print("数据长度不足以计算 HEMA")
            return []

        alpha = 2.0 / float(alpha_length + 1)
        gamma = 2.0 / float(gamma_length + 1)
     
        b: List[float] = [prices[0]]
        hema_arr: List[float] = [prices[0]]

        for i in range(1, len(prices)):
            prev_hema = hema_arr[-1]
            prev_b = b[-1]
            src_sum = prev_hema + prev_b
            new_hema = (1.0 - alpha) * src_sum + alpha * prices[i]
            new_b = (1.0 - gamma) * prev_b + gamma * (new_hema - prev_hema)
            hema_arr.append(new_heema)
            b.append(new_b)

        return hema_arr

    @staticmethod
    def ema(prices: List[float], length: int) -> Optional[float]:
        """计算 EMA"""
        if len(prices) < length:
            print("数据长度不足以计算 EMA")
            return None
        multiplier = 2.0 / float(length + 1)
        prev_ema = prices[0]
        for i in range(1, len(prices)):
            prev_ema = (prices[i] - prev_ema) * multiplier + prev_ema
        return prev_ema

    @staticmethod
    def rsi(prices: List[float], length: int) -> Optional[float]:
        """计算 RSI"""
        if len(prices) < length + 1:
            print("数据长度不足以计算 RSI")
            return None
        gains: List[float] = []
        losses: List[float] = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(-change)
        avg_gain = sum(gains[:length]) / float(length)
        avg_loss = sum(losses[:length]) / float(length)
        for i in range(length, len(prices) - 1):
            avg_gain = (avg_gain * (length - 1) + gains[i]) / float(length)
            avg_loss = (avg_loss * (length - 1) + losses[i]) / float(length)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

# 新算法转换

def new_algorithm(df: pd.DataFrame,
                  length: int = 14,
                  ind: str = 'RSI',
                  show: bool = True,
                  ovrb: float = 7.0,
                  ovrs: float = 1.0,
                  midl: float = 4.0,
                  pd: int = 22,
                  bbl: int = 20,
                  mult: float = 2.0,
                  lb: int = 50,
                  ph: float = 0.85,
                  hp: bool = False) -> pd.DataFrame:
    """
    将 Pine Script 中的新算法转换为 Python，计算并返回 DataFrame，
    包含 INDICATOR, INDICATOR1, INDICATOR2, nordates, fbnordates, jddate, jdhema 等列。
    """
    df = df.copy()
    ta = Ta()

    # 计算 INDICATOR 系列
    def get_ind(series: pd.Series, length: int) -> pd.Series:
        if ind == 'RSI':
            return series.rolling(window=length).apply(lambda x: ta.rsi(list(x), length))
        elif ind == 'CCI':
            sma = series.rolling(window=length).mean()
            mad = series.rolling(window=length).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            return (series - sma) / (0.015 * mad)
        elif ind == 'MFI':
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            pos_mf = positive_flow.rolling(window=length).sum()
            neg_mf = negative_flow.rolling(window=length).sum()
            mfr = pos_mf / neg_mf
            return 100 - (100 / (1 + mfr))
        elif ind == 'STOCHASTIC':
            lowest_low = df['low'].rolling(window=length).min()
            highest_high = df['high'].rolling(window=length).max()
            return 100 * (series - lowest_low) / (highest_high - lowest_low)
        elif ind == 'CMF':
            ad = ((2 * df['close'] - df['low'] - df['high']) /
                  (df['high'] - df['low']).replace(0, np.nan) * df['volume']).fillna(0)
            return ad.rolling(window=length).sum() / df['volume'].rolling(window=length).sum()
        elif ind == 'CMO':
            change = series.diff()
            gain = change.where(change > 0, 0)
            loss = (-change).where(change < 0, 0)
            sum_gain = gain.rolling(window=length).sum()
            sum_loss = loss.rolling(window=length).sum()
            return 100 * (sum_gain - sum_loss) / (sum_gain + sum_loss)
        else:
            return pd.Series(0.0, index=df.index)

    df['INDICATOR'] = get_ind(df['close'], length).round(2)
    df['INDICATOR1'] = get_ind(df['high'], length).round(2)
    df['INDICATOR2'] = get_ind(df['low'], length).round(2)

    # 成交量合成 (AD, MF)
    df['AD'] = np.where(df['high'] == df['low'], 0,
                         (2 * df['close'] - df['low'] - df['high']) /
                         (df['high'] - df['low']).replace(0, np.nan) * df['volume'])
    df['AD'] = df['AD'].fillna(0)
    df['MF'] = df['AD'].rolling(window=length).sum() / df['volume'].rolling(window=length).sum()

    # 成交量 RSI
    vol = df['volume']
    vol_change = vol.diff()
    vol_gain = vol_change.where(vol_change > 0, 0)
    vol_loss = (-vol_change).where(vol_change < 0, 0)
    avg_vol_gain = vol_gain.rolling(window=length).mean()
    avg_vol_loss = vol_loss.rolling(window=length).mean()
    vol_rs = avg_vol_gain / avg_vol_loss
    df['VOLUME_RSI'] = (100 - 100 / (1 + vol_rs)).fillna(0)
    df['norvolume'] = normalize(df['VOLUME_RSI'].round(2), 21)

    # 计算 bodx, emay, ema10, longnum, shortnum
    df['bodx'] = df['volume'] / (df['high'] + df['close'] - df['low'] * 2).replace(0, np.nan) / df['low'].replace(0, np.nan) * 100
    df['bodx'] = df['bodx'].replace([np.inf, -np.inf], 0).fillna(0)
    df['emay'] = df['bodx'].ewm(span=588, adjust=False).mean()
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['longnum'] = 0.0
    df['shortnum'] = 0.0
    for i in range(1, len(df)):
        if df.at[i, 'emay'] > df.at[i, 'bodx'] * 2:
            if df.at[i, 'close'] >= df.at[i, 'ema10']:
                df.at[i, 'longnum'] = df.at[i, 'high']
            else:
                df.at[i, 'shortnum'] = df.at[i, 'low']
        else:
            df.at[i, 'longnum'] = df.at[i-1, 'longnum']
            df.at[i, 'shortnum'] = df.at[i-1, 'shortnum']

    df['bodxp'] = df['bodx'] / df['emay'].replace(0, np.nan)
    df['bodxpx'] = df['bodxp'] * midl
    df['PLbo'] = df['bodxpx'].where(df['bodxpx'] < 100, 100)
    df['norplbo'] = normalize(df['bodxp'], 21)

    # Holt 指数移动平均 (HEMA)
    df['hemahigh'] = f_hema_series(df['high'], 20, 20)
    df['hemaclose'] = f_hema_series(df['close'], 20, 20)
    df['hemalow'] = f_hema_series(df['low'], 20, 20)
    df['fblqprice'] = 1 - (df['close'] - df['hemalow']) / (df['hemahigh'] - df['hemalow']).replace(0, np.nan)
    df[['norhhigh', 'norhclose', 'norhlow']] = df[['hemahigh', 'hemaclose', 'hemalow']].apply(lambda x: normalize(x, 21))
    df['norhigh'] = normalize(df['INDICATOR1'], 21)
    df['norclose'] = normalize(df['INDICATOR'], 21)
    df['norlow'] = normalize(df['INDICATOR2'], 21)

    # 组合值
    df['nordates'] = (df['norvolume'] + df['norplbo'] + df['norhigh'] +
                       df['norclose'] + df['norlow'] + df['norhhigh'] + df['norhclose'] + df['norhlow'])
    df['fbnordates'] = df['nordates'] + df['fblqprice']
    df['jddate'] = (df['nordates'] + df['fbnordates']) / 2
    df['jdhema'] = f_hema_series(df['jddate'], 35, 28)

    # 超买/超卖/中间线为常数，可用于绘图时参考
    df['overbought'] = ovrb
    df['oversold'] = ovrs
    df['middle'] = midl

    return df


def normalize(series: pd.Series, length: int) -> pd.Series:
    """将 series 归一化到 [0,1] 区间"""
    min_val = series.rolling(window=length, min_periods=1).min()
    max_val = series.rolling(window=length, min_periods=1).max()
    range_val = max_val - min_val
    return ((series - min_val) / range_val).fillna(0.5)


def f_hema_series(series: pd.Series, alpha_length: int, gamma_length: int) -> pd.Series:
    """对 Series 批量计算 HEMA，返回同长度 Series"""
    alpha = 2.0 / float(alpha_length + 1)
    gamma = 2.0 / float(gamma_length + 1)
    hema_vals = [series.iloc[0]]
    b_vals = [0.0]
    for i in range(1, len(series)):
        prev_hema = hema_vals[-1]
        prev_b = b_vals[-1]
        src_sum = prev_hema + prev_b
        new_hema = (1.0 - alpha) * src_sum + alpha * series.iat[i]
        new_b = (1.0 - gamma) * prev_b + gamma * (new_hema - prev_hema)
        hema_vals.append(new_hema)
        b_vals.append(new_b)
    return pd.Series(hema_vals, index=series.index)

# 注意：本函数依赖 pandas DataFrame，使用前请确保 df 包含 'high','low','close','volume' 等列。
