import math
from typing import List, Optional, Any


class Ta:
    @staticmethod
    def ema_series(prices: List[float], length: int) -> List[float]:
        """
        计算滑动窗口中的 EMA 序列。对于 prices 中每一个起点，取长度为 length 的子数组计算 EMA，
        返回所有窗口最后一个 EMA 值的列表。
        """
        # 过滤掉 NaN
        valid_prices = [p for p in prices if not math.isnan(p)]

        if length <= 0 or len(valid_prices) < length:
            print("数据长度不足以计算 EMA")
            return []

        ema_values: List[float] = []
        n = len(valid_prices)

        for start_idx in range(0, n - length + 1):
            sub = valid_prices[start_idx : start_idx + length]
            ema_val = Ta.ema(sub, length)
            if ema_val is not None:
                ema_values.append(ema_val)

        return ema_values

    @staticmethod
    def ema(prices: List[float], length: int) -> Optional[float]:
        """
        计算输入 prices（长度 >= length） 的 EMA，返回最后一个 EMA 值。
        """
        if len(prices) < length:
            return None

        multiplier = 2.0 / (length + 1)
        prev_ema = prices[0]

        for i in range(1, len(prices)):
            curr_ema = (prices[i] - prev_ema) * multiplier + prev_ema
            prev_ema = curr_ema

        return prev_ema

    @staticmethod
    def rsi_series(prices: List[float], length: int) -> List[float]:
        """
        计算滑动窗口中的 RSI 序列。对于 prices 中每一个起点，取长度为 length 的子数组计算 RSI，
        返回所有窗口最后一个 RSI 值的列表。
        """
        valid_prices = [p for p in prices if not math.isnan(p)]

        if length <= 0 or len(valid_prices) < length:
            print("数据长度不足以计算 RSI")
            return []

        rsi_values: List[float] = []
        n = len(valid_prices)

        for start_idx in range(0, n - length + 1):
            sub = valid_prices[start_idx : start_idx + length]
            rsi_val = Ta.rsi(sub, length)
            if rsi_val is not None:
                rsi_values.append(rsi_val)

        return rsi_values

    @staticmethod
    def rsi(prices: List[float], length: int) -> Optional[float]:
        """
        计算输入 prices（长度 >= length） 的最后一个 RSI 值。
        """
        if len(prices) < length:
            return None

        gains: List[float] = []
        losses: List[float] = []

        # 计算每个周期的涨跌
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(-change)

        # 初始平均涨跌（注意：gains/losses 长度为 len(prices)-1，但这里除以 length）
        avg_gain = sum(gains) / float(length)
        avg_loss = sum(losses) / float(length)

        # 初始 RS 和 RSI
        rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        rsi_val = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + rs))

        # 从第 length 个索引开始继续平滑计算
        for i in range(length, len(prices)):
            change = prices[i] - prices[i - 1]
            gain = change if change > 0 else 0.0
            loss = -change if change < 0 else 0.0

            avg_gain = (avg_gain * (length - 1) + gain) / float(length)
            avg_loss = (avg_loss * (length - 1) + loss) / float(length)

            rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
            rsi_val = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + rs))

        return rsi_val

    @staticmethod
    def hema_series(prices: List[float], alpha_length: int = 20, gamma_length: int = 20) -> List[float]:
        """
        计算滑动窗口中的 HEMA 序列。取每个窗口大小为 max(alpha_length, gamma_length) 的子数组，
        计算其最后一个 HEMA 值。
        """
        min_len = max(alpha_length, gamma_length)
        if len(prices) < min_len:
            print("数据长度不足以计算 HEMA")
            return []

        hema_values: List[float] = []
        n = len(prices)

        for start_idx in range(0, n - min_len + 1):
            sub = prices[start_idx : start_idx + min_len]
            hema_val = Ta.hema(sub, alpha_length, gamma_length)
            if hema_val is not None:
                hema_values.append(hema_val)

        return hema_values

    @staticmethod
    def hema(prices: List[float], alpha_length: int = 20, gamma_length: int = 20) -> Optional[float]:
        """
        计算输入 prices（长度 >= alpha_length）的 HEMA 并返回最后一个值。
        """
        if len(prices) < alpha_length:
            print("数据长度不足以计算 HEMA")
            return None

        alpha = 2.0 / float(alpha_length + 1)
        gamma = 2.0 / float(gamma_length + 1)

        b: List[float] = [prices[0]]
        hema_arr: List[float] = [prices[0]]

        def nz(val: Optional[float]) -> float:
            return val if val is not None else 0.0

        for i in range(1, len(prices)):
            prev_hema = nz(hema_arr[-1])
            prev_b = nz(b[-1])
            src_sum = prev_hema + prev_b
            new_hema = (1.0 - alpha) * src_sum + alpha * prices[i]
            new_b = (1.0 - gamma) * prev_b + gamma * (new_hema - prev_hema)

            hema_arr.append(new_hema)
            b.append(new_b)

        return hema_arr[-1]

    @staticmethod
    def mema(prices: Any, length: int) -> List[float]:
        """
        相当于 Swift 里的 mema：先做一个 EMA，然后如果结果长度大于 2，就删掉第一个元素。
        输入 prices 如果不是列表，返回 [0.0]。
        """
        if not isinstance(prices, list) or any(not isinstance(x, (int, float)) for x in prices):
            print("不支持的类型")
            return [0.0]

        prices_arr: List[float] = [float(x) for x in prices]
        n = len(prices_arr)

        if n < length:
            print("数据长度不足以计算 EMA")
            return [0.0]

        ema_arr: List[float] = [prices_arr[0]]

        for i in range(1, n):
            current_price = prices_arr[i]
            previous_ema = ema_arr[i - 1]
            weighted_price = 2.0 * current_price
            weighted_ema = float(length - 1) * previous_ema
            numerator = weighted_price + weighted_ema
            denominator = float(length + 1)
            new_ema = numerator / denominator
            ema_arr.append(new_ema)

        if len(ema_arr) > 2:
            # 删除第一个元素
            ema_arr.pop(0)

        return ema_arr

    @staticmethod
    def mlianjia(
        highs: Any, lows: Any, closes: Any, volumes: Any
    ) -> Optional[List[float]]:
        """
        对应 Swift 中的 mlianjia：给定 highs, lows, closes, volumes，计算预测价格并返回 [predictedPrice]，
        否则返回 None。
        """
        if not (
            isinstance(highs, list)
            and isinstance(lows, list)
            and isinstance(closes, list)
            and isinstance(volumes, list)
        ):
            print("不支持的类型")
            return None

        try:
            highs_arr = [float(x) for x in highs]
            lows_arr = [float(x) for x in lows]
            closes_arr = [float(x) for x in closes]
            volumes_arr = [float(x) for x in volumes]
        except Exception:
            print("不支持的类型")
            return None

        size = len(highs_arr)
        if not (len(lows_arr) == size == len(closes_arr) == len(volumes_arr)):
            print("数组长度不一致")
            return None

        # 计算波动线 bodxp
        bodxp: List[float] = []
        for i in range(size):
            denom = (highs_arr[i] + closes_arr[i] - lows_arr[i] * 2)
            if denom == 0 or lows_arr[i] == 0:
                bodxp.append(0.0)
            else:
                val = volumes_arr[i] / denom / lows_arr[i] * 100.0
                bodxp.append(val)

        # 计算波动线平均
        emay_arr = Ta.mema(bodxp, 144)
        emay = emay_arr[-1] if len(emay_arr) > 0 else 0.0

        # 计算收盘线平均
        emas_arr = Ta.mema(closes_arr, 12)
        emas = emas_arr[-1] if len(emas_arr) > 0 else 0.0

        # 深拷贝数组
        imhigh = highs_arr.copy()
        imlow = lows_arr.copy()
        imclose = closes_arr.copy()

        # 条件过滤
        for i in range(size):
            if not (emay > bodxp[i] * 2) or not (closes_arr[i] < emas):
                imhigh[i] = 0.0
            if not (emay > bodxp[i] * 2) or not (closes_arr[i] > emas):
                imlow[i] = 0.0
            if not (emay < bodxp[i] * 0.5):
                imclose[i] = 0.0

        # 计算多层最大值
        shurtnums: List[float] = []
        for i in range(size):
            shurtnums.append(max(imhigh[i], imlow[i]))

        shurtnums1: List[float] = []
        for i in range(size):
            shurtnums1.append(max(shurtnums[i], imclose[i]))

        shurtnums2: List[float] = []
        for i in range(size):
            shurtnums2.append(max(lows_arr[i], shurtnums1[i]))

        shortnum_arr = Ta.mema(shurtnums2, len(bodxp))
        shortnum = shortnum_arr[-1] if len(shortnum_arr) > 0 else 0.0

        # 计算 RSI
        rsi_arr = Ta.mrsi(closes_arr, 14)
        rsivalue = rsi_arr[-1] if len(rsi_arr) > 0 else 0.0

        # 计算权重与预测价格
        if shortnum == 0:
            # 避免除以 0
            return [0.0]

        weight = 1.0 / shortnum
        adjusted_weight = weight * (1.0 + rsivalue / 100.0)
        sum_weights = adjusted_weight
        weighted_sum = adjusted_weight * shortnum

        predicted_price = weighted_sum / sum_weights if sum_weights != 0 else 0.0
        return [predicted_price]

    @staticmethod
    def mhema(prices: Any, alpha_length: int = 20, gamma_length: int = 20) -> List[float]:
        """
        对应 Swift 中的 mhema。返回所有时点上的 HEMA 值列表。
        """
        if not isinstance(prices, list) or any(not isinstance(x, (int, float)) for x in prices):
            print("不支持的类型")
            return []

        prices_arr: List[float] = [float(x) for x in prices]
        n = len(prices_arr)
        if n < alpha_length:
            print("数据长度不足以计算 HEMA")
            return []

        alpha = 2.0 / float(alpha_length + 1)
        gamma = 2.0 / float(gamma_length + 1)

        b: List[float] = [prices_arr[0]]
        hema_arr: List[float] = [prices_arr[0]]

        for i in range(1, n):
            prev_b = hema_arr[-1]  # 注意：这里与 ema 不同，用 hema_arr 上一个元素作为 nzhema
            prev_hema = b[-1]     # 与 Swift 保持一致，nzhema 与 nzb 的顺序调换
            src_sum = prev_b + prev_hema
            new_hema = (1.0 - alpha) * src_sum + alpha * prices_arr[i]
            new_b = (1.0 - gamma) * prev_hema + gamma * (new_hema - prev_b)

            hema_arr.append(new_hema)
            b.append(new_b)

        return hema_arr

    @staticmethod
    def nz(value: Optional[float], default_value: float = 0.0) -> float:
        """
        如果 value 不是 None，就返回 value，否则返回 default_value。
        """
        return value if value is not None else default_value

    @staticmethod
    def mrsi(prices: Any, length: int) -> List[float]:
        """
        对应 Swift 中的 mrsi。返回长度为 len(prices)-length 的 RSI 序列。
        """
        if not isinstance(prices, list) or any(not isinstance(x, (int, float)) for x in prices):
            print("不支持的类型")
            return []

        prices_arr: List[float] = [float(x) for x in prices]
        n = len(prices_arr)
        if n < length + 1:
            print("数据长度不足以计算 RSI")
            return []

        gains: List[float] = []
        losses: List[float] = []

        # 计算初始涨跌
        for i in range(1, n):
            change = prices_arr[i] - prices_arr[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(-change)

        # 第一个 RSI
        rs_avg_gain = sum(gains[0:length]) / float(length)
        rs_avg_loss = sum(losses[0:length]) / float(length)
        rs = rs_avg_gain / rs_avg_loss if rs_avg_loss != 0 else float('inf')
        first_rsi = 100.0 if rs_avg_loss == 0 else 100.0 - (100.0 / (1.0 + rs))

        rsi_values: List[float] = [first_rsi]

        # 后续 RSI
        for i in range(length, n - 1):
            rs_avg_gain = (rs_avg_gain * (length - 1) + gains[i]) / float(length)
            rs_avg_loss = (rs_avg_loss * (length - 1) + losses[i]) / float(length)

            rs = rs_avg_gain / rs_avg_loss if rs_avg_loss != 0 else float('inf')
            curr_rsi = 100.0 if rs_avg_loss == 0 else 100.0 - (100.0 / (1.0 + rs))
            rsi_values.append(curr_rsi)

        return rsi_values

    @staticmethod
    def mone(array: List[float], return_type: Any = float) -> Any:
        """
        对应 Swift 中的 mone。尝试将 array 的最后一个值转换成 return_type，
        如果 array 为空或者转换失败，就返回 return_type(0)。
        """
        if not array:
            try:
                return return_type(0)
            except Exception:
                return 0  # 强制回退到 0

        last_val = array[-1]
        try:
            return return_type(last_val)
        except Exception:
            # 尝试用整数转换
            try:
                return return_type(int(last_val))
            except Exception:
                return return_type(0)

