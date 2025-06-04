import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class FibonacciPositionManager:
    """
    基于斐波那契水平的资金管理模块
    将highbi和lowbi分类结果转换为具体的仓位管理建议
    """
    
    def __init__(
        self,
        base_capital: float = 10000.0,
        max_risk_per_trade: float = 0.03,
        risk_levels: Dict[str, float] = None
    ):
        """
        初始化斐波那契资金管理器
        
        参数:
            base_capital: 基础资金量
            max_risk_per_trade: 单笔交易最大风险比例
            risk_levels: 自定义风险等级字典，格式为 {市场状态: 风险系数}
        """
        self.base_capital = base_capital
        self.max_risk_per_trade = max_risk_per_trade
        
        # 默认风险等级设置
        self.risk_levels = {
            "顺势": 1.0,      # 100%风险
            "正常": 0.67,     # 67%风险
            "逆势": 0.33,     # 33%风险
            "极端逆势": 0.17,  # 17%风险
        } if risk_levels is None else risk_levels
        
        # 斐波那契水平分类映射到市场状态
        self.fib_to_market_state = {
            "100%": "顺势",
            "50%": "正常",
            "44%": "正常",
            "38.2%": "正常",
            "25%": "逆势",
            "23.6%": "逆势",
            "14%": "逆势",
            "10%": "逆势",
            "8%": "极端逆势",
            "6%": "极端逆势",
            "3%": "极端逆势",
            "2%": "极端逆势",
            "1%": "极端逆势",
            "6U": "极端逆势",  # 默认极端情况
        }
        
        # 资金百分比映射
        self.market_state_to_capital_pct = {
            "顺势": 0.03,    # 3%资金
            "正常": 0.02,    # 2%资金
            "逆势": 0.01,    # 1%资金
            "极端逆势": 0.005  # 0.5%资金
        }
    
    def get_position_size(self, fib_level: str, price: float) -> Dict[str, Union[float, str, int]]:
        """
        根据斐波那契水平和当前价格计算仓位大小
        
        参数:
            fib_level: 斐波那契水平分类结果，如"100%", "50%"等
            price: 当前价格
            
        返回:
            包含仓位信息的字典
        """
        # 获取市场状态
        market_state = self.fib_to_market_state.get(fib_level, "极端逆势")
        
        # 获取风险系数
        risk_factor = self.risk_levels[market_state]
        
        # 计算资金百分比
        capital_pct = self.market_state_to_capital_pct[market_state]
        
        # 计算实际投入资金
        capital_amount = self.base_capital * capital_pct * risk_factor
        
        # 计算仓位数量
        position_size = capital_amount / price if price > 0 else 0
        
        return {
            "market_state": market_state,
            "risk_factor": risk_factor,
            "capital_percentage": capital_pct * 100,
            "capital_amount": capital_amount,
            "position_size": position_size,
            "recommendation": self._get_recommendation(market_state)
        }
    
    def _get_recommendation(self, market_state: str) -> str:
        """生成基于市场状态的操作建议"""
        recommendations = {
            "顺势": "积极建仓，利用趋势优势",
            "正常": "适度参与，谨慎持仓",
            "逆势": "轻仓观望，防守操作",
            "极端逆势": "仅保留少量仓位或全部观望"
        }
        return recommendations.get(market_state, "观望")
    
    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        分析DataFrame中的highbi和lowbi列，添加资金管理建议
        
        参数:
            df: 包含highbi和lowbi列的DataFrame
            
        返回:
            添加了资金管理建议的DataFrame
        """
        if 'highbi' not in df.columns or 'lowbi' not in df.columns or 'close' not in df.columns:
            raise ValueError("DataFrame必须包含'highbi', 'lowbi'和'close'列")
        
        # 创建结果列
        df['market_state'] = 'unknown'
        df['position_pct'] = 0.0
        df['recommendation'] = ''
        
        for idx, row in df.iterrows():
            # 优先使用highbi，如果为"6U"则使用lowbi
            fib_level = row['highbi'] if row['highbi'] != "6U" else row['lowbi']
            position_info = self.get_position_size(fib_level, row['close'])
            
            df.at[idx, 'market_state'] = position_info['market_state']
            df.at[idx, 'position_pct'] = position_info['capital_percentage']
            df.at[idx, 'recommendation'] = position_info['recommendation']
        
        return df
    
    def get_stop_loss_take_profit(
        self, 
        fib_level: str, 
        entry_price: float, 
        is_long: bool = True
    ) -> Tuple[float, float]:
        """
        根据斐波那契水平计算止损和止盈价格
        
        参数:
            fib_level: 斐波那契水平分类
            entry_price: 入场价格
            is_long: 是否为多头仓位
            
        返回:
            (止损价格, 止盈价格)的元组
        """
        # 市场状态对应的风险回报比
        risk_reward_ratios = {
            "顺势": 3.0,  # 风险回报比1:3
            "正常": 2.0,  # 风险回报比1:2
            "逆势": 1.5,  # 风险回报比1:1.5
            "极端逆势": 1.0  # 风险回报比1:1
        }
        
        # 市场状态对应的止损百分比
        stop_loss_pcts = {
            "顺势": 0.02,  # 2%止损
            "正常": 0.015, # 1.5%止损
            "逆势": 0.01,  # 1%止损
            "极端逆势": 0.005  # 0.5%止损
        }
        
        market_state = self.fib_to_market_state.get(fib_level, "极端逆势")
        stop_loss_pct = stop_loss_pcts[market_state]
        risk_reward_ratio = risk_reward_ratios[market_state]
        
        if is_long:
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + stop_loss_pct * risk_reward_ratio)
        else:
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - stop_loss_pct * risk_reward_ratio)
        
        return stop_loss, take_profit
    
    def adjust_position_on_signal(
        self, 
        current_position: float, 
        fib_level: str, 
        price: float, 
        signal: int
    ) -> Dict[str, Union[float, str]]:
        """
        根据信号调整仓位
        
        参数:
            current_position: 当前持仓数量
            fib_level: 斐波那契水平分类
            price: 当前价格
            signal: 信号，1为买入，0为持有，-1为卖出
            
        返回:
            调整后的仓位信息
        """
        position_info = self.get_position_size(fib_level, price)
        target_position = position_info['position_size']
        
        if signal == 0:  # 持有信号
            return {
                "action": "HOLD",
                "current_position": current_position,
                "target_position": current_position,
                "adjustment": 0,
                "market_state": position_info['market_state'],
                "recommendation": "保持当前仓位"
            }
        
        elif signal == 1:  # 买入信号
            if current_position <= 0:
                # 空仓或空头，转为多头
                adjustment = target_position
                action = "BUY_TO_OPEN"
            elif current_position < target_position:
                # 已有多头，但仓位小于目标仓位
                adjustment = target_position - current_position
                action = "BUY_TO_ADD"
            else:
                # 已有足够仓位
                adjustment = 0
                action = "HOLD"
            
            return {
                "action": action,
                "current_position": current_position,
                "target_position": target_position,
                "adjustment": adjustment,
                "market_state": position_info['market_state'],
                "recommendation": position_info['recommendation']
            }
            
        elif signal == -1:  # 卖出信号
            if current_position >= 0:
                # 多头或空仓，转为空头
                adjustment = -target_position - current_position
                action = "SELL_TO_OPEN"
            elif abs(current_position) < target_position:
                # 已有空头，但仓位小于目标仓位
                adjustment = -(target_position + current_position)
                action = "SELL_TO_ADD"
            else:
                # 已有足够仓位
                adjustment = 0
                action = "HOLD"
            
            return {
                "action": action,
                "current_position": current_position,
                "target_position": -target_position,
                "adjustment": adjustment,
                "market_state": position_info['market_state'],
                "recommendation": position_info['recommendation']
            }
        
        return {
            "action": "UNKNOWN",
            "current_position": current_position,
            "target_position": current_position,
            "adjustment": 0,
            "market_state": "unknown",
            "recommendation": "无效信号"
        }


# 使用示例
if __name__ == "__main__":
    # 创建资金管理器实例
    position_manager = FibonacciPositionManager(base_capital=10000.0)
    
    # 获取仓位建议
    position_info = position_manager.get_position_size("100%", 100.0)
    print(f"市场状态: {position_info['market_state']}")
    print(f"资金百分比: {position_info['capital_percentage']}%")
    print(f"资金金额: ${position_info['capital_amount']}")
    print(f"仓位大小: {position_info['position_size']} 单位")
    print(f"建议: {position_info['recommendation']}")
    
    # 计算止损止盈
    stop_loss, take_profit = position_manager.get_stop_loss_take_profit("100%", 100.0, True)
    print(f"止损价: ${stop_loss}")
    print(f"止盈价: ${take_profit}")
    
    # 根据信号调整仓位
    adjustment = position_manager.adjust_position_on_signal(0, "100%", 100.0, 1)
    print(f"操作: {adjustment['action']}")
    print(f"调整量: {adjustment['adjustment']} 单位") 