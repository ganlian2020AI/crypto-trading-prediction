import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
import json

# 添加项目根目录到sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from common.utils import TradingLogger, DataHandler, ReportGenerator, load_config


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, config_path: str = None, config: dict = None):
        """
        初始化回测引擎
        
        参数:
            config_path: 配置文件路径
            config: 配置字典，如果提供则优先使用
        """
        # 加载配置
        if config:
            self.config = config
        elif config_path:
            self.config = load_config(config_path)
        else:
            config_path = os.path.join(ROOT_DIR, 'config', 'trading_config.yaml')
            self.config = load_config(config_path)
        
        # 初始化日志
        log_level = self.config.get('general', {}).get('log_level', 'INFO')
        log_file = os.path.join(ROOT_DIR, 'logs', 'backtest.log')
        self.logger = TradingLogger('backtest', log_level, log_file)
        
        # 初始化数据处理器
        self.data_handler = DataHandler(self.config, self.logger)
        
        # 初始化报告生成器
        self.report_generator = ReportGenerator(self.config, self.logger)
        
        # 回测参数
        self.simulation_config = self.config.get('simulation', {})
        self.initial_capital = self.simulation_config.get('initial_capital', 10000)
        self.start_date = self.simulation_config.get('start_date', '2023-01-01')
        self.end_date = self.simulation_config.get('end_date', 'auto')
        self.include_fees = self.simulation_config.get('include_fees', True)
        self.fee_rate = self.simulation_config.get('fee_rate', 0.001)
        self.include_slippage = self.simulation_config.get('include_slippage', True)
        self.slippage_rate = self.simulation_config.get('slippage_rate', 0.0005)
        
        # 交易品种配置
        self.symbols_config = self.config.get('symbols', [])
        self.symbols = [s['name'] for s in self.symbols_config if s.get('enabled', True)]
        
        # 回测状态
        self.data = {}  # 存储每个品种的历史数据
        self.positions = {}  # 当前持仓
        self.trade_history = []  # 交易历史
        self.equity_curve = None  # 权益曲线
        self.performance_metrics = {}  # 性能指标
        
        # 初始化资金分配
        self.capital_per_symbol = self.initial_capital / len(self.symbols) if self.symbols else self.initial_capital
        
        self.logger.info(f"回测引擎初始化完成，交易品种: {', '.join(self.symbols)}")
    
    def load_data(self, custom_symbols: List[str] = None, start_date: str = None, end_date: str = None):
        """
        加载历史数据
        
        参数:
            custom_symbols: 自定义交易品种列表，如果为None则使用配置中的品种
            start_date: 开始日期，如果为None则使用配置中的日期
            end_date: 结束日期，如果为None则使用配置中的日期
        """
        symbols_to_load = custom_symbols if custom_symbols is not None else self.symbols
        start_date_to_use = start_date if start_date is not None else self.start_date
        end_date_to_use = end_date if end_date is not None else self.end_date
        
        self.logger.info(f"加载历史数据: {', '.join(symbols_to_load)}, 时间范围: {start_date_to_use} 至 {end_date_to_use}")
        
        # 清空之前的数据
        self.data = {}
        
        # 加载每个品种的数据
        for symbol in symbols_to_load:
            df = self.data_handler.load_historical_data(symbol, start_date_to_use, end_date_to_use)
            if not df.empty:
                # 预处理数据
                df = self.data_handler.preprocess_data(df)
                self.data[symbol] = df
            else:
                self.logger.warning(f"无法加载{symbol}的数据，将跳过该品种")
        
        if not self.data:
            self.logger.error("没有成功加载任何数据，回测无法进行")
            return False
        
        # 对齐多品种数据
        if len(self.data) > 1:
            self.aligned_data = self.data_handler.align_multi_symbol_data(self.data)
            self.logger.info(f"多品种数据对齐完成，形状: {self.aligned_data.shape}")
        else:
            # 单品种情况，不需要对齐
            symbol = list(self.data.keys())[0]
            self.aligned_data = self.data[symbol].copy()
            self.logger.info(f"单品种数据加载完成，形状: {self.aligned_data.shape}")
        
        return True
    
    def load_model_predictions(self, model_path: str):
        """
        加载模型预测结果
        
        参数:
            model_path: 模型路径或预测结果文件路径
        """
        try:
            # 检查文件类型
            if model_path.endswith('.csv'):
                # 直接加载CSV格式的预测结果
                predictions = pd.read_csv(model_path)
                
                # 确保timestamp列存在
                if 'timestamp' not in predictions.columns:
                    self.logger.error("预测结果缺少timestamp列")
                    return False
                
                # 转换timestamp为datetime并设为索引
                predictions['timestamp'] = pd.to_datetime(predictions['timestamp'])
                predictions.set_index('timestamp', inplace=True)
                
                self.predictions = predictions
                self.logger.info(f"已加载预测结果: {len(predictions)}行")
                return True
                
            elif model_path.endswith('.keras') or model_path.endswith('.h5'):
                # 需要使用模型生成预测
                self.logger.info("检测到模型文件，将使用模型生成预测")
                
                # 这里需要导入模型模块，根据项目实际情况调整
                try:
                    # 尝试导入主项目的模型
                    sys.path.append(os.path.dirname(ROOT_DIR))
                    from model import CryptoLSTMModel
                    
                    # 加载模型
                    model = CryptoLSTMModel(os.path.join(ROOT_DIR, 'config', 'trading_config.yaml'))
                    model.load(model_path)
                    
                    # 生成预测
                    # 这里需要根据实际模型的预测方法进行调整
                    # 假设模型需要特征数据作为输入
                    if not hasattr(self, 'aligned_data'):
                        self.logger.error("请先加载历史数据")
                        return False
                    
                    # 假设模型预测方法为predict(X)
                    # 这里需要根据实际模型的预测接口进行调整
                    predictions = model.predict(self.aligned_data)
                    
                    self.predictions = predictions
                    self.logger.info(f"已使用模型生成预测: {len(predictions)}行")
                    return True
                    
                except ImportError:
                    self.logger.error("无法导入模型模块，请确保模型文件可用")
                    return False
                except Exception as e:
                    self.logger.error(f"使用模型生成预测失败: {str(e)}")
                    return False
            else:
                self.logger.error(f"不支持的文件类型: {model_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"加载预测结果失败: {str(e)}")
            return False
    
    def run_backtest(self, custom_initial_capital: float = None, 
                    custom_symbols: List[str] = None,
                    start_date: str = None, 
                    end_date: str = None):
        """
        运行回测
        
        参数:
            custom_initial_capital: 自定义初始资金
            custom_symbols: 自定义交易品种列表
            start_date: 开始日期
            end_date: 结束日期
        
        返回:
            回测结果字典
        """
        # 使用自定义参数或默认参数
        if custom_initial_capital is not None:
            self.initial_capital = custom_initial_capital
            
        symbols_to_use = custom_symbols if custom_symbols is not None else self.symbols
        
        # 重新加载数据如果需要
        if start_date != self.start_date or end_date != self.end_date or custom_symbols is not None:
            self.load_data(custom_symbols, start_date, end_date)
        
        # 检查数据是否已加载
        if not hasattr(self, 'aligned_data') or self.aligned_data.empty:
            self.logger.error("没有可用的数据，请先加载数据")
            return None
        
        # 检查是否有预测结果
        if not hasattr(self, 'predictions') or self.predictions is None:
            self.logger.error("没有预测结果，请先加载模型预测")
            return None
        
        self.logger.info(f"开始回测: 初始资金={self.initial_capital}, 品种={', '.join(symbols_to_use)}")
        
        # 初始化回测状态
        self.positions = {symbol: [] for symbol in symbols_to_use}  # 每个品种的持仓列表
        self.trade_history = []  # 交易历史
        
        # 初始化权益曲线DataFrame
        equity_data = pd.DataFrame(index=self.aligned_data.index)
        
        # 为每个品种分配初始资金
        capital_per_symbol = self.initial_capital / len(symbols_to_use)
        symbol_equity = {symbol: capital_per_symbol for symbol in symbols_to_use}
        
        # 记录每个时间点的权益
        for timestamp in self.aligned_data.index:
            # 处理每个品种
            for symbol in symbols_to_use:
                # 获取当前价格
                if f"{symbol}_close" in self.aligned_data.columns:
                    current_price = self.aligned_data.loc[timestamp, f"{symbol}_close"]
                else:
                    # 如果没有对应的价格列，跳过
                    continue
                
                # 获取预测信号
                if f"{symbol}_signal" in self.predictions.columns and timestamp in self.predictions.index:
                    signal = self.predictions.loc[timestamp, f"{symbol}_signal"]
                else:
                    # 如果没有对应的信号，使用0（不操作）
                    signal = 0
                
                # 处理交易信号
                self._process_signal(symbol, timestamp, current_price, signal, symbol_equity)
                
                # 更新持仓价值
                position_value = 0
                for position in self.positions[symbol]:
                    position_value += position['quantity'] * current_price
                
                # 更新权益
                cash = symbol_equity[symbol]
                equity_data.loc[timestamp, f"{symbol}_equity"] = cash + position_value
                equity_data.loc[timestamp, f"{symbol}_cash"] = cash
                equity_data.loc[timestamp, f"{symbol}_position_value"] = position_value
        
        # 计算总体权益
        equity_data['total_equity'] = sum(equity_data[f"{symbol}_equity"] for symbol in symbols_to_use)
        equity_data['total_cash'] = sum(equity_data[f"{symbol}_cash"] for symbol in symbols_to_use)
        equity_data['total_position_value'] = sum(equity_data[f"{symbol}_position_value"] for symbol in symbols_to_use)
        
        self.equity_curve = equity_data
        
        # 计算性能指标
        self.performance_metrics = self._calculate_performance_metrics(equity_data, symbols_to_use)
        
        # 转换交易历史为DataFrame
        self.trade_history_df = pd.DataFrame(self.trade_history)
        
        self.logger.info(f"回测完成: 最终权益={self.performance_metrics['final_equity']:.2f}, 总收益率={self.performance_metrics['total_return_pct']:.2f}%")
        
        # 返回回测结果
        return {
            'equity_curve': self.equity_curve,
            'trade_history': self.trade_history_df,
            'performance_metrics': self.performance_metrics,
            'symbols': symbols_to_use,
            'start_date': self.aligned_data.index[0],
            'end_date': self.aligned_data.index[-1]
        }
    
    def _process_signal(self, symbol: str, timestamp, price: float, signal: int, symbol_equity: dict):
        """
        处理交易信号
        
        参数:
            symbol: 交易品种
            timestamp: 时间戳
            price: 当前价格
            signal: 交易信号 (1: 买入, -1: 卖出, 0: 不操作)
            symbol_equity: 每个品种的资金字典
        """
        # 获取品种配置
        symbol_config = next((s for s in self.symbols_config if s['name'] == symbol), None)
        if not symbol_config:
            return
        
        # 获取交易大小
        trade_size_usd = symbol_config.get('trade_size_usd', 100)
        max_position = symbol_config.get('max_position', 3)
        
        # 计算可交易数量
        quantity = trade_size_usd / price
        
        # 添加滑点
        if self.include_slippage:
            if signal > 0:  # 买入
                price *= (1 + self.slippage_rate)
            elif signal < 0:  # 卖出
                price *= (1 - self.slippage_rate)
        
        # 处理买入信号
        if signal > 0 and len(self.positions[symbol]) < max_position:
            # 检查是否有足够的资金
            cost = quantity * price
            if self.include_fees:
                cost *= (1 + self.fee_rate)
            
            if symbol_equity[symbol] >= cost:
                # 执行买入
                self.positions[symbol].append({
                    'entry_time': timestamp,
                    'entry_price': price,
                    'quantity': quantity,
                    'direction': 'long'
                })
                
                # 更新资金
                symbol_equity[symbol] -= cost
                
                # 记录交易
                self.trade_history.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'buy',
                    'price': price,
                    'quantity': quantity,
                    'cost': cost,
                    'fee': cost * self.fee_rate if self.include_fees else 0
                })
        
        # 处理卖出信号
        elif signal < 0 and self.positions[symbol]:
            # 卖出最早的持仓
            position = self.positions[symbol].pop(0)
            
            # 计算收益
            revenue = position['quantity'] * price
            if self.include_fees:
                revenue *= (1 - self.fee_rate)
            
            # 更新资金
            symbol_equity[symbol] += revenue
            
            # 计算盈亏
            entry_cost = position['quantity'] * position['entry_price']
            if self.include_fees:
                entry_cost *= (1 + self.fee_rate)
            
            profit_loss = revenue - entry_cost
            profit_loss_pct = (revenue / entry_cost - 1) * 100
            
            # 记录交易
            self.trade_history.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'sell',
                'price': price,
                'quantity': position['quantity'],
                'revenue': revenue,
                'fee': revenue * self.fee_rate if self.include_fees else 0,
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss_pct,
                'holding_period': (timestamp - position['entry_time']).total_seconds() / 86400  # 持仓天数
            })
    
    def _calculate_performance_metrics(self, equity_data: pd.DataFrame, symbols: List[str]) -> dict:
        """
        计算性能指标
        
        参数:
            equity_data: 权益曲线DataFrame
            symbols: 交易品种列表
            
        返回:
            性能指标字典
        """
        metrics = {}
        
        # 总体指标
        initial_equity = equity_data['total_equity'].iloc[0]
        final_equity = equity_data['total_equity'].iloc[-1]
        
        # 计算收益率
        total_return = final_equity - initial_equity
        total_return_pct = (final_equity / initial_equity - 1) * 100
        
        # 计算年化收益率
        days = (equity_data.index[-1] - equity_data.index[0]).days
        annual_return_pct = (final_equity / initial_equity) ** (365 / max(days, 1)) - 1
        annual_return_pct *= 100
        
        # 计算最大回撤
        cummax = equity_data['total_equity'].cummax()
        drawdown = (equity_data['total_equity'] - cummax) / cummax * 100
        max_drawdown = drawdown.min()
        
        # 计算夏普比率 (假设无风险利率为0)
        daily_returns = equity_data['total_equity'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 1 else 0
        
        # 计算交易统计
        if self.trade_history:
            trade_df = pd.DataFrame(self.trade_history)
            win_trades = trade_df[trade_df['action'] == 'sell']['profit_loss'] > 0
            win_rate = win_trades.sum() / len(win_trades) if len(win_trades) > 0 else 0
            
            profit_trades = trade_df[(trade_df['action'] == 'sell') & (trade_df['profit_loss'] > 0)]
            loss_trades = trade_df[(trade_df['action'] == 'sell') & (trade_df['profit_loss'] <= 0)]
            
            avg_profit = profit_trades['profit_loss'].mean() if len(profit_trades) > 0 else 0
            avg_loss = loss_trades['profit_loss'].mean() if len(loss_trades) > 0 else 0
            
            profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
            
            # 计算每个品种的交易次数
            trades_per_symbol = trade_df.groupby('symbol')['action'].count().to_dict()
        else:
            win_rate = 0
            profit_loss_ratio = 0
            trades_per_symbol = {symbol: 0 for symbol in symbols}
        
        # 汇总指标
        metrics.update({
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'annual_return_pct': annual_return_pct,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_trades': len(self.trade_history) // 2,  # 买入和卖出算一次完整交易
            'trades_per_symbol': trades_per_symbol
        })
        
        # 计算每个品种的指标
        for symbol in symbols:
            if f"{symbol}_equity" in equity_data.columns:
                symbol_initial_equity = equity_data[f"{symbol}_equity"].iloc[0]
                symbol_final_equity = equity_data[f"{symbol}_equity"].iloc[-1]
                
                symbol_return = symbol_final_equity - symbol_initial_equity
                symbol_return_pct = (symbol_final_equity / symbol_initial_equity - 1) * 100
                
                symbol_cummax = equity_data[f"{symbol}_equity"].cummax()
                symbol_drawdown = (equity_data[f"{symbol}_equity"] - symbol_cummax) / symbol_cummax * 100
                symbol_max_drawdown = symbol_drawdown.min()
                
                metrics[f"{symbol}_return"] = symbol_return
                metrics[f"{symbol}_return_pct"] = symbol_return_pct
                metrics[f"{symbol}_max_drawdown_pct"] = symbol_max_drawdown
        
        return metrics
    
    def generate_report(self, output_format: List[str] = None) -> str:
        """
        生成回测报告
        
        参数:
            output_format: 输出格式列表，如果为None则使用配置中的格式
            
        返回:
            报告文件路径
        """
        if not hasattr(self, 'equity_curve') or self.equity_curve is None:
            self.logger.error("没有回测结果，请先运行回测")
            return None
        
        if not hasattr(self, 'trade_history_df'):
            self.trade_history_df = pd.DataFrame(self.trade_history)
        
        # 使用自定义格式或配置中的格式
        formats = output_format if output_format else self.config.get('reporting', {}).get('report_format', ['csv', 'html'])
        
        # 生成报告
        report_path = self.report_generator.generate_trade_report(
            self.trade_history_df,
            self.equity_curve,
            self.performance_metrics,
            self.symbols,
            self.aligned_data.index[0].strftime('%Y-%m-%d'),
            self.aligned_data.index[-1].strftime('%Y-%m-%d')
        )
        
        self.logger.info(f"回测报告已生成: {report_path}")
        return report_path
    
    def plot_equity_curve(self, show: bool = True, save_path: str = None):
        """
        绘制权益曲线
        
        参数:
            show: 是否显示图表
            save_path: 保存路径，如果为None则不保存
        """
        if not hasattr(self, 'equity_curve') or self.equity_curve is None:
            self.logger.error("没有回测结果，请先运行回测")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.index, self.equity_curve['total_equity'], label='总权益')
        
        # 绘制每个品种的权益曲线
        for symbol in self.symbols:
            if f"{symbol}_equity" in self.equity_curve.columns:
                plt.plot(self.equity_curve.index, self.equity_curve[f"{symbol}_equity"], label=f"{symbol}权益", alpha=0.7)
        
        plt.title('回测权益曲线')
        plt.xlabel('日期')
        plt.ylabel('权益')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"权益曲线已保存到: {save_path}")
        
        if show:
            plt.show()
        
        plt.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='加密货币交易系统回测工具')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型文件或预测结果文件路径')
    parser.add_argument('--symbols', type=str, default=None, help='交易品种，用逗号分隔')
    parser.add_argument('--capital', type=float, default=None, help='初始资金')
    parser.add_argument('--start', type=str, default=None, help='开始日期')
    parser.add_argument('--end', type=str, default=None, help='结束日期')
    parser.add_argument('--report', action='store_true', help='生成报告')
    parser.add_argument('--plot', action='store_true', help='绘制权益曲线')
    
    args = parser.parse_args()
    
    # 创建回测引擎
    engine = BacktestEngine(args.config)
    
    # 解析交易品种
    symbols = args.symbols.split(',') if args.symbols else None
    
    # 加载数据
    engine.load_data(symbols, args.start, args.end)
    
    # 加载模型预测
    engine.load_model_predictions(args.model)
    
    # 运行回测
    results = engine.run_backtest(args.capital, symbols, args.start, args.end)
    
    # 生成报告
    if args.report:
        engine.generate_report()
    
    # 绘制权益曲线
    if args.plot:
        engine.plot_equity_curve()
    
    # 打印性能指标
    print("\n===== 回测结果 =====")
    print(f"初始资金: {results['performance_metrics']['initial_equity']:.2f}")
    print(f"最终权益: {results['performance_metrics']['final_equity']:.2f}")
    print(f"总收益率: {results['performance_metrics']['total_return_pct']:.2f}%")
    print(f"年化收益率: {results['performance_metrics']['annual_return_pct']:.2f}%")
    print(f"最大回撤: {results['performance_metrics']['max_drawdown_pct']:.2f}%")
    print(f"夏普比率: {results['performance_metrics']['sharpe_ratio']:.2f}")
    print(f"胜率: {results['performance_metrics']['win_rate']:.2f}")
    print(f"盈亏比: {results['performance_metrics']['profit_loss_ratio']:.2f}")
    print(f"总交易次数: {results['performance_metrics']['total_trades']}")
    print("=====================")


if __name__ == "__main__":
    main()