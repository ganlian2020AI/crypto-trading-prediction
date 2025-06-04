import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import yaml
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
from fibonacci_position_manager import FibonacciPositionManager


class ModelEvaluator:
    """
    模型评估模块：实现模型评估和回测功能
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        初始化评估器
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # 提取评估配置
        self.backtest_config = self.config['backtesting']
        self.metrics_config = self.backtest_config.get('metrics', {})
        
        # 初始化资金管理器
        self.position_manager = FibonacciPositionManager(
            base_capital=self.backtest_config.get('initial_capital', 10000.0)
        )
        
        # 初始化绘图目录
        self.plots_dir = 'evaluation_plots'
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        评估分类性能
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签或概率
            
        返回:
            评估指标字典
        """
        # 如果y_pred是概率，转换为类别
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            y_pred_classes = y_pred
        
        # 计算整体准确率
        accuracy = accuracy_score(y_true, y_pred_classes)
        
        # 计算F1分数
        f1 = f1_score(y_true, y_pred_classes, average='weighted')
        
        # 计算每个类别的准确率
        class_accuracies = {}
        for cls in np.unique(y_true):
            mask = y_true == cls
            if np.sum(mask) > 0:
                class_accuracies[f'class_{int(cls)}_accuracy'] = np.mean(y_pred_classes[mask] == cls)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred_classes)
        
        # 绘制混淆矩阵
        self._plot_confusion_matrix(cm)
        
        # 生成分类报告
        report = classification_report(y_true, y_pred_classes, output_dict=True)
        
        # 合并指标
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            **class_accuracies,
            'classification_report': report
        }
        
        return metrics
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """
        绘制混淆矩阵
        
        参数:
            cm: 混淆矩阵
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.plots_dir, f'confusion_matrix_{timestamp}.png'))
        plt.close()
    
    def backtest(self, 
                df: pd.DataFrame, 
                predictions: np.ndarray, 
                symbols: List[str],
                label_columns: List[str] = None) -> Dict[str, Any]:
        """
        回测交易策略
        
        参数:
            df: 包含价格数据的DataFrame
            predictions: 模型预测结果，形状 (样本数, 标签数)
            symbols: 交易品种列表
            label_columns: 标签列名列表，如果为None则自动生成
            
        返回:
            回测结果字典
        """
        if label_columns is None:
            label_columns = [f"{symbol}_label" for symbol in symbols]
        
        # 检查预测和标签数量是否匹配
        if len(label_columns) != predictions.shape[1]:
            raise ValueError(f"标签列数({len(label_columns)})与预测维度({predictions.shape[1]})不匹配")
        
        # 创建回测结果DataFrame
        backtest_df = df.copy()
        
        # 添加预测结果
        for i, col in enumerate(label_columns):
            backtest_df[f"{col}_pred"] = predictions[:, i]
        
        # 初始化回测变量
        initial_capital = self.backtest_config.get('initial_capital', 10000.0)
        commission = self.backtest_config.get('commission', 0.001)
        slippage = self.backtest_config.get('slippage', 0.0005)
        
        backtest_df['total_capital'] = initial_capital
        backtest_df['cash'] = initial_capital
        backtest_df['holdings'] = 0.0
        backtest_df['returns'] = 0.0
        
        # 为每个品种初始化仓位和交易列
        positions = {symbol: 0.0 for symbol in symbols}
        for symbol in symbols:
            backtest_df[f"{symbol}_position"] = 0.0
            backtest_df[f"{symbol}_trade"] = 0
            backtest_df[f"{symbol}_market_state"] = "unknown"
            backtest_df[f"{symbol}_capital_alloc"] = 0.0
        
        # 执行回测
        for i in range(1, len(backtest_df)):
            current_row = backtest_df.iloc[i]
            prev_row = backtest_df.iloc[i-1]
            
            # 更新总资本（前一行的现金+持仓）
            prev_capital = prev_row['cash'] + prev_row['holdings']
            backtest_df.at[backtest_df.index[i], 'total_capital'] = prev_capital
            backtest_df.at[backtest_df.index[i], 'cash'] = prev_row['cash']
            backtest_df.at[backtest_df.index[i], 'holdings'] = prev_row['holdings']
            
            # 遍历每个交易品种
            for j, symbol in enumerate(symbols):
                # 获取价格
                close_price = current_row[f"{symbol}_close"]
                
                # 获取当前预测和斐波那契水平
                prediction = current_row[f"{symbol}_label_pred"]
                fib_level = current_row[f"{symbol}_highbi"] if f"{symbol}_highbi" in current_row else "6U"
                
                # 获取当前仓位
                current_position = positions[symbol]
                
                # 根据预测生成信号（1=买入，0=持有，-1=卖出）
                signal = prediction
                
                # 使用资金管理器调整仓位
                position_info = self.position_manager.adjust_position_on_signal(
                    current_position=current_position,
                    fib_level=fib_level,
                    price=close_price,
                    signal=signal
                )
                
                # 更新仓位
                new_position = current_position + position_info['adjustment']
                positions[symbol] = new_position
                
                # 更新回测DataFrame
                backtest_df.at[backtest_df.index[i], f"{symbol}_position"] = new_position
                backtest_df.at[backtest_df.index[i], f"{symbol}_trade"] = 1 if position_info['adjustment'] != 0 else 0
                backtest_df.at[backtest_df.index[i], f"{symbol}_market_state"] = position_info['market_state']
                backtest_df.at[backtest_df.index[i], f"{symbol}_capital_alloc"] = position_info.get('capital_amount', 0.0)
                
                # 计算交易成本
                trade_cost = abs(position_info['adjustment']) * close_price * (commission + slippage)
                
                # 更新现金和持仓
                if position_info['adjustment'] != 0:
                    cash_change = -position_info['adjustment'] * close_price - trade_cost
                    backtest_df.at[backtest_df.index[i], 'cash'] += cash_change
                    backtest_df.at[backtest_df.index[i], 'holdings'] += position_info['adjustment'] * close_price
            
            # 计算回报率
            new_capital = backtest_df.at[backtest_df.index[i], 'cash'] + backtest_df.at[backtest_df.index[i], 'holdings']
            backtest_df.at[backtest_df.index[i], 'total_capital'] = new_capital
            backtest_df.at[backtest_df.index[i], 'returns'] = new_capital / prev_capital - 1
        
        # 计算累积回报
        backtest_df['cumulative_returns'] = (1 + backtest_df['returns']).cumprod()
        
        # 计算回测指标
        results = self._calculate_backtest_metrics(backtest_df)
        
        # 绘制回测结果
        self._plot_backtest_results(backtest_df)
        
        # 添加回测DataFrame到结果
        results['backtest_df'] = backtest_df
        
        return results
    
    def _calculate_backtest_metrics(self, backtest_df: pd.DataFrame) -> Dict[str, float]:
        """
        计算回测指标
        
        参数:
            backtest_df: 回测结果DataFrame
            
        返回:
            回测指标字典
        """
        # 总收益率
        total_return = backtest_df['cumulative_returns'].iloc[-1] - 1
        
        # 年化收益率（假设交易日为252天）
        n_days = (backtest_df.index[-1] - backtest_df.index[0]).days
        annual_return = (1 + total_return) ** (365 / max(n_days, 1)) - 1
        
        # 最大回撤
        peak = backtest_df['cumulative_returns'].cummax()
        drawdown = (backtest_df['cumulative_returns'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # 计算Sharpe比率（假设无风险收益率为0，半小时数据，年化）
        returns = backtest_df['returns'].dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(48 * 365)  # 48个半小时 = 1天
        
        # 计算Sortino比率（只考虑负回报的波动率）
        negative_returns = returns[returns < 0]
        sortino_ratio = returns.mean() / negative_returns.std() * np.sqrt(48 * 365) if len(negative_returns) > 0 else np.inf
        
        # 计算胜率
        win_rate = (returns > 0).mean()
        
        # 计算平均盈亏比
        gain = returns[returns > 0].mean() if any(returns > 0) else 0
        loss = abs(returns[returns < 0].mean()) if any(returns < 0) else 0
        profit_loss_ratio = gain / loss if loss != 0 else np.inf
        
        # 计算交易次数
        trade_columns = [col for col in backtest_df.columns if col.endswith('_trade')]
        total_trades = backtest_df[trade_columns].sum().sum()
        
        # 返回指标
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_trades': total_trades
        }
        
        return metrics
    
    def _plot_backtest_results(self, backtest_df: pd.DataFrame):
        """
        绘制回测结果
        
        参数:
            backtest_df: 回测结果DataFrame
        """
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # 绘制累积回报
        backtest_df['cumulative_returns'].plot(ax=axes[0], color='blue')
        axes[0].set_title('累积回报')
        axes[0].set_ylabel('回报倍数')
        axes[0].grid(True)
        
        # 计算回撤
        peak = backtest_df['cumulative_returns'].cummax()
        drawdown = (backtest_df['cumulative_returns'] - peak) / peak
        
        # 绘制回撤
        drawdown.plot(ax=axes[1], color='red')
        axes[1].set_title('回撤')
        axes[1].set_ylabel('回撤比例')
        axes[1].set_ylim(drawdown.min() * 1.1, 0.01)
        axes[1].grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.plots_dir, f'backtest_results_{timestamp}.png'))
        plt.close()
    
    def calculate_comprehensive_score(self, classification_metrics: Dict[str, float], backtest_metrics: Dict[str, float]) -> float:
        """
        计算综合评分
        
        参数:
            classification_metrics: 分类评估指标
            backtest_metrics: 回测评估指标
            
        返回:
            综合评分
        """
        # 获取权重
        accuracy_weight = self.metrics_config.get('accuracy_weight', 0.3)
        f1_weight = self.metrics_config.get('f1_score_weight', 0.2)
        return_weight = self.metrics_config.get('return_weight', 0.3)
        drawdown_weight = self.metrics_config.get('drawdown_weight', 0.2)
        
        # 获取指标
        accuracy = classification_metrics.get('accuracy', 0)
        f1 = classification_metrics.get('f1_score', 0)
        total_return = backtest_metrics.get('total_return', 0)
        max_drawdown = backtest_metrics.get('max_drawdown', 0)
        
        # 计算归一化回报分数（假设目标回报为100%）
        return_score = min(total_return / 1.0, 1.0) if total_return > 0 else 0
        
        # 计算归一化回撤分数（回撤越小越好）
        drawdown_score = 1.0 - min(abs(max_drawdown) / 0.3, 1.0)
        
        # 计算综合得分
        score = (
            accuracy_weight * accuracy +
            f1_weight * f1 +
            return_weight * return_score +
            drawdown_weight * drawdown_score
        )
        
        return score
    
    def generate_evaluation_report(self, 
                                 classification_metrics: Dict[str, float], 
                                 backtest_metrics: Dict[str, float]) -> str:
        """
        生成评估报告
        
        参数:
            classification_metrics: 分类评估指标
            backtest_metrics: 回测评估指标
            
        返回:
            评估报告字符串
        """
        # 计算综合得分
        score = self.calculate_comprehensive_score(classification_metrics, backtest_metrics)
        
        # 创建报告
        report = f"""
        # 模型评估报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        ## 综合得分: {score:.4f}
        
        ## 分类性能指标:
        - 准确率: {classification_metrics.get('accuracy', 0):.4f}
        - F1分数: {classification_metrics.get('f1_score', 0):.4f}
        
        ## 回测性能指标:
        - 总收益率: {backtest_metrics.get('total_return', 0):.4f}
        - 年化收益率: {backtest_metrics.get('annual_return', 0):.4f}
        - 最大回撤: {backtest_metrics.get('max_drawdown', 0):.4f}
        - Sharpe比率: {backtest_metrics.get('sharpe_ratio', 0):.4f}
        - Sortino比率: {backtest_metrics.get('sortino_ratio', 0):.4f}
        - 胜率: {backtest_metrics.get('win_rate', 0):.4f}
        - 盈亏比: {backtest_metrics.get('profit_loss_ratio', 0):.4f}
        - 交易次数: {backtest_metrics.get('total_trades', 0)}
        
        ## 类别准确率:
        """
        
        # 添加类别准确率
        for key, value in classification_metrics.items():
            if key.startswith('class_') and key.endswith('_accuracy'):
                class_name = key.replace('class_', '').replace('_accuracy', '')
                report += f"- 类别 {class_name}: {value:.4f}\n"
        
        # 保存报告
        report_dir = 'reports'
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f'evaluation_report_{timestamp}.md')
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"评估报告已保存到: {report_path}")
        
        return report
    
    def simple_backtest(self, 
                      data: pd.DataFrame, 
                      predictions: np.ndarray, 
                      symbols: List[str]) -> Dict[str, Any]:
        """
        简单回测策略：每个品种初始资金1000，每次交易100，不考虑滑点和费率
        涨时做多，到平或跌时平仓；跌时做空，到平或涨时平仓
        
        参数:
            data: 包含价格数据的DataFrame
            predictions: 模型预测结果，形状为(-1)的一维数组，值为-1(跌)/0(平)/1(涨)
            symbols: 交易品种列表
            
        返回:
            回测结果字典
        """
        print(f"开始简单回测，品种数量: {len(symbols)}")
        print(f"预测数据形状: {predictions.shape}")
        print(f"价格数据形状: {data.shape}")
        
        # 确保数据对齐
        # 如果predictions比data短，我们需要对齐数据
        if len(predictions) < len(data):
            print(f"预测数据({len(predictions)})比价格数据({len(data)})短，进行对齐")
            # 假设predictions对应data的后半部分
            data = data.iloc[-len(predictions):]
        elif len(predictions) > len(data):
            print(f"预测数据({len(predictions)})比价格数据({len(data)})长，进行截断")
            predictions = predictions[-len(data):]
        
        print(f"对齐后数据形状: 价格数据={data.shape}, 预测数据={predictions.shape}")
        
        # 创建回测结果DataFrame
        results = {}
        all_symbol_results = pd.DataFrame(index=data.index)
        all_symbol_results['total_equity'] = 0
        
        # 对每个品种单独回测
        for symbol_idx, symbol in enumerate(symbols):
            print(f"回测品种: {symbol}")
            
            # 检查必要的列是否存在
            required_cols = [f"{symbol}_close"]
            if not all(col in data.columns for col in required_cols):
                print(f"警告: {symbol}缺少必要列，跳过回测")
                continue
            
            # 创建该品种的回测DataFrame
            symbol_data = pd.DataFrame(index=data.index)
            symbol_data['close'] = data[f"{symbol}_close"]
            symbol_data['prediction'] = predictions[:, symbol_idx] if predictions.ndim > 1 else predictions
            
            # 初始化回测变量
            initial_capital = 1000.0  # 每个品种初始资金1000
            trade_amount = 100.0      # 每次交易100
            
            symbol_data['cash'] = initial_capital
            symbol_data['position'] = 0  # 0表示无仓位，1表示多头，-1表示空头
            symbol_data['position_value'] = 0.0
            symbol_data['equity'] = initial_capital
            symbol_data['trade'] = 0  # 0表示不交易，1表示开仓，-1表示平仓
            
            # 执行回测
            for i in range(1, len(symbol_data)):
                current_price = symbol_data['close'].iloc[i]
                prev_price = symbol_data['close'].iloc[i-1]
                current_prediction = symbol_data['prediction'].iloc[i]
                prev_prediction = symbol_data['prediction'].iloc[i-1]
                
                # 继承前一天的状态
                symbol_data.loc[symbol_data.index[i], 'cash'] = symbol_data['cash'].iloc[i-1]
                symbol_data.loc[symbol_data.index[i], 'position'] = symbol_data['position'].iloc[i-1]
                symbol_data.loc[symbol_data.index[i], 'position_value'] = symbol_data['position'].iloc[i] * current_price * trade_amount / prev_price
                
                # 交易逻辑
                trade = 0
                
                # 当前无仓位
                if symbol_data['position'].iloc[i] == 0:
                    if current_prediction == 1:  # 预测涨，做多
                        symbol_data.loc[symbol_data.index[i], 'position'] = 1
                        symbol_data.loc[symbol_data.index[i], 'cash'] -= trade_amount
                        symbol_data.loc[symbol_data.index[i], 'position_value'] = trade_amount
                        trade = 1
                    elif current_prediction == -1:  # 预测跌，做空
                        symbol_data.loc[symbol_data.index[i], 'position'] = -1
                        symbol_data.loc[symbol_data.index[i], 'cash'] += trade_amount
                        symbol_data.loc[symbol_data.index[i], 'position_value'] = -trade_amount
                        trade = 1
                
                # 当前持有多头
                elif symbol_data['position'].iloc[i] == 1:
                    if current_prediction <= 0:  # 预测平或跌，平仓
                        symbol_data.loc[symbol_data.index[i], 'position'] = 0
                        symbol_data.loc[symbol_data.index[i], 'cash'] += symbol_data['position_value'].iloc[i]
                        symbol_data.loc[symbol_data.index[i], 'position_value'] = 0
                        trade = -1
                
                # 当前持有空头
                elif symbol_data['position'].iloc[i] == -1:
                    if current_prediction >= 0:  # 预测平或涨，平仓
                        symbol_data.loc[symbol_data.index[i], 'position'] = 0
                        symbol_data.loc[symbol_data.index[i], 'cash'] -= symbol_data['position_value'].iloc[i]
                        symbol_data.loc[symbol_data.index[i], 'position_value'] = 0
                        trade = -1
                
                symbol_data.loc[symbol_data.index[i], 'trade'] = trade
                
                # 计算权益
                symbol_data.loc[symbol_data.index[i], 'equity'] = symbol_data['cash'].iloc[i] + symbol_data['position_value'].iloc[i]
            
            # 计算收益指标
            final_equity = symbol_data['equity'].iloc[-1]
            total_return = (final_equity / initial_capital - 1) * 100
            
            # 计算最大回撤
            symbol_data['cumulative_max'] = symbol_data['equity'].cummax()
            symbol_data['drawdown'] = (symbol_data['equity'] - symbol_data['cumulative_max']) / symbol_data['cumulative_max'] * 100
            max_drawdown = symbol_data['drawdown'].min()
            
            # 计算交易次数
            total_trades = (symbol_data['trade'] != 0).sum()
            winning_trades = ((symbol_data['equity'] - symbol_data['equity'].shift(1)) * symbol_data['trade'].shift(1) > 0).sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 保存结果
            results[symbol] = {
                'initial_capital': initial_capital,
                'final_equity': final_equity,
                'total_return_pct': total_return,
                'max_drawdown_pct': max_drawdown,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'data': symbol_data
            }
            
            # 添加到总体结果
            all_symbol_results[f'{symbol}_equity'] = symbol_data['equity']
            all_symbol_results['total_equity'] += symbol_data['equity']
            
            print(f"{symbol} 回测完成: 收益率={total_return:.2f}%, 最大回撤={max_drawdown:.2f}%, 交易次数={total_trades}, 胜率={win_rate:.2f}")
        
        # 计算整体指标
        initial_total = len(symbols) * 1000.0
        final_total = all_symbol_results['total_equity'].iloc[-1]
        total_return_pct = (final_total / initial_total - 1) * 100
        
        all_symbol_results['cumulative_max'] = all_symbol_results['total_equity'].cummax()
        all_symbol_results['drawdown'] = (all_symbol_results['total_equity'] - all_symbol_results['cumulative_max']) / all_symbol_results['cumulative_max'] * 100
        total_max_drawdown = all_symbol_results['drawdown'].min()
        
        # 绘制总体权益曲线
        plt.figure(figsize=(12, 6))
        plt.plot(all_symbol_results.index, all_symbol_results['total_equity'])
        plt.title('总体权益曲线')
        plt.xlabel('日期')
        plt.ylabel('权益')
        plt.grid(True)
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.plots_dir, f'simple_backtest_equity_{timestamp}.png'))
        plt.close()
        
        # 绘制每个品种的权益曲线
        plt.figure(figsize=(15, 10))
        for symbol in symbols:
            if f'{symbol}_equity' in all_symbol_results.columns:
                plt.plot(all_symbol_results.index, all_symbol_results[f'{symbol}_equity'], label=symbol)
        plt.title('各品种权益曲线')
        plt.xlabel('日期')
        plt.ylabel('权益')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plt.savefig(os.path.join(self.plots_dir, f'simple_backtest_symbol_equity_{timestamp}.png'))
        plt.close()
        
        # 汇总结果
        summary = {
            'initial_total_capital': initial_total,
            'final_total_equity': final_total,
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': total_max_drawdown,
            'symbol_results': results,
            'equity_data': all_symbol_results
        }
        
        print(f"总体回测结果: 初始资金={initial_total:.2f}, 最终权益={final_total:.2f}, 收益率={total_return_pct:.2f}%, 最大回撤={total_max_drawdown:.2f}%")
        
        return summary


# 使用示例
if __name__ == "__main__":
    # 模拟数据
    dates = pd.date_range(start='2022-01-01', periods=1000, freq='30min')
    df = pd.DataFrame(index=dates)
    
    symbols = ['BTC', 'ETH']
    for symbol in symbols:
        df[f"{symbol}_close"] = np.random.random(len(df)) * 1000 + 30000
        df[f"{symbol}_high"] = df[f"{symbol}_close"] * (1 + np.random.random(len(df)) * 0.01)
        df[f"{symbol}_low"] = df[f"{symbol}_close"] * (1 - np.random.random(len(df)) * 0.01)
        df[f"{symbol}_volume"] = np.random.random(len(df)) * 100
        df[f"{symbol}_label"] = np.random.randint(-1, 2, len(df))
        df[f"{symbol}_highbi"] = np.random.choice(["100%", "50%", "25%", "10%"], len(df))
    
    # 模拟预测
    predictions = np.random.randint(-1, 2, (len(df), len(symbols)))
    
    # 创建评估器
    evaluator = ModelEvaluator('config.yaml')
    
    # 评估分类性能
    y_true = np.concatenate([df[f"{symbol}_label"].values for symbol in symbols])
    y_pred = predictions.flatten()
    
    classification_metrics = evaluator.evaluate_classification(y_true, y_pred)
    print("分类指标:", classification_metrics)
    
    # 回测
    backtest_results = evaluator.backtest(df, predictions, symbols)
    print("回测指标:", backtest_results)
    
    # 生成评估报告
    report = evaluator.generate_evaluation_report(classification_metrics, backtest_results)
    print(report)