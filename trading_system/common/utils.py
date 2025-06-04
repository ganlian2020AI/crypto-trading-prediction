#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import yaml
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class TradingLogger:
    """交易系统日志类"""
    
    def __init__(self, name: str, log_level: str = 'INFO', log_file: str = None):
        """
        初始化日志记录器
        
        参数:
            name: 日志记录器名称
            log_level: 日志级别
            log_file: 日志文件路径
        """
        self.logger = logging.getLogger(name)
        
        # 设置日志级别
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        self.logger.setLevel(level_map.get(log_level.upper(), logging.INFO))
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 如果指定了日志文件，创建文件处理器
        if log_file:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """记录调试信息"""
        self.logger.debug(message)
        
    def info(self, message: str):
        """记录一般信息"""
        self.logger.info(message)
        
    def warning(self, message: str):
        """记录警告信息"""
        self.logger.warning(message)
        
    def error(self, message: str):
        """记录错误信息"""
        self.logger.error(message)
        
    def critical(self, message: str):
        """记录严重错误信息"""
        self.logger.critical(message)


class AlertManager:
    """警报管理类"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional['TradingLogger'] = None):
        """
        初始化警报管理器
        
        参数:
            config: 警报配置信息
            logger: 日志记录器，如果为None则创建新的
        """
        self.config = config
        
        # 使用提供的logger或创建新的
        if logger:
            self.logger = logger
        else:
            log_level = config.get('general', {}).get('log_level', 'INFO')
            log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'alert_manager.log')
            self.logger = TradingLogger('alert_manager', log_level, log_file)
            
        self.alert_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'alerts')
        
        # 确保警报目录存在
        if not os.path.exists(self.alert_dir):
            os.makedirs(self.alert_dir)
    
    def send_alert(self, alert_type: str, message: str, level: str = 'INFO', data: Dict = None):
        """
        发送警报
        
        参数:
            alert_type: 警报类型
            message: 警报消息
            level: 警报级别
            data: 附加数据
        """
        # 记录到日志
        if level == 'INFO':
            self.logger.info(f"{alert_type}: {message}")
        elif level == 'WARNING':
            self.logger.warning(f"{alert_type}: {message}")
        elif level == 'ERROR':
            self.logger.error(f"{alert_type}: {message}")
        elif level == 'CRITICAL':
            self.logger.critical(f"{alert_type}: {message}")
        
        # 保存到警报文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alert_file = os.path.join(self.alert_dir, f"{alert_type}_{timestamp}.json")
        
        alert_data = {
            'type': alert_type,
            'message': message,
            'level': level,
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }
        
        with open(alert_file, 'w', encoding='utf-8') as f:
            json.dump(alert_data, f, indent=2)
        
        # 发送邮件警报
        if self.config.get('alerts', {}).get('email', {}).get('enabled', False):
            self._send_email_alert(alert_data)
        
        # 发送短信警报
        if self.config.get('alerts', {}).get('sms', {}).get('enabled', False) and level in ['ERROR', 'CRITICAL']:
            self._send_sms_alert(alert_data)
    
    def _send_email_alert(self, alert_data: Dict):
        """发送邮件警报"""
        try:
            email_config = self.config.get('alerts', {}).get('email', {})
            
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = email_config.get('sender_email')
            msg['To'] = email_config.get('recipient_email')
            msg['Subject'] = f"交易系统警报: {alert_data['type']} - {alert_data['level']}"
            
            # 邮件正文
            body = f"""
            <html>
            <body>
                <h2>交易系统警报</h2>
                <p><strong>类型:</strong> {alert_data['type']}</p>
                <p><strong>级别:</strong> {alert_data['level']}</p>
                <p><strong>时间:</strong> {alert_data['timestamp']}</p>
                <p><strong>消息:</strong> {alert_data['message']}</p>
                <hr>
                <p><strong>附加数据:</strong></p>
                <pre>{json.dumps(alert_data['data'], indent=2)}</pre>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # 连接到SMTP服务器并发送邮件
            with smtplib.SMTP(email_config.get('smtp_server'), email_config.get('smtp_port')) as server:
                server.starttls()
                server.login(email_config.get('sender_email'), email_config.get('sender_password'))
                server.send_message(msg)
                
            self.logger.info(f"已发送邮件警报: {alert_data['type']}")
            
        except Exception as e:
            self.logger.error(f"发送邮件警报失败: {str(e)}")
    
    def _send_sms_alert(self, alert_data: Dict):
        """发送短信警报"""
        try:
            sms_config = self.config.get('alerts', {}).get('sms', {})
            provider = sms_config.get('provider', '').lower()
            
            if provider == 'twilio':
                # 需要安装twilio库
                try:
                    from twilio.rest import Client
                    
                    # 创建Twilio客户端
                    client = Client(sms_config.get('account_sid'), sms_config.get('auth_token'))
                    
                    # 发送短信
                    message = client.messages.create(
                        body=f"交易系统警报: {alert_data['type']} - {alert_data['level']}: {alert_data['message']}",
                        from_=sms_config.get('from_number'),
                        to=sms_config.get('to_number')
                    )
                    
                    self.logger.info(f"已发送短信警报: {message.sid}")
                    
                except ImportError:
                    self.logger.error("未安装twilio库，无法发送短信警报")
                except Exception as e:
                    self.logger.error(f"发送Twilio短信警报失败: {str(e)}")
            else:
                self.logger.warning(f"不支持的短信提供商: {provider}")
                
        except Exception as e:
            self.logger.error(f"发送短信警报失败: {str(e)}")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    参数:
        config_path: 配置文件路径，如果为None则返回默认配置
        
    返回:
        配置字典
    """
    # 默认配置
    default_config = {
        'general': {
            'log_level': 'INFO',
            'timezone': 'Asia/Shanghai',
            'enable_alerts': True,
            'alert_methods': ['console'],
            'check_interval_minutes': 30
        },
        'symbols': [
            {'name': 'BTC', 'enabled': True, 'trade_size_usd': 100, 'max_position': 3},
            {'name': 'ETH', 'enabled': True, 'trade_size_usd': 100, 'max_position': 3}
        ],
        'simulation': {
            'initial_capital': 10000,
            'start_date': '2023-01-01',
            'end_date': 'auto',
            'include_fees': True,
            'fee_rate': 0.001
        },
        'live_trading': {
            'exchange': 'binance',
            'api_key': '',
            'api_secret': '',
            'testnet': True
        },
        'risk_management': {
            'max_open_trades': 5,
            'max_daily_trades': 10,
            'max_drawdown_pct': 0.1
        },
        'database': {
            'type': 'sqlite',
            'path': 'data/trading_system.db',
            'log_level': 'INFO'
        },
        'model': {
            'model_dir': '../models',
            'default_model': ''
        }
    }
    
    if not config_path:
        return default_config
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        # 合并默认配置和加载的配置
        merged_config = default_config.copy()
        
        # 递归更新配置
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = update_dict(d[k].copy(), v)
                else:
                    d[k] = v
            return d
            
        merged_config = update_dict(merged_config, config)
        return merged_config
        
    except Exception as e:
        logger = TradingLogger('config_loader')
        logger.error(f"加载配置文件失败: {str(e)}")
        return default_config


def get_database_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    从配置中获取数据库配置
    
    参数:
        config: 完整配置字典
        
    返回:
        数据库配置字典
    """
    db_config = config.get('database', {})
    
    # 确保必要的配置项存在
    if 'type' not in db_config:
        db_config['type'] = 'sqlite'
        
    if db_config['type'] == 'sqlite' and 'path' not in db_config:
        db_config['path'] = 'data/trading_system.db'
        
    return db_config


def plot_equity_curve(equity_data: pd.DataFrame, title: str = "Equity Curve", show: bool = True, save_path: str = None):
    """
    绘制权益曲线
    
    参数:
        equity_data: 包含权益数据的DataFrame
        title: 图表标题
        show: 是否显示图表
        save_path: 保存图表的路径
    """
    plt.figure(figsize=(12, 6))
    sns.set_style("darkgrid")
    
    # 绘制权益曲线
    plt.plot(equity_data.index, equity_data['total_equity'], label='Total Equity', linewidth=2)
    
    if 'cash' in equity_data.columns:
        plt.plot(equity_data.index, equity_data['cash'], label='Cash', alpha=0.7)
        
    if 'position_value' in equity_data.columns:
        plt.plot(equity_data.index, equity_data['position_value'], label='Position Value', alpha=0.7)
    
    # 添加标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Equity (USD)', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图表
    if show:
        plt.show()
    else:
        plt.close()


def calculate_performance_metrics(equity_curve: pd.DataFrame) -> Dict[str, float]:
    """
    计算性能指标
    
    参数:
        equity_curve: 权益曲线DataFrame
        
    返回:
        性能指标字典
    """
    # 确保索引是日期时间类型
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        equity_curve.index = pd.to_datetime(equity_curve.index)
    
    # 计算日收益率
    equity_curve['daily_return'] = equity_curve['total_equity'].pct_change()
    
    # 初始和最终权益
    initial_equity = equity_curve['total_equity'].iloc[0]
    final_equity = equity_curve['total_equity'].iloc[-1]
    
    # 总收益和年化收益
    total_return = final_equity / initial_equity - 1
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1
    
    # 计算最大回撤
    equity_curve['cummax'] = equity_curve['total_equity'].cummax()
    equity_curve['drawdown'] = (equity_curve['total_equity'] / equity_curve['cummax'] - 1)
    max_drawdown = equity_curve['drawdown'].min()
    
    # 计算夏普比率
    risk_free_rate = 0.0  # 假设无风险利率为0
    excess_returns = equity_curve['daily_return'] - risk_free_rate / 365
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
    
    # 交易统计
    if 'trades' in equity_curve.columns:
        total_trades = equity_curve['trades'].sum()
        winning_trades = equity_curve[equity_curve['trade_return'] > 0]['trades'].sum() if 'trade_return' in equity_curve.columns else 0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
    else:
        total_trades = 0
        win_rate = 0
    
    # 计算盈亏比
    if 'trade_return' in equity_curve.columns:
        winning_returns = equity_curve[equity_curve['trade_return'] > 0]['trade_return']
        losing_returns = equity_curve[equity_curve['trade_return'] < 0]['trade_return']
        
        avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
        avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
        
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    else:
        profit_loss_ratio = 0
    
    # 返回性能指标
    return {
        'initial_equity': initial_equity,
        'final_equity': final_equity,
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'annual_return': annual_return,
        'annual_return_pct': annual_return * 100,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown * 100,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio
    }


def generate_html_report(equity_curve: pd.DataFrame, trades: pd.DataFrame, performance_metrics: Dict[str, float], 
                         config: Dict[str, Any], output_path: str) -> str:
    """
    生成HTML报告
    
    参数:
        equity_curve: 权益曲线DataFrame
        trades: 交易记录DataFrame
        performance_metrics: 性能指标字典
        config: 配置信息
        output_path: 输出路径
        
    返回:
        报告文件路径
    """
    # 创建报告目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 生成权益曲线图
    equity_plot_path = os.path.join(os.path.dirname(output_path), 'equity_curve.png')
    plot_equity_curve(equity_curve, show=False, save_path=equity_plot_path)
    
    # 创建HTML报告内容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading System Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #0066cc;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .metric {{
                font-weight: bold;
            }}
            .positive {{
                color: green;
            }}
            .negative {{
                color: red;
            }}
            .chart {{
                max-width: 100%;
                height: auto;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <h1>Trading System Performance Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Performance Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Initial Equity</td>
                <td>${performance_metrics['initial_equity']:.2f}</td>
            </tr>
            <tr>
                <td>Final Equity</td>
                <td>${performance_metrics['final_equity']:.2f}</td>
            </tr>
            <tr>
                <td>Total Return</td>
                <td class="{('positive' if performance_metrics['total_return_pct'] >= 0 else 'negative')}">{performance_metrics['total_return_pct']:.2f}%</td>
            </tr>
            <tr>
                <td>Annual Return</td>
                <td class="{('positive' if performance_metrics['annual_return_pct'] >= 0 else 'negative')}">{performance_metrics['annual_return_pct']:.2f}%</td>
            </tr>
            <tr>
                <td>Maximum Drawdown</td>
                <td class="negative">{performance_metrics['max_drawdown_pct']:.2f}%</td>
            </tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td>{performance_metrics['sharpe_ratio']:.2f}</td>
            </tr>
            <tr>
                <td>Total Trades</td>
                <td>{performance_metrics['total_trades']}</td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td>{performance_metrics['win_rate']*100:.2f}%</td>
            </tr>
            <tr>
                <td>Profit/Loss Ratio</td>
                <td>{performance_metrics['profit_loss_ratio']:.2f}</td>
            </tr>
        </table>
        
        <h2>Equity Curve</h2>
        <img src="equity_curve.png" class="chart" alt="Equity Curve">
        
        <h2>Trading Configuration</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
    """
    
    # 添加配置信息
    for section, params in config.items():
        if isinstance(params, dict):
            for param, value in params.items():
                if isinstance(value, (str, int, float, bool)):
                    html_content += f"""
                    <tr>
                        <td>{section}.{param}</td>
                        <td>{value}</td>
                    </tr>
                    """
        elif isinstance(params, list):
            html_content += f"""
            <tr>
                <td>{section}</td>
                <td>{', '.join([str(item) if not isinstance(item, dict) else str(item.get('name', '')) for item in params])}</td>
            </tr>
            """
    
    html_content += """
        </table>
    """
    
    # 添加交易记录
    if not trades.empty:
        html_content += """
        <h2>Trade History</h2>
        <table>
            <tr>
                <th>Date</th>
                <th>Symbol</th>
                <th>Type</th>
                <th>Price</th>
                <th>Amount</th>
                <th>Value</th>
                <th>Fee</th>
            </tr>
        """
        
        for _, trade in trades.iterrows():
            trade_type_class = "positive" if trade['trade_type'].lower() == 'buy' else "negative"
            html_content += f"""
            <tr>
                <td>{trade['timestamp']}</td>
                <td>{trade['symbol']}</td>
                <td class="{trade_type_class}">{trade['trade_type']}</td>
                <td>${trade['price']:.2f}</td>
                <td>{trade['amount']:.6f}</td>
                <td>${trade['price']*trade['amount']:.2f}</td>
                <td>${trade.get('fee', 0):.2f}</td>
            </tr>
            """
        
        html_content += """
        </table>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # 写入HTML文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path 