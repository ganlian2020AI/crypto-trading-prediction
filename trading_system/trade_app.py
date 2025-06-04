#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from datetime import datetime
import yaml
import pandas as pd
import json

# 添加项目根目录到sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from common.utils import TradingLogger, load_config
from backtest.backtest_engine import BacktestEngine
from live.live_trader import LiveTrader
from common.data_downloader import DataDownloader
from common.database import DatabaseManager


def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


def show_header():
    """显示标题"""
    print("=" * 60)
    print("     加密货币交易预测系统 - 实时交易与回测平台")
    print("=" * 60)
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)


def show_menu():
    """显示菜单"""
    print("\n请选择操作：")
    print("1. 模拟交易")
    print("2. 实盘交易")
    print("3. 查看历史报告")
    print("4. 系统设置")
    print("5. 安装依赖")
    print("6. 数据下载")
    print("0. 退出")
    print("-" * 60)


def run_backtest():
    """运行模拟交易"""
    clear_screen()
    show_header()
    print("【模拟交易】\n")
    
    # 加载配置
    config_path = os.path.join(ROOT_DIR, 'config', 'trading_config.yaml')
    config = load_config(config_path)
    
    # 创建回测引擎
    engine = BacktestEngine(config_path)
    
    # 选择模型
    model_dir = os.path.join(os.path.dirname(ROOT_DIR), 'models')
    if not os.path.exists(model_dir):
        print(f"错误：找不到模型目录 {model_dir}")
        input("\n按回车键返回主菜单...")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras') or f.endswith('.h5')]
    if not model_files:
        print("错误：找不到模型文件，请先训练模型")
        input("\n按回车键返回主菜单...")
        return
    
    print("可用模型：")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    print("\n输入模型编号，或按回车使用最新模型：")
    choice = input("> ")
    
    model_path = None
    if choice.strip():
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(model_files):
                model_path = os.path.join(model_dir, model_files[idx])
            else:
                print("无效的选择，使用最新模型")
                model_path = os.path.join(model_dir, sorted(model_files)[-1])
        except ValueError:
            print("无效的输入，使用最新模型")
            model_path = os.path.join(model_dir, sorted(model_files)[-1])
    else:
        model_path = os.path.join(model_dir, sorted(model_files)[-1])
    
    print(f"\n使用模型: {model_path}")
    
    # 选择交易品种
    symbols_config = config.get('symbols', [])
    enabled_symbols = [s['name'] for s in symbols_config if s.get('enabled', True)]
    
    print("\n可用交易品种：")
    for i, symbol in enumerate(enabled_symbols):
        print(f"{i+1}. {symbol}")
    print(f"{len(enabled_symbols)+1}. 全部")
    
    print("\n选择交易品种（输入编号，多个用逗号分隔）：")
    choice = input("> ")
    
    selected_symbols = []
    if choice.strip():
        try:
            if int(choice) == len(enabled_symbols) + 1:
                selected_symbols = enabled_symbols
            else:
                for idx in choice.split(','):
                    idx = int(idx.strip()) - 1
                    if 0 <= idx < len(enabled_symbols):
                        selected_symbols.append(enabled_symbols[idx])
        except ValueError:
            print("无效的输入，使用全部品种")
            selected_symbols = enabled_symbols
    else:
        selected_symbols = enabled_symbols
    
    print(f"\n选择的交易品种: {', '.join(selected_symbols)}")
    
    # 设置初始资金
    print("\n输入初始资金（USD）[默认10000]：")
    initial_capital = input("> ")
    
    if not initial_capital.strip():
        initial_capital = 10000
    else:
        try:
            initial_capital = float(initial_capital)
        except ValueError:
            print("无效的输入，使用默认值10000")
            initial_capital = 10000
    
    # 设置回测时间范围
    print("\n输入回测开始日期（YYYY-MM-DD）[默认2023-01-01]：")
    start_date = input("> ")
    
    if not start_date.strip():
        start_date = "2023-01-01"
    
    print("\n输入回测结束日期（YYYY-MM-DD）[默认auto]：")
    end_date = input("> ")
    
    if not end_date.strip():
        end_date = "auto"
    
    print("\n开始回测，这可能需要几分钟...\n")
    
    # 加载数据
    engine.load_data(selected_symbols, start_date, end_date)
    
    # 加载模型预测
    engine.load_model_predictions(model_path)
    
    # 运行回测
    results = engine.run_backtest(initial_capital, selected_symbols, start_date, end_date)
    
    if results:
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
        
        # 询问是否生成报告
        print("\n是否生成详细报告? (y/n)")
        choice = input("> ")
        
        if choice.lower() == 'y':
            report_path = engine.generate_report()
            print(f"\n报告已生成: {report_path}")
        
        # 询问是否显示权益曲线
        print("\n是否显示权益曲线? (y/n)")
        choice = input("> ")
        
        if choice.lower() == 'y':
            # 保存权益曲线图
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(ROOT_DIR, 'reports', f"equity_curve_{timestamp}.png")
            engine.plot_equity_curve(show=False, save_path=plot_path)
            print(f"\n权益曲线已保存到: {plot_path}")
            
            # 在某些环境中可能无法直接显示图表，提供保存选项
            print("\n图表已保存，请在文件管理器中查看")
    else:
        print("\n回测失败，请检查日志获取详细信息")
    
    input("\n按回车键返回主菜单...")


def run_live_trading():
    """运行实盘交易"""
    clear_screen()
    show_header()
    print("【实盘交易】\n")
    
    # 加载配置
    config_path = os.path.join(ROOT_DIR, 'config', 'trading_config.yaml')
    config = load_config(config_path)
    
    # 检查API配置
    live_config = config.get('live_trading', {})
    api_key = live_config.get('api_key', '')
    api_secret = live_config.get('api_secret', '')
    
    if not api_key or not api_secret:
        print("错误：未配置API密钥，请先在配置文件中设置API密钥")
        
        print("\n是否现在设置API密钥? (y/n)")
        choice = input("> ")
        
        if choice.lower() == 'y':
            print("\n输入API Key:")
            api_key = input("> ")
            
            print("\n输入API Secret:")
            api_secret = input("> ")
            
            # 更新配置
            config['live_trading']['api_key'] = api_key
            config['live_trading']['api_secret'] = api_secret
            
            # 保存配置
            with open(config_path, 'w', encoding='utf-8') as file:
                yaml.dump(config, file)
            
            print("\nAPI密钥已更新")
        else:
            input("\n按回车键返回主菜单...")
            return
    
    # 选择模型
    model_dir = os.path.join(os.path.dirname(ROOT_DIR), 'models')
    if not os.path.exists(model_dir):
        print(f"错误：找不到模型目录 {model_dir}")
        input("\n按回车键返回主菜单...")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras') or f.endswith('.h5')]
    if not model_files:
        print("错误：找不到模型文件，请先训练模型")
        input("\n按回车键返回主菜单...")
        return
    
    print("可用模型：")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    print("\n输入模型编号，或按回车使用最新模型：")
    choice = input("> ")
    
    model_path = None
    if choice.strip():
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(model_files):
                model_path = os.path.join(model_dir, model_files[idx])
            else:
                print("无效的选择，使用最新模型")
                model_path = os.path.join(model_dir, sorted(model_files)[-1])
        except ValueError:
            print("无效的输入，使用最新模型")
            model_path = os.path.join(model_dir, sorted(model_files)[-1])
    else:
        model_path = os.path.join(model_dir, sorted(model_files)[-1])
    
    print(f"\n使用模型: {model_path}")
    
    # 选择交易品种
    symbols_config = config.get('symbols', [])
    enabled_symbols = [s['name'] for s in symbols_config if s.get('enabled', True)]
    
    print("\n可用交易品种：")
    for i, symbol in enumerate(enabled_symbols):
        print(f"{i+1}. {symbol}")
    print(f"{len(enabled_symbols)+1}. 全部")
    
    print("\n选择交易品种（输入编号，多个用逗号分隔）：")
    choice = input("> ")
    
    selected_symbols = []
    if choice.strip():
        try:
            if int(choice) == len(enabled_symbols) + 1:
                selected_symbols = enabled_symbols
            else:
                for idx in choice.split(','):
                    idx = int(idx.strip()) - 1
                    if 0 <= idx < len(enabled_symbols):
                        selected_symbols.append(enabled_symbols[idx])
        except ValueError:
            print("无效的输入，使用全部品种")
            selected_symbols = enabled_symbols
    else:
        selected_symbols = enabled_symbols
    
    print(f"\n选择的交易品种: {', '.join(selected_symbols)}")
    
    # 设置检查间隔
    print("\n输入检查间隔（分钟）[默认30]：")
    interval = input("> ")
    
    if not interval.strip():
        interval = 30
    else:
        try:
            interval = int(interval)
        except ValueError:
            print("无效的输入，使用默认值30分钟")
            interval = 30
    
    # 确认启动
    print("\n实盘交易将使用以下配置：")
    print(f"- 模型: {os.path.basename(model_path)}")
    print(f"- 交易品种: {', '.join(selected_symbols)}")
    print(f"- 检查间隔: {interval}分钟")
    print(f"- 交易所: {live_config.get('exchange', 'binance')}")
    print(f"- 测试网络: {'是' if live_config.get('testnet', True) else '否'}")
    
    print("\n确认启动实盘交易? (y/n)")
    choice = input("> ")
    
    if choice.lower() != 'y':
        print("\n已取消启动")
        input("\n按回车键返回主菜单...")
        return
    
    print("\n正在启动实盘交易...")
    
    # 创建实盘交易实例
    trader = LiveTrader(config_path)
    
    # 设置交易品种
    trader.symbols = selected_symbols
    
    # 启动交易服务
    try:
        trader.start_trading(model_path, interval)
    except KeyboardInterrupt:
        print("\n收到中断信号，正在停止交易服务...")
        trader.stop_trading()
    
    input("\n按回车键返回主菜单...")


def view_reports():
    """查看历史报告"""
    clear_screen()
    show_header()
    print("【查看历史报告】\n")
    
    report_dir = os.path.join(ROOT_DIR, 'reports')
    if not os.path.exists(report_dir):
        print("错误：找不到报告目录")
        input("\n按回车键返回主菜单...")
        return
    
    report_files = []
    for ext in ['.html', '.csv']:
        report_files.extend([f for f in os.listdir(report_dir) if f.endswith(ext)])
    
    if not report_files:
        print("错误：找不到报告文件")
        input("\n按回车键返回主菜单...")
        return
    
    # 按时间排序
    report_files.sort(reverse=True)
    
    # 显示可用报告
    print("可用报告：")
    for i, report_file in enumerate(report_files):
        print(f"{i+1}. {report_file}")
    
    print("\n输入报告编号查看详情，或按回车返回：")
    choice = input("> ")
    
    if not choice.strip():
        return
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(report_files):
            report_path = os.path.join(report_dir, report_files[idx])
            
            if report_path.endswith('.html'):
                # 打开HTML报告
                print(f"\n正在打开报告: {report_path}")
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(report_path)}")
            elif report_path.endswith('.csv'):
                # 显示CSV报告
                try:
                    df = pd.read_csv(report_path)
                    print(f"\n报告内容: {report_path}\n")
                    print(df)
                except Exception as e:
                    print(f"\n无法读取报告: {str(e)}")
        else:
            print("无效的选择")
    except ValueError:
        print("无效的输入")
    
    input("\n按回车键返回主菜单...")


def system_settings():
    """系统设置"""
    clear_screen()
    show_header()
    print("【系统设置】\n")
    
    config_path = os.path.join(ROOT_DIR, 'config', 'trading_config.yaml')
    config = load_config(config_path)
    
    print("当前配置：")
    
    # 显示主要配置项
    print("\n1. 基本配置")
    print(f"   - 日志级别: {config.get('general', {}).get('log_level', 'INFO')}")
    print(f"   - 时区: {config.get('general', {}).get('timezone', 'Asia/Shanghai')}")
    print(f"   - 启用警报: {'是' if config.get('general', {}).get('enable_alerts', True) else '否'}")
    
    print("\n2. 模拟交易配置")
    print(f"   - 初始资金: {config.get('simulation', {}).get('initial_capital', 10000)}")
    print(f"   - 包含费用: {'是' if config.get('simulation', {}).get('include_fees', True) else '否'}")
    print(f"   - 费率: {config.get('simulation', {}).get('fee_rate', 0.001)}")
    
    print("\n3. 实盘交易配置")
    print(f"   - 交易所: {config.get('live_trading', {}).get('exchange', 'binance')}")
    print(f"   - 测试网络: {'是' if config.get('live_trading', {}).get('testnet', True) else '否'}")
    print(f"   - API密钥: {'已设置' if config.get('live_trading', {}).get('api_key', '') else '未设置'}")
    
    print("\n4. 风险管理配置")
    print(f"   - 最大同时开仓数: {config.get('risk_management', {}).get('max_open_trades', 5)}")
    print(f"   - 每日最大交易次数: {config.get('risk_management', {}).get('max_daily_trades', 10)}")
    print(f"   - 最大回撤百分比: {config.get('risk_management', {}).get('max_drawdown_pct', 0.1)}")
    
    print("\n选择要修改的配置项（输入编号），或按回车返回：")
    choice = input("> ")
    
    if not choice.strip():
        return
    
    try:
        section = int(choice)
        
        if section == 1:  # 基本配置
            print("\n修改基本配置:")
            
            print("\n输入日志级别 [DEBUG/INFO/WARNING/ERROR]:")
            log_level = input("> ")
            if log_level.strip() in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
                config['general']['log_level'] = log_level
            
            print("\n启用警报? [y/n]:")
            enable_alerts = input("> ")
            if enable_alerts.lower() == 'y':
                config['general']['enable_alerts'] = True
            elif enable_alerts.lower() == 'n':
                config['general']['enable_alerts'] = False
            
        elif section == 2:  # 模拟交易配置
            print("\n修改模拟交易配置:")
            
            print("\n输入初始资金:")
            initial_capital = input("> ")
            if initial_capital.strip():
                try:
                    config['simulation']['initial_capital'] = float(initial_capital)
                except ValueError:
                    print("无效的输入，保持原值")
            
            print("\n包含费用? [y/n]:")
            include_fees = input("> ")
            if include_fees.lower() == 'y':
                config['simulation']['include_fees'] = True
            elif include_fees.lower() == 'n':
                config['simulation']['include_fees'] = False
            
            if config['simulation']['include_fees']:
                print("\n输入费率:")
                fee_rate = input("> ")
                if fee_rate.strip():
                    try:
                        config['simulation']['fee_rate'] = float(fee_rate)
                    except ValueError:
                        print("无效的输入，保持原值")
            
        elif section == 3:  # 实盘交易配置
            print("\n修改实盘交易配置:")
            
            print("\n输入交易所 [binance/huobi/okex/...]:")
            exchange = input("> ")
            if exchange.strip():
                config['live_trading']['exchange'] = exchange
            
            print("\n使用测试网络? [y/n]:")
            testnet = input("> ")
            if testnet.lower() == 'y':
                config['live_trading']['testnet'] = True
            elif testnet.lower() == 'n':
                config['live_trading']['testnet'] = False
            
            print("\n设置API密钥? [y/n]:")
            set_api = input("> ")
            if set_api.lower() == 'y':
                print("\n输入API Key:")
                api_key = input("> ")
                
                print("\n输入API Secret:")
                api_secret = input("> ")
                
                config['live_trading']['api_key'] = api_key
                config['live_trading']['api_secret'] = api_secret
            
        elif section == 4:  # 风险管理配置
            print("\n修改风险管理配置:")
            
            print("\n输入最大同时开仓数:")
            max_open_trades = input("> ")
            if max_open_trades.strip():
                try:
                    config['risk_management']['max_open_trades'] = int(max_open_trades)
                except ValueError:
                    print("无效的输入，保持原值")
            
            print("\n输入每日最大交易次数:")
            max_daily_trades = input("> ")
            if max_daily_trades.strip():
                try:
                    config['risk_management']['max_daily_trades'] = int(max_daily_trades)
                except ValueError:
                    print("无效的输入，保持原值")
            
            print("\n输入最大回撤百分比 (0-1):")
            max_drawdown_pct = input("> ")
            if max_drawdown_pct.strip():
                try:
                    config['risk_management']['max_drawdown_pct'] = float(max_drawdown_pct)
                except ValueError:
                    print("无效的输入，保持原值")
        
        # 保存配置
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file)
        
        print("\n配置已更新")
        
    except ValueError:
        print("无效的输入")
    except Exception as e:
        print(f"更新配置时出错: {str(e)}")
    
    input("\n按回车键返回主菜单...")


def install_dependencies():
    """安装依赖"""
    clear_screen()
    show_header()
    print("【安装依赖】\n")
    
    # 创建requirements.txt
    requirements = """numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tensorflow>=2.6.0
pyyaml>=6.0
ccxt>=2.0.0
schedule>=1.1.0
requests>=2.26.0
"""
    
    req_path = os.path.join(ROOT_DIR, "requirements.txt")
    with open(req_path, 'w') as f:
        f.write(requirements)
    
    print("将安装以下依赖：")
    print(requirements)
    
    print("\n确认安装？(y/n)")
    confirm = input("> ")
    
    if confirm.lower() != 'y':
        print("\n已取消安装")
        input("\n按回车键返回主菜单...")
        return
    
    print("\n开始安装依赖...\n")
    
    import subprocess
    try:
        subprocess.run(f"pip install -r {req_path}", shell=True, check=True)
        print("\n依赖安装完成！")
    except subprocess.CalledProcessError:
        print("\n安装过程中出现错误")
    
    input("\n按回车键返回主菜单...")


def download_data():
    """下载数据"""
    clear_screen()
    show_header()
    print("【数据下载】\n")
    
    # 加载配置
    config_path = os.path.join(ROOT_DIR, 'config', 'trading_config.yaml')
    config = load_config(config_path)
    
    # 创建数据库管理器
    db_config = config.get('database', {})
    db_manager = DatabaseManager(db_config)
    db_manager.connect()
    
    # 检查API代理前缀设置
    api_proxy_prefix = config.get('data_download', {}).get('api_proxy_prefix', '')
    
    try:
        # 显示下载选项
        print("请选择下载选项：")
        print("1. 下载单个交易对数据")
        print("2. 下载多个交易对数据")
        print("3. 下载配置文件中的所有交易对数据")
        print("4. 设置API代理前缀")
        print("0. 返回主菜单")
        
        choice = input("\n请输入选项: ")
        
        if choice == '1':
            # 下载单个交易对数据
            print("\n请输入要下载的交易对符号（例如BTC）：")
            symbol = input("> ")
            
            if not symbol.strip():
                print("未输入交易对符号，操作取消")
                input("\n按回车键返回主菜单...")
                return
            
            # 选择数据源
            print("\n请选择数据源：")
            print("1. 币安 (Binance)")
            print("2. Alpha Vantage")
            
            source_choice = input("> ")
            source = 'binance'
            if source_choice == '2':
                source = 'alphavantage'
                
                # 检查API密钥
                api_key = config.get('data_download', {}).get('sources', {}).get('alphavantage', {}).get('api_key', '')
                if not api_key:
                    print("\n未配置Alpha Vantage API密钥，请先在配置文件中设置API密钥")
                    print("是否现在设置API密钥? (y/n)")
                    set_key = input("> ")
                    
                    if set_key.lower() == 'y':
                        print("\n输入Alpha Vantage API Key:")
                        api_key = input("> ")
                        
                        # 更新配置
                        if 'data_download' not in config:
                            config['data_download'] = {}
                        if 'sources' not in config['data_download']:
                            config['data_download']['sources'] = {}
                        if 'alphavantage' not in config['data_download']['sources']:
                            config['data_download']['sources']['alphavantage'] = {}
                        
                        config['data_download']['sources']['alphavantage']['api_key'] = api_key
                        config['data_download']['sources']['alphavantage']['enabled'] = True
                        
                        # 保存配置
                        with open(config_path, 'w', encoding='utf-8') as file:
                            yaml.dump(config, file)
                        
                        print("\nAlpha Vantage API密钥已更新")
                    else:
                        print("\n未设置API密钥，将使用币安作为数据源")
                        source = 'binance'
            
            # 选择时间间隔
            interval = '1d'
            print("\n请选择时间间隔：")
            if source == 'binance':
                print("1. 1分钟 (1m)")
                print("2. 5分钟 (5m)")
                print("3. 15分钟 (15m)")
                print("4. 1小时 (1h)")
                print("5. 4小时 (4h)")
                print("6. 1天 (1d) [默认]")
                print("7. 1周 (1w)")
                
                interval_choice = input("> ")
                if interval_choice == '1':
                    interval = '1m'
                elif interval_choice == '2':
                    interval = '5m'
                elif interval_choice == '3':
                    interval = '15m'
                elif interval_choice == '4':
                    interval = '1h'
                elif interval_choice == '5':
                    interval = '4h'
                elif interval_choice == '7':
                    interval = '1w'
            else:
                print("1. 1分钟 (1min)")
                print("2. 5分钟 (5min)")
                print("3. 15分钟 (15min)")
                print("4. 30分钟 (30min)")
                print("5. 60分钟 (60min)")
                print("6. 每日 (daily) [默认]")
                print("7. 每周 (weekly)")
                print("8. 每月 (monthly)")
                
                interval_choice = input("> ")
                if interval_choice == '1':
                    interval = '1min'
                elif interval_choice == '2':
                    interval = '5min'
                elif interval_choice == '3':
                    interval = '15min'
                elif interval_choice == '4':
                    interval = '30min'
                elif interval_choice == '5':
                    interval = '60min'
                elif interval_choice == '6':
                    interval = 'daily'
                elif interval_choice == '7':
                    interval = 'weekly'
                elif interval_choice == '8':
                    interval = 'monthly'
            
            # 设置时间范围
            print("\n输入开始日期（YYYY-MM-DD）[默认2023-01-01]：")
            start_date = input("> ")
            
            if not start_date.strip():
                start_date = "2023-01-01"
            
            print("\n输入结束日期（YYYY-MM-DD）[默认当前日期]：")
            end_date = input("> ")
            
            # 设置保存选项
            print("\n是否保存到CSV文件? (y/n) [默认y]")
            save_to_csv = input("> ")
            save_to_csv = save_to_csv.lower() != 'n'
            
            print("\n是否保存到数据库? (y/n) [默认y]")
            save_to_db = input("> ")
            save_to_db = save_to_db.lower() != 'n'
            
            # 显示当前API代理前缀设置
            if api_proxy_prefix:
                print(f"\n当前API代理前缀: {api_proxy_prefix}")
            
            # 开始下载
            print(f"\n开始下载 {symbol} 数据，请稍候...\n")
            
            # 创建数据下载器
            downloader = DataDownloader(config, db_manager)
            
            kwargs = {}
            if start_date:
                kwargs['start_time'] = start_date
            if end_date:
                kwargs['end_time'] = end_date
                
            df = downloader.download_data(symbol, source, interval, save_to_csv, save_to_db, **kwargs)
            
            if not df.empty:
                print(f"\n成功下载 {len(df)} 条 {symbol} 数据")
                print("\n数据预览：")
                print(df.head())
            else:
                print(f"\n下载 {symbol} 数据失败，请检查参数和网络连接")
            
        elif choice == '2':
            # 下载多个交易对数据
            print("\n请输入要下载的交易对符号，多个用逗号分隔（例如BTC,ETH,ADA）：")
            symbols_input = input("> ")
            
            if not symbols_input.strip():
                print("未输入交易对符号，操作取消")
                input("\n按回车键返回主菜单...")
                return
            
            symbols = [s.strip() for s in symbols_input.split(',')]
            
            # 选择数据源
            print("\n请选择数据源：")
            print("1. 币安 (Binance) [默认]")
            print("2. Alpha Vantage")
            
            source_choice = input("> ")
            source = 'binance'
            if source_choice == '2':
                source = 'alphavantage'
                
                # 检查API密钥
                api_key = config.get('data_download', {}).get('sources', {}).get('alphavantage', {}).get('api_key', '')
                if not api_key:
                    print("\n未配置Alpha Vantage API密钥，请先在配置文件中设置API密钥")
                    print("\n将使用币安作为数据源")
                    source = 'binance'
            
            # 选择默认时间间隔
            interval = '1d'
            print("\n请选择时间间隔：")
            if source == 'binance':
                print("1. 1天 (1d) [默认]")
                print("2. 4小时 (4h)")
                print("3. 1小时 (1h)")
                
                interval_choice = input("> ")
                if interval_choice == '2':
                    interval = '4h'
                elif interval_choice == '3':
                    interval = '1h'
            else:
                print("1. 每日 (daily) [默认]")
                print("2. 每周 (weekly)")
                
                interval_choice = input("> ")
                if interval_choice == '2':
                    interval = 'weekly'
            
            # 设置时间范围
            print("\n输入开始日期（YYYY-MM-DD）[默认2023-01-01]：")
            start_date = input("> ")
            
            if not start_date.strip():
                start_date = "2023-01-01"
            
            print("\n输入结束日期（YYYY-MM-DD）[默认当前日期]：")
            end_date = input("> ")
            
            # 显示当前API代理前缀设置
            if api_proxy_prefix:
                print(f"\n当前API代理前缀: {api_proxy_prefix}")
            
            # 开始下载
            print(f"\n开始下载 {len(symbols)} 个交易对数据，这可能需要一些时间...\n")
            
            # 创建数据下载器
            downloader = DataDownloader(config, db_manager)
            
            kwargs = {}
            if start_date:
                kwargs['start_time'] = start_date
            if end_date:
                kwargs['end_time'] = end_date
                
            results = downloader.download_multiple_symbols(symbols, source, interval, True, True, **kwargs)
            
            print(f"\n下载完成，成功下载 {len(results)} 个交易对的数据")
            
        elif choice == '3':
            # 下载配置文件中的所有交易对数据
            symbols_config = config.get('symbols', [])
            enabled_symbols = [s['name'] for s in symbols_config if s.get('enabled', True)]
            
            if not enabled_symbols:
                print("\n配置文件中没有启用的交易对，请先在配置文件中启用交易对")
                input("\n按回车键返回主菜单...")
                return
            
            print(f"\n将下载以下 {len(enabled_symbols)} 个交易对的数据：")
            print(', '.join(enabled_symbols))
            
            # 显示当前API代理前缀设置
            if api_proxy_prefix:
                print(f"\n当前API代理前缀: {api_proxy_prefix}")
            
            print("\n确认下载? (y/n)")
            confirm = input("> ")
            
            if confirm.lower() != 'y':
                print("\n已取消下载")
                input("\n按回车键返回主菜单...")
                return
            
            # 开始下载
            print("\n开始下载数据，这可能需要一些时间...\n")
            
            # 创建数据下载器
            downloader = DataDownloader(config, db_manager)
            
            results = downloader.download_symbols_from_config()
            
            print(f"\n下载完成，成功下载 {len(results)} 个交易对的数据")
        
        elif choice == '4':
            # 设置API代理前缀
            print("\n当前API代理前缀: " + (api_proxy_prefix if api_proxy_prefix else "未设置"))
            print("\n请输入新的API代理前缀（例如 https://youusl/api/proxy/）：")
            print("留空则清除当前设置")
            new_prefix = input("> ")
            
            # 更新配置
            if 'data_download' not in config:
                config['data_download'] = {}
            
            config['data_download']['api_proxy_prefix'] = new_prefix.strip()
            
            # 保存配置
            with open(config_path, 'w', encoding='utf-8') as file:
                yaml.dump(config, file)
            
            if new_prefix.strip():
                print(f"\nAPI代理前缀已更新为: {new_prefix}")
            else:
                print("\nAPI代理前缀已清除")
            
            input("\n按回车键返回主菜单...")
            return
            
    except Exception as e:
        print(f"\n下载数据时出错: {str(e)}")
    finally:
        # 关闭数据库连接
        if db_manager:
            db_manager.disconnect()
    
    input("\n按回车键返回主菜单...")


def main():
    """主函数"""
    # 确保目录存在
    os.makedirs(os.path.join(ROOT_DIR, 'config'), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, 'alerts'), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, 'data'), exist_ok=True)
    
    # 确保配置文件存在
    config_path = os.path.join(ROOT_DIR, 'config', 'trading_config.yaml')
    if not os.path.exists(config_path):
        # 复制默认配置文件
        default_config_path = os.path.join(ROOT_DIR, 'config', 'trading_config.yaml.default')
        if os.path.exists(default_config_path):
            import shutil
            shutil.copy(default_config_path, config_path)
        else:
            # 创建默认配置
            from common.utils import load_config
            config = load_config(None)
            with open(config_path, 'w', encoding='utf-8') as file:
                yaml.dump(config, file)
    
    while True:
        clear_screen()
        show_header()
        show_menu()
        
        choice = input("\n请输入选项: ")
        
        if choice == '1':
            run_backtest()
        elif choice == '2':
            run_live_trading()
        elif choice == '3':
            view_reports()
        elif choice == '4':
            system_settings()
        elif choice == '5':
            install_dependencies()
        elif choice == '6':
            download_data()
        elif choice == '0':
            print("\n感谢使用，再见！")
            sys.exit(0)
        else:
            print("\n无效的选择，请重试")
            input("\n按回车键继续...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已被用户中断")
        sys.exit(0)