#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import time
from datetime import datetime
import yaml
import pandas as pd
import json

# 确保当前工作目录是脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 全局变量，控制是否使用GPU
use_gpu = False


def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


def show_header():
    """显示标题"""
    print("=" * 60)
    print("     加密货币交易预测系统 - 基于深度学习的多品种交易信号")
    print("=" * 60)
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU模式: {'开启' if use_gpu else '关闭'}")
    print("-" * 60)


def show_menu():
    """显示菜单"""
    print("\n请选择操作：")
    print("1. 训练模型")
    print("2. 恢复训练")
    print("3. 回测模型")
    print("4. 生成预测")
    print("5. 部署交易服务")
    print("6. 查看历史报告")
    print("7. 检查系统状态")
    print("8. 切换GPU/CPU模式")
    print("9. 安装依赖")
    print("0. 退出")
    print("-" * 60)


def train_model():
    """训练模型"""
    clear_screen()
    show_header()
    print("【训练模型】\n")
    
    # 检查是否有检查点可以恢复
    checkpoint_dir = 'checkpoints'
    model_state_file = os.path.join(checkpoint_dir, 'model_state.keras')
    training_state_file = os.path.join(checkpoint_dir, 'training_state.json')
    
    can_resume = (os.path.exists(model_state_file) and 
                 os.path.exists(training_state_file))
    
    if can_resume:
        try:
            with open(training_state_file, 'r') as f:
                training_state = json.load(f)
            
            last_epoch = training_state.get('epoch', 0)
            last_timestamp = training_state.get('timestamp', 'unknown')
            
            print(f"发现可恢复的训练检查点:")
            print(f"- 上次训练时间: {last_timestamp}")
            print(f"- 已完成轮次: {last_epoch}")
            
            print("\n是否从上次中断处恢复训练? (y/n)")
            choice = input("> ")
            
            resume = choice.lower() == 'y'
        except Exception as e:
            print(f"读取训练状态时出错: {str(e)}")
            resume = False
    else:
        resume = False
    
    print("\n开始训练模型，这可能需要几个小时...\n")
    print("提示: 如果需要中断训练，可以按Ctrl+C，程序会安全地保存检查点，以便下次恢复\n")
    
    # 调用main.py的train子命令，添加resume参数
    cmd = f"python main.py train {'--resume' if resume else ''}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("\n模型训练完成！")
    except subprocess.CalledProcessError:
        print("\n模型训练过程中出现错误。")
    except KeyboardInterrupt:
        print("\n训练被用户中断。您可以稍后通过选择'恢复训练'选项继续训练。")
    
    input("\n按回车键返回主菜单...")


def resume_training():
    """恢复训练"""
    clear_screen()
    show_header()
    print("【恢复训练】\n")
    
    # 检查是否有检查点可以恢复
    checkpoint_dir = 'checkpoints'
    model_state_file = os.path.join(checkpoint_dir, 'model_state.keras')
    training_state_file = os.path.join(checkpoint_dir, 'training_state.json')
    
    if not os.path.exists(model_state_file) or not os.path.exists(training_state_file):
        print("错误：找不到可恢复的训练检查点。")
        input("\n按回车键返回主菜单...")
        return
    
    try:
        with open(training_state_file, 'r') as f:
            training_state = json.load(f)
        
        last_epoch = training_state.get('epoch', 0)
        last_timestamp = training_state.get('timestamp', 'unknown')
        
        print(f"找到训练检查点:")
        print(f"- 上次训练时间: {last_timestamp}")
        print(f"- 已完成轮次: {last_epoch}")
        
        print("\n确认恢复训练? (y/n)")
        choice = input("> ")
        
        if choice.lower() != 'y':
            print("\n已取消恢复训练。")
            input("\n按回车键返回主菜单...")
            return
        
        print("\n开始恢复训练，这可能需要几个小时...\n")
        print("提示: 如果需要中断训练，可以按Ctrl+C，程序会安全地保存检查点，以便下次恢复\n")
        
        # 调用main.py的train子命令，添加resume参数
        cmd = f"python main.py train --resume"
        subprocess.run(cmd, shell=True, check=True)
        print("\n模型训练完成！")
        
    except Exception as e:
        print(f"\n恢复训练失败: {str(e)}")
    
    input("\n按回车键返回主菜单...")


def backtest_model():
    """回测模型"""
    clear_screen()
    show_header()
    print("【回测模型】\n")
    
    # 检查模型文件
    model_dir = 'models'
    if not os.path.exists(model_dir):
        print("错误：找不到模型目录。请先训练模型。")
        input("\n按回车键返回主菜单...")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('_final.keras')]
    if not model_files:
        print("错误：找不到训练好的模型文件。请先训练模型。")
        input("\n按回车键返回主菜单...")
        return
    
    # 显示可用模型
    print("可用模型：")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    print("\n输入模型编号进行回测，或按回车使用最新模型：")
    choice = input("> ")
    
    model_path = None
    if choice.strip():
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(model_files):
                model_path = os.path.join(model_dir, model_files[idx])
            else:
                print("无效的选择，使用最新模型。")
                model_path = os.path.join(model_dir, sorted(model_files)[-1])
        except ValueError:
            print("无效的输入，使用最新模型。")
            model_path = os.path.join(model_dir, sorted(model_files)[-1])
    else:
        model_path = os.path.join(model_dir, sorted(model_files)[-1])
    
    print(f"\n使用模型: {model_path}")
    
    # 选择回测策略
    print("\n选择回测策略：")
    print("1. 简单回测策略（每个品种初始资金1000，每次交易100，不考虑滑点和费率）")
    print("2. 标准回测策略（使用配置文件中的设置）")
    
    strategy_choice = input("\n请选择 [1/2]: ")
    use_simple_backtest = strategy_choice != "2"  # 默认使用简单回测
    
    strategy_name = "简单回测策略" if use_simple_backtest else "标准回测策略"
    print(f"\n使用{strategy_name}进行回测，这可能需要几分钟...\n")
    print("提示: 如果需要中断回测，可以按Ctrl+C，程序会安全地返回主菜单\n")
    
    cmd = f"python main.py backtest --model {model_path} {'--simple' if use_simple_backtest else '--standard'}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("\n回测完成！")
    except subprocess.CalledProcessError:
        print("\n回测过程中出现错误。")
    except KeyboardInterrupt:
        print("\n回测被用户中断。")
    
    input("\n按回车键返回主菜单...")


def generate_prediction():
    """生成预测"""
    clear_screen()
    show_header()
    print("【生成预测】\n")
    
    # 检查模型文件
    model_dir = 'models'
    if not os.path.exists(model_dir):
        print("错误：找不到模型目录。请先训练模型。")
        input("\n按回车键返回主菜单...")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('_final.keras')]
    if not model_files:
        print("错误：找不到训练好的模型文件。请先训练模型。")
        input("\n按回车键返回主菜单...")
        return
    
    # 使用最新模型
    model_path = os.path.join(model_dir, sorted(model_files)[-1])
    print(f"使用模型: {model_path}")
    print("生成预测，这可能需要几分钟...\n")
    print("提示: 如果需要中断预测，可以按Ctrl+C，程序会安全地返回主菜单\n")
    
    cmd = f"python main.py predict --model {model_path}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("\n预测生成完成！")
    except subprocess.CalledProcessError:
        print("\n预测过程中出现错误。")
    except KeyboardInterrupt:
        print("\n预测被用户中断。")
    
    input("\n按回车键返回主菜单...")


def deploy_service():
    """部署交易服务"""
    clear_screen()
    show_header()
    print("【部署交易服务】\n")
    
    # 检查模型文件
    model_dir = 'models'
    if not os.path.exists(model_dir):
        print("错误：找不到模型目录。请先训练模型。")
        input("\n按回车键返回主菜单...")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('_final.keras')]
    if not model_files:
        print("错误：找不到训练好的模型文件。请先训练模型。")
        input("\n按回车键返回主菜单...")
        return
    
    # 使用最新模型
    model_path = os.path.join(model_dir, sorted(model_files)[-1])
    
    # 询问更新频率
    print("请输入更新频率（分钟）[默认30]：")
    interval = input("> ")
    
    if not interval.strip():
        interval = "30"
    
    try:
        interval = int(interval)
    except ValueError:
        print("无效的输入，使用默认值30分钟。")
        interval = 30
    
    print(f"\n使用模型: {model_path}")
    print(f"更新频率: {interval}分钟")
    print("\n确认部署交易服务？(y/n)")
    confirm = input("> ")
    
    if confirm.lower() != 'y':
        print("\n已取消部署。")
        input("\n按回车键返回主菜单...")
        return
    
    print("\n开始部署交易服务...\n")
    print("交易服务将在后台运行。按Ctrl+C可以停止服务。\n")
    
    cmd = f"python main.py deploy --model {model_path} --interval {interval}"
    try:
        # 使用Popen启动进程并立即返回
        process = subprocess.Popen(cmd, shell=True)
        print(f"交易服务已启动，进程ID: {process.pid}")
        print("服务正在后台运行中...")
    except Exception as e:
        print(f"\n部署过程中出现错误: {str(e)}")
    except KeyboardInterrupt:
        print("\n部署被用户中断。")
    
    input("\n按回车键返回主菜单...")


def view_reports():
    """查看历史报告"""
    clear_screen()
    show_header()
    print("【查看历史报告】\n")
    
    report_dir = 'reports'
    if not os.path.exists(report_dir):
        print("错误：找不到报告目录。")
        input("\n按回车键返回主菜单...")
        return
    
    report_files = [f for f in os.listdir(report_dir) if f.endswith('.md')]
    if not report_files:
        print("错误：找不到报告文件。")
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
            
            # 读取报告内容
            with open(report_path, 'r') as f:
                report_content = f.read()
            
            clear_screen()
            show_header()
            print(f"【报告】{report_files[idx]}\n")
            print(report_content)
        else:
            print("无效的选择。")
    except ValueError:
        print("无效的输入。")
    
    input("\n按回车键返回主菜单...")


def check_system_status():
    """检查系统状态"""
    clear_screen()
    show_header()
    print("【系统状态】\n")
    
    # 检查配置文件
    print("配置文件状态：")
    config_path = 'config.yaml'
    if os.path.exists(config_path):
        print(f"{config_path:20} ✓ 存在")
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                yaml.safe_load(file)
            print(f"{' '*20} ✓ 格式正确")
        except Exception as e:
            print(f"{' '*20} ✗ 格式错误: {str(e)}")
    else:
        print(f"{config_path:20} ✗ 不存在")
    
    # 检查目录
    dirs = ['data', 'models', 'reports', 'logs', 'signals', 'processed_data', 'evaluation_plots', 'backtest_results', 'plots']
    print("\n目录状态：")
    for d in dirs:
        status = "✓ 存在" if os.path.exists(d) else "✗ 不存在"
        if os.path.exists(d):
            # 检查是否可写
            try:
                test_file = os.path.join(d, 'test_write.tmp')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                status += " (可写)"
            except Exception:
                status += " (不可写)"
        print(f"{d:20} {status}")
    
    # 检查数据文件
    print("\n数据文件：")
    if os.path.exists('data'):
        data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        if data_files:
            print(f"找到{len(data_files)}个CSV文件：{', '.join(data_files[:5])}{'...' if len(data_files) > 5 else ''}")
            
            # 检查CSV文件格式
            print("\n抽样检查CSV文件格式：")
            for sample_file in data_files[:2]:  # 只检查前两个文件
                file_path = os.path.join('data', sample_file)
                try:
                    df = pd.read_csv(file_path)
                    print(f"{sample_file:20} ✓ 可读取 ({len(df)}行)")
                    
                    # 检查是否包含必要的列
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col.lower() not in [c.lower() for c in df.columns]]
                    
                    if missing_cols:
                        print(f"{' '*20} ✗ 缺少必要列: {', '.join(missing_cols)}")
                    else:
                        print(f"{' '*20} ✓ 包含所有必要列")
                except Exception as e:
                    print(f"{sample_file:20} ✗ 读取失败: {str(e)}")
        else:
            print("未找到CSV数据文件。")
    else:
        print("数据目录不存在。")
    
    # 检查模型文件
    print("\n模型文件：")
    if os.path.exists('models'):
        model_files = [f for f in os.listdir('models') if f.endswith('.h5')]
        if model_files:
            print(f"找到{len(model_files)}个模型文件：{', '.join(model_files[:3])}{'...' if len(model_files) > 3 else ''}")
        else:
            print("未找到模型文件。")
    else:
        print("模型目录不存在。")
    
    # 检查Python环境
    print("\nPython环境：")
    try:
        import tensorflow as tf
        print(f"TensorFlow: {tf.__version__}")
    except ImportError:
        print("TensorFlow: 未安装")
    
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except ImportError:
        print("NumPy: 未安装")
    
    try:
        import pandas as pd
        print(f"Pandas: {pd.__version__}")
    except ImportError:
        print("Pandas: 未安装")
    
    # 检查进程状态
    print("\n进程状态：")
    try:
        if os.name == 'nt':  # Windows
            output = subprocess.check_output("tasklist | findstr python", shell=True).decode()
        else:  # Linux/Mac
            output = subprocess.check_output("ps aux | grep python | grep -v grep", shell=True).decode()
        
        if output.strip():
            print("Python进程正在运行：")
            for line in output.strip().split('\n')[:5]:
                print(f"  {line}")
            if len(output.strip().split('\n')) > 5:
                print("  ...")
        else:
            print("未检测到相关Python进程。")
    except subprocess.CalledProcessError:
        print("未检测到相关Python进程。")
    
    input("\n按回车键返回主菜单...")


def ensure_directories_exist():
    """确保所有必要的目录存在"""
    required_dirs = [
        'data',              # 原始数据目录
        'models',            # 模型存储目录
        'reports',           # 报告目录
        'logs',              # 日志目录
        'signals',           # 交易信号目录
        'processed_data',    # 处理后的数据目录
        'evaluation_plots',  # 评估图表目录
        'backtest_results',  # 回测结果目录
        'plots'              # 图表目录
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"创建目录: {directory}")
            os.makedirs(directory, exist_ok=True)
    
    # 特殊检查：确保data目录中有示例文件，如果为空则提示
    if os.path.exists('data') and not os.listdir('data'):
        print("警告: 数据目录为空，请添加加密货币数据文件。")
        print("数据文件应为CSV格式，文件名为交易品种符号，如BTC.csv, ETH.csv等")


def toggle_gpu_mode():
    """切换GPU/CPU模式"""
    global use_gpu
    
    clear_screen()
    show_header()
    print("【GPU/CPU模式设置】\n")
    
    print(f"当前模式: {'GPU' if use_gpu else 'CPU'}")
    
    # 动态检测系统资源
    print("\n您的系统资源:")
    
    # 检测CPU信息
    try:
        import psutil
        import platform
        
        # CPU信息
        cpu_count = psutil.cpu_count(logical=False)
        cpu_logical_count = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_info = f"{cpu_count}核心/{cpu_logical_count}线程 (使用率: {cpu_percent}%)"
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_total = memory.total / (1024 * 1024 * 1024)  # GB
        memory_available = memory.available / (1024 * 1024 * 1024)  # GB
        memory_info = f"{memory_available:.1f}GB可用/{memory_total:.1f}GB总计 (使用率: {memory.percent}%)"
        
        # GPU信息
        gpu_info = "未检测到GPU"
        try:
            import subprocess
            
            # 尝试使用nvidia-smi工具检测NVIDIA GPU
            nvidia_output = subprocess.check_output("nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader", shell=True).decode()
            if nvidia_output.strip():
                gpu_info = "已检测到NVIDIA GPU:\n"
                for i, line in enumerate(nvidia_output.strip().split('\n')):
                    gpu_info += f"    GPU {i}: {line}\n"
        except:
            # 尝试使用tensorflow检测GPU
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    gpu_info = f"已检测到{len(gpus)}个GPU设备"
                    for i, gpu in enumerate(gpus):
                        gpu_info += f"\n    GPU {i}: {gpu.name}"
                else:
                    gpu_info = "通过TensorFlow未检测到GPU设备"
            except:
                gpu_info = "未检测到GPU或未安装GPU驱动"
    
    except ImportError:
        cpu_info = "无法获取 (需安装psutil库)"
        memory_info = "无法获取 (需安装psutil库)"
        gpu_info = "无法获取 (需安装相关依赖)"
    except Exception as e:
        cpu_info = f"检测时出错: {str(e)}"
        memory_info = "检测时出错"
        gpu_info = "检测时出错"
    
    print(f"- CPU: {cpu_info}")
    print(f"- 内存: {memory_info}")
    print(f"- GPU: {gpu_info}")
    
    print("\n请选择运行模式:")
    print("1. CPU模式 (适合资源有限的环境或无GPU系统，稳定但速度较慢)")
    print("2. GPU模式 (需要CUDA环境和显卡支持，适合大规模数据训练，速度更快)")
    
    choice = input("\n请选择 [1/2]: ")
    
    if choice == '2':
        use_gpu = True
        print("\n已切换至GPU模式。注意：如果系统不支持GPU，程序可能会报错。")
    else:
        use_gpu = False
        print("\n已切换至CPU模式。")
    
    input("\n按回车键返回主菜单...")


def ensure_config_exists():
    """确保配置文件存在"""
    config_path = 'config.yaml'
    if os.path.exists(config_path):
        return
    
    print(f"配置文件{config_path}不存在，正在创建默认配置...")
    
    default_config = """# 加密货币交易系统配置文件

# 数据相关配置
data:
  input_dir: "data/"                                   # 数据文件目录
  symbols: ["AAVE", "ADA", "BCH", "BNB", "BTC", "DOGE", "ETH", "LTC", "SOL", "XRP"]  # 交易品种
  timeframe: "30min"                                   # 时间框架
  start_date: "2024-01-01"                             # 数据起始日期
  end_date: "2024-12-31"                               # 数据结束日期
  train_test_split: 0.8                                # 训练集比例
  validation_split: 0.2                                # 验证集比例（从训练集中划分）
  
# 特征工程相关配置
features:
  # 传统技术指标
  ema_periods: [21, 144, 169]                          # EMA周期
  rsi_period: 14                                       # RSI计算周期
  atr_period: 14                                       # ATR计算周期
  hema_alpha_length: 20                                # HEMA alpha长度
  hema_gamma_length: 20                                # HEMA gamma长度
  
  # 自定义指标
  custom_indicator1_length: 576                        # 自定义指标1长度(predictedPrice)
  custom_algorithm2_enabled: true                      # 是否启用自定义算法2
  custom_algorithm3_enabled: true                      # 是否启用自定义算法3
  
  # 预处理参数
  normalization: "z-score"                             # 标准化方法：z-score/min-max
  missing_value_method: "interpolation"                # 缺失值处理方法：interpolation/ffill/bfill
  clip_outliers: true                                  # 是否裁剪异常值
  outlier_threshold: 3.0                               # 异常值裁剪阈值（标准差的倍数）
  
# 模型相关配置
model:
  type: "lstm"                                         # 模型类型：lstm/gru/transformer
  lookback_period: 48                                  # 回溯期（1天 = 48个半小时）
  lstm_units: [256, 128]                               # LSTM层神经元数量
  dropout: 0.3                                         # Dropout比例
  recurrent_dropout: 0.2                               # 循环层Dropout比例
  batch_size: 64                                       # 批处理大小
  epochs: 150                                          # 训练轮数
  early_stopping_patience: 15                          # 早停耐心值
  learning_rate: 0.001                                 # 学习率
  optimizer: "adam"                                    # 优化器
  
# 标签生成相关配置
labels:
  forward_period: 15                                   # 预测未来15小时
  label_type: "dynamic_atr"                            # 标签类型：fixed/dynamic_atr
  atr_multiple: 0.8                                    # 动态阈值ATR倍数
  fixed_threshold: 0.005                               # 固定阈值（如果使用）
  
# 资金管理相关配置
capital_management:
  base_capital: 10000.0                                # 基础资金量
  risk_profile: "balanced"                             # 风险偏好：conservative/balanced/aggressive
  max_risk_per_trade: 0.03                             # 单笔交易最大风险
  position_sizing_method: "fibonacci"                  # 仓位大小方法：fixed/fibonacci/risk_parity
  
  # 斐波那契资金管理配置
  fibonacci:
    trending_allocation: 0.03                          # 顺势资金分配比例
    normal_allocation: 0.02                            # 正常资金分配比例
    counter_trend_allocation: 0.01                     # 逆势资金分配比例
    extreme_allocation: 0.005                          # 极端逆势资金分配比例
  
  # 止损止盈配置
  stop_loss:
    method: "dynamic_atr"                              # 止损方法：fixed/dynamic_atr/fibonacci
    fixed_percentage: 0.02                             # 固定止损百分比
    atr_multiple: 2.0                                  # ATR倍数
  
  take_profit:
    method: "risk_reward"                              # 止盈方法：fixed/risk_reward
    fixed_percentage: 0.05                             # 固定止盈百分比
    risk_reward_ratio: 2.0                             # 风险回报比
  
# 回测相关配置
backtesting:
  start_date: "2022-01-01"                             # 回测开始日期
  end_date: "2023-12-31"                               # 回测结束日期
  commission: 0.001                                    # 交易手续费
  slippage: 0.0005                                     # 滑点
  initial_capital: 10000.0                             # 初始资金
  
  # 评估指标
  metrics:
    accuracy_weight: 0.3                               # 准确率权重
    f1_score_weight: 0.2                               # F1分数权重
    return_weight: 0.3                                 # 收益率权重
    drawdown_weight: 0.2                               # 回撤权重
  
# 交易相关配置
trading:
  mode: "paper"                                        # 交易模式：paper/live
  update_frequency: "manual"                           # 更新频率：manual/daily/weekly/monthly
  signal_threshold: 0.6                                # 信号阈值（概率）
  webhook_url: "https://your-webhook-endpoint.com"     # Webhook URL
  
  # 币安API配置
  binance:
    api_key: "YOUR_API_KEY"                            # API密钥（请替换为实际值）
    api_secret: "YOUR_API_SECRET"                      # API密钥（请替换为实际值）
    testnet: true                                      # 是否使用测试网络
  
  # 通知配置
  notifications:
    enabled: true                                      # 是否启用通知
    send_on_signal: true                               # 信号生成时发送
    send_on_trade: true                                # 交易执行时发送
    send_on_error: true                                # 错误发生时发送

# 日志配置
logging:
  level: "INFO"                                        # 日志级别：DEBUG/INFO/WARNING/ERROR
  file: "logs/trading_system.log"                      # 日志文件路径
  max_file_size: 10485760                              # 最大日志文件大小（10MB）
  backup_count: 5                                      # 备份文件数量
"""
    
    # 写入配置文件
    with open(config_path, 'w', encoding='utf-8') as file:
        file.write(default_config)
    
    print(f"已创建默认配置文件: {config_path}")


def check_dependencies():
    """检查必要的Python库是否已安装"""
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'tensorflow': 'tensorflow',
        'scikit-learn': 'scikit-learn',
        'pyyaml': 'PyYAML',
        'scipy': 'scipy',
        'requests': 'requests'
    }
    
    missing_packages = []
    
    print("检查必要的Python库...")
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("\n请安装以下缺失的Python库：")
        install_cmd = f"pip install {' '.join(missing_packages)}"
        print(f"\n{install_cmd}\n")
        
        # 询问是否自动安装
        print("是否要自动安装这些库？(y/n)")
        choice = input("> ")
        
        if choice.lower() == 'y':
            print("\n正在安装缺失的库...")
            try:
                subprocess.run(install_cmd, shell=True, check=True)
                print("库安装完成！")
            except subprocess.CalledProcessError:
                print("安装过程中出现错误，请手动安装。")
                return False
        else:
            print("\n请手动安装缺失的库后再运行程序。")
            return False
    
    return True


def install_dependencies():
    """安装所有依赖"""
    clear_screen()
    show_header()
    print("【安装依赖】\n")
    
    # 检查requirements.txt是否存在
    req_path = "requirements.txt"
    if not os.path.exists(req_path):
        print(f"错误：找不到{req_path}文件")
        input("\n按回车键返回主菜单...")
        return
    
    print(f"将从{req_path}安装所有依赖，这可能需要几分钟时间...\n")
    
    # 显示requirements.txt内容
    with open(req_path, 'r') as f:
        requirements = f.read()
    print("将安装以下依赖：")
    print(requirements)
    
    print("\n确认安装？(y/n)")
    confirm = input("> ")
    
    if confirm.lower() != 'y':
        print("\n已取消安装。")
        input("\n按回车键返回主菜单...")
        return
    
    print("\n开始安装依赖...\n")
    cmd = f"pip install -r {req_path}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("\n依赖安装完成！")
    except subprocess.CalledProcessError:
        print("\n安装过程中出现错误。")
    
    input("\n按回车键返回主菜单...")


def configure_system_resources():
    """配置系统资源"""
    clear_screen()
    show_header()
    print("【系统资源配置】\n")
    
    # 检测系统资源
    try:
        import psutil
        total_cpu_cores = psutil.cpu_count(logical=False)
        total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        has_gpu = False
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            has_gpu = len(gpus) > 0
        except:
            pass
    except:
        total_cpu_cores = 1
        total_memory_gb = 1
        has_gpu = False
    
    print(f"检测到的系统资源：")
    print(f"- CPU核心数：{total_cpu_cores}")
    print(f"- 系统内存：{total_memory_gb:.1f}GB")
    print(f"- GPU可用：{'是' if has_gpu else '否'}\n")
    
    # 询问CPU核心数
    print(f"请输入要使用的CPU核心数（1-{total_cpu_cores}）[默认: 1]：")
    cpu_cores = input("> ").strip()
    try:
        cpu_cores = int(cpu_cores)
        if cpu_cores < 1 or cpu_cores > total_cpu_cores:
            print(f"无效的输入，使用默认值：1")
            cpu_cores = 1
    except ValueError:
        print(f"无效的输入，使用默认值：1")
        cpu_cores = 1
    
    # 询问内存限制
    print(f"\n请输入要使用的内存大小（GB，1-{int(total_memory_gb)}）[默认: 1]：")
    memory_limit = input("> ").strip()
    try:
        memory_limit = float(memory_limit)
        if memory_limit < 1 or memory_limit > total_memory_gb:
            print(f"无效的输入，使用默认值：1")
            memory_limit = 1
    except ValueError:
        print(f"无效的输入，使用默认值：1")
        memory_limit = 1
    
    # 询问是否启用GPU
    if has_gpu:
        print("\n是否启用GPU？(y/n) [默认: n]：")
        use_gpu = input("> ").strip().lower() == 'y'
    else:
        use_gpu = False
    
    # 更新配置文件
    config_path = 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # 添加或更新系统资源配置
    if 'system_resources' not in config:
        config['system_resources'] = {}
    
    config['system_resources'].update({
        'cpu': {
            'num_threads': cpu_cores,
            'thread_affinity': False,
            'priority': 'normal'
        },
        'gpu': {
            'enabled': use_gpu,
            'device_ids': [0] if use_gpu else [],
            'memory_limit': 0.8,
            'mixed_precision': True,
            'xla_acceleration': True
        },
        'memory': {
            'limit_gb': memory_limit,
            'data_loading_batch': 10000,
            'prefetch_buffer_size': 5,
            'cache_data_in_memory': True,
            'garbage_collection_threshold': 0.8
        }
    })
    
    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    
    print("\n系统资源配置已更新！")
    input("\n按回车键继续...")

def export_data_from_database():
    """从数据库导出数据"""
    clear_screen()
    show_header()
    print("【数据库导出】\n")
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # 获取数据库配置
    if 'database' not in config:
        print("错误：配置文件中未找到数据库配置")
        input("\n按回车键继续...")
        return
    
    db_config = config['database']
    
    print("请选择数据源：")
    print("1. MySQL")
    print("2. PostgreSQL")
    print("3. SQLite")
    print("4. MongoDB")
    print("0. 返回")
    
    choice = input("\n请选择: ")
    
    if choice == '0':
        return
    
    # 获取数据库连接信息
    print("\n请输入数据库连接信息：")
    
    if choice in ['1', '2']:  # MySQL或PostgreSQL
        host = input("主机地址 [localhost]: ").strip() or 'localhost'
        port = input("端口 [3306/5432]: ").strip() or ('3306' if choice == '1' else '5432')
        database = input("数据库名称: ").strip()
        username = input("用户名: ").strip()
        password = input("密码: ").strip()
        
        try:
            if choice == '1':  # MySQL
                import mysql.connector
                conn = mysql.connector.connect(
                    host=host,
                    port=int(port),
                    database=database,
                    user=username,
                    password=password
                )
            else:  # PostgreSQL
                import psycopg2
                conn = psycopg2.connect(
                    host=host,
                    port=int(port),
                    database=database,
                    user=username,
                    password=password
                )
        except Exception as e:
            print(f"\n连接数据库失败: {str(e)}")
            input("\n按回车键继续...")
            return
            
    elif choice == '3':  # SQLite
        database = input("数据库文件路径: ").strip()
        try:
            import sqlite3
            conn = sqlite3.connect(database)
        except Exception as e:
            print(f"\n连接数据库失败: {str(e)}")
            input("\n按回车键继续...")
            return
            
    elif choice == '4':  # MongoDB
        host = input("主机地址 [localhost]: ").strip() or 'localhost'
        port = input("端口 [27017]: ").strip() or '27017'
        database = input("数据库名称: ").strip()
        collection = input("集合名称: ").strip()
        username = input("用户名 (可选): ").strip()
        password = input("密码 (可选): ").strip()
        
        try:
            from pymongo import MongoClient
            if username and password:
                uri = f"mongodb://{username}:{password}@{host}:{port}"
            else:
                uri = f"mongodb://{host}:{port}"
            client = MongoClient(uri)
            db = client[database]
            collection = db[collection]
        except Exception as e:
            print(f"\n连接数据库失败: {str(e)}")
            input("\n按回车键继续...")
            return
    
    # 导出数据
    print("\n正在导出数据...")
    try:
        os.makedirs('data', exist_ok=True)
        
        if choice in ['1', '2', '3']:  # SQL数据库
            cursor = conn.cursor()
            # 获取所有交易品种
            cursor.execute("SELECT DISTINCT symbol FROM crypto_data")
            symbols = [row[0] for row in cursor.fetchall()]
            
            for symbol in symbols:
                print(f"正在导出 {symbol} 的数据...")
                query = f"SELECT * FROM crypto_data WHERE symbol = %s ORDER BY timestamp"
                cursor.execute(query, (symbol,))
                
                # 将数据写入CSV文件
                filename = os.path.join('data', f'{symbol}.csv')
                with open(filename, 'w') as f:
                    # 写入表头
                    columns = [desc[0] for desc in cursor.description]
                    f.write(','.join(columns) + '\n')
                    
                    # 写入数据
                    for row in cursor:
                        f.write(','.join(map(str, row)) + '\n')
                
            cursor.close()
            conn.close()
            
        elif choice == '4':  # MongoDB
            symbols = collection.distinct('symbol')
            for symbol in symbols:
                print(f"正在导出 {symbol} 的数据...")
                cursor = collection.find({'symbol': symbol}).sort('timestamp', 1)
                
                # 将数据写入CSV文件
                filename = os.path.join('data', f'{symbol}.csv')
                df = pd.DataFrame(list(cursor))
                df.to_csv(filename, index=False)
        
        print("\n数据导出完成！")
        
    except Exception as e:
        print(f"\n导出数据时出错: {str(e)}")
    
    input("\n按回车键继续...")

def main():
    """主函数"""
    # 检查依赖
    if not check_dependencies():
        print("程序无法运行，因为缺少必要的库。")
        sys.exit(1)
    
    # 确保所有必要的目录存在
    ensure_directories_exist()
    
    # 确保配置文件存在
    ensure_config_exists()
    
    # 配置系统资源
    configure_system_resources()
    
    # 询问是否需要从数据库导出数据
    print("\n是否需要从数据库导出数据？(y/n) [默认: n]：")
    if input("> ").strip().lower() == 'y':
        export_data_from_database()
    
    while True:
        clear_screen()
        show_header()
        show_menu()
        
        choice = input("\n请输入选项: ")
        
        if choice == '1':
            train_model()
        elif choice == '2':
            resume_training()
        elif choice == '3':
            backtest_model()
        elif choice == '4':
            generate_prediction()
        elif choice == '5':
            deploy_service()
        elif choice == '6':
            view_reports()
        elif choice == '7':
            check_system_status()
        elif choice == '8':
            toggle_gpu_mode()
        elif choice == '9':
            install_dependencies()
        elif choice == '0':
            print("\n感谢使用，再见！")
            sys.exit(0)
        else:
            print("\n无效的选择，请重试。")
            time.sleep(1)


# 这里的if语句确保只有在直接运行此脚本时才执行main()函数
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已被用户中断。")
        sys.exit(0)