#!/bin/bash

# 加密货币交易系统安装脚本

echo "====================================================="
echo "      加密货币交易系统安装脚本"
echo "====================================================="

# 检查Python版本
echo "检查Python版本..."
if command -v python3 &>/dev/null; then
    python_version=$(python3 --version)
    echo "发现 $python_version"
else
    echo "错误: 未找到Python 3，请先安装Python 3.8或更高版本"
    exit 1
fi

# 创建虚拟环境
echo "创建虚拟环境..."
if [ -d "venv" ]; then
    echo "虚拟环境已存在，跳过创建"
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "错误: 创建虚拟环境失败"
        exit 1
    fi
    echo "虚拟环境创建成功"
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "错误: 激活虚拟环境失败"
    exit 1
fi

# 安装依赖
echo "安装依赖..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "警告: 安装依赖时出现错误，请检查日志"
else
    echo "依赖安装完成"
fi

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p data
mkdir -p logs
mkdir -p models
mkdir -p reports
mkdir -p trading_system/logs
mkdir -p trading_system/reports
mkdir -p trading_system/alerts
mkdir -p trading_system/data

# 复制配置文件模板
echo "设置配置文件..."
if [ ! -f "trading_system/config/trading_config.yaml" ]; then
    cp trading_system/config/trading_config.yaml.template trading_system/config/trading_config.yaml
    echo "已创建交易系统配置文件，请编辑 trading_system/config/trading_config.yaml 填写您的配置"
else
    echo "交易系统配置文件已存在，跳过创建"
fi

# 安装完成
echo "====================================================="
echo "安装完成!"
echo "使用方法:"
echo "1. 激活虚拟环境: source venv/bin/activate"
echo "2. 编辑配置文件: trading_system/config/trading_config.yaml"
echo "3. 运行交易应用: python trading_system/trade_app.py"
echo "====================================================="