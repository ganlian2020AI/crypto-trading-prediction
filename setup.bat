@echo off
echo =====================================================
echo       加密货币交易系统安装脚本 (Windows)
echo =====================================================

REM 检查Python版本
echo 检查Python版本...
python --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 未找到Python，请先安装Python 3.8或更高版本
    exit /b 1
)

REM 创建虚拟环境
echo 创建虚拟环境...
if exist venv (
    echo 虚拟环境已存在，跳过创建
) else (
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo 错误: 创建虚拟环境失败
        exit /b 1
    )
    echo 虚拟环境创建成功
)

REM 激活虚拟环境
echo 激活虚拟环境...
call venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 激活虚拟环境失败
    exit /b 1
)

REM 安装依赖
echo 安装依赖...
pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo 警告: 安装依赖时出现错误，请检查日志
) else (
    echo 依赖安装完成
)

REM 创建必要的目录
echo 创建必要的目录...
if not exist data mkdir data
if not exist logs mkdir logs
if not exist models mkdir models
if not exist reports mkdir reports
if not exist trading_system\logs mkdir trading_system\logs
if not exist trading_system\reports mkdir trading_system\reports
if not exist trading_system\alerts mkdir trading_system\alerts
if not exist trading_system\data mkdir trading_system\data

REM 复制配置文件模板
echo 设置配置文件...
if not exist trading_system\config\trading_config.yaml (
    copy trading_system\config\trading_config.yaml.template trading_system\config\trading_config.yaml
    echo 已创建交易系统配置文件，请编辑 trading_system\config\trading_config.yaml 填写您的配置
) else (
    echo 交易系统配置文件已存在，跳过创建
)

REM 安装完成
echo =====================================================
echo 安装完成!
echo 使用方法:
echo 1. 激活虚拟环境: venv\Scripts\activate.bat
echo 2. 编辑配置文件: trading_system\config\trading_config.yaml
echo 3. 运行交易应用: python trading_system\trade_app.py
echo =====================================================

pause