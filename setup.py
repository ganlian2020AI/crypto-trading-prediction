#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# 基础依赖包
base_requirements = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "pyyaml>=6.0",
    "requests>=2.26.0",
]

# 实盘交易系统依赖包
live_requirements = [
    "ccxt>=2.0.0",
    "schedule>=1.1.0",
    "PySocks>=1.7.1",  # 用于SOCKS代理支持
]

# 回测系统依赖包
backtest_requirements = [
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
]

# 训练系统依赖包
training_requirements = [
    "scikit-learn>=1.0.0",
    "tensorflow>=2.6.0",
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
    "scipy>=1.7.0",
    "joblib>=1.1.0",
]

# 数据下载和处理依赖包
data_requirements = [
    "ccxt>=2.0.0",
    "PySocks>=1.7.1",
    "aiohttp>=3.8.1",
    "pymysql>=1.0.2",
    "psycopg2-binary>=2.9.3",
]

# 警报和通知依赖包
alert_requirements = [
    "twilio>=7.0.0",
]

# 模型管理依赖包
model_requirements = [
    "requests>=2.26.0",
]

# 全部依赖包
all_requirements = list(set(
    base_requirements +
    live_requirements +
    backtest_requirements +
    training_requirements +
    data_requirements +
    alert_requirements +
    model_requirements
))

setup(
    name="crypto_trading_system",
    version="0.1.0",
    description="加密货币交易系统",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=base_requirements,
    extras_require={
        "live": live_requirements,
        "backtest": backtest_requirements,
        "training": training_requirements,
        "data": data_requirements,
        "alerts": alert_requirements,
        "model": model_requirements,
        "all": all_requirements,
    },
    entry_points={
        "console_scripts": [
            "crypto-live=trading_system.live.cli:main [live]",
            "crypto-backtest=trading_system.backtest.cli:main [backtest]",
            "crypto-train=trading_system.training.cli:main [training]",
            "crypto-data=trading_system.common.data_downloader:main [data]",
            "crypto-model=trading_system.models.model_manager:main [model]",
            "crypto-app=trading_system.trade_app:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 