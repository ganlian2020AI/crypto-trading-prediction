# 加密货币交易预测系统

这是一个完整的加密货币交易预测系统，包含数据处理、特征工程、模型训练、回测和实盘交易功能。系统基于深度学习模型预测未来价格走势，并通过自定义指标和斐波那契资金管理策略优化交易决策。

## 系统特点

- **多品种支持**：同时处理多种加密货币的数据和交易
- **自定义技术指标**：包含EMA、RSI、ATR及多种自定义指标
- **深度学习模型**：使用LSTM网络预测价格走势
- **斐波那契资金管理**：根据市场状态动态调整仓位
- **完整回测系统**：考虑交易成本的多品种回测，详细的绩效评估
- **实盘交易支持**：连接交易所API，执行实时交易
- **多数据库支持**：支持SQLite、MySQL和PostgreSQL

## 系统架构

```
crypto_project/
├── data/                   # 原始和处理后的数据
├── models/                 # 训练好的模型
├── checkpoints/            # 训练检查点
├── reports/                # 回测报告
├── logs/                   # 日志文件
├── trading_system/         # 交易系统
│   ├── alerts/             # 警报记录
│   ├── backtest/           # 回测模块
│   ├── common/             # 公共工具
│   ├── config/             # 配置文件
│   ├── data/               # 交易数据
│   ├── docs/               # 文档
│   ├── live/               # 实盘交易
│   ├── logs/               # 交易日志
│   ├── models/             # 模型引用
│   ├── reports/            # 交易报告
│   └── training/           # 训练模块
├── data_processor.py       # 数据处理模块
├── feature_engineering.py  # 特征工程模块
├── model.py                # 模型定义
├── main.py                 # 主程序
├── evaluator.py            # 评估模块
├── trader.py               # 交易模块
├── rsivolema.py            # RSI/量/EMA指标计算
└── fibonacci_position_manager.py  # 斐波那契仓位管理
```

## 特征列表及说明

| 特征名称     | 说明                       | 长度（周期） | 预处理/备注                    |
| -------- | ------------------------ | ------ | ------------------------- |
| EMA144   | 收盘价指数移动平均线               | 144    | 标准计算                      |
| EMA169   | 收盘价指数移动平均线               | 169    | 标准计算                      |
| EMA21    | 收盘价指数移动平均线               | 21     | 标准计算                      |
| HEMA     | Holt指数平滑EMA，长度20，步长20    | 20     | 标准计算                      |
| RSI      | 相对强弱指标                   | 通常14   | 标准计算                      |
| ATR      | 平均真实波幅                   | 通常14   | 标准计算                      |
| 自定义指标1   | 预测价格（连续值）                | 576    | 标准计算                      |
| 扩展自定义指标1 | 基于自定义指标1的高低价格区间转换为离散标签   | 576    | 标签值为{1,0,-1}，基于当前价与区间比较生成 |

## 安装指南

### 系统要求

- Python 3.8+
- 足够的内存（推荐8GB以上）
- 支持的操作系统：Windows、macOS、Linux

### 安装步骤

#### Linux/macOS

```bash
# 克隆仓库
git clone https://github.com/yourusername/crypto_project.git
cd crypto_project

# 运行安装脚本
chmod +x setup.sh
./setup.sh
```

#### Windows

```bash
# 克隆仓库
git clone https://github.com/yourusername/crypto_project.git
cd crypto_project

# 运行安装脚本
setup.bat
```

## 使用指南

### 配置

编辑`trading_system/config/trading_config.yaml`文件，设置您的交易参数、API密钥等。

### 数据下载

```bash
python trading_system/trade_app.py
# 在菜单中选择"6. 数据下载"
```

### 模型训练

```bash
python main.py --train
```

### 回测

```bash
python trading_system/trade_app.py
# 在菜单中选择"1. 模拟交易"
```

### 实盘交易

```bash
python trading_system/trade_app.py
# 在菜单中选择"2. 实盘交易"
```

## 资金管理策略

根据市场状态动态调整仓位比例：

| 状态   | 资金% | 说明                   |
|------|-----|----------------------|
| 顺势   | 3%  | 利用趋势优势积极建仓           |
| 正常   | 2%  | 谨慎持仓，适度参与市场         |
| 逆势   | 1%  | 控制风险，轻仓观望或防守操作     |
| 极端逆势 | 0.5% | 仅保留少量仓位或全部观望 |

## 常见问题

### 内存不足问题

如果遇到内存不足问题，可以尝试以下解决方案：

1. 减小`config.yaml`中的`batch_size`
2. 减少处理的交易品种数量
3. 缩短回溯期（lookback_period）
4. 使用`system_resources.memory.data_loading_batch`参数控制数据加载批次大小

### API连接问题

如果无法连接到交易所API，请检查：

1. API密钥是否正确
2. 网络连接是否正常
3. 是否需要使用代理（可在配置文件中设置）

## 开发计划

请参考[TODO.md](TODO.md)文件了解项目开发计划和进度。

## 许可证

[MIT License](LICENSE)
