# 加密货币交易系统

一个完整的加密货币交易解决方案，支持模拟交易和实盘交易功能，并提供多种数据库支持。

## 系统特点

- **模拟交易**：使用历史数据进行回测，评估交易策略的有效性
- **实盘交易**：连接交易所API，执行实时交易操作
- **多品种支持**：支持BTC、ETH、ADA、BNB、SOL、XRP、DOGE等多种加密货币
- **风险管理**：内置风险控制机制，包括止损、资金管理等
- **报告生成**：生成详细的交易报告和权益曲线
- **数据库支持**：支持SQLite、MySQL/MariaDB和PostgreSQL数据库存储交易数据

## 目录结构

```
trading_system/
├── backtest/                # 回测模块
├── common/                  # 公共工具模块
│   ├── utils.py            # 工具类和函数
│   └── database.py         # 数据库支持模块
├── config/                  # 配置文件目录
│   └── trading_config.yaml # 主配置文件
├── data/                    # 数据目录
├── docs/                    # 文档目录
├── live/                    # 实盘交易模块
├── logs/                    # 日志目录
├── models/                  # 模型目录
├── reports/                 # 报告目录
└── trade_app.py             # 主应用程序
```

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行应用：
   ```bash
   python trade_app.py
   ```

3. 在主菜单中选择操作：
   - 模拟交易
   - 实盘交易
   - 查看历史报告
   - 系统设置

## 数据库支持

系统支持多种数据库类型，可以根据需要在配置文件中设置：

```yaml
database:
  # 数据库类型: sqlite, mysql, mariadb, postgresql
  type: "sqlite"
  # SQLite配置
  path: "data/trading_system.db"
  # MySQL/MariaDB/PostgreSQL配置
  host: "localhost"
  port: 3306  # MySQL/MariaDB默认端口，PostgreSQL为5432
  user: "root"
  password: ""
  database: "trading_system"
  log_level: "INFO"
```

数据库表结构详情请参考 `docs/数据库说明.md`。

## 模型文件

模型文件默认存放在 `models/` 目录下，支持以下格式：

- TensorFlow/Keras模型：`.keras`、`.h5`、`.pb`
- 其他机器学习模型：`.pkl`、`.joblib`

建议的模型命名格式：`model_[日期]_[版本]_[特征描述].[扩展名]`

模型使用详情请参考 `docs/模型说明.md`。

## 配置说明

系统的主要配置文件位于 `config/trading_config.yaml`，包含以下主要配置项：

- 基本配置（日志级别、时区等）
- 交易品种配置
- 模拟交易配置
- 实盘交易配置
- 风险管理配置
- 数据库配置
- 模型配置

## 文档

详细文档请参考 `docs/` 目录：

- [使用指南](docs/使用指南.md)
- [系统架构](docs/系统架构.md)
- [开发指南](docs/开发指南.md)
- [数据库说明](docs/数据库说明.md)
- [模型说明](docs/模型说明.md)

## 注意事项

- 实盘交易前请确保已正确配置API密钥
- 建议先在测试网络上进行测试
- 定期查看系统日志和报告
- 定期备份数据库文件

## 许可证

本项目采用 MIT 许可证。 