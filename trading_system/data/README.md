# 数据目录

此目录用于存放历史数据文件和数据库文件。

## 历史数据文件

系统支持CSV格式的历史数据文件，命名格式为`{symbol}.csv`，例如`BTC.csv`、`ETH.csv`等。

### CSV文件格式要求

每个CSV文件应包含以下列：

| 列名 | 类型 | 说明 |
|-----|------|-----|
| timestamp | datetime | 时间戳，格式为ISO日期时间或Unix时间戳 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| volume | float | 交易量 |

示例：
```
timestamp,open,high,low,close,volume
2023-01-01T00:00:00,16500.23,16750.45,16480.12,16720.34,1250.67
2023-01-01T01:00:00,16720.34,16800.56,16690.78,16750.23,980.45
...
```

## 数据库文件

系统默认使用SQLite数据库，数据库文件`trading_system.db`将存放在此目录中。

如果配置使用MySQL、MariaDB或PostgreSQL等外部数据库，则需要在相应的数据库服务器中创建数据库。

### 数据库配置

在`config/trading_config.yaml`中配置数据库：

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

### 数据库表结构

系统使用以下表结构存储数据：

1. **market_data**：市场数据表
2. **trades**：交易记录表
3. **predictions**：预测结果表
4. **account_state**：账户状态表

详细的表结构说明请参考`docs/数据库说明.md`。

## 注意事项

1. 确保历史数据的时间序列完整，无大量缺失值
2. 定期备份数据库文件，尤其是在实盘交易中
3. 对于大量历史数据，考虑使用数据分区或定期归档策略
4. 如使用外部数据库，确保数据库服务器有足够的磁盘空间和内存资源 