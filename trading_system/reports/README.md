# 报告目录

此目录用于存放系统生成的各类报告文件，包括回测报告、训练评估报告和交易报告等。

## 报告类型

### 回测报告

回测报告包含回测结果的详细信息，包括：

- 性能指标（收益率、最大回撤、夏普比率等）
- 交易记录
- 权益曲线图
- 持仓分析
- 交易统计

文件命名格式：`backtest_report_{策略名称}_{开始日期}_{结束日期}.{格式}`

例如：
- `backtest_report_lstm_strategy_20230101_20231231.html`
- `backtest_report_lstm_strategy_20230101_20231231.pdf`
- `backtest_report_lstm_strategy_20230101_20231231.json`

### 训练评估报告

训练评估报告包含模型训练和评估的结果，包括：

- 模型性能指标（准确率、精确率、召回率、F1分数等）
- 学习曲线
- 混淆矩阵
- 特征重要性
- 超参数优化结果

文件命名格式：`training_report_{模型名称}_{训练日期}.{格式}`

例如：
- `training_report_lstm_btc_eth_ada_20230101.html`
- `training_report_lstm_btc_eth_ada_20230101.pdf`

### 交易报告

交易报告包含实盘交易的结果，包括：

- 交易记录
- 盈亏分析
- 权益曲线
- 持仓分析
- 交易统计

文件命名格式：`trading_report_{策略名称}_{开始日期}_{结束日期}.{格式}`

例如：
- `trading_report_lstm_strategy_20230101_20231231.html`
- `trading_report_lstm_strategy_20230101_20231231.pdf`

## 报告格式

系统支持以下报告格式：

- HTML：交互式网页报告，包含图表和表格
- PDF：适合打印和分享的静态报告
- JSON：结构化数据，适合进一步处理和分析
- CSV：表格数据，适合导入到电子表格软件

## 报告生成

使用相应的命令行工具生成报告：

```bash
# 生成回测报告
crypto-backtest --config config/trading_config.yaml --report --plot

# 生成训练评估报告
crypto-train evaluate --model models/saved_models/my_model.keras --report

# 生成交易报告
crypto-live report --start-date 2023-01-01 --end-date 2023-12-31 --report
```

## 注意事项

1. 报告文件可能较大，定期清理不需要的报告
2. HTML报告需要浏览器打开，PDF报告需要PDF阅读器
3. 重要的报告应定期备份
4. 报告中可能包含敏感信息，注意保护