# 模型存储目录

此目录用于存放训练好的模型文件和模型元数据。

## 目录结构

```
saved_models/
  ├── models_metadata.json    # 模型元数据文件
  ├── model_name_1.keras      # 模型文件
  ├── model_name_2.keras      # 模型文件
  └── ...
```

## 模型元数据

`models_metadata.json` 文件包含所有模型的元数据信息，格式如下：

```json
{
  "model_id_1": {
    "filename": "model_name_1.keras",
    "name": "模型名称1",
    "version": "1.0",
    "type": "LSTM",
    "symbols": ["BTC", "ETH", "ADA"],
    "created_at": "2023-01-01T12:00:00",
    "downloaded_at": "2023-01-02T08:30:00",
    "file_hash": "sha256_hash_value",
    "url": "https://example.com/models/model_name_1.keras"
  },
  "model_id_2": {
    // ...
  }
}
```

## 模型管理

使用 `crypto-model` 命令行工具管理模型：

```bash
# 列出所有可用模型
crypto-model list

# 从远程服务器下载模型
crypto-model download --model-id lstm_btc_eth_ada_v1 --url https://example.com/models/lstm_btc_eth_ada_v1.keras

# 验证模型完整性
crypto-model verify --model-id lstm_btc_eth_ada_v1

# 删除模型
crypto-model delete --model-id lstm_btc_eth_ada_v1
```

## 注意事项

1. 不要直接修改 `models_metadata.json` 文件，使用 `crypto-model` 工具管理模型
2. 模型文件可能较大，确保有足够的磁盘空间
3. 定期备份重要的模型文件 