# 模型文件目录

此目录用于存放训练好的模型文件，系统将从这里加载模型用于预测和交易。

## 支持的模型格式

系统支持以下模型格式：

1. **TensorFlow/Keras模型**
   - `.keras`文件格式（TensorFlow 2.6+推荐格式）
   - `.h5`文件格式（HDF5格式，兼容旧版本）
   - `.pb`文件格式（SavedModel格式）

2. **其他机器学习模型**
   - `.pkl`文件格式（使用pickle序列化的scikit-learn模型）
   - `.joblib`文件格式（使用joblib序列化的模型）

## 命名规范

为了便于管理和版本控制，请使用以下命名规范：

```
model_[日期]_[版本]_[特征描述].[扩展名]
```

例如：
- `model_20230101_v1_basic.keras`
- `model_20230215_v2_with_sentiment.h5`
- `model_20230320_v3_ensemble.keras`

## 模型选择

系统会根据以下规则选择模型：

1. 如果在配置文件中指定了`default_model`，则使用该模型
2. 如果未指定，则使用此目录中最新的模型文件（按文件名排序）
3. 在用户界面中，可以手动选择要使用的模型

## 模型结构要求

### TensorFlow/Keras模型

模型应该满足以下输入/输出要求：

1. **输入层**：
   - 形状为`(None, sequence_length, features)`的3D张量
   - `sequence_length`为时间窗口长度
   - `features`为特征数量

2. **输出层**：
   - 对于回归模型：形状为`(None, prediction_horizon)`的2D张量
   - 对于分类模型：形状为`(None, num_classes)`的2D张量

## 注意事项

1. 请勿在此目录中存放非模型文件
2. 定期清理旧的或不再使用的模型文件
3. 保留模型训练的相关记录，如训练参数、性能指标等
4. 确保模型与系统使用的TensorFlow版本兼容
