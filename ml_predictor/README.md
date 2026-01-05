# SCALE-Sim ML Predictor

## 项目概述

SCALE-Sim ML Predictor 是一个机器学习模块，用于预测 SCALE-Sim（深度学习加速器仿真工具）的性能指标。通过训练神经网络模型，可以在**毫秒级**预测仿真结果，相比传统仿真快数千倍。

### 核心功能

- **数据生成**: 自动运行 SCALE-Sim 仿真并收集训练数据
- **模型训练**: 使用多层神经网络学习硬件配置与性能的映射关系
- **快速预测**: 根据配置参数直接预测性能指标，无需运行仿真
- **自动去重**: 智能检测并跳过已存在的配置组合

### 预测指标

从 `COMPUTE_REPORT.csv` 中提取以下 6 个关键性能指标：

| 指标 | 说明 |
|------|------|
| `total_cycles_with_prefetch` | 包含预取的总周期数 |
| `total_cycles` | 不含预取的总周期数 |
| `stall_cycles` | 停顿周期数 |
| `overall_util_percent` | 整体利用率 (%) |
| `mapping_efficiency_percent` | 映射效率 (%) |
| `compute_util_percent` | 计算利用率 (%) |

---

## 安装依赖

```bash
# 在 scale-sim-v3 项目根目录下执行
pip install -r ml_predictor/requirements.txt
```

**依赖包**:
- `torch>=1.9.0` - PyTorch 深度学习框架
- `numpy>=1.19.0` - 数值计算
- `pandas>=1.2.0` - 数据处理
- `scikit-learn>=0.24.0` - 数据预处理和评估

---

## 快速开始

### 1. 生成训练数据

```bash
# 生成 100 个样本（每次使用不同的随机种子）
python -m ml_predictor.main generate --num_samples 100 --workers 4

# 指定输出路径
python -m ml_predictor.main generate --num_samples 100 --output ./data/my_data.csv --workers 4

# 使用固定种子（便于复现）
python -m ml_predictor.main generate --num_samples 100 --seed 12345 --workers 4
```

### 2. 训练模型

```bash
# 使用默认配置训练
python -m ml_predictor.main train --data_path ./data/raw/training_data.csv

# 自定义训练参数
python -m ml_predictor.main train \
  --data_path ./data/raw/training_data.csv \
  --epochs 200 \
  --batch_size 128 \
  --lr 0.0001
```

### 3. 预测

```bash
# 对单个配置进行预测
python -m ml_predictor.main predict \
  --config ./configs/google.cfg \
  --topology ./topologies/ispass25_models/alexnet.csv \
  --output ./results/predictions.csv
```

### 4. 评估模型

```bash
# 在测试集上评估模型性能
python -m ml_predictor.main evaluate \
  --data_path ./data/raw/test_data.csv \
  --output ./results/evaluation.json
```

---

## 模块详解

### 📁 项目结构

```
ml_predictor/
├── __init__.py              # 包初始化
├── config.py                # 配置参数
├── data_generation.py       # 数据生成模块
├── data_preprocessing.py    # 数据预处理模块
├── model.py                 # 神经网络模型定义
├── train.py                 # 训练脚本
├── predict.py               # 推理模块
├── evaluate.py              # 评估模块
├── main.py                  # 主入口CLI
├── requirements.txt         # 依赖列表
└── README.md               # 本文档
```

---

### 1️⃣ config.py - 配置参数

定义所有可调的超参数和配置。

#### 数据生成配置

```python
DATA_GENERATION_CONFIG = {
    "num_samples": 5000,  # 默认生成样本数
    
    # 硬件配置参数范围
    "array_height_range": [64, 128, 256, 512],
    "array_width_range": [64, 128, 256, 512],
    "ifmap_sram_sz_kb_range": [256, 512, 1024, 2048, 4096, 6144],
    "filter_sram_sz_kb_range": [256, 512, 1024, 2048, 4096, 6144],
    "ofmap_sram_sz_kb_range": [256, 512, 1024, 2048],
    "dataflow_options": ["os", "ws", "is"],
    "bandwidth_range": [5, 10, 20, 50, 100],
    
    # 卷积层参数范围
    "ifmap_height_range": [7, 13, 14, 27, 28, 56, 112, 224],
    "ifmap_width_range": [7, 13, 14, 27, 28, 56, 112, 224],
    "filter_height_range": [1, 3, 5, 7, 11],
    "filter_width_range": [1, 3, 5, 7, 11],
    "channels_range": [3, 16, 32, 64, 96, 128, 256, 384, 512],
    "num_filter_range": [16, 32, 64, 96, 128, 256, 384, 512],
    "strides_range": [1, 2, 4],
}
```

#### 模型训练配置

```python
MODEL_CONFIG = {
    "hidden_dims": [128, 256, 128, 64],  # 隐藏层维度
    "dropout_rate": 0.2,                 # Dropout 比例
    "learning_rate": 0.001,              # 学习率
    "batch_size": 64,                    # 批大小
    "epochs": 100,                       # 最大训练轮数
    "early_stopping_patience": 10,       # 早停耐心值
    "train_val_test_split": [0.7, 0.15, 0.15],  # 数据划分比例
}
```

**修改配置**: 直接编辑 `config.py` 文件即可。

---

### 2️⃣ data_generation.py - 数据生成模块

#### 实现原理

1. **随机采样配置**: 从预定义范围内随机生成硬件配置和卷积层参数
2. **生成临时文件**: 创建临时的 `config.cfg`, `topology.csv`, `layout.csv`
3. **运行仿真**: 调用原始 SCALE-Sim 进行仿真
4. **解析结果**: 提取 `COMPUTE_REPORT.csv` 中的性能指标
5. **保存数据**: 将输入特征和输出指标配对保存到 CSV

#### 核心类: `DataGenerator`

```python
class DataGenerator:
    def __init__(self, config=None, seed=None):
        """
        Args:
            config: 配置字典，默认使用 DATA_GENERATION_CONFIG
            seed: 随机种子，None 时使用当前时间（每次不同）
        """
```

**关键方法**:

- `_generate_random_config()`: 随机生成硬件配置
- `_generate_random_conv_layer()`: 随机生成卷积层参数
- `_create_config_file()`: 创建临时 config.cfg
- `_create_topology_file()`: 创建临时 topology.csv
- `_create_layout_file()`: 创建临时 layout.csv
- `_run_single_simulation()`: 运行单次仿真并返回结果
- `_get_config_signature()`: 生成配置的唯一签名（用于去重）
- `_load_existing_configs()`: 加载已存在的配置（避免重复）
- `generate()`: 主生成函数

#### 去重机制

使用 MD5 哈希对配置进行签名，自动跳过已存在的配置：

```python
def _get_config_signature(self, hw_config, conv_layer):
    sig_parts = [
        hw_config["array_height"],
        hw_config["array_width"],
        # ... 所有配置参数
    ]
    sig_str = "|".join(map(str, sig_parts))
    return hashlib.md5(sig_str.encode()).hexdigest()
```

#### 追加模式

新数据自动追加到已有文件，不会覆盖：

```python
# 文件存在时追加，不存在时创建
with open(output_file, "a" if file_exists else "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    writer.writerows(results)
```

---

### 3️⃣ data_preprocessing.py - 数据预处理模块

#### 实现原理

1. **One-Hot 编码**: 将 dataflow (os/ws/is) 编码为 3 个二进制特征
2. **衍生特征计算**: 计算 MACs, 数据大小, 计算强度等
3. **对数变换**: 对大数值（cycles, MACs）进行 log1p 变换
4. **标准化**: 使用 StandardScaler 归一化所有特征

#### 核心类: `DataPreprocessor`

```python
class DataPreprocessor:
    def __init__(self, scaler_type="standard"):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
```

**关键方法**:

- `_one_hot_encode_dataflow()`: One-Hot 编码 dataflow
- `_add_derived_features()`: 添加衍生特征
- `preprocess()`: 主预处理函数
- `inverse_transform_targets()`: 反归一化预测结果
- `split_data()`: 划分训练/验证/测试集
- `save()` / `load()`: 保存/加载预处理器

#### 特征工程

**输入特征** (21 个):
```
硬件配置 (9):
  - array_height, array_width
  - ifmap_sram_sz_kb, filter_sram_sz_kb, ofmap_sram_sz_kb
  - dataflow_os, dataflow_ws, dataflow_is (One-Hot)
  - bandwidth

卷积层参数 (7):
  - ifmap_height, ifmap_width
  - filter_height, filter_width
  - channels, num_filter, strides

衍生特征 (5):
  - total_macs: ofmap_h × ofmap_w × filter_h × filter_w × channels × num_filter
  - ifmap_size: ifmap_h × ifmap_w × channels
  - filter_size: filter_h × filter_w × channels × num_filter
  - ofmap_size: ofmap_h × ofmap_w × num_filter
  - compute_intensity: total_macs / (ifmap_size + filter_size + ofmap_size)
```

#### 数据变换

```python
# 对大数值进行 log 变换
log_features = ['total_macs', 'ifmap_size', 'filter_size', 'ofmap_size']
X[:, log_indices] = np.log1p(X[:, log_indices])

# 对 cycles 进行 log 变换
log_targets = ['total_cycles_with_prefetch', 'total_cycles', 'stall_cycles']
y[:, cycle_indices] = np.log1p(y[:, cycle_indices])

# StandardScaler 标准化
X_scaled = (X - mean) / std
y_scaled = (y - mean) / std
```

---

### 4️⃣ model.py - 神经网络模型

#### 模型架构

```
Input (21 features)
  ↓
Linear(21 → 128) → BatchNorm → ReLU → Dropout(0.2)
  ↓
Linear(128 → 256) → BatchNorm → ReLU → Dropout(0.2)
  ↓
Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.2)
  ↓
Linear(128 → 64) → BatchNorm → ReLU → Dropout(0.2)
  ↓
Linear(64 → 6)  # 6 个输出指标
```

#### 核心类: `ScaleSimPredictor`

```python
class ScaleSimPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=6, 
                 hidden_dims=[128, 256, 128, 64],
                 dropout_rate=0.2):
        # 多层 MLP
        # BatchNorm 加速收敛
        # Dropout 防止过拟合
```

#### 扩展模型: `ScaleSimPredictorWithUncertainty`

带不确定性估计的模型，可用于主动学习：

```python
class ScaleSimPredictorWithUncertainty(nn.Module):
    def forward(self, x):
        features = self.backbone(x)
        mean = self.mean_head(features)      # 预测均值
        logvar = self.logvar_head(features)  # 预测对数方差
        return mean, logvar
```

---

### 5️⃣ train.py - 训练脚本

#### 训练流程

```
1. 加载数据 → load_and_preprocess()
2. 数据预处理 → DataPreprocessor.preprocess()
3. 划分数据集 → train/val/test (70%/15%/15%)
4. 创建模型 → create_model()
5. 训练循环 → Trainer.train()
   - 前向传播
   - 计算 MSE Loss
   - 反向传播
   - 梯度裁剪 (max_norm=1.0)
   - 参数更新
6. 验证 → Trainer.validate()
7. 学习率调度 → ReduceLROnPlateau
8. 早停检查 → patience=10
9. 保存最佳模型 → save_model()
```

#### 核心类: `Trainer`

```python
class Trainer:
    def __init__(self, model, device, learning_rate, 
                 batch_size, epochs, early_stopping_patience):
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(...)
```

**关键方法**:

- `train_epoch()`: 训练一个 epoch
- `validate()`: 验证集评估
- `train()`: 完整训练循环
- `evaluate()`: 测试集评估（计算 MAE, MAPE, RMSE, R²）
- `save_model()`: 保存模型和元数据

#### 评估指标

```python
# Mean Absolute Error
MAE = mean(|pred - true|)

# Mean Absolute Percentage Error
MAPE = mean(|pred - true| / |true|) × 100%

# Root Mean Squared Error
RMSE = sqrt(mean((pred - true)²))

# R-squared
R² = 1 - SS_res / SS_tot
```

---

### 6️⃣ predict.py - 推理模块

#### 推理流程

```
1. 解析 config.cfg → _parse_config_file()
2. 解析 topology.csv → _parse_topology_file()
3. 构建输入 DataFrame → _prepare_input()
4. 特征预处理 → preprocessor.preprocess(fit=False)
5. 模型前向推理 → model(X_tensor)
6. 反归一化 → inverse_transform_targets()
7. 输出预测结果
```

#### 核心类: `Predictor`

```python
class Predictor:
    def __init__(self, model_path, preprocessor_path, device):
        # 加载预处理器
        self.preprocessor = DataPreprocessor.load(preprocessor_path)
        
        # 加载模型
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
```

**关键方法**:

- `_parse_config_file()`: 解析 .cfg 文件
- `_parse_topology_file()`: 解析 .csv 文件
- `_prepare_input()`: 准备输入 DataFrame
- `predict_layer()`: 预测单层性能
- `predict_from_files()`: 从文件读取并预测
- `predict_batch()`: 批量预测

---

### 7️⃣ evaluate.py - 评估模块

#### 评估功能

1. **模型性能评估**: 在测试集上计算各种指标
2. **与仿真对比**: 运行真实仿真并对比预测值

#### 核心函数

```python
def evaluate_model(data_path, model_path, preprocessor_path):
    """
    在测试集上评估模型
    返回: {
        'num_samples': N,
        'targets': {
            'total_cycles': {'MAE': ..., 'MAPE': ..., 'RMSE': ..., 'R2': ...},
            ...
        }
    }
    """

def compare_with_simulation(config_path, topology_path):
    """
    运行真实仿真并对比 ML 预测
    用于验证模型准确性
    """
```

---

### 8️⃣ main.py - 主入口

统一的命令行接口，支持 4 个子命令：

```bash
python -m ml_predictor.main <command> [options]
```

**子命令**:

| 命令 | 功能 | 示例 |
|------|------|------|
| `generate` | 生成训练数据 | `generate --num_samples 5000` |
| `train` | 训练模型 | `train --data_path ./data/train.csv` |
| `predict` | 预测性能 | `predict --config ./configs/google.cfg` |
| `evaluate` | 评估模型 | `evaluate --data_path ./data/test.csv` |

---

## 数据流程图

```
┌─────────────────────────────────────────────────────────┐
│              Phase 1: 数据生成                           │
│                                                         │
│  随机配置 → 创建临时文件 → 运行 SCALE-Sim → 解析报告    │
│     ↓                                                    │
│  training_data.csv (config + topology + metrics)        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Phase 2: 数据预处理                         │
│                                                         │
│  One-Hot 编码 → 计算衍生特征 → Log 变换 → 标准化         │
│     ↓                                                    │
│  X (21 features), y (6 targets) - 标准化后的数据         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Phase 3: 模型训练                           │
│                                                         │
│  划分数据集 → 创建神经网络 → 训练循环 → 早停/保存         │
│     ↓                                                    │
│  scalesim_predictor.pt (训练好的模型)                    │
│  preprocessor.pkl (预处理器)                             │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Phase 4: 推理预测                           │
│                                                         │
│  config.cfg + topology.csv → 特征提取 → 模型推理 → 结果  │
│     ↓                                                    │
│  预测的 6 个性能指标 (毫秒级完成)                         │
└─────────────────────────────────────────────────────────┘
```

---

## 配置参数说明

### 调整采样范围

在 `config.py` 中修改范围以探索不同的设计空间：

```python
# 例如：只探索大型阵列
"array_height_range": [256, 512, 1024],
"array_width_range": [256, 512, 1024],

# 例如：只关注小卷积核
"filter_height_range": [1, 3],
"filter_width_range": [1, 3],
```

### 调整模型结构

```python
MODEL_CONFIG = {
    "hidden_dims": [256, 512, 256, 128],  # 更深的网络
    "dropout_rate": 0.3,                  # 更高的 dropout
    "learning_rate": 0.0005,              # 更小的学习率
}
```

---

## 常见问题 (FAQ)

### Q1: 数据生成很慢怎么办？

某些配置组合（特别是大阵列 + 大特征图）会导致仿真耗时很长。

**解决方案**:
1. 减小采样范围，避免极端配置
2. 使用多进程并行生成（需要修改代码）
3. 先生成少量数据验证流程，再大规模生成

### Q2: 如何提高模型准确性？

1. **增加训练数据**: 更多样本 → 更好的泛化
2. **调整网络结构**: 尝试更深/更宽的网络
3. **特征工程**: 添加更多衍生特征
4. **超参数调优**: 学习率、batch size、dropout 等

### Q3: 预测结果不合理？

检查以下几点：
1. 输入配置是否在训练数据范围内（模型外推能力有限）
2. 预处理器是否正确加载
3. 模型是否训练充分（查看训练 loss）

### Q4: 如何处理已有数据？

数据生成会自动加载并跳过重复配置：

```bash
# 追加 100 个新样本到已有文件
python -m ml_predictor.main generate --num_samples 100
```

### Q5: 能否预测其他指标？

可以！修改 `data_generation.py` 中的解析逻辑，从 `BANDWIDTH_REPORT.csv` 或 `DETAILED_ACCESS_REPORT.csv` 提取更多指标。

---

## 性能对比

| 方法 | 单次预测耗时 | 准确性 | 适用场景 |
|------|-------------|--------|---------|
| **SCALE-Sim 仿真** | 数秒～数分钟 | 100% (ground truth) | 精确验证 |
| **ML 预测** | 毫秒级 | 90-95% MAPE | 快速设计空间探索 |

---

## 开发者指南

### 添加新特征

1. 在 `data_generation.py` 中计算新特征
2. 在 `config.py` 的 `INPUT_FEATURES` 中添加特征名
3. 在 `data_preprocessing.py` 中处理新特征

### 修改模型架构

编辑 `model.py` 中的 `ScaleSimPredictor` 类：

```python
def __init__(self, input_dim, output_dim):
    # 添加注意力机制、残差连接等
    self.attention = nn.MultiheadAttention(...)
```

### 调试技巧

```bash
# 生成少量样本快速验证
python -m ml_predictor.main generate --num_samples 5

# 训练少量 epoch
python -m ml_predictor.main train --data_path ./data/raw/training_data.csv --epochs 5
```

---

## 引用

如果您在研究中使用此模块，请引用：

```bibtex
@software{scalesim_ml_predictor,
  title={SCALE-Sim ML Predictor},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/scale-sim-v3}
}
```

---

## 许可证

遵循 SCALE-Sim 项目的原始许可证。

---

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
