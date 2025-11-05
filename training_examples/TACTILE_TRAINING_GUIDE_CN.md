# 使用触觉数据训练CNEP/CNMP模型

本指南展示如何使用触觉图像和三维力作为条件输入，通过CNEP和CNMP模型学习机器人末端执行器轨迹基元。

## 概述

本实现允许您：
- 使用触觉图像作为视觉条件信号
- 使用三维力测量数据作为额外的条件输入
- 建模三维机器人末端执行器轨迹
- 训练CNEP（具有多个专家解码器）或CNMP模型

## 数据结构

您的数据应按以下方式组织：

```
data_folder/
├── demo_0/
│   ├── img/
│   │   ├── 000.jpg
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   └── data.csv
├── demo_1/
│   ├── img/
│   │   └── ...
│   └── data.csv
└── ...
```

### CSV文件格式

CSV文件应包含轨迹和力数据，具有以下结构：

```csv
x,y,z,fx,fy,fz
0.1,0.2,0.3,0.5,0.1,0.2
0.11,0.21,0.31,0.52,0.12,0.21
...
```

- 列 0-2: 三维机器人末端执行器位置 (x, y, z)
- 列 3-5: 三维力测量 (fx, fy, fz)

您可以在数据加载器中自定义使用哪些列。

### 图像文件

- 图像应按顺序命名：`000.jpg`、`001.jpg`、`002.jpg` 等
- 支持的格式：`.jpg`、`.jpeg`、`.png`
- 图像会自动调整大小并预处理
- 使用预训练的CNN模型（默认为MobileNetV2）提取特征

## 使用方法

### 1. 使用数据加载器

```python
from data.tactile_data_loader import TactileDataset, create_feature_extractor
import torch

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 创建特征提取器
img_feature_extractor = create_feature_extractor('mobilenet_v2', device=device)

# 加载数据集
dataset = TactileDataset(
    data_folder='./your_data_folder',      # 您的数据文件夹路径
    demo_folders=['demo_0', 'demo_1', 'demo_2'],  # 演示文件夹列表
    img_subfolder='img',                   # 图像子文件夹名称
    csv_filename='data.csv',               # CSV文件名
    trajectory_cols=[0, 1, 2],             # CSV中x,y,z列的索引
    force_cols=[3, 4, 5],                  # CSV中fx,fy,fz列的索引
    num_timesteps=200,                     # 采样到200个时间步
    normalize=True,                        # 将数据归一化到[-1, 1]
    img_feature_extractor=img_feature_extractor,
    device=device
)

# 获取数据
trajectories, forces, img_features = dataset.get_all_data()
print(f"轨迹: {trajectories.shape}")        # (N, T, 3)
print(f"力: {forces.shape}")                # (N, T, 3)
print(f"图像特征: {img_features.shape}")    # (N, T, D)
```

### 2. 训练CNEP模型

```bash
cd training_examples
python train_cnep_on_tactile.py
```

编辑脚本以配置：
- `data_folder`：您的数据路径
- `demo_folders`：演示文件夹列表
- `num_timesteps`：要采样的时间步数
- `feature_extractor_name`：'mobilenet_v2'、'resnet18' 或 'resnet50'

### 3. 训练CNMP模型

```bash
cd training_examples
python train_cnmp_on_tactile.py
```

配置选项与CNEP训练相同。

## 自定义配置

### 使用不同的CSV列

如果您的CSV文件结构不同，请指定列索引：

```python
dataset = TactileDataset(
    data_folder='./data',
    demo_folders=['demo_0'],
    trajectory_cols=[1, 2, 3],  # 轨迹的不同列
    force_cols=[7, 8, 9],       # 力的不同列
    # ... 其他参数
)
```

### 使用不同的图像特征提取器

更改特征提取器模型：

```python
# 选项：'mobilenet_v2'、'resnet18'、'resnet50'
img_feature_extractor = create_feature_extractor('resnet18', device=device)
```

### 自定义图像转换

提供您自己的图像预处理：

```python
from torchvision import transforms

custom_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

dataset = TactileDataset(
    # ... 其他参数
    img_transform=custom_transform,
)
```

## 模型架构

### 输入结构

模型接收：
- **观测值**：`[时间, 力, 图像特征, 轨迹]`
  - 时间：归一化的时间戳（0到1）
  - 力：三维力测量
  - 图像特征：提取的视觉特征
  - 轨迹：三维位置（用于条件输入）

- **目标**：`[时间, 力, 图像特征]` → 预测轨迹

### CNEP vs CNMP

- **CNEP**（条件神经专家过程）：使用具有门控机制的多个专家解码器，更适合多模态轨迹分布
- **CNMP**（条件神经运动基元）：单个解码器，更简单更快，适合单模态分布

## 快速开始示例

### 步骤1：生成示例数据

```bash
cd data
python generate_example_tactile_data.py
```

这将创建包含20个演示的合成数据集，用于测试。

### 步骤2：测试数据加载器

```bash
cd data
python test_tactile_data_loader.py
```

这将验证数据加载器是否正常工作。

### 步骤3：训练模型

```bash
cd training_examples
# 编辑train_cnep_on_tactile.py，设置data_folder路径
python train_cnep_on_tactile.py
```

## 完整示例：从数据到训练

```python
from data.tactile_data_loader import TactileDataset, create_feature_extractor
from models.cnep import CNEP
import torch

device = 'cuda'

# 1. 加载数据
feature_extractor = create_feature_extractor('mobilenet_v2', device)
dataset = TactileDataset(
    data_folder='./tactile_data',
    demo_folders=[f'demo_{i}' for i in range(20)],
    num_timesteps=200,
    img_feature_extractor=feature_extractor,
    device=device
)

trajectories, forces, img_features = dataset.get_all_data()

# 2. 创建模型
input_dim = 1 + 3 + img_features.shape[-1]  # 时间 + 力 + 图像特征
output_dim = 3  # 三维轨迹

model = CNEP(
    input_dim=input_dim,
    output_dim=output_dim,
    n_max=20,
    m_max=20,
    encoder_hidden_dims=[512, 512, 512],
    num_decoders=4,
    decoder_hidden_dims=[256, 256],
    batch_size=8,
    device=device
)

# 3. 训练模型（参见训练脚本以获取完整实现）
# ...
```

## 推理示例

```python
# 加载训练好的模型
model.load_state_dict(torch.load('saved_model.pt'))
model.eval()

# 准备观测和目标数据
# obs: (batch, n_obs, input_dim + output_dim)
# tar: (batch, n_tar, input_dim)
# obs_mask: (batch, n_obs)

with torch.no_grad():
    pred, gate = model.val(obs, tar, obs_mask)
    
    # 对于CNEP，选择最佳解码器
    dec_id = torch.argmax(gate.squeeze(1), dim=-1)
    pred_traj = pred[dec_id, torch.arange(batch_size), :, :3]
    
    # 反归一化回原始尺度
    pred_traj = dataset.denormalize_trajectory(pred_traj)
```

## 常见问题

### 内存不足

- 减小 `batch_size`
- 减小 `num_timesteps`
- 使用更小的特征提取器（例如 'mobilenet_v2' 而不是 'resnet50'）

### 训练速度慢

- 使用更小的隐藏维度
- 减少解码器数量（对于CNEP）
- 使用 `torch.compile()`（PyTorch 2.0+）

### 性能不佳

- 增加模型容量（更多层，更大的隐藏维度）
- 增加演示数量
- 调整学习率
- 对于CNEP：调整损失系数

## 文件结构

```
cnep/
├── data/
│   ├── tactile_data_loader.py           # 数据加载工具
│   ├── generate_example_tactile_data.py # 生成示例数据
│   ├── test_tactile_data_loader.py      # 测试数据加载器
│   └── TACTILE_TRAINING_GUIDE_CN.md     # 本文件（中文指南）
├── models/
│   ├── cnep.py                           # CNEP模型
│   └── cnmp.py                           # CNMP模型
├── training_examples/
│   ├── train_cnep_on_tactile.py         # CNEP训练脚本
│   ├── train_cnmp_on_tactile.py         # CNMP训练脚本
│   └── TACTILE_TRAINING_GUIDE.md        # 英文指南
└── outputs/                              # 训练输出
```

## 技术细节

### 数据归一化

- 轨迹和力数据自动归一化到 [-1, 1] 范围
- 使用 `denormalize_trajectory()` 和 `denormalize_force()` 方法恢复原始尺度

### 特征提取

- MobileNetV2: 1280维特征
- ResNet18: 512维特征
- ResNet50: 2048维特征

### 训练技巧

1. 从较小的模型开始测试
2. 使用验证集监控过拟合
3. 保存最佳模型检查点
4. 绘制训练曲线分析收敛情况

## 参考资料

有关CNEP和CNMP的更多信息，请参阅：
- 论文："Conditional Neural Expert Processes for Learning Movement Primitives From Demonstration"
- 仓库根目录的主 README.md

## 支持

如有问题或需要帮助，请：
1. 查看 `TACTILE_TRAINING_GUIDE.md`（英文详细指南）
2. 检查示例脚本中的注释
3. 在GitHub上提交issue
