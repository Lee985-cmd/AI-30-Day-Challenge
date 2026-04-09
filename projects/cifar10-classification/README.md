# CIFAR-10 图像分类项目

## 📖 项目简介

使用 CNN 对 CIFAR-10 数据集进行图像分类，达到 80%+ 准确率。

## 🎯 学习目标

- 掌握完整的深度学习项目流程
- 理解数据增强的重要性
- 学会模型设计和调优
- 能够分析和可视化结果

## 📂 项目结构

```
cifar10-classification/
├── main.py              # 主程序入口
├── model.py             # 模型定义
├── train.py             # 训练脚本
├── evaluate.py          # 评估脚本
├── utils.py             # 工具函数
├── requirements.txt     # 项目依赖
├── README.md            # 本文件
└── demo.ipynb           # Jupyter 演示
```

## 🚀 快速开始

### 前置要求

- Python 3.7+
- 推荐：GPU（可加速训练 5-10 倍）
- 磁盘空间：至少 2GB（用于数据集和模型）

### 1. 克隆项目

```bash
git clone https://github.com/Lee985-cmd/AI-30-Day-Challenge.git
cd AI-30-Day-Challenge/projects/cifar10-classification
```

### 2. 创建虚拟环境（推荐）

**Windows:**
```bash
python -m venv cifar-env
cifar-env\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv cifar-env
source cifar-env/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> 💡 **国内用户加速：**
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 4. 运行训练

```bash
python main.py
```

首次运行会自动下载 CIFAR-10 数据集（约 163MB）。

### 3. 查看结果

训练完成后会生成：
- `training_curves.png` - 训练曲线
- `confusion_matrix.png` - 混淆矩阵
- `predictions.png` - 预测示例
- `cifar_best.pth` - 最佳模型权重

## 📊 预期结果

### 训练时间
- **CPU**: 约 30-60 分钟
- **GPU**: 约 5-10 分钟

### 性能指标
- **测试准确率**: 75-85%
- **模型大小**: 约 5 MB
- **推理速度**: ~100 张/秒（GPU）

### 生成文件
训练完成后会生成：
- `training_curves.png` - 训练曲线图
- `confusion_matrix.png` - 混淆矩阵热力图
- `predictions.png` - 预测示例图片
- `cifar_best.pth` - 最佳模型权重文件

## 🔧 可调参数

在 `main.py` 中修改：

```python
# 训练参数
BATCH_SIZE = 128        # 批次大小
NUM_EPOCHS = 30         # 训练轮数
LEARNING_RATE = 0.001   # 学习率

# 模型参数
NUM_CLASSES = 10        # 类别数
```

## 💡 改进建议

### 提升准确率的方法

1. **数据增强**
   ```python
   transforms.RandomRotation(10)
   transforms.ColorJitter(brightness=0.2, contrast=0.2)
   ```

2. **更深的网络**
   ```python
   # 添加更多卷积层
   self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
   ```

3. **学习率调度**
   ```python
   scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
   ```

4. **使用预训练模型**
   ```python
   from torchvision import models
   model = models.resnet18(pretrained=True)
   ```

## 🐛 常见问题

### Q: CUDA out of memory

**A:** 减小 batch size
```python
BATCH_SIZE = 64  # 或 32
```

### Q: 准确率太低 (<60%)

**A:** 检查以下几点：
- 数据是否正确加载
- 学习率是否合适
- 训练轮数是否足够
- 模型是否有 bug

### Q: 训练很慢

**A:** 
- 使用 GPU
- 增大批次大小
- 减少 workers 数量

## 📚 相关资源

- [CIFAR-10 官网](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Vision](https://pytorch.org/vision/)
- [Day14 教程](../../Day14/)

## 📄 许可证

MIT License
