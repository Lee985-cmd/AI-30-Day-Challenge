"""
Day11 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day11_examples.py

注意: 某些代码可能需要安装额外的库
"""

# 导入必要的库
import sys
import os

# 尝试导入常用库
try:
    import numpy as np
except ImportError:
    print("提示: 需要安装 numpy: pip install numpy")
    np = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("提示: 需要安装 matplotlib: pip install matplotlib")
    plt = None

try:
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("提示: 需要安装 scikit-learn: pip install scikit-learn")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("提示: 需要安装 PyTorch: pip install torch torchvision")

print("=" * 60)
print("Day11 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("👁️ CNN 基础：卷积层详解")
print("=" * 50)

# 1. 创建一个简单的"图片"
print("\n【1. 理解卷积操作】")

# 灰度图：(batch, channel, height, width)
image = torch.zeros(1, 1, 6, 6)

# 画一个十字形图案
image[0, 0, 2, :] = 1  # 中间一行
image[0, 0, :, 2] = 1  # 中间一列

print(f"原始图片形状：{image.shape}")
print("原始图片:")
print(image[0, 0])

# 2. 定义卷积核
kernel = torch.tensor([
    [-1, -1, -1],
    [ 2,  2,  2],
    [-1, -1, -1]
], dtype=torch.float32).view(1, 1, 3, 3)

print(f"\n卷积核形状：{kernel.shape}")
print("卷积核（水平边缘检测）:")
print(kernel[0, 0])

# 3. 应用卷积
conv_layer = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=0
)

# 设置卷积核的权重
conv_layer.weight = nn.Parameter(kernel)

# 进行卷积
output = conv_layer(image)

print(f"\n卷积结果形状：{output.shape}")
print("卷积结果（检测到了水平边缘！）:")
print(output[0, 0])

print(f"\n💡 说明:")
print(f"- 卷积核在图片上滑动")
print(f"- 每个位置计算点积")
print(f"- 得到新的特征图")
print(f"- 这个核检测到了水平边缘")

# ===== 代码块 2 =====

print("=" * 50)
print("【2. 池化层（Pooling）】")
print("=" * 50)

# 创建一个特征图
feature_map = torch.tensor([
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
     [13, 14, 15, 16]]
], dtype=torch.float32).view(1, 1, 4, 4)

print("原始特征图 (4×4):")
print(feature_map[0, 0])

# 最大池化
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
max_output = max_pool(feature_map)

print(f"\n最大池化结果 (2×2):")
print(max_output[0, 0])
print("每个 2×2 区域取最大值")

# 平均池化
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
avg_output = avg_pool(feature_map)

print(f"\n平均池化结果 (2×2):")
print(avg_output[0, 0])
print("每个 2×2 区域取平均值")

print(f"\n💡 池化的作用:")
print(f"1. 降维（减少计算量）")
print(f"2. 保留主要特征")
print(f"3. 防止过拟合")
print(f"\n常用最大池化（取最明显的特征）")

# ===== 代码块 3 =====

print("=" * 50)
print("【3. 搭建 LeNet-5 架构】")
print("=" * 50)

class LeNet5(nn.Module):
    """LeNet-5：第一个成功的 CNN 架构"""
    
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        print("\nLeNet-5 架构:")
        
        # 第 1 个卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        print("✓ Conv1: 1 通道 → 6 通道，5×5 卷积")
        
        self.pool1 = nn.MaxPool2d(2, 2)
        print("✓ Pool1: 2×2 最大池化")
        
        # 第 2 个卷积层
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        print("✓ Conv2: 6 通道 → 16 通道，5×5 卷积")
        
        self.pool2 = nn.MaxPool2d(2, 2)
        print("✓ Pool2: 2×2 最大池化")
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        print("✓ FC1: 全连接层 400 → 120")
        
        self.fc2 = nn.Linear(120, 84)
        print("✓ FC2: 全连接层 120 → 84")
        
        self.fc3 = nn.Linear(84, num_classes)
        print(f"✓ FC3: 全连接层 84 → {num_classes} (输出)")
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 第 1 个卷积 + 池化
        x = self.pool1(self.relu(self.conv1(x)))
        
        # 第 2 个卷积 + 池化
        x = self.pool2(self.relu(self.conv2(x)))
        
        # 展平
        x = x.view(-1, 16 * 5 * 5)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# 创建模型
model = LeNet5(num_classes=10)
print(f"\n{'='*50}")
print("模型结构总结:")
print(model)

# 测试
test_input = torch.randn(1, 1, 32, 32)
output = model(test_input)
print(f"\n测试:")
print(f"  输入形状：{test_input.shape}")
print(f"  输出形状：{output.shape}")

# ===== 代码块 4 =====

print("=" * 50)
print("✍️ CNN 实战：MNIST 手写数字识别")
print("=" * 50)

import torch.optim as optim
from torchvision import datasets, transforms

# 1. 加载数据
print("\n【1. 准备 MNIST 数据集】")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False
)

print(f"✓ 训练集：{len(train_dataset)} 张")
print(f"✓ 测试集：{len(test_dataset)} 张")

# 显示一些样本
fig, axes = plt.subplots(2, 5, figsize=(10, 3))
axes = axes.ravel()

for i in range(10):
    image, label = train_dataset[i]
    axes[i].imshow(image.squeeze(), cmap='gray')
    axes[i].set_title(f'标签：{label}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# 2. 创建模型
print("\n" + "=" * 50)
print("【2. 创建 CNN 模型】")
print("=" * 50)

model = LeNet5(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"损失函数：CrossEntropyLoss")
print(f"优化器：Adam (lr=0.001)")

# 3. 训练模型
print("\n" + "=" * 50)
print("【3. 开始训练】")
print("=" * 50)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # 打印这轮的结果
    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total * 100
    
    print(f"第{epoch+1}/{num_epochs}轮 - "
          f"损失：{avg_loss:.4f} - "
          f"准确率：{accuracy:.2f}%")

# 4. 评估模型
print("\n" + "=" * 50)
print("【4. 评估模型】")
print("=" * 50)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total * 100
print(f"✓ 测试集准确率：{test_accuracy:.2f}%")

if test_accuracy > 98:
    print("🎉 优秀！超过 98%！")
elif test_accuracy > 95:
    print("👍 很好！超过 95%！")
else:
    print("💪 不错！多训练几轮会更好！")

# 5. 实际预测
print("\n" + "=" * 50)
print("【5. 实际预测示例】")
print("=" * 50)

sample_indices = [0, 100, 200, 300, 400]

fig, axes = plt.subplots(1, 5, figsize=(12, 3))

for i, idx in enumerate(sample_indices):
    image, true_label = test_dataset[idx]
    
    # 预测
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output.data, 1)
        pred_label = predicted.item()
    
    # 显示
    axes[i].imshow(image.squeeze(), cmap='gray')
    color = 'green' if pred_label == true_label else 'red'
    axes[i].set_title(f'真:{true_label} 预:{pred_label}', 
                     color=color, fontsize=12)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

print(f"\n{'='*50}")
print("🎊 恭喜！你用 CNN 完成了 MNIST 分类！")
print(f"{'='*50}")

print("""
CNN vs 普通神经网络:

CNN 的优势:
✓ 参数更少（效率高）
✓ 考虑空间结构（效果好）
✓ 平移不变性（更鲁棒）

这就是为什么 CNN 在图像领域这么成功！
""")