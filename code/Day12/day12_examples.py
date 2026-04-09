"""
Day12 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day12_examples.py

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
print("Day12 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

print("=" * 50)
print("🏆 经典 CNN 架构详解")
print("=" * 50)

# 1. 加载预训练模型
print("\n【1. 加载经典模型】")

# AlexNet
alexnet = models.alexnet(pretrained=True)
print("✓ AlexNet 加载完成")

# VGG16
vgg16 = models.vgg16(pretrained=True)
print("✓ VGG16 加载完成")

# ResNet18
resnet18 = models.resnet18(pretrained=True)
print("✓ ResNet18 加载完成")

print(f"\n{'='*50}")
print("模型对比:")
print(f"{'='*50}")

print(f"""
AlexNet:
✓ 2012 年，深度学习革命的开始
✓ 5 个卷积层，3 个全连接层
✓ 参数量：60M
✓ 优点：开创性
✗ 缺点：现在看不算深

VGG16:
✓ 2014 年，优雅简洁
✓ 全部 3×3 卷积核
✓ 参数量：138M
✓ 优点：结构简单，效果好
✗ 缺点：参数太多

ResNet18:
✓ 2015 年，残差网络
✓ 18 层，有跳跃连接
✓ 参数量：11M
✓ 优点：可以很深，不梯度消失
✗ 缺点：稍微复杂
""")

# ===== 代码块 2 =====

print("=" * 50)
print("🎯 迁移学习：站在巨人肩膀上")
print("=" * 50)

print("""
什么是迁移学习？

就像你学会了骑自行车：
- 再学电动车 → 很快
- 因为平衡感是通用的

深度学习也一样：
- 在大数据集（ImageNet）上训练
- 学到通用特征（边缘、纹理）
- 用到自己的小数据集上
- 只需要微调最后几层

好处：
✓ 少量数据也能训练
✓ 训练速度快
✓ 效果好
""")

# 实战：用预训练的 ResNet 做猫狗分类
print("\n【实战：猫狗分类】")

# 1. 修改 ResNet 做二分类
# ResNet 原本是 1000 类，改成 2 类（猫、狗）

num_ftrs = resnet18.fc.in_features  # 获取全连接层输入特征数
resnet18.fc = nn.Linear(num_ftrs, 2)  # 改成 2 分类

print(f"✓ 修改 ResNet 最后一层")
print(f"  原来：{num_ftrs} → 1000")
print(f"  现在：{num_ftrs} → 2 (猫 or 狗)")

# 2. 冻结前面的层（只训练最后一层）
for param in resnet18.parameters():
    param.requires_grad = False

# 只让最后一层可训练
for param in resnet18.fc.parameters():
    param.requires_grad = True

print(f"\n✓ 冻结了前面的层")
print(f"  只训练最后一层（快速！）")

# 3. 准备数据（模拟）
print(f"\n{'='*50}")
print("准备数据...")
print(f"{'='*50}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

print("✓ 数据预处理配置完成")
print("  - 缩放到 224×224")
print("  - 标准化（和 ImageNet 一致）")

# 4. 测试一下
print(f"\n{'='*50}")
print("测试模型...")
print(f"{'='*50}")

# 创建一个假图片
dummy_input = torch.randn(1, 3, 224, 224)

# 切换到评估模式
resnet18.eval()

with torch.no_grad():
    output = resnet18(dummy_input)
    probabilities = torch.softmax(output, dim=1)
    
print(f"✓ 模型测试成功")
print(f"  输出形状：{output.shape}")
print(f"  预测概率：{probabilities}")

print(f"\n💡 说明:")
print(f"- 这是预训练模型的威力")
print(f"- 还没用自己的数据就能用了")
print(f"- 微调后会更好！")

# ===== 代码块 3 =====

print("=" * 50)
print("🐱🆚🐶 猫狗大战完整项目")
print("=" * 50)

# 注意：这里用模拟数据演示
# 实际项目需要真实的猫狗图片数据集

import os
from torch.utils.data import Dataset, DataLoader

# 自定义数据集
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 模拟一些数据
        for i in range(100):  # 假装有 100 张图
            self.images.append(f"image_{i}.jpg")
            self.labels.append(i % 2)  # 0=猫，1=狗
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 实际项目中这里要加载图片
        # 这里用随机 tensor 模拟
        image = torch.randn(3, 224, 224)
        label = self.labels[idx]
        return image, label

print("\n【1. 创建数据集】")

# 创建数据集
train_dataset = CatDogDataset(root_dir='./data/train', transform=transform)
test_dataset = CatDogDataset(root_dir='./data/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"✓ 训练集：{len(train_dataset)} 张")
print(f"✓ 测试集：{len(test_dataset)} 张")

# 2. 配置训练
print(f"\n{'='*50}")
print("【2. 配置训练参数】")
print(f"{'='*50}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet18.fc.parameters(), lr=0.001)

print(f"损失函数：CrossEntropyLoss")
print(f"优化器：Adam (lr=0.001)")
print(f"只训练最后一层（快！）")

# 3. 训练
print(f"\n{'='*50}")
print("【3. 开始训练】")
print(f"{'='*50}")

num_epochs = 5

for epoch in range(num_epochs):
    resnet18.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        # 前向传播
        outputs = resnet18(images)
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
    
    # 打印这轮结果
    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total * 100
    
    print(f"第{epoch+1}/{num_epochs}轮 - "
          f"损失：{avg_loss:.4f} - "
          f"准确率：{accuracy:.2f}%")

# 4. 测试
print(f"\n{'='*50}")
print("【4. 评估模型】")
print(f"{'='*50}")

resnet18.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = resnet18(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total * 100
print(f"✓ 测试集准确率：{test_accuracy:.2f}%")

if test_accuracy > 95:
    print("🎉 优秀！超过 95%！")
elif test_accuracy > 90:
    print("👍 很好！超过 90%！")
else:
    print("💪 不错！多训练或调参会更好！")

print(f"\n{'='*50}")
print("🎊 恭喜！你用迁移学习完成了猫狗分类！")
print(f"{'='*50}")

print("""
总结迁移学习的优势:

✓ 不用从零训练（省时间）
✓ 少量数据也能用（省数据）
✓ 效果好（站在巨人肩膀上）
✓ 工业界标准做法

这就是专业 AI 工程师的工作方式！
""")