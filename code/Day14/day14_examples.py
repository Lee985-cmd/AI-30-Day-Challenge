"""
Day14 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day14_examples.py

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
print("Day14 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

print("=" * 50)
print("🎓 Week 2 毕业项目：CIFAR-10 图像分类")
print("=" * 50)

# 1. 加载数据
print("\n【1. 准备 CIFAR-10 数据集】")

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机翻转（数据增强）
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载并加载数据集
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

# 类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print(f"✓ 训练集：{len(trainset)} 张")
print(f"✓ 测试集：{len(testset)} 张")
print(f"✓ 类别数：{len(classes)} 类")
print(f"  {classes}")

# 显示一些样本
fig, axes = plt.subplots(2, 5, figsize=(12, 3))
axes = axes.ravel()

for i in range(10):
    image, label = trainset[i]
    # 反标准化
    img = image.numpy().transpose(1, 2, 0)
    img = img * 0.5 + 0.5  # 反归一化
    img = np.clip(img, 0, 1)
    
    axes[i].imshow(img)
    axes[i].set_title(f'标签：{classes[label]}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# ===== 代码块 2 =====

print("=" * 50)
print("【2. 构建 CNN 模型】")
print("=" * 50)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 第 1 个卷积块
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 第 2 个卷积块
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # 第 3 个卷积块
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # 32x32 → 16x16
        
        # Conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # 16x16 → 8x8
        
        # Conv3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)  # 8x8 → 4x4
        
        # 展平
        x = x.view(-1, 256 * 4 * 4)
        
        # 全连接
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# 创建模型
model = SimpleCNN()
print("✓ 创建了 SimpleCNN 模型")
print(f"\n模型结构:")
print(model)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量：{total_params:,} ({total_params/1e6:.2f}M)")

# 配置训练参数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

print(f"\n损失函数：CrossEntropyLoss")
print(f"优化器：Adam (lr=0.001, weight_decay=1e-4)")
print(f"学习率调度：每 10 轮 ×0.1")

# ===== 代码块 3 =====

print("=" * 50)
print("【3. 训练模型】")
print("=" * 50)

num_epochs = 30
best_acc = 0.0

train_losses = []
train_accs = []
test_accs = []

for epoch in range(num_epochs):
    # 训练
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in trainloader:
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
    
    # 计算训练集指标
    train_loss = running_loss / len(trainloader)
    train_acc = correct / total * 100
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # 测试
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = correct / total * 100
    test_accs.append(test_acc)
    
    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_cifar10_model.pth')
    
    # 打印进度
    scheduler.step()  # 更新学习率
    
    print(f"第{epoch+1:2d}/{num_epochs}轮 | "
          f"训练损失：{train_loss:.4f} | "
          f"训练准确率：{train_acc:5.2f}% | "
          f"测试准确率：{test_acc:5.2f}% | "
          f"最佳：{best_acc:.2f}%")

# 画训练曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 损失曲线
ax1.plot(train_losses, 'b-', linewidth=2, label='训练损失')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('训练损失曲线')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 准确率曲线
ax2.plot(train_accs, 'g-', linewidth=2, label='训练准确率')
ax2.plot(test_accs, 'r-', linewidth=2, label='测试准确率')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('准确率曲线')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n✓ 训练完成！")
print(f"✓ 最佳测试准确率：{best_acc:.2f}%")

# ===== 代码块 4 =====

print("=" * 50)
print("【4. 模型评估】")
print("=" * 50)

# 加载最佳模型
model.load_state_dict(torch.load('best_cifar10_model.pth'))
model.eval()

# 在测试集上评估
correct = 0
total = 0
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 各类别的准确率
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# 总体准确率
overall_acc = correct / total * 100
print(f"✓ 测试集总体准确率：{overall_acc:.2f}%")

# 各类别准确率
print(f"\n各类别准确率:")
for i in range(10):
    if class_total[i] > 0:
        acc = 100 * class_correct[i] / class_total[i]
        bar = '█' * int(acc / 10)
        print(f"  {classes[i]:10} {acc:5.2f}% {bar}")

# 混淆矩阵
from sklearn.metrics import confusion_matrix
import seaborn as sns

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 实际预测示例
print(f"\n{'='*50}")
print("【5. 实际预测示例】")
print(f"{'='*50}")

fig, axes = plt.subplots(3, 5, figsize=(12, 8))
axes = axes.ravel()

sample_indices = np.random.choice(len(testset), 15)

for i, idx in enumerate(sample_indices):
    image, true_label = testset[idx]
    
    # 预测
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, pred_label = torch.max(output.data, 1)
        pred_label = pred_label.item()
    
    # 显示
    img = image.numpy().transpose(1, 2, 0)
    img = img * 0.5 + 0.5
    img = np.clip(img, 0, 1)
    
    axes[i].imshow(img)
    color = 'green' if pred_label == true_label else 'red'
    axes[i].set_title(f'真:{classes[true_label]}\n预:{classes[pred_label]}',
                     color=color, fontsize=8)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# ===== 代码块 5 =====

print("=" * 50)
print("📊 项目总结报告")
print("=" * 50)

report = f"""
# CIFAR-10 图像分类项目报告

## 1. 项目概述
- 任务：图像分类（10 类物体）
- 数据集：CIFAR-10（60,000 张 32×32 彩色图片）
- 目标：训练一个 CNN 模型进行分类

## 2. 数据准备
- 训练集：50,000 张
- 测试集：10,000 张
- 数据增强：随机翻转、随机裁剪
- 标准化：均值 0.5，标准差 0.5

## 3. 模型架构
SimpleCNN:
- Conv1: 3→64, BN, ReLU, Pool
- Conv2: 64→128, BN, ReLU, Pool
- Conv3: 128→256, BN, ReLU, Pool
- FC1: 256*4*4 → 512, Dropout
- FC2: 512 → 10

总参数量：约 1.5M

## 4. 训练配置
- 损失函数：CrossEntropyLoss
- 优化器：Adam (lr=0.001, weight_decay=1e-4)
- Batch Size: 128
- Epochs: 30
- 学习率调度：StepLR (每 10 轮×0.1)

## 5. 实验结果
最佳测试准确率：{best_acc:.2f}%

各类别表现:
"""

print(report)

# 添加各类别表现
for i in range(10):
    if class_total[i] > 0:
        acc = 100 * class_correct[i] / class_total[i]
        print(f"  - {classes[i]:10}: {acc:.2f}%")

conclusion = f"""

## 6. 结论
✓ 成功训练了一个 CNN 模型
✓ 达到了 {_:.2f}% 的测试准确率
✓ 模型能够识别 10 类常见物体

## 7. 改进方向
1. 使用更深的网络（ResNet、DenseNet）
2. 更多数据增强
3. 更长的训练时间
4. 学习率调优
5. 集成学习（多个模型投票）

## 8. 学到的技能
✓ 数据加载和预处理
✓ CNN 架构设计
✓ BatchNorm、Dropout 的使用
✓ 训练过程监控
✓ 模型评估和可视化
✓ 混淆矩阵分析

这是一个完整的深度学习项目流程！
"""

print(conclusion)

print("\n" + "=" * 50)
print("🎊 恭喜！你完成了 Week 2 的毕业项目！")
print("=" * 50)

print("""
Week 2 学习成果总结:

理论层面:
✓ 理解了神经网络的工作原理
✓ 掌握了 CNN 和 RNN 的核心思想
✓ 知道经典架构的特点

实践层面:
✓ 熟练使用 PyTorch 框架
✓ 能搭建和训练各种网络
✓ 完成了 5+ 个完整项目

技能层面:
✓ 图像处理（CNN）
✓ 文本处理（RNN/LSTM）
✓ 迁移学习
✓ 模型调优

你已经具备了初级 AI 工程师的能力！
继续深入学习特定方向吧！💪✨
""")