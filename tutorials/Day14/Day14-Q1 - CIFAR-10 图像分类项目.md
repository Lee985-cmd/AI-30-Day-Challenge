# Day14-Q1 - CIFAR-10 图像分类项目

## 🎓 Week2 毕业项目：完整实战

### 项目目标

用 CNN 对 CIFAR-10 数据集进行分类，达到 80%+ 准确率！

**CIFAR-10 是什么?**
- 60,000 张彩色小图片 (32×32)
- 10 个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
- 每个类别 6,000 张

**难度:** ⭐⭐⭐  
**时间:** 2-3 小时  
**预期准确率:** 75-85%

---

## 一、完整代码实现

### Step 1: 导入库和准备数据

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("🎓 Week2 毕业项目：CIFAR-10 图像分类")
print("=" * 60)

# ============================================================================
# 第 1 步：加载和预处理数据
# ============================================================================
print("\n【1. 准备 CIFAR-10 数据集】")

# 数据增强（训练集）
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # 随机水平翻转
    transforms.RandomCrop(32, padding=4),   # 随机裁剪
    transforms.ToTensor(),                   # 转为 Tensor
    transforms.Normalize((0.5, 0.5, 0.5),   # 标准化
                        (0.5, 0.5, 0.5))
])

# 测试集不做增强
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载并加载数据集
print("正在下载 CIFAR-10 数据集...")
trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform_train
)

testset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform_test
)

# 创建 DataLoader
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=128,     # 每批 128 张图片
    shuffle=True,       # 打乱顺序
    num_workers=2       # 2 个工作进程
)

testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=100, 
    shuffle=False, 
    num_workers=2
)

# 类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print(f"✓ 训练集：{len(trainset):,} 张")
print(f"✓ 测试集：{len(testset):,} 张")
print(f"✓ 类别数：{len(classes)} 类")
print(f"  {classes}")

# 可视化样本
def show_samples():
    """显示一些训练样本"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(10):
        image, label = trainset[i]
        
        # 反标准化（还原为原始像素值）
        img = image.numpy().transpose(1, 2, 0)
        img = img * 0.5 + 0.5  # 从 [-1,1] 变回 [0,1]
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f'{classes[label]}', fontsize=12)
        axes[i].axis('off')
    
    plt.suptitle('CIFAR-10 样本展示', fontsize=16)
    plt.tight_layout()
    plt.savefig('cifar_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ 样本图已保存为 cifar_samples.png")

show_samples()

print("\n💡 为什么要数据增强？")
print("-" * 60)
print("• 随机翻转/裁剪 → 增加数据多样性")
print("• 防止过拟合 → 让模型更鲁棒")
print("• 就像给小孩看不同角度、不同位置的猫")
print("• 模型学到的不是'某只特定的猫'，而是'猫的特征'")
```

### Step 2: 构建 CNN 模型

```python
# ============================================================================
# 第 2 步：构建 CNN 模型
# ============================================================================
print("\n" + "=" * 60)
print("【2. 构建 CNN 模型】")
print("=" * 60)

class SimpleCNN(nn.Module):
    """
    简单的 CNN 模型
    
    架构设计思路：
    1. 3 个卷积块，逐步提取特征
    2. 每个卷积块后接 BatchNorm 稳定训练
    3. MaxPool 降低空间维度
    4. 全连接层做分类
    5. Dropout 防止过拟合
    """
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # === 第 1 个卷积块 ===
        # 输入: 3×32×32 (RGB 图像)
        # 输出: 64×16×16
        self.conv1 = nn.Conv2d(
            in_channels=3,      # 输入通道 (RGB)
            out_channels=64,    # 输出通道 (64 个滤波器)
            kernel_size=3,      # 卷积核大小 3×3
            padding=1           # 填充，保持尺寸
        )
        self.bn1 = nn.BatchNorm2d(64)  # 批归一化
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)  # 2×2 最大池化
        
        # === 第 2 个卷积块 ===
        # 输入: 64×16×16
        # 输出: 128×8×8
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # === 第 3 个卷积块 ===
        # 输入: 128×8×8
        # 输出: 256×4×4
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # === 全连接层 ===
        # 输入: 256×4×4 = 4096 维
        # 输出: 10 类
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)  # 50% Dropout
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        """前向传播"""
        
        # Conv Block 1
        x = self.conv1(x)       # 3×32×32 → 64×32×32
        x = self.bn1(x)         # 批归一化
        x = self.relu(x)        # ReLU 激活
        x = self.pool(x)        # 64×32×32 → 64×16×16
        
        # Conv Block 2
        x = self.conv2(x)       # 64×16×16 → 128×16×16
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)        # 128×16×16 → 128×8×8
        
        # Conv Block 3
        x = self.conv3(x)       # 128×8×8 → 256×8×8
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)        # 256×8×8 → 256×4×4
        
        # Flatten (展平)
        x = x.view(-1, 256 * 4 * 4)  # → 4096 维向量
        
        # Fully Connected
        x = self.dropout(self.relu(self.fc1(x)))  # 4096 → 512
        x = self.fc2(x)          # 512 → 10
        
        return x

# 创建模型
model = SimpleCNN()

print("✓ 模型结构:")
print(model)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n📊 模型统计:")
print(f"  总参数量：{total_params:,}")
print(f"  可训练参数：{trainable_params:,}")
print(f"  模型大小：约 {total_params * 4 / 1024 / 1024:.2f} MB")

# 检查是否有 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"\n✓ 使用设备：{device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
```

### Step 3: 定义损失函数和优化器

```python
# ============================================================================
# 第 3 步：定义损失函数和优化器
# ============================================================================
print("\n" + "=" * 60)
print("【3. 定义损失函数和优化器】")
print("=" * 60)

# 损失函数：交叉熵（多分类标准选择）
criterion = nn.CrossEntropyLoss()

# 优化器：Adam（自适应学习率）
optimizer = optim.Adam(
    model.parameters(), 
    lr=0.001,              # 初始学习率
    weight_decay=1e-4      # L2 正则化（权重衰减）
)

# 学习率调度器：每 10 轮降低学习率
scheduler = optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=10,   # 每 10 个 epoch
    gamma=0.1       # 学习率 × 0.1
)

print(f"✓ 损失函数：CrossEntropyLoss")
print(f"  - 适合多分类问题")
print(f"  - 结合了 Softmax 和负对数似然")
print(f"\n✓ 优化器：Adam")
print(f"  - 学习率：0.001")
print(f"  - 权重衰减：1e-4（防止过拟合）")
print(f"  - 自适应调整每个参数的学习率")
print(f"\n✓ 学习率调度：StepLR")
print(f"  - 每 10 个 epoch，学习率 × 0.1")
print(f"  - 后期用小学习率精细调整")
```

### Step 4: 训练模型

```python
# ============================================================================
# 第 4 步：训练模型
# ============================================================================
print("\n" + "=" * 60)
print("【4. 开始训练模型】")
print("=" * 60)

num_epochs = 30
best_acc = 0.0

# 记录训练历史
train_losses = []
train_acces = []
test_acces = []

print(f"训练配置:")
print(f"  - Epochs: {num_epochs}")
print(f"  - Batch Size: 128")
print(f"  - 设备: {device}")
print(f"\n开始训练...\n")

for epoch in range(num_epochs):
    # === 训练阶段 ===
    model.train()  # 设置为训练模式（启用 Dropout、BatchNorm）
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # 移到设备
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # 计算训练集指标
    train_loss = running_loss / len(trainloader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_acces.append(train_acc)
    
    # === 测试阶段 ===
    model.eval()  # 设置为评估模式（关闭 Dropout）
    
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不计算梯度（节省内存）
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_acc = correct / total
    test_acces.append(test_acc)
    
    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'cifar_best.pth')
    
    # 打印进度
    print(f'Epoch [{epoch+1:2d}/{num_epochs}] | '
          f'Train Loss: {train_loss:.3f} | '
          f'Train Acc: {train_acc:.3f} | '
          f'Test Acc: {test_acc:.3f} | '
          f'Best: {best_acc:.3f}')
    
    # 调整学习率
    scheduler.step()

print(f"\n✅ 训练完成！")
print(f"🏆 最佳测试准确率：{best_acc*100:.2f}%")
print(f"💾 最佳模型已保存为 cifar_best.pth")
```

### Step 5: 可视化训练过程

```python
# ============================================================================
# 第 5 步：可视化训练过程
# ============================================================================
print("\n" + "=" * 60)
print("📊 可视化训练过程")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图：损失曲线
ax1.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
ax1.set_title('Training Loss Curve', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, num_epochs)

# 右图：准确率曲线
ax2.plot(train_acces, 'g-', linewidth=2, label='Training Accuracy')
ax2.plot(test_acces, 'r-', linewidth=2, label='Test Accuracy')
ax2.set_title('Accuracy Comparison', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, num_epochs)
ax2.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ 训练曲线已保存为 training_curves.png")

# 分析训练过程
print("\n📈 训练分析:")
print(f"  - 最终训练准确率：{train_acces[-1]*100:.2f}%")
print(f"  - 最终测试准确率：{test_acces[-1]*100:.2f}%")
print(f"  - 最佳测试准确率：{best_acc*100:.2f}%")

gap = train_acces[-1] - test_acces[-1]
if gap > 0.1:
    print(f"\n⚠️  警告：训练-测试差距较大 ({gap*100:.2f}%)")
    print("   可能存在过拟合，建议：")
    print("   • 增加数据增强")
    print("   • 增加 Dropout")
    print("   • 使用正则化")
elif gap < 0.02:
    print(f"\n✓ 训练-测试差距很小 ({gap*100:.2f}%)")
    print("  模型泛化能力良好！")
```

### Step 6: 查看预测结果

```python
# ============================================================================
# 第 6 步：查看预测结果
# ============================================================================
print("\n" + "=" * 60)
print("【6. 查看预测结果】")
print("=" * 60)

# 加载最佳模型
model.load_state_dict(torch.load('cifar_best.pth'))
model.eval()

# 取一批测试数据
inputs, targets = next(iter(testloader))
inputs, targets = inputs.to(device), targets.to(device)

with torch.no_grad():
    outputs = model(inputs)
    _, predictions = outputs.max(1)

# 可视化前 10 个预测
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

correct_count = 0
for i in range(10):
    # 反标准化
    img = inputs[i].cpu().numpy().transpose(1, 2, 0)
    img = img * 0.5 + 0.5
    img = np.clip(img, 0, 1)
    
    pred = predictions[i].item()
    true = targets[i].item()
    
    is_correct = pred == true
    if is_correct:
        correct_count += 1
    
    axes[i].imshow(img)
    color = 'green' if is_correct else 'red'
    axes[i].set_title(
        f'Pred: {classes[pred]}\nTrue: {classes[true]}', 
        color=color, 
        fontsize=10
    )
    axes[i].axis('off')

plt.suptitle(f'Predictions (Correct: {correct_count}/10)', fontsize=14)
plt.tight_layout()
plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ 预测结果已保存为 predictions.png")

# 详细评估
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 获取所有测试集预测
all_preds = []
all_targets = []

model.eval()
with torch.no_grad():
    for inputs, targets in testloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.numpy())

# 分类报告
print("\n📊 详细分类报告:")
print(classification_report(
    all_targets, 
    all_preds, 
    target_names=classes,
    digits=4
))

# 混淆矩阵
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes
)
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ 混淆矩阵已保存为 confusion_matrix.png")

print(f"\n🎊 恭喜！你完成了 Week2 毕业项目！")
print(f"🏆 最终测试准确率：{best_acc*100:.2f}%")
print("=" * 60)
```

---

## 二、费曼输出任务

### 任务1: 向朋友解释这个项目

**场景:** 你要向不懂 AI 的朋友介绍你的毕业项目

**要求:**
- 用大白话解释 CIFAR-10 是什么
- 说明为什么需要数据增强
- 解释 CNN 的工作原理（用比喻）
- 展示最终成果

**参考模板:**
```
"我做了一个 AI 图片识别系统，
它能认出 10 种东西：飞机、汽车、小鸟...

就像教小孩认东西一样，
我给它看了 6 万张图片，
它还学会了'举一反三'——
即使图片翻转、裁剪了也能认出来。

最后它的准确率达到了 80%，
相当于 10 道题能对 8 道！"
```

**⏰ 时间:** 15 分钟

---

### 任务2: 当一次技术经理

**场景:** 向老板汇报项目成果

**要覆盖:**
1. 项目背景和目标
2. 技术方案选型理由
3. 关键挑战和解决方案
4. 结果分析和改进方向
5. 实际应用场景

**提示:**
- 架构选择：为什么用 3 层卷积？
- 数据增强：如何提升泛化能力？
- BatchNorm：为什么能加速训练？
- Dropout：如何防止过拟合？

**⏰ 时间:** 30 分钟

---

## 三、常见问题和调试

### 问题1: 准确率太低 (<60%)

**可能原因:**
- 学习率太高或太低
- 训练轮数不够
- 模型太简单

**解决:**
```python
# 尝试调整学习率
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 降低

# 增加训练轮数
num_epochs = 50

# 增加模型复杂度
self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
```

### 问题2: 过拟合（训练准确率高，测试低）

**症状:** Train Acc > 90%, Test Acc < 70%

**解决:**
```python
# 1. 增加数据增强
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 新增
    transforms.ToTensor(),
    transforms.Normalize(...)
])

# 2. 增加 Dropout
self.dropout = nn.Dropout(0.5)  # 从 0.3 增加到 0.5

# 3. 增加权重衰减
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
```

### 问题3: 训练太慢

**优化:**
```python
# 1. 使用 GPU
device = torch.device('cuda')
model = model.to(device)

# 2. 增大批次
trainloader = DataLoader(..., batch_size=256)

# 3. 减少 workers（如果 CPU 弱）
trainloader = DataLoader(..., num_workers=0)
```

---

## 四、进阶挑战

### 挑战1: 提升到 85%+

**方法:**
```python
# 使用更深的网络（ResNet 风格）
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = torch.relu(self.bn(self.conv1(x)))
        x = self.conv2(x)
        x += residual  # 残差连接
        return torch.relu(x)
```

### 挑战2: 迁移学习

```python
# 使用预训练的 ResNet
from torchvision import models

model = models.resnet18(pretrained=True)
# 修改最后一层
model.fc = nn.Linear(512, 10)
```

### 挑战3: 部署为 Web 服务

```python
# 用 Flask 提供 API
from flask import Flask, request
import base64

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # 接收图片，返回预测
    ...
```

---

## 五、本章小结

### 核心要点

✅ **数据处理:**
- 数据增强提升泛化
- 标准化加速收敛
- 可视化理解数据

✅ **模型设计:**
- 卷积提取特征
- BatchNorm 稳定训练
- Dropout 防止过拟合

✅ **训练技巧:**
- Adam 优化器
- 学习率调度
- 保存最佳模型

✅ **评估方法:**
- 准确率曲线
- 混淆矩阵
- 分类报告

### 下一步

现在你已经完成了 Week2 的所有内容！

**准备好进入 Week3 了吗？**
- Day15: 目标检测 (YOLO)
- Day16: 图像分割
- Day17: GAN 生成对抗网络
- ...

**继续前进！** 🚀

---

## 📱 关于作者 & 获取更多资源

本教程由 **Lee（职场宝爸）** 创建，记录从零基础到独立完成 AI 项目的真实历程。

### 关注公众号，获取独家内容

**公众号名称：Lee 的成长日记**

微信搜索关注，获取：
- ✅ **AI 学习路线规划**：零基础如何系统学习 AI
- ✅ **项目实战源码**：完整可运行的项目代码
- ✅ **深度技术解析**：前沿技术原理 + 手写代码实现
- ✅ **职场成长心得**：一个宝爸的 AI 逆袭之路

**关注福利**：
- 回复「**路线**」→ 获取 30 天 AI 学习计划表
- 回复「**项目**」→ 获取 GitHub 项目源码合集
- 回复「**资料**」→ 获取零基础学习资源推荐

**扫码关注公众号**：

![公众号二维码](../../images/logos/ewm.jpg)

### 其他平台

- 📂 **GitHub**：https://github.com/Lee985-cmd/AI-30Days-Challenge
- 📝 **CSDN 博客**：https://blog.csdn.net/m0_67081842
- 💬 **公众号**：微信搜索「Lee 的成长日记」

---

> 💡 **学习建议**
> 
> 如果本篇教程对你有帮助，欢迎：
> 1. **Star GitHub 项目**：https://github.com/Lee985-cmd/AI-30Days-Challenge
> 2. **关注公众号**获取更多独家内容
> 3. **留言交流**你的学习困惑
> 
> **一起学习，一起进步！** 🤝
