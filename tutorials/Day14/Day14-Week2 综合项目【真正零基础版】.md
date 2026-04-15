# 🎓 AI 入门 30 天挑战 - Day 14 真正零基础版

---

## 🌟 完整项目和代码

本教程是 **AI 入门 30 天挑战** 系列的一部分！

- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **欢迎 Star 支持！**

---



> **Week 2 的最后一天！**  
> **做一个完整的深度学习项目！**  
> **每个步骤都详细说明！**

---

## 📖 Week 2 完整回顾

### Day 8 - 神经网络初探
```
✓ 生物神经元 → 人工神经元
✓ 激活函数（ReLU、Sigmoid）
✓ 感知机
✓ 前向传播
```

### Day 9 - 多层神经网络
```
✓ 为什么需要多层
✓ 反向传播（学习的关键）
✓ 梯度下降优化
```

### Day 10 - PyTorch 入门
```
✓ Tensor 基础
✓ 自动求导
✓ nn.Module
✓ 完整训练流程
```

### Day 11 - CNN 基础
```
✓ 卷积层（提取特征）
✓ 池化层（降维）
✓ LeNet-5 架构
```

### Day 12 - 经典 CNN 架构
```
✓ AlexNet、VGG、ResNet
✓ 迁移学习
✓ 站在巨人肩膀上
```

### Day 13 - RNN 和 LSTM
```
✓ 序列数据处理
✓ RNN（有记忆）
✓ LSTM（长期依赖）
✓ 情感分析
```

如果这些都记得，我们开始今天的毕业项目！

---

## 🎯 项目三选一

今天你可以从以下三个项目中选择一个完成：

### 选项 A：图像分类进阶（CIFAR-10）🖼️

```
难度：⭐⭐⭐☆☆
有趣度：⭐⭐⭐⭐☆
实用性：⭐⭐⭐⭐☆

数据集：CIFAR-10
- 60,000 张彩色图片
- 10 个类别（飞机、汽车、鸟、猫...）
- 每张图片 32×32 像素

目标：
✓ 准确率 > 70%（及格）
✓ 准确率 > 80%（良好）
✓ 准确率 > 90%（优秀）

技能：
✓ CNN 架构设计
✓ 数据增强
✓ Dropout、BatchNorm
```

### 选项 B：文本生成（唐诗宋词）📜

```
难度：⭐⭐⭐⭐☆
有趣度：⭐⭐⭐⭐⭐
实用性：⭐⭐⭐☆☆

数据集：唐诗宋词
- 几千首古诗词
- 学习韵律和格式

目标：
✓ 生成合理的诗句
✓ 符合平仄韵律
✓ 有一定意境

技能：
✓ LSTM/GRU
✓ 文本生成
✓ 温度采样
```

### 选项 C：人脸识别 👤

```
难度：⭐⭐⭐⭐⭐
有趣度：⭐⭐⭐⭐⭐
实用性：⭐⭐⭐⭐⭐

数据集：LFW 或 CelebA
- 名人面部图片
- 区分不同的人

目标：
✓ 准确识别人脸
✓ 对光照、角度鲁棒

技能：
✓ CNN + Triplet Loss
✓ 面部特征提取
✓ 相似度计算
```

**建议选择：**
- 想做图像 → 选 A 或 C
- 想做文字 → 选 B
- 想挑战自己 → 选 C
- 想快速完成 → 选 A

---

## 💻 完整项目演示：CIFAR-10 图像分类

### 第 1 步：问题定义和数据准备

```python
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
```

---

### 第 2 步：构建模型

```python
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
```

---

### 第 3 步：训练模型

```python
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
```

---

### 第 4 步：评估和可视化

```python
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
```

---

## 📊 项目总结报告

```python
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
```

---

## 📝 Week 2 完整总结

### ✅ 这两周你学到了：

**第 8 天：神经网络基础**
- 神经元、激活函数、前向传播

**第 9 天：多层网络**
- 反向传播、梯度下降

**第 10 天：PyTorch**
- Tensor、自动求导、nn.Module

**第 11 天：CNN**
- 卷积、池化、LeNet-5

**第 12 天：经典架构**
- AlexNet、VGG、ResNet
- 迁移学习

**第 13 天：RNN/LSTM**
- 序列数据、情感分析

**第 14 天：综合项目**
- 完整的深度学习流程

---

## 🎁 下一步建议

### 选项 A：继续 Week 3-4
```
Week 3: 计算机视觉进阶
- 目标检测（YOLO、Faster R-CNN）
- 图像分割（U-Net、Mask R-CNN）
- GAN（生成对抗网络）

Week 4: 自然语言处理
- Transformer
- BERT
- GPT
```

### 选项 B：巩固基础
```
✓ 复习 Week 1-2 所有内容
✓ 多做项目练习
✓ 参加 Kaggle 比赛
✓ 读论文学习最新技术
```

### 选项 C：实战应用
```
✓ 找实习/工作
✓ 做个人项目
✓ 写技术博客
✓ 建立作品集
```

---

## 🌟 最后的鼓励

**恭喜你完成了前两周的学习！** 🎉

```
回头看：
两周前，你可能还不懂什么是 AI
现在，你已经能独立完成深度学习项目了！

看看你的成长：
Week 1: 7 种机器学习算法
Week 2: 深度学习全套技能

这不仅仅是知识的增长
更是能力的飞跃！

向前看：
AI 的世界很大很深：
- 计算机视觉
- 自然语言处理
- 强化学习
- 生成模型
- ...

继续前进吧！
你已经证明了你有能力学会这些！
相信自己！坚持下去！

AI 工程师之路，你已经在走了！💪✨

加油！我们在 Week 3 等你！🚀
```

---

## 📞 毕业打卡模板

```
日期：___________
Week 2 总学习时长：_______ 小时

掌握程度：⭐⭐⭐⭐⭐

最有收获的一天：


最难理解的概念：


最喜欢的项目：


Week 2 整体感受：


Week 3 的目标：


对自己说：


```

**Week 2 完成！给自己一个大大的奖励吧！** 🎊🎉🚀

---

## 🔗 相关链接

### 🌐 项目资源
- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **如果觉得有帮助，请给 GitHub 仓库 Star 支持！**

### 📚 学习路径
- [← Day13](../Day13/README.md)
- [→ Day15](../Day15/README.md)

---

*本教程属于 [AI 入门 30 天挑战](https://github.com/Lee985-cmd/AI-30-Day-Challenge) 系列*

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

![公众号二维码](../../../images/logos/ewm.jpg)

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
