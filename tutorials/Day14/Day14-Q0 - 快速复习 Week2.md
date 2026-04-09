# Day14-Q0 - 快速复习 Week2

## 🔄 Week2 核心要点回顾

### 知识地图

```
Week2: 深度学习入门
│
├─ Day8: 神经网络初探
│  ├─ 神经元模型
│  ├─ 激活函数 (Sigmoid, ReLU, Tanh)
│  └─ 前向传播
│
├─ Day9: 多层神经网络
│  ├─ 隐藏层的作用
│  ├─ 反向传播算法
│  └─ 梯度下降优化
│
├─ Day10: PyTorch 入门
│  ├─ Tensor 操作
│  ├─ 自动求导 (autograd)
│  └─ nn.Module 构建模型
│
├─ Day11: CNN 基础
│  ├─ 卷积层 (Conv2d)
│  ├─ 池化层 (MaxPool)
│  └─ CNN 架构设计
│
├─ Day12: 经典 CNN 架构
│  ├─ LeNet, AlexNet
│  ├─ VGG, ResNet
│  └─ 迁移学习
│
└─ Day13: RNN 和 LSTM
   ├─ 循环神经网络
   ├─ LSTM 单元
   └─ 序列数据处理
```

---

## 📝 自测清单

### Day8-9: 神经网络基础
- [ ] 能解释神经元的工作原理
- [ ] 知道常见激活函数的区别
- [ ] 理解反向传播的核心思想
- [ ] 会用 PyTorch 搭建简单网络

### Day10: PyTorch
- [ ] 会创建和操作 Tensor
- [ ] 理解 autograd 机制
- [ ] 能用 nn.Module 定义模型
- [ ] 知道完整的训练流程

### Day11-12: CNN
- [ ] 理解卷积操作的原理
- [ ] 知道 padding、stride 的作用
- [ ] 熟悉经典 CNN 架构
- [ ] 会使用预训练模型

### Day13: RNN/LSTM
- [ ] 理解 RNN 的循环结构
- [ ] 知道 LSTM 如何解决长期依赖
- [ ] 能处理序列数据
- [ ] 做过文本分类或生成

---

## 💻 核心代码回顾

### 简单神经网络

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.network(x)
```

### CNN 模型

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 16 * 16, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 16 * 16)
        return self.fc(x)
```

### 训练循环

```python
model = SimpleNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 🎯 从 Week2 到 Day14

**Week2 我们学会了:**
- ✅ 神经网络的原理
- ✅ PyTorch 框架使用
- ✅ CNN 处理图像
- ✅ RNN 处理序列

**Day14 我们要做:**
- 🎓 综合运用所有知识
- 🎓 完成一个完整项目
- 🎓 实践费曼学习法
- 🎓 为 Week3 做准备

**类比:**
```
Week2: 学习了各种武功招式
   ↓
Day14: 实战演练，融会贯通
```

---

## 🔗 相关链接

- [← Day13-Q6 - RNN 进阶](../Day13/Day13-Q6%20-%20RNN%20进阶应用.md)
- [→ Day14-Q1 - 项目选择和规划](./Day14-Q1%20-%20项目选择和规划.md)
