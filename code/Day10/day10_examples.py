"""
Day10 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day10_examples.py

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
print("Day10 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 50)
print("🔥 我的第一个 PyTorch 程序！")
print("=" * 50)

# 检查 PyTorch 版本
print(f"\nPyTorch 版本：{torch.__version__}")

# 1. Tensor（张量）- PyTorch 的基本单位
print("\n【1. Tensor 基础】")

# 创建一个 Tensor（就像 NumPy 数组）
x = torch.tensor([1.0, 2.0, 3.0])
print(f"创建一个 Tensor: {x}")
print(f"形状：{x.shape}")
print(f"数据类型：{x.dtype}")

# 各种创建方法
zeros = torch.zeros(2, 3)      # 全 0
ones = torch.ones(2, 3)        # 全 1
random_t = torch.rand(2, 3)    # 随机数

print(f"\n全 0 矩阵:\n{zeros}")
print(f"\n全 1 矩阵:\n{ones}")
print(f"\n随机数矩阵:\n{random_t}")

# Tensor 运算
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"\nTensor 运算:")
print(f"a + b = {a + b}")
print(f"a × b = {a * b}")
print(f"a · b (点积) = {torch.dot(a, b)}")

# 2. 自动求导（Autograd）- PyTorch 的超能力！
print("\n" + "=" * 50)
print("【2. 自动求导 - 超级方便！】")
print("=" * 50)

# 创建一个需要求导的 Tensor
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1

# 自动计算导数！
y.backward()

print(f"函数：y = x² + 3x + 1")
print(f"当 x = 2 时:")
print(f"  y = {y.item()}")
print(f"  dy/dx = {x.grad.item()}")  # 应该等于 2x + 3 = 7

print("\n💡 这就是自动求导！")
print("不用手动推导数！PyTorch 帮你算！")
print("这是训练神经网络的关键！")

# 3. 用 PyTorch 实现一个简单的神经网络
print("\n" + "=" * 50)
print("【3. 搭建神经网络】")
print("=" * 50)

class SimpleNet(nn.Module):
    """一个简单的神经网络"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        
        # 定义网络层
        self.layer1 = nn.Linear(input_size, hidden_size)  # 输入层→隐藏层
        self.relu = nn.ReLU()                             # ReLU 激活
        self.layer2 = nn.Linear(hidden_size, output_size) # 隐藏层→输出层
        self.sigmoid = nn.Sigmoid()                       # Sigmoid 激活
        
        print(f"✓ 创建了神经网络:")
        print(f"  输入层：{input_size} 个神经元")
        print(f"  隐藏层：{hidden_size} 个神经元 (ReLU)")
        print(f"  输出层：{output_size} 个神经元 (Sigmoid)")
    
    def forward(self, x):
        """前向传播"""
        x = self.layer1(x)    # 第 1 层
        x = self.relu(x)      # ReLU 激活
        x = self.layer2(x)    # 第 2 层
        x = self.sigmoid(x)   # Sigmoid 激活
        return x

# 创建网络实例
net = SimpleNet(input_size=2, hidden_size=4, output_size=1)

print(f"\n网络结构：")
print(net)

# 测试一下
test_input = torch.tensor([[0.5, 0.3]], dtype=torch.float32)
output = net(test_input)

print(f"\n测试:")
print(f"输入：{test_input}")
print(f"输出：{output.item():.4f}")

print("\n🎉 你已经用 PyTorch 搭建了第一个神经网络！")

# ===== 代码块 2 =====

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

print("=" * 50)
print("✍️ PyTorch 实战：MNIST 手写数字识别")
print("=" * 50)

# 1. 准备数据
print("\n【1. 加载 MNIST 数据集】")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),           # 转成 Tensor
    transforms.Normalize((0.5,), (0.5,))  # 标准化
])

# 下载并加载训练集
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

print(f"✓ 训练集大小：{len(train_dataset)} 张图片")
print(f"✓ 测试集大小：{len(test_dataset)} 张图片")

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

# 2. 构建神经网络模型
print("\n" + "=" * 50)
print("【2. 构建神经网络】")
print("=" * 50)

class MNISTNet(nn.Module):
    """MNIST 分类网络"""
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # 卷积层（提取特征）
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.relu = nn.ReLU()
        
        # 全连接层（分类）
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 个数字
        self.dropout = nn.Dropout(0.5)  # 防止过拟合
    
    def forward(self, x):
        # 卷积层
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 → 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 → 7x7
        
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# 创建模型
model = MNISTNet()
print("✓ 创建了 CNN 模型")
print(model)

# 3. 定义损失函数和优化器
print("\n" + "=" * 50)
print("【3. 配置训练参数】")
print("=" * 50)

criterion = nn.CrossEntropyLoss()  # 交叉熵损失（分类常用）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

print(f"损失函数：CrossEntropyLoss")
print(f"优化器：Adam (学习率=0.001)")

# 4. 训练模型
print("\n" + "=" * 50)
print("【4. 开始训练模型】")
print("=" * 50)

num_epochs = 5
loss_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()         # 反向传播
        optimizer.step()        # 更新权重
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # 计算平均损失和准确率
    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total * 100
    loss_history.append(avg_loss)
    
    print(f"第{epoch+1}/{num_epochs}轮 - "
          f"损失：{avg_loss:.4f} - "
          f"准确率：{accuracy:.2f}%")

# 画损失曲线
plt.figure(figsize=(10, 4))
plt.plot(loss_history, 'bo-', linewidth=2, markersize=8)
plt.xlabel('训练轮数')
plt.ylabel('平均损失')
plt.title('训练过程损失曲线')
plt.grid(True, alpha=0.3)
plt.show()

# 5. 评估模型
print("\n" + "=" * 50)
print("【5. 评估模型性能】")
print("=" * 50)

model.eval()  # 切换到评估模式
correct = 0
total = 0

with torch.no_grad():  # 不计算梯度（节省内存）
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total * 100
print(f"✓ 测试集准确率：{test_accuracy:.2f}%")

if test_accuracy > 98:
    print("🎉 太棒了！超过 98%！")
elif test_accuracy > 95:
    print("👍 很好！超过 95%！")
else:
    print("💪 不错！继续训练会更好！")

# 6. 实际预测
print("\n" + "=" * 50)
print("🔮 【6. 实际预测示例】")
print("=" * 50)

model.eval()
sample_indices = [0, 100, 200, 300, 400]

fig, axes = plt.subplots(1, 5, figsize=(12, 3))
if len(sample_indices) == 1:
    axes = [axes]

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

print("\n" + "=" * 50)
print("🎊 恭喜！你用 PyTorch 完成了 MNIST 项目！")
print("=" * 50)

print("""
你学会了：
✓ PyTorch 基础（Tensor、自动求导）
✓ 搭建神经网络
✓ 训练完整流程
✓ 评估和预测

这已经是工业级的技能了！👏
""")