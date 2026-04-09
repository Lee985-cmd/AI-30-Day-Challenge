"""
Day09 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day09_examples.py

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
print("Day09 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("🏢 我的第一个多层神经网络！")
print("=" * 50)

# 定义激活函数
def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid 激活函数（用于输出层）"""
    return 1 / (1 + np.exp(-x))

# 创建一个简单的 3 层神经网络
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化神经网络
        
        参数:
        - input_size: 输入特征数（比如图片有多少像素）
        - hidden_size: 隐藏层神经元数
        - output_size: 输出类别数（比如 0-9 就是 10）
        """
        print("\n【创建神经网络】")
        
        # 第 1 层权重（输入层 → 隐藏层）
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        
        # 第 2 层权重（隐藏层 → 输出层）
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        print(f"✓ 输入层：{input_size} 个神经元")
        print(f"✓ 隐藏层：{hidden_size} 个神经元")
        print(f"✓ 输出层：{output_size} 个神经元")
        print(f"\n网络结构：{input_size} → {hidden_size} → {output_size}")
    
    def forward(self, X):
        """
        前向传播（信息从前往后流动）
        
        就像传球游戏：
        输入层 → 隐藏层 → 输出层
        """
        # 第 1 层：输入 → 隐藏
        self.z1 = np.dot(X, self.W1) + self.b1  # 加权求和
        self.a1 = relu(self.z1)                 # ReLU 激活
        
        # 第 2 层：隐藏 → 输出
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)              # Sigmoid 激活（输出概率）
        
        return self.a2
    
    def predict(self, X):
        """预测类别"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# 测试网络
print("\n" + "=" * 50)
print("🔮 测试神经网络")
print("=" * 50)

# 创建一个简单的网络
# 输入：2 个特征
# 隐藏：4 个神经元
# 输出：2 个类别
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=2)

# 一些测试数据
X_test = np.array([
    [0.5, 0.3],
    [0.8, 0.6],
    [0.2, 0.9]
])

# 前向传播
output = nn.forward(X_test)

print("\n测试结果：")
for i in range(len(X_test)):
    print(f"样本{i+1}: 输入={X_test[i]} → 输出概率={output[i]}")

print("\n💡 工作原理：")
print("""
1. 输入数据进入网络
2. 第 1 层：加权求和 + ReLU 激活
   - 提取简单特征
3. 第 2 层：再次加权求和 + Sigmoid 激活
   - 组合特征，输出概率
4. 选概率最大的作为预测结果

这就是"前向传播"！
""")

# ===== 代码块 2 =====

print("=" * 50)
print("🔄 反向传播详解")
print("=" * 50)

class NeuralNetworkWithBackprop(NeuralNetwork):
    """带反向传播的神经网络"""
    
    def train(self, X_train, y_train, learning_rate=0.1, epochs=1000):
        """
        训练神经网络
        
        参数:
        - X_train: 训练数据
        - y_train: 真实标签
        - learning_rate: 学习率（每次调整多少）
        - epochs: 训练轮数
        """
        print(f"\n开始训练...")
        print(f"学习率：{learning_rate}")
        print(f"训练{epochs}轮\n")
        
        loss_history = []
        
        for epoch in range(epochs):
            # 1. 前向传播
            output = self.forward(X_train)
            
            # 2. 计算误差（损失）
            loss = -np.mean(y_train * np.log(output + 1e-9) + 
                          (1 - y_train) * np.log(1 - output + 1e-9))
            loss_history.append(loss)
            
            # 3. 反向传播（计算梯度）
            # 输出层误差
            dz2 = output - y_train
            dW2 = np.dot(self.a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            # 隐藏层误差
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * (self.z1 > 0)  # ReLU 的导数
            dW1 = np.dot(X_train.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            # 4. 更新权重（梯度下降）
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            
            # 打印进度
            if (epoch + 1) % 200 == 0:
                acc = np.mean(np.argmax(output, axis=1) == np.argmax(y_train, axis=1))
                print(f"第{epoch+1:4d}轮 - 损失：{loss:.4f} - 准确率：{acc*100:.1f}%")
        
        print("\n训练完成！✅")
        
        # 画损失曲线
        plt.figure(figsize=(10, 4))
        plt.plot(loss_history, linewidth=2)
        plt.xlabel('训练轮数')
        plt.ylabel('损失')
        plt.title('训练过程损失曲线')
        plt.grid(True, alpha=0.3)
        plt.show()

# 创建一个简单的数据集（异或问题）
print("\n" + "=" * 50)
print("🎯 实战：解决异或问题")
print("=" * 50)

# XOR 数据
X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# One-hot 编码（把类别转成向量）
y_xor_onehot = np.array([
    [1, 0],  # 0 → [1, 0]
    [0, 1],  # 1 → [0, 1]
    [0, 1],  # 1 → [0, 1]
    [1, 0]   # 0 → [1, 0]
])

# 创建网络
print("\n创建神经网络...")
nn_xor = NeuralNetworkWithBackprop(input_size=2, hidden_size=4, output_size=2)

# 训练
nn_xor.train(X_xor, y_xor_onehot, learning_rate=0.5, epochs=2000)

# 测试
print("\n" + "=" * 50)
print("🔮 测试结果")
print("=" * 50)

predictions = nn_xor.predict(X_xor)

for i in range(len(X_xor)):
    input_val = X_xor[i]
    pred = predictions[i]
    # 转回数字
    pred_num = 0 if pred == 0 else 1
    true_num = 0 if np.argmax(y_xor_onehot[i]) == 0 else 1
    
    status = "✅" if pred_num == true_num else "❌"
    print(f"输入：{input_val} → 预测：{pred_num} (真实：{true_num}) {status}")

print("\n🎉 多层神经网络完美解决了异或问题！")
print("这是单层网络做不到的！")

# ===== 代码块 3 =====

print("=" * 50)
print("👁️ 可视化神经网络内部")
print("=" * 50)

# 创建一个更大的网络来可视化
nn_viz = NeuralNetworkWithBackprop(input_size=2, hidden_size=8, output_size=2)

# 训练一下
nn_viz.train(X_xor, y_xor_onehot, learning_rate=0.5, epochs=1000)

# 可视化权重
fig, axes = plt.subplots(2, 4, figsize=(16, 4))
axes = axes.ravel()

print("\n隐藏层每个神经元的学习到的权重：")

for i in range(8):
    # 画出这个神经元的权重
    w1 = nn_viz.W1[:, i]
    axes[i].imshow(w1.reshape(1, -1), cmap='coolwarm', aspect='auto')
    axes[i].set_title(f'神经元{i+1}')
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.tight_layout()
plt.show()

print("\n💡 说明：")
print("- 每个神经元学到了不同的权重模式")
print("- 红色 = 正权重（兴奋）")
print("- 蓝色 = 负权重（抑制）")
print("- 不同的神经元关注不同的特征")