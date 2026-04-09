"""
Day08 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day08_examples.py

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
print("Day08 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

import numpy as np

print("=" * 50)
print("🧠 我的第一个人工神经元！")
print("=" * 50)

# 定义神经元参数
print("\n【神经元的配置】")

# 权重（每个输入的重要性）
weights = np.array([0.3, 0.5, 0.2])
print(f"权重：{weights}")
print("  - 天气冷的权重：{weights[0]}")
print("  - 有钱的权重：{weights[1]}")
print("  - 有人陪的权重：{weights[2]}")

# 偏置（基础倾向）
bias = -0.5
print(f"\n偏置：{bias}")
print("（负数表示倾向于不去）")

# 激活函数（决定输出）
def step_function(x):
    """阶跃函数：大于 0 输出 1，否则输出 0"""
    return 1 if x > 0 else 0

# 测试不同的输入情况
print("\n" + "=" * 50)
print("🔮 测试不同情况")
print("=" * 50)

# 情况 1：天气冷、有钱、有人陪
print("\n【情况 1】天气冷 + 有钱 + 有人陪")
inputs1 = np.array([1, 1, 1])
weighted_sum1 = np.dot(inputs1, weights) + bias
output1 = step_function(weighted_sum1)
print(f"加权和：{weighted_sum1:.2f}")
print(f"输出：{'去吃火锅！✅' if output1 == 1 else '不去吃 ❌'}")

# 情况 2：天气不冷、有钱、有人陪
print("\n【情况 2】天气好 + 有钱 + 有人陪")
inputs2 = np.array([0, 1, 1])
weighted_sum2 = np.dot(inputs2, weights) + bias
output2 = step_function(weighted_sum2)
print(f"加权和：{weighted_sum2:.2f}")
print(f"输出：{'去吃火锅！✅' if output2 == 1 else '不去吃 ❌'}")

# 情况 3：天气冷、没钱、没人陪
print("\n【情况 3】天气冷 + 没钱 + 没人陪")
inputs3 = np.array([1, 0, 0])
weighted_sum3 = np.dot(inputs3, weights) + bias
output3 = step_function(weighted_sum3)
print(f"加权和：{weighted_sum3:.2f}")
print(f"输出：{'去吃火锅！✅' if output3 == 1 else '不去吃 ❌'}")

print("\n" + "=" * 50)
print("💡 神经元工作原理：")
print("=" * 50)
print("""
1. 接收输入（多个信号）
2. 加权求和（重要的信号权重大）
3. 加上偏置（基础倾向）
4. 激活函数判断（是否超过阈值）
5. 产生输出（做决定）

就像你做决定一样：
- 考虑多个因素
- 每个因素重要性不同
- 有个基础倾向
- 最后拍板决定！
""")

# ===== 代码块 2 =====

def step_function(x):
    """最简单的激活函数"""
    return 1 if x > 0 else 0

# 就像开关：非 0 即 1

# ===== 代码块 3 =====

def sigmoid(x):
    """S 型函数，输出 0-1 之间"""
    return 1 / (1 + np.exp(-x))

# 输出是概率：
# 接近 0 → 不太可能
# 接近 1 → 很可能

# ===== 代码块 4 =====

def relu(x):
    """修正线性单元"""
    return np.maximum(0, x)

# 小于 0 就输出 0
# 大于 0 就原样输出
# 就像单向阀

# ===== 代码块 5 =====

import matplotlib.pyplot as plt

print("=" * 50)
print("📈 激活函数可视化")
print("=" * 50)

# 生成 x 值
x = np.linspace(-5, 5, 100)

# 计算各种激活函数
y_step = np.where(x > 0, 1, 0)
y_sigmoid = 1 / (1 + np.exp(-x))
y_relu = np.maximum(0, x)

# 画图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 阶跃函数
axes[0].plot(x, y_step, linewidth=2)
axes[0].set_title('阶跃函数', fontsize=12)
axes[0].set_xlabel('输入 x')
axes[0].set_ylabel('输出 y')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

# Sigmoid 函数
axes[1].plot(x, y_sigmoid, linewidth=2, color='green')
axes[1].set_title('Sigmoid 函数', fontsize=12)
axes[1].set_xlabel('输入 x')
axes[1].set_ylabel('输出 y')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

# ReLU 函数
axes[2].plot(x, y_relu, linewidth=2, color='blue')
axes[2].set_title('ReLU 函数', fontsize=12)
axes[2].set_xlabel('输入 x')
axes[2].set_ylabel('输出 y')
axes[2].grid(True, alpha=0.3)
axes[2].axvline(x=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print("\n三种激活函数对比：")
print("1. 阶跃函数：")
print("   - 简单粗暴（非 0 即 1）")
print("   - 但不够平滑（不能求导）")
print()
print("2. Sigmoid 函数：")
print("   - 输出 0-1 之间的概率")
print("   - 平滑可导")
print("   - 适合二分类")
print()
print("3. ReLU 函数（最常用）:")
print("   - 计算快")
print("   - 效果好")
print("   - 深度学习首选")

# ===== 代码块 6 =====

print("=" * 50)
print("🎯 感知机解决异或问题")
print("=" * 50)

class Perceptron:
    """感知机类"""
    
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        """初始化"""
        # 随机初始化权重
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def predict(self, X):
        """预测"""
        weighted_sum = np.dot(X, self.weights) + self.bias
        return step_function(weighted_sum)
    
    def train(self, X_train, y_train):
        """训练"""
        print("开始训练...")
        
        for epoch in range(self.epochs):
            errors = 0
            
            for i in range(len(X_train)):
                # 前向传播
                prediction = self.predict(X_train[i])
                
                # 计算误差
                error = y_train[i] - prediction
                
                if error != 0:
                    errors += 1
                    
                    # 更新权重（学习）
                    self.weights += self.learning_rate * error * X_train[i]
                    self.bias += self.learning_rate * error
            
            # 打印训练进度
            if (epoch + 1) % 20 == 0:
                print(f"  第{epoch+1}轮 - 错误数：{errors}")
        
        print("训练完成！")

# 异或问题的数据
X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_xor = np.array([0, 1, 1, 0])

# 创建并训练感知机
print("\n创建一个感知机...")
perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=100)
perceptron.train(X_xor, y_xor)

# 测试
print("\n" + "=" * 50)
print("🔮 测试结果")
print("=" * 50)

for i in range(len(X_xor)):
    pred = perceptron.predict(X_xor[i])
    print(f"输入：{X_xor[i]} → 预测：{pred}, 真实：{y_xor[i]}")

print("\n⚠️ 注意：")
print("单个感知机无法完美解决异或问题！")
print("需要多层神经网络（下节课讲）")