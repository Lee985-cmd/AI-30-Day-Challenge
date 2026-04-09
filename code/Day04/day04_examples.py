"""
Day04 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day04_examples.py

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
print("Day04 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

from sklearn.svm import SVC
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

print("=" * 50)
print("🎯 我的第一个 SVM 模型！")
print("=" * 50)

# 1. 创建数据集（两类容易分开的数据）
X, y = make_blobs(
    n_samples=100,      # 100 个样本
    centers=2,          # 2 个中心（两类）
    n_features=2,       # 2 个特征（方便画图）
    cluster_std=1.0,    # 每个类的分散程度
    random_state=42
)

print(f"\n数据集信息：")
print(f"样本数：{len(X)}")
print(f"特征数：{X.shape[1]}")
print(f"类别数：{len(np.unique(y))}")

# 2. 可视化数据
plt.figure(figsize=(8, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='类别 0', alpha=0.6)
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='类别 1', alpha=0.6)
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.title('原始数据分布')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 3. 创建 SVM 模型（线性核）
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)

# 4. 训练模型
print("\n正在训练线性 SVM...")
svm_linear.fit(X, y)
print("✅ 训练完成！")

# 5. 评估
accuracy = svm_linear.score(X, y)
print(f"\n训练集准确率：{accuracy*100:.2f}%")

# 6. 可视化决策边界
print("\n正在绘制决策边界...")

# 创建网格
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测整个网格
Z = svm_linear.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 画图
plt.figure(figsize=(10, 8))

# 画背景（决策区域）
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# 画数据点
plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='类别 0', edgecolors='k', s=100)
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='类别 1', edgecolors='k', s=100)

# 画决策边界
contour = plt.contour(xx, yy, Z, colors='black', linewidths=2)

# 标记支持向量（最关键的点）
support_vectors = svm_linear.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
           c='yellow', s=200, edgecolors='k', 
           marker='*', label='支持向量', zorder=10)

plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.title('SVM 线性分类 - 决策边界和支持向量')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n支持向量数量：{len(support_vectors)}")
print("看到黄色的星星了吗？这些就是支持向量！")
print("它们决定了分界线的位置！")

print("\n" + "=" * 50)
print("🎊 恭喜！你学会了线性 SVM！")
print("=" * 50)

# ===== 代码块 2 =====

from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

print("=" * 50)
print("🔬 实验：不同核函数的效果对比")
print("=" * 50)

# 创建非线性可分的数据（月牙形）
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# 画原始数据
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[y==0, 0], X[y==0, 1], c='red', alpha=0.6)
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', alpha=0.6)
plt.title('月牙形数据 - 线性不可分')
plt.grid(True, alpha=0.3)

# 1. 线性核
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X, y)

# 计算决策边界
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svm_linear.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(1, 3, 2)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=100)
plt.title('线性核 - 直线分不开')
plt.grid(True, alpha=0.3)

# 2. RBF 核
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X, y)

Z = svm_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(1, 3, 3)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=100)
plt.title('RBF 核 - 曲线完美分开')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n结论：")
print("✓ 线性核：只能用直线，分不开月牙形")
print("✓ RBF 核：可以用曲线，完美分开！")
print("✓ 这就是核技巧的魔力！")