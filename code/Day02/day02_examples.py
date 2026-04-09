"""
Day02 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day02_examples.py

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
print("Day02 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

age = 25        # 年龄盒子
name = "小明"   # 名字盒子

# ===== 代码块 2 =====

fruits = ["苹果", "香蕉", "橙子"]
#       第 0 个  第 1 个  第 2 个

# ===== 代码块 3 =====

person = {
    "name": "张三",
    "age": 25
}

# ===== 代码块 4 =====

if score >= 60:
    print("及格")
else:
    print("不及格")

# ===== 代码块 5 =====

for i in range(5):
    print("第", i+1, "次")

# ===== 代码块 6 =====

# 导入必要的工具包
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

print("=" * 50)
print("🎉 我的第一个机器学习程序！")
print("=" * 50)

# 准备训练数据
# 假设有 4 个水果的数据
# 每个水果有两个特征：[重量（克）, 甜度（1-10）]

# 训练数据（已知答案）
X_train = [
    [150, 7],    # 苹果（重，比较甜）
    [160, 8],    # 苹果
    [140, 6],    # 苹果
    [50, 3],     # 柠檬（轻，不太甜）
    [60, 2],     # 柠檬
    [55, 4],     # 柠檬
]

# 标签（每个水果的类别）
# 0 = 苹果，1 = 柠檬
y_train = [
    0, 0, 0,     # 前 3 个是苹果
    1, 1, 1      # 后 3 个是柠檬
]

print("\n训练数据：")
print("苹果（0）:", X_train[0:3])
print("柠檬（1）:", X_train[3:6])

# 创建 K 近邻模型
# K=3 表示看最近的 3 个邻居
knn = KNeighborsClassifier(n_neighbors=3)

# 训练模型
print("\n正在训练模型...")
knn.fit(X_train, y_train)
print("✅ 训练完成！")

# 测试模型
# 假设有一个新的水果：[155, 7.5]
new_fruit = [[155, 7.5]]

prediction = knn.predict(new_fruit)

print("\n" + "=" * 50)
print("🔮 预测结果")
print("=" * 50)
print(f"新水果的特征：重量={new_fruit[0][0]}克，甜度={new_fruit[0][1]}")

if prediction[0] == 0:
    print("预测结果：这是苹果 🍎")
else:
    print("预测结果：这是柠檬 🍋")

# ===== 代码块 7 =====

from sklearn.neighbors import KNeighborsClassifier
# sklearn = 机器学习工具箱（里面有很多工具）
# neighbors = 邻居模块（包含 K 近邻算法）
# KNeighborsClassifier = K 近邻分类器（专门用于分类）
# import = 导入这个工具

import numpy as np
# numpy = 科学计算工具箱
# np = 给它起个小名（方便后面使用）

# ===== 代码块 8 =====

X_train = [
    [150, 7],    # 苹果
    [160, 8],    # 苹果
    [140, 6],    # 苹果
    [50, 3],     # 柠檬
    [60, 2],     # 柠檬
    [55, 4],     # 柠檬
]
# X_train = 训练数据（特征）
# [] = 列表
# 每个 [] 里是一个水果的数据
# [重量，甜度] = 两个特征

y_train = [0, 0, 0, 1, 1, 1]
# y_train = 标签（答案）
# 0 = 苹果，1 = 柠檬
# 顺序要和 X_train 对应

# ===== 代码块 9 =====

knn = KNeighborsClassifier(n_neighbors=3)
# knn = 模型的名字（你可以自己起）
# KNeighborsClassifier() = 创建一个 K 近邻分类器
# n_neighbors=3 = K=3，看最近的 3 个邻居

# ===== 代码块 10 =====

knn.fit(X_train, y_train)
# fit = 拟合、训练
# 把训练数据和答案告诉模型
# 让它学习规律

# ===== 代码块 11 =====

new_fruit = [[155, 7.5]]
# 新水果的数据
# 注意：要用两层 [[]]，因为要表示"一批数据"

prediction = knn.predict(new_fruit)
# predict = 预测
# 用训练好的模型预测新水果是什么

# ===== 代码块 12 =====

import matplotlib.pyplot as plt

# 把数据画出来看看
plt.figure(figsize=(8, 6))

# 画苹果（红色圆点）
apple_x = [150, 160, 140]
apple_y = [7, 8, 6]
plt.scatter(apple_x, apple_y, c='red', s=100, label='苹果', marker='o')

# 画柠檬（黄色圆点）
lemon_x = [50, 60, 55]
lemon_y = [3, 2, 4]
plt.scatter(lemon_x, lemon_y, c='yellow', s=100, label='柠檬', 
           edgecolors='black', marker='^')

# 画新水果（绿色星星）
new_x = [155]
new_y = [7.5]
plt.scatter(new_x, new_y, c='green', s=200, label='新水果', 
           marker='*', edgecolors='black')

# 设置图表
plt.xlabel('重量（克）', fontsize=12)
plt.ylabel('甜度（1-10）', fontsize=12)
plt.title('K 近邻算法示意图（K=3）', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("看图说明：")
print("✓ 红色圆点 = 苹果（训练数据）")
print("✓ 黄色三角 = 柠檬（训练数据）")
print("✓ 绿色星星 = 新水果（要预测的）")
print("\n可以看到：")
print("新水果离苹果更近")
print("所以它应该是苹果！🍎")

# ===== 代码块 13 =====

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

print("=" * 50)
print("🌸 鸢尾花分类 - 机器学习经典案例")
print("=" * 50)

# 1. 加载数据
iris = load_iris()
X = iris.data      # 特征（花萼和花瓣的尺寸）
y = iris.target    # 标签（花的品种）

print("\n数据集信息：")
print(f"样本数量：{len(X)} 朵花")
print(f"特征数量：{X.shape[1]} 个")
print(f"特征名称：{iris.feature_names}")
print(f"类别名称：{iris.target_names}")

# 2. 划分训练集和测试集
# 80% 用来训练，20% 用来测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n训练集：{len(X_train)} 朵花")
print(f"测试集：{len(X_test)} 朵花")

# 3. 创建模型
knn = KNeighborsClassifier(n_neighbors=3)

# 4. 训练模型
print("\n正在训练模型...")
knn.fit(X_train, y_train)
print("✅ 训练完成！")

# 5. 测试准确率
predictions = knn.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n测试集准确率：{accuracy * 100:.2f}%")

# 6. 实际使用
print("\n" + "=" * 50)
print("🔮 实际应用：预测一朵新的花")
print("=" * 50)

# 假设有一朵新的鸢尾花
new_flower = [[5.0, 3.5, 1.5, 0.3]]
# 特征顺序：花萼长度、花萼宽度、花瓣长度、花瓣宽度

prediction = knn.predict(new_flower)
species_name = iris.target_names[prediction[0]]

print(f"\n新花的特征：")
print(f"  花萼：{new_flower[0][0]}cm × {new_flower[0][1]}cm")
print(f"  花瓣：{new_flower[0][2]}cm × {new_flower[0][3]}cm")
print(f"\n预测结果：这是 {species_name} 鸢尾花")

print("\n" + "=" * 50)
print("🎊 恭喜！你已经完成了第一个真正的 AI 项目！")
print("=" * 50)

# ===== 代码块 14 =====

# 把数据分成两部分
训练集（80%）→ 用来学习
测试集（20%）→ 用来考试

准确率 = 答对的题数 / 总题数

准确率 90% = 100 道题对了 90 道

# ===== 代码块 15 =====

print("=" * 50)
print("🔧 实验：不同的 K 值会怎样？")
print("=" * 50)

# 尝试不同的 K 值
for k in [1, 3, 5, 10]:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    acc = knn_k.score(X_test, y_test)
    print(f"K={k:2d} → 准确率：{acc*100:.2f}%")

print("\n结论：")
print("✓ K 太小（如 K=1）→ 容易受个别数据影响")
print("✓ K 太大（如 K=100）→ 可能把不同类别的混在一起")
print("✓ K 适中（如 K=3,5）→ 通常比较好")