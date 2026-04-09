"""
Day05 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day05_examples.py

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
print("Day05 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

print("=" * 50)
print("🔮 K-means 聚类初体验！")
print("=" * 50)

# 1. 创建数据集（模拟真实数据）
X, y_true = make_blobs(
    n_samples=300,      # 300 个样本
    centers=4,          # 4 个中心（真实的类别数）
    n_features=2,       # 2 个特征（方便画图）
    cluster_std=0.8,    # 每个类的分散程度
    random_state=42
)

print(f"\n数据集信息：")
print(f"样本数：{len(X)}")
print(f"特征数：{X.shape[1]}")
print(f"真实类别数：{len(np.unique(y_true))}")

# 2. 可视化原始数据（假装我们不知道标签）
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, s=50)
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.title('原始数据（没有标签）')
plt.grid(True, alpha=0.3)
plt.show()

# 3. 创建 K-means 模型（假设我们知道 K=4）
kmeans = KMeans(
    n_clusters=4,       # 分成 4 类
    init='k-means++',   # 智能初始化（更快收敛）
    n_init=10,          # 运行 10 次，选最好的
    max_iter=300,       # 最多迭代 300 次
    random_state=42
)

# 4. 训练模型（拟合数据）
print("\n正在训练 K-means...")
kmeans.fit(X)
print("✅ 训练完成！")

# 5. 获取聚类结果
labels = kmeans.labels_        # 每个样本的类别
centers = kmeans.cluster_centers_  # 聚类中心

print(f"\n聚类结果：")
print(f"找到 {len(np.unique(labels))} 个类别")
print(f"聚类中心坐标：\n{centers}")

# 6. 可视化聚类结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图：真实标签
ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', 
           alpha=0.6, s=50, edgecolors='k')
ax1.set_title('真实类别（上帝视角）', fontsize=12)
ax1.set_xlabel('特征 1')
ax1.set_ylabel('特征 2')
ax1.grid(True, alpha=0.3)

# 右图：K-means 预测
ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
           alpha=0.6, s=50, edgecolors='k')
ax2.scatter(centers[:, 0], centers[:, 1], c='red', 
           s=200, marker='*', label='聚类中心')
ax2.set_title('K-means 聚类结果', fontsize=12)
ax2.set_xlabel('特征 1')
ax2.set_ylabel('特征 2')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("🎊 恭喜！你学会了 K-means 聚类！")
print("=" * 50)

# ===== 代码块 2 =====

kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
# kmeans = 模型的名字
# KMeans() = 创建 K-means 聚类器
# n_clusters=4 = 分成 4 类（K=4）
# init='k-means++' = 智能初始化
#   - 不是完全随机选中心
#   - 让初始中心尽量分开
#   - 更快收敛，结果更好
# n_init=10 = 运行 10 次
#   - 每次随机初始中心
#   - 选效果最好的一次
# max_iter=300 = 最多迭代 300 次
# random_state=42 = 随机种子（保证结果可重复）

# ===== 代码块 3 =====

labels = kmeans.labels_
# labels_ = 每个样本的类别标签
# 比如：[0, 0, 1, 1, 2, 2, 3, ...]
# 表示第 1、2 个样本是第 0 类，第 3、4 个是第 1 类...

centers = kmeans.cluster_centers_
# cluster_centers_ = 聚类中心的坐标
# 每个类一个中心点

# ===== 代码块 4 =====

print("=" * 50)
print("❓ 怎么确定 K 值？（肘部法则）")
print("=" * 50)

# 尝试不同的 K 值
inertias = []  # 记录每个 K 值的误差
K_range = range(1, 11)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X)
    inertias.append(kmeans_temp.inertia_)  # 记录簇内误差平方和

# 画图
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('K 值（类别数）', fontsize=12)
plt.ylabel('误差平方和（Inertia）', fontsize=12)
plt.title('肘部法则 - 选择最优 K 值', fontsize=14)
plt.grid(True, alpha=0.3)

# 标记"肘部"
plt.axvline(x=4, color='r', linestyle='--', label='肘部位置（推荐 K=4）')
plt.legend()
plt.show()

print("\n怎么看这个图？")
print("✓ 找拐点（像手肘的地方）")
print("✓ K 太小 → 误差大（欠拟合）")
print("✓ K 太大 → 过细（过拟合）")
print("✓ 肘部位置 → 最佳平衡点")
print("\n在这个例子中，K=4 是最优的！")

# ===== 代码块 5 =====

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

print("=" * 50)
print("👥 实战：商场客户分群")
print("=" * 50)

# 1. 创建模拟数据
print("\n正在创建模拟数据...")

np.random.seed(42)
n_customers = 200

# 模拟商场客户数据
customers = pd.DataFrame({
    '年龄': np.random.normal(35, 10, n_customers).clip(18, 70),
    '年收入（万元）': np.random.exponential(20, n_customers).clip(5, 100),
    '消费评分': np.random.normal(50, 20, n_customers).clip(0, 100),
    '逛商场频率': np.random.poisson(5, n_customers).clip(0, 20),
})

print(f"✓ 数据创建完成！")
print(f"客户数：{len(customers)}")
print(f"特征数：{customers.shape[1]}")

# 2. 数据可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 年龄分布
axes[0, 0].hist(customers['年龄'], bins=20, alpha=0.7, color='skyblue')
axes[0, 0].set_xlabel('年龄')
axes[0, 0].set_ylabel('人数')
axes[0, 0].set_title('年龄分布')
axes[0, 0].grid(True, alpha=0.3)

# 收入分布
axes[0, 1].hist(customers['年收入（万元）'], bins=20, alpha=0.7, color='lightgreen')
axes[0, 1].set_xlabel('年收入（万元）')
axes[0, 1].set_ylabel('人数')
axes[0, 1].set_title('收入分布')
axes[0, 1].grid(True, alpha=0.3)

# 年龄 vs 收入
axes[1, 0].scatter(customers['年龄'], customers['年收入（万元）'], 
                  alpha=0.6, s=50)
axes[1, 0].set_xlabel('年龄')
axes[1, 0].set_ylabel('年收入（万元）')
axes[1, 0].set_title('年龄与收入关系')
axes[1, 0].grid(True, alpha=0.3)

# 消费评分 vs 逛商场频率
axes[1, 1].scatter(customers['消费评分'], customers['逛商场频率'], 
                  alpha=0.6, s=50)
axes[1, 1].set_xlabel('消费评分')
axes[1, 1].set_ylabel('逛商场频率（次/月）')
axes[1, 1].set_title('消费行为分析')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3. 数据预处理（标准化）
print("\n正在预处理数据...")

# 为什么要标准化？
# 因为"年收入"数值大（几万），"消费评分"数值小（0-100）
# 不标准化的话，年收入会主导结果
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customers)

print("✓ 数据标准化完成！")

# 4. 肘部法则选 K
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('K 值')
plt.ylabel('误差平方和')
plt.title('肘部法则 - 客户分群')
plt.grid(True, alpha=0.3)
plt.show()

# 5. 训练最终模型（假设选 K=4）
print("\n训练最终模型（K=4）...")
kmeans_customers = KMeans(n_clusters=4, random_state=42, n_init=10)
customer_labels = kmeans_customers.fit_predict(X_scaled)

# 添加到原数据
customers['客户类别'] = customer_labels

print("✅ 聚类完成！")

# 6. 分析每个类别的特征
print("\n" + "=" * 50)
print("📊 各类客户特征分析")
print("=" * 50)

group_stats = customers.groupby('客户类别').mean()
print("\n各类别平均值：")
print(group_stats)

# 给客户画像
print("\n" + "=" * 50)
print("🎯 客户画像")
print("=" * 50)

for cluster_id in range(4):
    cluster_data = customers[customers['客户类别'] == cluster_id]
    
    print(f"\n【客户群体 {cluster_id}】")
    print(f"  人数：{len(cluster_data)} ({len(cluster_data)/len(customers)*100:.1f}%)")
    print(f"  平均年龄：{cluster_data['年龄'].mean():.1f} 岁")
    print(f"  平均年收入：{cluster_data['年收入（万元）'].mean():.1f} 万元")
    print(f"  平均消费评分：{cluster_data['消费评分'].mean():.1f}")
    print(f"  平均逛商场频率：{cluster_data['逛商场频率'].mean():.1f} 次/月")
    
    # 给这个群体起个名字
    if (cluster_data['年收入（万元）'].mean() > 25 and 
        cluster_data['消费评分'].mean() > 60):
        print(f"  👉 标签：高价值客户")
    elif (cluster_data['年龄'].mean() < 30):
        print(f"  👉 标签：年轻潜力客户")
    elif (cluster_data['逛商场频率'].mean() > 6):
        print(f"  👉 标签：高频客户")
    else:
        print(f"  👉 标签：普通客户")

print("\n" + "=" * 50)
print("🎊 恭喜！你完成了客户分群项目！")
print("=" * 50)