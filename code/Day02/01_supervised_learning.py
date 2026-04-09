"""
Day 2: 监督学习算法
KNN、K-Means、数据预处理
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

print("=" * 60)
print("Day 2: 监督学习算法")
print("=" * 60)

# ============================================================================
# 示例 1: 加载和探索数据
# ============================================================================
print("\n【1. 加载鸢尾花数据集】")

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"数据形状: {X.shape}")
print(f"标签形状: {y.shape}")
print(f"特征名称: {feature_names}")
print(f"类别名称: {list(target_names)}")
print(f"\n前5个样本:")
for i in range(5):
    print(f"  样本 {i+1}: {X[i]}, 标签: {target_names[y[i]]}")

# ============================================================================
# 示例 2: 数据预处理
# ============================================================================
print("\n【2. 数据预处理】")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n标准化前 - 均值: {X_train[:, 0].mean():.2f}, 标准差: {X_train[:, 0].std():.2f}")
print(f"标准化后 - 均值: {X_train_scaled[:, 0].mean():.2f}, 标准差: {X_train_scaled[:, 0].std():.2f}")

# ============================================================================
# 示例 3: KNN 分类器
# ============================================================================
print("\n【3. KNN 分类器】")

# 尝试不同的 K 值
k_values = [1, 3, 5, 7, 9, 11]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"K={k:2d}: 准确率 = {acc:.4f}")

# 选择最佳 K
best_k = k_values[np.argmax(accuracies)]
print(f"\n最佳 K 值: {best_k}")

# 使用最佳 K 训练最终模型
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
y_pred_final = best_knn.predict(X_test_scaled)

print(f"\n详细分类报告:")
print(classification_report(y_test, y_pred_final, target_names=target_names))

# 可视化 K 值 vs 准确率
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8)
plt.xlabel('K Value', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('KNN: K Value vs Accuracy', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.tight_layout()
plt.savefig('knn_k_vs_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ K值分析图已保存")

# ============================================================================
# 示例 4: K-Means 聚类（无监督）
# ============================================================================
print("\n【4. K-Means 聚类】")

# 使用肘部法则找到最佳 K
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled if 'X_scaled' in dir() else X)
    inertias.append(kmeans.inertia_)

# 绘制肘部曲线
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.title('Elbow Method for Optimal K', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(K_range)
plt.tight_layout()
plt.savefig('kmeans_elbow.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ 肘部法则图已保存")

# 使用 K=3 进行聚类（因为鸢尾花有3个类别）
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X)

print(f"\n聚类中心:\n{kmeans_final.cluster_centers_}")
print(f"每个簇的样本数: {np.bincount(clusters)}")

# 可视化聚类结果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6, edgecolors='k')
centers = kmeans_final.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, edgecolors='black', linewidths=2, label='Centroids')
plt.xlabel(feature_names[0], fontsize=12)
plt.ylabel(feature_names[1], fontsize=12)
plt.title('K-Means Clustering Result', fontsize=14)
plt.legend()
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_clustering.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ 聚类结果图已保存")

# ============================================================================
# 示例 5: 预测新样本
# ============================================================================
print("\n【5. 预测新样本】")

# 创建一个新样本
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
new_sample_scaled = scaler.transform(new_sample)

prediction = best_knn.predict(new_sample_scaled)
probabilities = best_knn.predict_proba(new_sample_scaled)

print(f"新样本: {new_sample[0]}")
print(f"预测类别: {target_names[prediction[0]]}")
print(f"各类别概率:")
for name, prob in zip(target_names, probabilities[0]):
    print(f"  {name}: {prob:.4f}")

print("\n" + "=" * 60)
print("✅ Day 2 监督学习完成！")
print("=" * 60)
