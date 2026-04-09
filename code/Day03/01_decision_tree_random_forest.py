"""
Day 3: 决策树和随机森林
信息增益、基尼系数、集成学习
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("Day 3: 决策树和随机森林")
print("=" * 60)

# ============================================================================
# 示例 1: 加载数据
# ============================================================================
print("\n【1. 加载鸢尾花数据集】")

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# ============================================================================
# 示例 2: 决策树分类器
# ============================================================================
print("\n【2. 决策树分类器】")

# 创建决策树（使用信息熵）
dt_entropy = DecisionTreeClassifier(
    criterion='entropy',  # 信息熵
    max_depth=3,          # 最大深度
    random_state=42
)
dt_entropy.fit(X_train, y_train)

# 预测
y_pred_entropy = dt_entropy.predict(X_test)
acc_entropy = accuracy_score(y_test, y_pred_entropy)
print(f"决策树 (Entropy) 准确率: {acc_entropy:.4f}")

# 创建决策树（使用基尼系数）
dt_gini = DecisionTreeClassifier(
    criterion='gini',     # 基尼系数
    max_depth=3,
    random_state=42
)
dt_gini.fit(X_train, y_train)

y_pred_gini = dt_gini.predict(X_test)
acc_gini = accuracy_score(y_test, y_pred_gini)
print(f"决策树 (Gini) 准确率: {acc_gini:.4f}")

# 打印决策规则
print("\n决策规则:")
tree_rules = export_text(dt_entropy, feature_names=list(feature_names))
print(tree_rules[:500])  # 只显示前500字符

# ============================================================================
# 示例 3: 可视化决策树
# ============================================================================
print("\n【3. 可视化决策树】")

plt.figure(figsize=(20, 10))
plot_tree(
    dt_entropy,
    feature_names=feature_names,
    class_names=target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree (max_depth=3)', fontsize=16)
plt.tight_layout()
plt.savefig('decision_tree_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ 决策树可视化已保存")

# ============================================================================
# 示例 4: 不同深度的影响
# ============================================================================
print("\n【4. 不同深度的影响（过拟合演示）】")

depths = range(1, 11)
train_accuracies = []
test_accuracies = []

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc = accuracy_score(y_test, dt.predict(X_test))
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    
    print(f"Depth {depth:2d}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(depths, train_accuracies, 'bo-', label='Training Accuracy', linewidth=2)
plt.plot(depths, test_accuracies, 'ro-', label='Test Accuracy', linewidth=2)
plt.xlabel('Max Depth', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Decision Tree: Depth vs Accuracy', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.xticks(depths)
plt.tight_layout()
plt.savefig('decision_tree_depth_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ 深度分析图已保存")

# ============================================================================
# 示例 5: 特征重要性
# ============================================================================
print("\n【5. 特征重要性】")

dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

importances = dt_full.feature_importances_
indices = np.argsort(importances)[::-1]

print("特征重要性排序:")
for i, idx in enumerate(indices):
    print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.title('Feature Importances (Decision Tree)', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance_dt.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ 特征重要性图已保存")

# ============================================================================
# 示例 6: 随机森林
# ============================================================================
print("\n【6. 随机森林分类器】")

# 尝试不同的树数量
n_estimators_list = [1, 5, 10, 20, 50, 100, 200]
rf_accuracies = []

for n_est in n_estimators_list:
    rf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=5,
        random_state=42,
        n_jobs=-1  # 使用所有CPU核心
    )
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    rf_accuracies.append(acc)
    print(f"Trees={n_est:3d}: Accuracy = {acc:.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_list, rf_accuracies, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Trees', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Random Forest: Number of Trees vs Accuracy', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.tight_layout()
plt.savefig('random_forest_trees_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ 随机森林分析图已保存")

# ============================================================================
# 示例 7: 随机森林特征重要性
# ============================================================================
print("\n【7. 随机森林特征重要性】")

best_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
best_rf.fit(X_train, y_train)

rf_importances = best_rf.feature_importances_
rf_indices = np.argsort(rf_importances)[::-1]

print("随机森林特征重要性排序:")
for i, idx in enumerate(rf_indices):
    print(f"  {i+1}. {feature_names[idx]}: {rf_importances[idx]:.4f}")

# 对比决策树和随机森林的特征重要性
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 决策树
ax1.bar(range(len(importances)), importances[indices], align='center', color='steelblue')
ax1.set_xticks(range(len(importances)))
ax1.set_xticklabels([feature_names[i] for i in indices], rotation=45)
ax1.set_title('Decision Tree Feature Importance', fontsize=12)
ax1.set_ylabel('Importance')

# 随机森林
ax2.bar(range(len(rf_importances)), rf_importances[rf_indices], align='center', color='coral')
ax2.set_xticks(range(len(rf_importances)))
ax2.set_xticklabels([feature_names[i] for i in rf_indices], rotation=45)
ax2.set_title('Random Forest Feature Importance', fontsize=12)
ax2.set_ylabel('Importance')

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ 特征重要性对比图已保存")

# ============================================================================
# 示例 8: 模型对比
# ============================================================================
print("\n【8. 模型性能对比】")

print("\n详细分类报告 - 决策树:")
print(classification_report(y_test, y_pred_entropy, target_names=target_names))

print("\n详细分类报告 - 随机森林:")
y_pred_rf = best_rf.predict(X_test)
print(classification_report(y_test, y_pred_rf, target_names=target_names))

# 总结
print("\n" + "=" * 60)
print("📊 性能总结:")
print(f"  决策树 (Entropy): {acc_entropy:.4f}")
print(f"  决策树 (Gini):    {acc_gini:.4f}")
print(f"  随机森林 (100树): {accuracy_score(y_test, y_pred_rf):.4f}")
print("=" * 60)

print("\n✅ Day 3 决策树和随机森林完成！")
print("\n💡 关键要点:")
print("  1. 决策树易于理解和解释")
print("  2. 容易过拟合，需要限制深度")
print("  3. 随机森林通过集成减少过拟合")
print("  4. 特征重要性帮助理解数据")
