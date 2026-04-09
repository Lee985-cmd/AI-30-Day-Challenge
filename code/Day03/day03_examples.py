"""
Day03 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day03_examples.py

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
print("Day03 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

print("=" * 50)
print("🌳 我的第一个决策树模型！")
print("=" * 50)

# 1. 加载数据（鸢尾花数据集）
iris = load_iris()
X = iris.data      # 特征：花萼长、宽，花瓣长、宽
y = iris.target    # 标签：花的品种（0,1,2）

print("\n数据集信息：")
print(f"样本数：{len(X)} 朵花")
print(f"特征：{iris.feature_names}")
print(f"类别：{iris.target_names}")

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\n训练集：{len(X_train)} 朵花")
print(f"测试集：{len(X_test)} 朵花")

# 3. 创建决策树模型
clf = DecisionTreeClassifier(
    max_depth=3,        # 最大深度（防止过拟合）
    random_state=42     # 随机种子（保证结果可重复）
)

# 4. 训练模型
print("\n正在训练决策树...")
clf.fit(X_train, y_train)
print("✅ 训练完成！")

# 5. 评估模型
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)

print(f"\n📊 模型性能：")
print(f"训练集准确率：{train_score*100:.2f}%")
print(f"测试集准确率：{test_score*100:.2f}%")

# 6. 实际预测
print("\n" + "=" * 50)
print("🔮 实际应用：预测一朵新的花")
print("=" * 50)

new_flower = [[5.0, 3.5, 1.5, 0.3]]
prediction = clf.predict(new_flower)
species_name = iris.target_names[prediction[0]]

print(f"\n新花的特征：{new_flower[0]}")
print(f"预测结果：这是 {species_name} 鸢尾花")

print("\n" + "=" * 50)
print("🎊 恭喜！你学会了决策树！")
print("=" * 50)

# ===== 代码块 2 =====

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
# clf = 分类器（classifier 的缩写）
# DecisionTreeClassifier() = 创建决策树分类器
# max_depth=3 = 树的最大深度为 3 层
#   - 太深容易死记硬背（过拟合）
#   - 太浅学不到东西（欠拟合）
#   - 3 是个适中的值
# random_state=42 = 随机种子
#   - 保证每次运行结果一样

# ===== 代码块 3 =====

clf.fit(X_train, y_train)
# fit = 拟合、训练
# 把训练数据给模型，让它学习

# ===== 代码块 4 =====

prediction = clf.predict(new_flower)
# predict = 预测
# 用训练好的模型预测新数据

# ===== 代码块 5 =====

from sklearn.ensemble import RandomForestClassifier

print("=" * 50)
print("🌲🌲🌲 随机森林的力量！")
print("=" * 50)

# 使用之前的数据
# X_train, X_test, y_train, y_test 已经有了

# 1. 创建随机森林模型
rf = RandomForestClassifier(
    n_estimators=100,   # 树的数量（100 棵）
    max_depth=10,       # 每棵树的最大深度
    random_state=42
)

# 2. 训练模型
print("\n正在训练随机森林（100 棵树）...")
rf.fit(X_train, y_train)
print("✅ 训练完成！")

# 3. 评估
rf_train_score = rf.score(X_train, y_train)
rf_test_score = rf.score(X_test, y_test)

print(f"\n📊 模型性能：")
print(f"训练集准确率：{rf_train_score*100:.2f}%")
print(f"测试集准确率：{rf_test_score*100:.2f}%")

# 4. 对比单棵决策树
print("\n" + "=" * 50)
print("📊 决策树 vs 随机森林")
print("=" * 50)

# 重新训练一棵决策树（用同样的数据）
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)

print(f"单棵决策树 - 测试集准确率：{dt.score(X_test, y_test)*100:.2f}%")
print(f"随机森林     - 测试集准确率：{rf_test_score*100:.2f}%")

improvement = (rf_test_score - dt.score(X_test, y_test)) * 100
print(f"\n提升：{improvement:.2f}%")
print("看到了吗？随机森林通常比单棵树更好！")

# ===== 代码块 6 =====

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("=" * 50)
print("🚢 泰坦尼克号生存预测实战")
print("=" * 50)

# 1. 创建模拟数据（因为真实数据要下载）
print("\n正在创建模拟数据...")

np.random.seed(42)
n_passengers = 500

# 创建乘客信息
data = pd.DataFrame({
    'Pclass': np.random.choice([1, 2, 3], n_passengers),  # 船票等级
    'Sex': np.random.choice(['male', 'female'], n_passengers),  # 性别
    'Age': np.random.normal(30, 15, n_passengers).clip(0, 80),  # 年龄
    'SibSp': np.random.randint(0, 6, n_passengers),  # 兄弟姐妹/配偶数
    'Parch': np.random.randint(0, 6, n_passengers),  # 父母/子女数
    'Fare': np.random.exponential(30, n_passengers).clip(0, 500),  # 票价
})

# 创建目标变量（是否存活）
# 模拟历史事实：女性、一等舱、年轻人更容易存活
survival_prob = (
    0.3 * (data['Sex'] == 'female') +  # 女性加分
    0.2 * (data['Pclass'] == 1) +      # 一等舱加分
    0.1 * (data['Age'] < 18) +         # 未成年人加分
    np.random.random(n_passengers) * 0.3  # 随机因素
)
data['Survived'] = (survival_prob > 0.5).astype(int)

print(f"✓ 数据创建完成！")
print(f"乘客总数：{len(data)} 人")
print(f"存活人数：{sum(data['Survived']==1)} 人")
print(f"遇难人数：{sum(data['Survived']==0)} 人")

# 2. 数据预处理
print("\n正在处理数据...")

# 把文字转成数字（机器只懂数字）
le_sex = LabelEncoder()
data['Sex'] = le_sex.fit_transform(data['Sex'])
# male=1, female=0

# 准备特征和标签
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = data[features]
y = data['Survived']

# 填补缺失值（如果有）
X = X.fillna(X.median())

print("✓ 数据预处理完成！")
print(f"特征：{features}")

# 3. 划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n训练集：{len(X_train)} 人")
print(f"测试集：{len(X_test)} 人")

# 4. 训练随机森林模型
print("\n正在训练随机森林模型...")
rf_titanic = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,  # 至少 10 个样本才继续分
    random_state=42
)
rf_titanic.fit(X_train, y_train)
print("✅ 模型训练完成！")

# 5. 评估
train_acc = rf_titanic.score(X_train, y_train)
test_acc = rf_titanic.score(X_test, y_test)

print(f"\n📊 模型性能：")
print(f"训练集准确率：{train_acc*100:.2f}%")
print(f"测试集准确率：{test_acc*100:.2f}%")

# 6. 实际预测几个乘客
print("\n" + "=" * 50)
print("🔮 预测具体乘客的生存概率")
print("=" * 50)

# 创建几个虚拟乘客
passengers = pd.DataFrame({
    'Pclass': [1, 3, 1, 3],
    'Sex': le_sex.transform(['female', 'male', 'male', 'female']),
    'Age': [25, 25, 40, 5],
    'SibSp': [1, 0, 0, 2],
    'Parch': [0, 0, 0, 2],
    'Fare': [100, 10, 200, 50]
}, columns=features)

passenger_names = [
    "乘客 A（一等舱女乘客，25 岁）",
    "乘客 B（三等舱男乘客，25 岁）",
    "乘客 C（一等舱男乘客，40 岁）",
    "乘客 D（三等舱女儿童，5 岁）"
]

predictions = rf_titanic.predict(passengers)
probabilities = rf_titanic.predict_proba(passengers)

for i, name in enumerate(passenger_names):
    pred = predictions[i]
    prob_survive = probabilities[i][1] * 100
    
    status = "✅ 存活" if pred == 1 else "❌ 遇难"
    
    print(f"\n{name}:")
    print(f"  预测结果：{status}")
    print(f"  存活概率：{prob_survive:.1f}%")

# 7. 特征重要性分析
print("\n" + "=" * 50)
print("📊 影响生存的关键因素")
print("=" * 50)

feat_names = X.columns
importances = rf_titanic.feature_importances_

# 排序
indices = np.argsort(importances)[::-1]

print("\n按重要性排序：")
for i in range(len(feat_names)):
    feat_idx = indices[i]
    bar = "█" * int(importances[feat_idx] * 20)
    print(f"{feat_names[feat_idx]:12} {bar} {importances[feat_idx]:.4f}")

print("\n结论：")
if importances[indices[0]] > 0.3:
    print(f"✓ 最重要的因素是：{feat_names[indices[0]]}")

print("\n" + "=" * 50)
print("🎊 恭喜！你完成了泰坦尼克号预测项目！")
print("=" * 50)