"""
Day06 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day06_examples.py

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
print("Day06 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 50)
print("📊 混淆矩阵详解")
print("=" * 50)

# 加载数据（乳腺癌数据集）
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target  # 0=恶性，1=良性

print(f"\n数据集信息：")
print(f"样本数：{len(X)}")
print(f"恶性（0）: {sum(y==0)} 例")
print(f"良性（1）: {sum(y==1)} 例")

# 划分数据集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 训练模型
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)

print("\n混淆矩阵：")
print(cm)

# 可视化
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['预测恶性', '预测良性'],
            yticklabels=['实际恶性', '实际良性'])
plt.xlabel('预测值')
plt.ylabel('真实值')
plt.title('混淆矩阵')
plt.show()

# 解释四个数字
TN = cm[0, 0]  # 真阴性
FP = cm[0, 1]  # 假阳性
FN = cm[1, 0]  # 假阴性
TP = cm[1, 1]  # 真阳性

print("\n详细解读：")
print(f"真阴性（TN）: {TN} - 恶性肿瘤 correctly 识别为恶性 ✅")
print(f"假阳性（FP）: {FP} - 恶性肿瘤 incorrectly 识别为良性 ❌（漏诊！危险！）")
print(f"假阴性（FN）: {FN} - 良性肿瘤 incorrectly 识别为恶性 ❌（虚惊一场）")
print(f"真阳性（TP）: {TP} - 良性肿瘤 correctly 识别为良性 ✅")

# ===== 代码块 2 =====

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("=" * 50)
print("📈 四大核心评估指标")
print("=" * 50)

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n1️⃣ 准确率（Accuracy）: {accuracy*100:.2f}%")
print(f"   含义：所有预测中，猜对的比例")

print(f"\n2️⃣ 精确率（Precision）: {precision*100:.2f}%")
print(f"   含义：预测为阳性的中，真正阳性的比例")
print(f"   关注：宁缺毋滥")

print(f"\n3️⃣ 召回率（Recall）: {recall*100:.2f}%")
print(f"   含义：所有阳性中，被正确找出的比例")
print(f"   关注：宁可错杀，不可放过")

print(f"\n4️⃣ F1 分数（F1-Score）: {f1*100:.2f}%")
print(f"   含义：精确率和召回率的平衡")

# 完整报告
print("\n" + "=" * 50)
print("📋 完整分类报告")
print("=" * 50)
print(classification_report(y_test, y_pred, 
                           target_names=['恶性', '良性']))

# ===== 代码块 3 =====

from sklearn.metrics import roc_curve, auc, roc_auc_score

print("=" * 50)
print("🎯 ROC 曲线和 AUC 值")
print("=" * 50)

# 获取预测概率
y_prob = model.predict_proba(X_test)[:, 1]

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# 计算 AUC
roc_auc = auc(fpr, tpr)

print(f"\nAUC 值：{roc_auc:.4f}")
print(f"AUC 含义：")
print(f"  0.5 = 随机猜测")
print(f"  0.7-0.8 = 还可以")
print(f"  0.8-0.9 = 很好")
print(f"  0.9+ = 优秀")

# 画图
plt.figure(figsize=(10, 8))

# ROC 曲线
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC 曲线 (AUC = {roc_auc:.2f})')

# 对角线（随机猜测）
plt.plot([0, 1], [0, 1], color='navy', lw=2, 
         linestyle='--', label='随机猜测')

# 填充面积
plt.fill_between(fpr, tpr, alpha=0.3, color='darkorange')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率（FPR）', fontsize=12)
plt.ylabel('真阳性率（TPR）', fontsize=12)
plt.title('受试者工作特征曲线（ROC Curve）', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# ===== 代码块 4 =====

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import numpy as np

print("=" * 50)
print("🎨 过拟合 vs 欠拟合 vs 恰到好处")
print("=" * 50)

# 生成数据
np.random.seed(42)
X = np.linspace(0, 10, 20).reshape(-1, 1)
y_true = 2 * X.squeeze() ** 2 - 3 * X.squeeze() + 1
y = y_true + np.random.normal(0, 10, len(X))  # 加噪声

# 三种模型
models = {
    '欠拟合': Pipeline([
        ('poly', PolynomialFeatures(degree=1)),
        ('linear', LinearRegression())
    ]),
    '恰到好处': Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ]),
    '过拟合': Pipeline([
        ('poly', PolynomialFeatures(degree=10)),
        ('linear', LinearRegression())
    ])
}

# 画图对比
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X, y)
    
    X_smooth = np.linspace(0, 10, 100).reshape(-1, 1)
    y_pred = model.predict(X_smooth)
    
    ax.scatter(X, y, c='blue', s=50, label='训练数据', alpha=0.6)
    ax.plot(X_smooth, y_pred, 'r-', linewidth=2, label='模型预测')
    ax.plot(X_smooth, y_true, 'g--', linewidth=2, label='真实规律', alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(name, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n三种情况对比：")
print("""
❌ 欠拟合:
- 模型太简单
- 训练集表现差
- 测试集表现也差
- 解决办法：增加复杂度、增加特征

✅ 恰到好处:
- 模型复杂度适中
- 训练集表现好
- 测试集表现也好
- 这是我们的目标！

❌ 过拟合:
- 模型太复杂
- 训练集表现太好了（死记硬背）
- 测试集表现差（不会举一反三）
- 解决办法：简化模型、正则化、增加数据
""")

# ===== 代码块 5 =====

from sklearn.model_selection import cross_val_score

print("=" * 50)
print("🔄 交叉验证 - 更可靠的评估")
print("=" * 50)

from sklearn.datasets import load_iris
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# 方法 1：简单划分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42
)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)
simple_score = model.score(X_test, y_test)

print(f"\n简单划分法：准确率 = {simple_score*100:.2f}%")

# 方法 2:5 折交叉验证
cv_scores = cross_val_score(
    model, X_iris, y_iris, 
    cv=5,  # 5 折
    scoring='accuracy'
)

print(f"\n5 折交叉验证:")
print(f"  第 1 折：{cv_scores[0]*100:.2f}%")
print(f"  第 2 折：{cv_scores[1]*100:.2f}%")
print(f"  第 3 折：{cv_scores[2]*100:.2f}%")
print(f"  第 4 折：{cv_scores[3]*100:.2f}%")
print(f"  第 5 折：{cv_scores[4]*100:.2f}%")
print(f"  平均分：{cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")