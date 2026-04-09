"""
Day07 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day07_examples.py

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
print("Day07 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_auc_score, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("💳 信用卡欺诈检测系统")
print("=" * 50)

# 1. 加载数据（使用模拟数据）
print("\n正在加载数据...")

np.random.seed(42)
n_transactions = 10000

# 创建模拟数据
data = pd.DataFrame({
    'Amount': np.random.exponential(50, n_transactions).clip(0, 1000),
    'Time': np.random.randint(0, 24*3600, n_transactions),
})

# 添加一些特征工程后的列（PCA 变换）
for i in range(1, 10):
    data[f'V{i}'] = np.random.randn(n_transactions)

# 创建目标变量（欺诈标记）
fraud_prob = (
    (data['Amount'] > 200).astype(float) * 0.3 +
    ((data['Time'] % 86400 < 6*3600).astype(float)) * 0.2 +
    np.random.random(n_transactions) * 0.3
)
data['Class'] = (fraud_prob > 0.6).astype(int)

print(f"✓ 数据加载完成！")
print(f"\n数据集信息：")
print(f"  总交易数：{len(data)}")
print(f"  正常交易：{sum(data['Class']==0)} ({sum(data['Class']==0)/len(data)*100:.2f}%)")
print(f"  欺诈交易：{sum(data['Class']==1)} ({sum(data['Class']==1)/len(data)*100:.2f}%)")
print(f"\n这是一个典型的类别不平衡问题！")

# 2. 探索性数据分析
print("\n" + "=" * 50)
print("📊 探索性数据分析")
print("=" * 50)

# 查看基本统计
print("\n数据基本统计：")
print(data.describe())

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. 类别分布
axes[0, 0].pie([sum(data['Class']==0), sum(data['Class']==1)], 
               labels=['正常', '欺诈'], autopct='%1.1f%%',
               colors=['skyblue', 'salmon'])
axes[0, 0].set_title('交易类别分布')

# 2. 金额分布
data[data['Class']==0]['Amount'].hist(bins=50, ax=axes[0, 1], 
                                     alpha=0.5, label='正常', color='skyblue')
data[data['Class']==1]['Amount'].hist(bins=50, ax=axes[0, 1], 
                                     alpha=0.5, label='欺诈', color='salmon')
axes[0, 1].legend()
axes[0, 1].set_title('交易金额分布')
axes[0, 1].set_xlabel('金额')

# 3. 时间分布
hours = data['Time'] // 3600
axes[0, 2].hist(hours[data['Class']==0], bins=24, alpha=0.5, 
               label='正常', color='skyblue')
axes[0, 2].hist(hours[data['Class']==1], bins=24, alpha=0.5, 
               label='欺诈', color='salmon')
axes[0, 2].legend()
axes[0, 2].set_title('交易时间分布（小时）')
axes[0, 2].set_xlabel('小时')

# 4. V1 特征分布
axes[1, 0].hist(data[data['Class']==0]['V1'], bins=50, alpha=0.5, 
               label='正常', color='skyblue', density=True)
axes[1, 0].hist(data[data['Class']==1]['V1'], bins=50, alpha=0.5, 
               label='欺诈', color='salmon', density=True)
axes[1, 0].legend()
axes[1, 0].set_title('特征 V1 分布')

# 5. 相关性热图
corr_cols = ['Amount', 'V1', 'V2', 'V3', 'Class']
corr_matrix = data[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('特征相关性')

# 6. 箱线图检查异常值
data.boxplot(column='Amount', by='Class', ax=axes[1, 2])
axes[1, 2].set_title('金额箱线图')
axes[1, 2].set_xlabel('类别')
plt.suptitle('')

plt.tight_layout()
plt.show()

# 3. 数据预处理
print("\n" + "=" * 50)
print("🔧 数据预处理")
print("=" * 50)

# 分离特征和目标
X = data.drop('Class', axis=1)
y = data['Class']

# 标准化（很重要！）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✓ 数据标准化完成")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集：{len(X_train)} 个样本")
print(f"测试集：{len(X_test)} 个样本")

# 4. 训练多个模型
print("\n" + "=" * 50)
print("🤖 训练多个模型")
print("=" * 50)

models = {
    '逻辑回归': LogisticRegression(max_iter=1000, class_weight='balanced'),
    '随机森林': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}

results = []

for name, model in models.items():
    print(f"\n正在训练 {name}...")
    
    # 训练
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 评估
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        '模型': name,
        '准确率': acc,
        '精确率': prec,
        '召回率': rec,
        'F1 分数': f1,
        'AUC': auc
    })
    
    print(f"  准确率：{acc*100:.2f}%")
    print(f"  精确率：{prec*100:.2f}%")
    print(f"  召回率：{rec*100:.2f}%")
    print(f"  F1 分数：{f1*100:.2f}%")
    print(f"  AUC: {auc:.4f}")

# 对比结果
results_df = pd.DataFrame(results)
print("\n" + "=" * 50)
print("📊 模型性能对比")
print("=" * 50)
print(results_df.to_string(index=False))

# 5. 最佳模型深入分析
print("\n" + "=" * 50)
print("🔍 最佳模型深入分析")
print("=" * 50)

# 选 F1 最高的模型
best_model_name = results_df.loc[results_df['F1 分数'].idxmax(), '模型']
best_model = models[best_model_name]
print(f"最佳模型：{best_model_name}")

# 重新预测
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_best)

print("\n混淆矩阵：")
print(cm)

# 详细报告
print("\n详细分类报告：")
print(classification_report(y_test, y_pred_best, 
                           target_names=['正常', '欺诈']))

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 混淆矩阵热图
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['预测正常', '预测欺诈'],
            yticklabels=['实际正常', '实际欺诈'],
            ax=axes[0, 0])
axes[0, 0].set_title('混淆矩阵')
axes[0, 0].set_ylabel('真实值')
axes[0, 0].set_xlabel('预测值')

# 2. ROC 曲线
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_prob_best)
roc_auc = auc(fpr, tpr)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC 曲线 (AUC = {roc_auc:.2f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('假阳性率')
axes[0, 1].set_ylabel('真阳性率')
axes[0, 1].set_title('ROC 曲线')
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(True, alpha=0.3)

# 3. 精确率 - 召回率曲线
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob_best)
axes[1, 0].plot(recall_curve, precision_curve, color='blue', lw=2)
axes[1, 0].set_xlabel('召回率')
axes[1, 0].set_ylabel('精确率')
axes[1, 0].set_title('精确率 - 召回率曲线')
axes[1, 0].grid(True, alpha=0.3)

# 4. 特征重要性（如果是树模型）
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_names = X.columns[:len(importances)]
    
    indices = np.argsort(importances)[::-1][:10]
    
    axes[1, 1].bar(range(len(indices)), importances[indices], align='center')
    axes[1, 1].set_xticks(range(len(indices)))
    axes[1, 1].set_xticklabels(feature_names[indices], rotation=45)
    axes[1, 1].set_title('Top 10 重要特征')
    axes[1, 1].grid(True, alpha=0.3)
else:
    axes[1, 1].text(0.5, 0.5, '该模型不支持\n特征重要性分析', 
                   ha='center', va='center', fontsize=14)
    axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# 6. 业务建议
print("\n" + "=" * 50)
print("💼 业务建议和部署方案")
print("=" * 50)

print(f"""
基于以上分析，我们建议：

1. 模型选择：
   ✓ 使用 {best_model_name} 作为主要模型
   ✓ 召回率达到 {recall_score(y_test, y_pred_best, zero_division=0)*100:.1f}%
   ✓ 意味着能抓住 {recall_score(y_test, y_pred_best, zero_division=0)*100:.1f}% 的欺诈交易

2. 阈值调整：
   ✓ 当前默认阈值为 0.5
   ✓ 如果想提高召回率：降低阈值（如 0.3）
   ✓ 如果想提高精确率：提高阈值（如 0.7）

3. 实际应用流程：
   ✓ 实时监测每笔交易
   ✓ 模型评分超过阈值 → 触发警报
   ✓ 人工审核高风险交易
   ✓ 确认为欺诈 → 冻结账户

4. 持续优化：
   ✓ 定期用新数据重新训练
   ✓ 监控模型性能下降
   ✓ 收集新的欺诈模式
   ✓ 更新特征工程

5. 成本控制：
   ✓ 每次误报的成本：客服时间 + 用户体验
   ✓ 每次漏报的成本：金钱损失 + 声誉损害
   ✓ 找到最佳平衡点
""")

print("\n" + "=" * 50)
print("🎊 恭喜！你完成了第一个完整的 AI 项目！")
print("=" * 50)

# ===== 代码块 2 =====

print("=" * 50)
print("🏆 Week 1 学习成果总结")
print("=" * 50)

print("""
【技能清单】

✅ 编程能力：
   - Python 基础语法
   - NumPy 数组操作
   - Pandas 数据处理
   - Matplotlib/Seaborn 可视化

✅ 机器学习算法：
   - K 近邻（KNN）
   - 线性回归
   - 逻辑回归
   - 决策树
   - 随机森林
   - SVM
   - K-means 聚类

✅ 模型评估：
   - 混淆矩阵
   - 准确率、精确率、召回率、F1
   - ROC 曲线、AUC
   - 交叉验证
   - 学习曲线

✅ 实战经验：
   - 鸢尾花分类
   - 手写数字识别
   - 客户分群
   - 信用卡欺诈检测

【项目作品集】
1. 鸢尾花分类器
2. 手写数字识别系统
3. 商场客户分群分析
4. 信用卡欺诈检测系统

你已经是个初级数据科学家了！👏
""")