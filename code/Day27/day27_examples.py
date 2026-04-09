"""
Day27 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day27_examples.py

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
print("Day27 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

"""
偏见从哪里来？

来源 1：训练数据
历史数据本身就包含偏见
→ 招聘数据：男性程序员多
→ 贷款数据：某些地区通过率低
→ 犯罪数据：某些族裔被捕率高

AI 学到这些模式
→ 认为这是"正常"的
→ 继续强化这些偏见
→ 形成恶性循环

来源 2：特征选择
选择的特征可能间接导致歧视
→ 用邮编判断信用
→ 某些邮编对应少数族裔聚居区
→ 实际上是在变相歧视

来源 3：标签定义
标签本身可能有问题
→ "优秀员工"的定义
→ 可能基于主观评价
→ 评价者本身有偏见
"""

# ===== 代码块 2 =====

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("=" * 60)
print("⚖️ AI 偏见检测和缓解")
print("=" * 60)

# ============================================================================
# 第一部分：创建模拟数据
# ============================================================================
print("\n【1. 创建模拟招聘数据】")

np.random.seed(42)
n_samples = 1000

# 生成数据
data = {
    'gender': np.random.choice(['male', 'female'], n_samples),
    'education': np.random.choice(['bachelor', 'master', 'phd'], n_samples),
    'experience': np.random.randint(0, 20, n_samples),
    'skills_score': np.random.uniform(50, 100, n_samples),
}

# 引入偏见：同等条件下，男性通过率更高
base_rate = 0.5
gender_bias = {'male': 0.2, 'female': -0.2}

hired = []
for i in range(n_samples):
    base_prob = base_rate
    base_prob += (data['skills_score'][i] - 75) / 100
    base_prob += data['experience'][i] / 50
    base_prob += gender_bias[data['gender'][i]]
    
    # 限制在 0-1 之间
    prob = max(0, min(1, base_prob))
    hired.append(1 if np.random.random() < prob else 0)

data['hired'] = hired

df = pd.DataFrame(data)

print(f"✓ 数据集大小：{len(df)}")
print(f"\n数据统计:")
print(f"  总录取率：{df['hired'].mean():.1%}")
print(f"  男性录取率：{df[df['gender']=='male']['hired'].mean():.1%}")
print(f"  女性录取率：{df[df['gender']=='female']['hired'].mean():.1%}")

# ============================================================================
# 第二部分：训练有偏见的模型
# ============================================================================
print("\n" + "=" * 60)
print("【2. 训练模型（包含偏见）】")
print("=" * 60)

# 准备特征
X = pd.get_dummies(df[['gender', 'education', 'experience', 'skills_score']], drop_first=True)
y = df['hired']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
model_biased = LogisticRegression(random_state=42)
model_biased.fit(X_train, y_train)

print("✓ 模型训练完成")
print(f"  训练集准确率：{accuracy_score(y_train, model_biased.predict(X_train)):.1%}")
print(f"  测试集准确率：{accuracy_score(y_test, model_biased.predict(X_test)):.1%}")

# ============================================================================
# 第三部分：检测偏见
# ============================================================================
print("\n" + "=" * 60)
print("【3. 检测模型偏见】")
print("=" * 60)

# 在测试集上预测
y_pred = model_biased.predict(X_test)
y_proba = model_biased.predict_proba(X_test)[:, 1]

# 按性别分组统计
test_df = X_test.copy()
test_df['true_label'] = y_test.values
test_df['predicted'] = y_pred
test_df['probability'] = y_proba
test_df['gender_male'] = X_test['gender_male'].values

male_preds = test_df[test_df['gender_male']==1]
female_preds = test_df[test_df['gender_male']==0]

print("\n分组统计:")
print(f"  男性预测通过率：{male_preds['predicted'].mean():.1%}")
print(f"  女性预测通过率：{female_preds['predicted'].mean():.1%}")
print(f"  差异：{(male_preds['predicted'].mean() - female_preds['predicted'].mean()):.1%}")

# 计算公平性指标
def demographic_parity_ratio(group1_rate, group2_rate):
    """人口统计学 parity 比率"""
    return min(group1_rate, group2_rate) / max(group1_rate, group2_rate)

dpr = demographic_parity_ratio(
    male_preds['predicted'].mean(),
    female_preds['predicted'].mean()
)

print(f"\n公平性指标:")
print(f"  人口统计学 Parity 比率：{dpr:.3f}")
print(f"  解释：越接近 1 越公平，<0.8 通常认为有偏见")

if dpr < 0.8:
    print(f"  ⚠️  警告：检测到明显的性别偏见!")

# ============================================================================
# 第四部分：缓解偏见
# ============================================================================
print("\n" + "=" * 60)
print("【4. 缓解偏见】")
print("=" * 60)

print("""
方法 1：去除敏感特征
→ 去掉 gender 列
→ 但可能有代理特征（如专业、爱好）
→ 效果有限

方法 2：重新加权
→ 给少数群体样本更高权重
→ 让模型更关注他们
→ 平衡不同群体的影响

方法 3：对抗去偏
→ 训练一个判别器检测偏见
→ 主模型要骗过判别器
→ 迫使模型学习无偏表示

方法 4：后处理
→ 调整不同群体的阈值
→ 达到相同的通过率
→ 简单但有效

我们用方法 2：重新加权
""")

# 计算样本权重
sample_weights = np.ones(len(y_train))

# 找出女性在训练集中的索引
female_indices = np.where(X_train['gender_male'].values == 0)[0]
male_indices = np.where(X_train['gender_male'].values == 1)[0]

# 计算比例
n_female = len(female_indices)
n_male = len(male_indices)
total = len(y_train)

# 给少数群体更高权重
if n_female < n_male:
    sample_weights[female_indices] = n_male / n_female
else:
    sample_weights[male_indices] = n_female / n_male

print(f"✓ 样本权重设置完成")
print(f"  男性样本数：{n_male}, 权重：1.0")
print(f"  女性样本数：{n_female}, 权重：{sample_weights[female_indices[0]]:.2f}")

# 重新训练模型
model_debiased = LogisticRegression(random_state=42)
model_debiased.fit(X_train, y_train, sample_weight=sample_weights)

print("\n✓ 去偏模型训练完成")

# ============================================================================
# 第五部分：对比结果
# ============================================================================
print("\n" + "=" * 60)
print("【5. 对比去偏前后】")
print("=" * 60)

# 原模型
y_pred_biased = model_biased.predict(X_test)
male_biased = y_pred_biased[X_test['gender_male'].values == 1].mean()
female_biased = y_pred_biased[X_test['gender_male'].values == 0].mean()

# 去偏模型
y_pred_debiased = model_debiased.predict(X_test)
male_debiased = y_pred_debiased[X_test['gender_male'].values == 1].mean()
female_debiased = y_pred_debiased[X_test['gender_male'].values == 0].mean()

print("\n通过率对比:")
print(f"             男性      女性      差异")
print(f"去偏前：   {male_biased:.1%}     {female_biased:.1%}     {abs(male_biased-female_biased):.1%}")
print(f"去偏后：   {male_debiased:.1%}     {female_debiased:.1%}     {abs(male_debiased-female_debiased):.1%}")

dpr_before = demographic_parity_ratio(male_biased, female_biased)
dpr_after = demographic_parity_ratio(male_debiased, female_debiased)

print(f"\n公平性对比:")
print(f"  去偏前 DPR: {dpr_before:.3f}")
print(f"  去偏后 DPR: {dpr_after:.3f}")
print(f"  改善：{(dpr_after - dpr_before):.3f}")

if dpr_after > dpr_before:
    print(f"  ✅ 去偏有效！公平性提升了")
else:
    print(f"  ⚠️  去偏效果不明显")

# 准确率对比
acc_biased = accuracy_score(y_test, y_pred_biased)
acc_debiased = accuracy_score(y_test, y_pred_debiased)

print(f"\n准确率对比:")
print(f"  去偏前：{acc_biased:.1%}")
print(f"  去偏后：{acc_debiased:.1%}")
print(f"  变化：{(acc_debiased - acc_biased):+.1%}")

print("\n💡 结论:")
print("  - 去偏通常会略微降低准确率")
print("  - 但这是为了公平必须付出的代价")
print("  - 关键是找到准确率和公平性的平衡点")

print("\n🎊 偏见检测和缓解演示完成!")
print("=" * 60)

# ===== 代码块 3 =====

"""
差分隐私（Differential Privacy）是什么？

目标:
从数据中学习规律
但不泄露任何个人的信息

核心思想:
添加适量的随机噪声
→ 单个数据点的变化不影响结果
→ 攻击者无法推断具体个人
→ 但整体统计规律仍然准确

例子:
调查：你有某种疾病吗？

传统方法:
直接回答 → 泄露隐私

差分隐私:
抛硬币决定
正面→如实回答
反面→随机说是/否
→ 没人知道你是真是假
→ 但可以统计整体患病率

应用:
✓ Apple 收集用户行为
✓ Google 输入法改进
✓ Census 人口普查
"""

# ===== 代码块 4 =====

"""
联邦学习（Federated Learning）是什么？

问题:
数据集中到服务器有风险
→ 可能被黑客攻击
→ 可能被滥用
→ 违反 GDPR 等法规

解决:
数据不动模型动!

流程:
1. 服务器发送初始模型到各设备
2. 各设备用本地数据训练
3. 只上传模型更新（梯度）
4. 服务器聚合所有更新
5. 得到全局模型

好处:
✓ 数据永远在本地
✓ 不会泄露隐私
✓ 符合法规要求

应用:
✓ Gboard 输入法
✓ 微信语音识别
✓ 医疗影像分析
"""

# ===== 代码块 5 =====

"""
案例 1：COMPAS 再犯预测系统

背景:
美国法院用 AI 预测罪犯再犯概率
用于量刑和假释决策

问题:
发现对黑人系统性歧视
→ 同样罪行，黑人得分更高
→ 假释通过率更低

原因:
训练数据包含历史判决
→ 历史上就存在种族歧视
→ AI 学会了歧视

教训:
⚠️ 司法领域要格外谨慎
⚠️ 不能用有偏见的数据
⚠️ 需要人工监督和申诉机制
"""

"""
案例 2：Amazon 招聘 AI

背景:
Amazon 开发 AI 筛选简历
目标是自动化招聘流程

问题:
发现对女性求职者不公平
→ 简历中出现"女子学院"就被扣分
→ 出现"国际象棋社"（男性主导）就加分

原因:
用过去 10 年的简历训练
→ IT 行业男性占主导
→ AI 认为男性更合适

结果:
Amazon 放弃这个项目
→ 公开承认偏见问题
→ 引起业界重视

教训:
⚠️ 招聘是高风险应用
⚠️ 历史数据可能包含歧视
⚠️ 需要多元化团队审查
"""

"""
案例 3：Deepfake 换脸

技术:
用 GAN 生成逼真的假视频
可以替换任何人的脸

正面应用:
✓ 电影后期制作
✓ 教育娱乐
✓ 艺术创作

负面应用:
✗ 制造假新闻
✗ 色情内容
✗ 诽谤他人

应对:
→ 开发 Deepfake 检测工具
→ 立法禁止恶意使用
→ 平台加强审核
→ 提高公众媒介素养

思考:
这项技术应该被禁止吗？
→ 完全禁止不可能
→ 技术发展不可阻挡
→ 关键是如何规范和引导
"""

print("=" * 60)
print("🤔 AI 伦理案例讨论")
print("=" * 60)

cases = [
    {
        'name': 'COMPAS 再犯预测',
        'domain': '司法',
        'issue': '种族歧视',
        'lesson': '司法领域要格外谨慎'
    },
    {
        'name': 'Amazon 招聘 AI',
        'domain': '人力资源',
        'issue': '性别歧视',
        'lesson': '招聘是高风险应用'
    },
    {
        'name': 'Deepfake 换脸',
        'domain': '媒体',
        'issue': '虚假信息',
        'lesson': '需要规范和引导'
    }
]

for i, case in enumerate(cases, 1):
    print(f"\n【案例 {i}】{case['name']}")
    print(f"  领域：{case['domain']}")
    print(f"  问题：{case['issue']}")
    print(f"  教训：{case['lesson']}")

print("\n" + "=" * 60)
print("伦理原则总结")
print("=" * 60)

principles = """
1. 公平性（Fairness）
   ✓ 不歧视任何群体
   ✓ 平等对待所有人
   ✓ 定期检测偏见

2. 透明度（Transparency）
   ✓ 公开 AI 如何工作
   ✓ 解释决策依据
   ✓ 接受社会监督

3. 隐私保护（Privacy）
   ✓ 最小化数据收集
   ✓ 加密存储数据
   ✓ 用户知情同意

4. 安全性（Safety）
   ✓ 防止被攻击
   ✓ 防止被滥用
   ✓ 有应急预案

5. 可问责性（Accountability）
   ✓ 明确责任主体
   ✓ 建立追责机制
   ✓ 提供申诉渠道

6. 可控性（Controllability）
   ✓ 人类可以随时接管
   ✓ 有关闭开关
   ✓ 不会自主进化

7. 社会责任（Social Responsibility）
   ✓ 为社会创造价值
   ✓ 考虑长远影响
   ✓ 促进人类福祉
"""

print(principles)

print("\n💡 作为 AI 开发者，我们应该:")
print("  1. 时刻牢记伦理原则")
print("  2. 主动检测潜在问题")
print("  3. 接受社会监督")
print("  4. 为后果承担责任")
print("  5. 让 AI 真正造福人类")

print("\n🎊 AI 伦理讨论完成!")
print("=" * 60)