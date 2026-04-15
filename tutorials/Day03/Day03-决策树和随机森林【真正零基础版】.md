# 🎓 AI 入门 30 天挑战 - Day 3 真正零基础版

---

## 🌟 完整项目和代码

本教程是 **AI 入门 30 天挑战** 系列的一部分！

- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **欢迎 Star 支持！**

---



> **今天学习决策树和随机森林！**  
> **像玩游戏一样做决策！**  
> **每个概念都用生活例子解释！**

---

## 📖 先复习一下前两天的内容

### Day 1 回顾
- 变量 = 装数据的盒子
- if 判断 = 做选择
- for 循环 = 重复做事

### Day 2 回顾
- 机器学习 = 从数据中学习规律
- K 近邻 = 近朱者赤，近墨者黑
- 完整流程：数据 → 训练 → 预测

如果这些都记得，我们开始今天的内容！

---

## 🌳 什么是决策树？

### 故事时间 📚

想象你在玩一个**猜人游戏**：

```
你的朋友心里想了一个人，你要猜是谁

你问的问题：
1. 是男的吗？
   ├─ 是 → 继续问 2
   └─ 否 → 继续问 3
   
2. 戴眼镜吗？（针对男性）
   ├─ 是 → 可能是小明
   └─ 否 → 可能是小刚
   
3. 长头发吗？（针对女性）
   ├─ 是 → 可能是小红
   └─ 否 → 可能是小丽
```

这个过程就是一棵**决策树**！

### 决策树的结构

```
           [是男的吗？]     ← 根节点（第一个问题）
           /        \
      是 /          \ 否
       /            \
   [戴眼镜吗？]    [长头发吗？]  ← 内部节点（中间问题）
    /    \          /    \
 是/      \否     是/      \否
  /        \       /        \
[小明]   [小刚]  [小红]    [小丽]  ← 叶节点（最终答案）
```

**名词解释：**
- **根节点**：第一个问题（树的根部）
- **内部节点**：中间的问题
- **叶节点**：最终的答案（不再有分支）
- **分支**：从一个节点到另一个节点的连线

---

## 💻 决策树代码实现

### 第 1 步：准备环境

**在命令行输入：**

```bash
pip install scikit-learn matplotlib seaborn
```

**等安装完成！**

---

### 第 2 步：第一个决策树模型

**打开 Jupyter Notebook，新建笔记本，输入：**

```python
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
```

**按 Shift + Enter 运行！**

---

### 逐行解释

**创建模型部分：**

```python
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
# clf = 分类器（classifier 的缩写）
# DecisionTreeClassifier() = 创建决策树分类器
# max_depth=3 = 树的最大深度为 3 层
#   - 太深容易死记硬背（过拟合）
#   - 太浅学不到东西（欠拟合）
#   - 3 是个适中的值
# random_state=42 = 随机种子
#   - 保证每次运行结果一样
```

**训练模型：**

```python
clf.fit(X_train, y_train)
# fit = 拟合、训练
# 把训练数据给模型，让它学习
```

**预测：**

```python
prediction = clf.predict(new_flower)
# predict = 预测
# 用训练好的模型预测新数据
```

---

## 🌲🌲🌲 什么是随机森林？

### 思想：三个臭皮匠，顶个诸葛亮

**生活中的例子：**

```
考试答题：

一个人答 → 可能答错 ❌

100 个人一起答：
- 每个人投票选一个答案
- 统计票数，选最多的
- 正确率更高 ✅

这就是随机森林的思想！
```

### 随机森林的工作原理

```
森林 = 很多棵树

训练过程：
1. 从数据中随机抽取一部分样本
2. 从特征中随机选择一部分
3. 用这些样本和特征训练一棵树
4. 重复 1-3 步，训练 100 棵树
5. 这 100 棵树组成"森林"

预测过程：
1. 每棵树都给出自己的答案
2. 统计所有树的答案
3. 选票数最多的那个
```

**为什么更准确？**

```
一棵树 → 可能看走眼
多棵树 → 集体智慧，更可靠

就像：
- 一个人判断 → 可能主观
- 100 个人投票 → 更客观
```

---

### 随机森林代码实现

```python
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
```

---

## 🚢 实战项目：泰坦尼克号生存预测

### 背景介绍

```
1912 年，泰坦尼克号沉没

数据包含乘客信息：
- 年龄、性别
- 船票等级（一等舱、二等舱、三等舱）
- 是否有兄弟姐妹/配偶
- 是否有父母/子女
- 票价等

目标：预测谁能存活

历史事实：
- 妇女和儿童优先上救生艇
- 一等舱乘客存活率更高
```

### 完整项目代码

```python
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
```

---

## 📝 今日总结

### ✅ 你今天学到了：

**1. 决策树**
- 像玩游戏一样做决策
- if-else 判断的升级版
- 有根节点、内部节点、叶节点

**2. 随机森林**
- 多棵树投票
- 三个臭皮匠顶个诸葛亮
- 通常比单棵树更准确

**3. 完整项目**
- 泰坦尼克号生存预测
- 数据预处理
- 特征重要性分析

---

## 🎁 明日预告

**明天你将学习：**

```
主题：支持向量机（SVM）

内容：
✓ 找最优分界线
✓ 核技巧的魔力
✓ 处理复杂数据
✓ 实战：手写数字识别

需要准备：
✓ 复习今天的决策树
✓ 了解什么是"边界"
✓ 保持好奇心！
```

---

## 🆘 常见问题

### Q1: 决策树和随机森林选哪个？

```
决策树：
✓ 简单易懂（能画出来）
✓ 适合解释给非技术人员
✗ 容易过拟合

随机森林：
✓ 准确率高
✓ 不容易过拟合
✗ 黑箱模型（不知道为啥）

建议：
需要解释 → 决策树
追求准确率 → 随机森林
```

### Q2: 树的数量设多少？

```
经验值：
✓ 小数据集：50-100 棵树
✓ 中等数据：100-200 棵树
✓ 大数据集：200-500 棵树

注意：
树越多越慢，但不一定更准
超过一定数量后提升不明显
```

---

## 🌟 鼓励的话

**第三天完成了！** 🎉

```
你已经学会了：
✓ K 近邻（Day 2）
✓ 决策树（Day 3）
✓ 随机森林（Day 3）

三种不同的算法！
你现在是个真正的机器学习初学者了！

继续加油！明天学习更高级的算法！💪
```

---

## 📞 打卡模板

```
日期：___________
学习时长：_______ 小时
掌握程度：⭐⭐⭐⭐⭐

今天学会了：


遇到的问题：


明天的目标：


```

**明天见！继续加油！** ✨

---

## 🔗 相关链接

### 🌐 项目资源
- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **如果觉得有帮助，请给 GitHub 仓库 Star 支持！**

### 📚 学习路径
- [← Day02](../Day02/README.md)
- [→ Day04](../Day04/README.md)

---

*本教程属于 [AI 入门 30 天挑战](https://github.com/Lee985-cmd/AI-30-Day-Challenge) 系列*

---

## 📱 关于作者 & 获取更多资源

本教程由 **Lee（职场宝爸）** 创建，记录从零基础到独立完成 AI 项目的真实历程。

### 关注公众号，获取独家内容

**公众号名称：Lee 的成长日记**

微信搜索关注，获取：
- ✅ **AI 学习路线规划**：零基础如何系统学习 AI
- ✅ **项目实战源码**：完整可运行的项目代码
- ✅ **深度技术解析**：前沿技术原理 + 手写代码实现
- ✅ **职场成长心得**：一个宝爸的 AI 逆袭之路

**关注福利**：
- 回复「**路线**」→ 获取 30 天 AI 学习计划表
- 回复「**项目**」→ 获取 GitHub 项目源码合集
- 回复「**资料**」→ 获取零基础学习资源推荐

**扫码关注公众号**：

![公众号二维码](../../images/logos/ewm.jpg)

### 其他平台

- 📂 **GitHub**：https://github.com/Lee985-cmd/AI-30Days-Challenge
- 📝 **CSDN 博客**：https://blog.csdn.net/m0_67081842
- 💬 **公众号**：微信搜索「Lee 的成长日记」

---

> 💡 **学习建议**
> 
> 如果本篇教程对你有帮助，欢迎：
> 1. **Star GitHub 项目**：https://github.com/Lee985-cmd/AI-30Days-Challenge
> 2. **关注公众号**获取更多独家内容
> 3. **留言交流**你的学习困惑
> 
> **一起学习，一起进步！** 🤝
