# 🎯 AI 入门 30 天挑战 - Day 5 真正零基础版

---

## 🌟 完整项目和代码

本教程是 **AI 入门 30 天挑战** 系列的一部分！

- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **欢迎 Star 支持！**

---



> **今天学习 K-means 聚类（无监督学习）！**  
> **没有答案也能学习！**  
> **每个概念都用生活例子解释！**

---

## 📖 先复习一下前几天的内容

### 监督学习回顾
```
监督学习 = 有数据 + 有答案

例如：
- 给很多苹果和橙子的图片（数据）
- 告诉电脑哪个是苹果，哪个是橙子（答案）
- 电脑学会后能认新的水果

我们学过的监督学习算法：
✓ K 近邻（Day 2）
✓ 决策树 + 随机森林（Day 3）
✓ SVM（Day 4）
```

### 今天要学的：无监督学习

```
无监督学习 = 有数据 + 没答案

例如：
- 给电脑一堆水果图片（数据）
- 不告诉它是什么（没答案）
- 让它自己发现：哦～这些长得像，那些长得像

核心思想：物以类聚！
```

如果准备好了，我们开始今天的内容！

---

## 🔮 什么是 K-means 聚类？

### 故事时间 📚

想象你在**整理书架上的书**：

```
你有一堆书，但不知道如何分类：

方法 1：随便放 → 乱七八糟 ❌

方法 2：K-means 聚类 ✅

步骤：
第 1 步：随便选 3 个位置放书（选 3 个中心）
        ↓
第 2 步：把相似的书放到一起
        小说放一堆，科普放一堆，历史放一堆
        ↓
第 3 步：调整位置，让同类书更集中
        把每堆书往中间靠靠
        ↓
第 4 步：重复 2-3 步，直到稳定
        不再变化了，就分好了！
        
结果：三类书分好了！✅
```

### K-means 的工作原理

```
算法步骤：

1. 选择 K 值（要分成几类）
   ●     ●       ← 随机选 2 个中心
   
2. 分配样本到最近的中心
   🔴🔴●🔵🔵
   红色归左边，蓝色归右边
   
3. 更新中心点（移到平均值位置）
      ●           ●
   🔴🔴🔴       🔵🔵🔵
   
4. 重复 2-3 步，直到不再变化
   收敛完成！✅
```

**名词解释：**
- **K 值** = 要分成的类别数（比如 K=3 就是分成 3 类）
- **质心** = 每个类的中心点（所有点的平均位置）
- **距离** = 通常用直线距离（欧氏距离）

---

## 💻 K-means 代码实现

### 第 1 步：准备环境

**在命令行输入：**

```bash
pip install scikit-learn matplotlib
```

---

### 第 2 步：第一个 K-means 模型

**打开 Jupyter Notebook，新建笔记本，输入：**

```python
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
```

**按 Shift + Enter 运行！**

---

### 逐行解释

**创建模型部分：**

```python
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
```

**聚类结果：**

```python
labels = kmeans.labels_
# labels_ = 每个样本的类别标签
# 比如：[0, 0, 1, 1, 2, 2, 3, ...]
# 表示第 1、2 个样本是第 0 类，第 3、4 个是第 1 类...

centers = kmeans.cluster_centers_
# cluster_centers_ = 聚类中心的坐标
# 每个类一个中心点
```

---

## ❓ 怎么确定 K 值？

### 肘部法则

**就像选衣服：**
```
衣服太小 → 不舒服（欠拟合）
衣服太大 → 不好看（过拟合）
刚刚好 → 完美！（最优 K 值）
```

**代码演示：**

```python
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
```

---

## 👥 实战项目：客户分群

### 完整的 K-means 项目

```python
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
```

---

## 📝 今日总结

### ✅ 你今天学到了：

**1. K-means 聚类原理**
- 无监督学习（没有答案）
- 物以类聚的思想
- 迭代更新质心

**2. 肘部法则**
- 选择最优 K 值
- 找拐点（像手肘）

**3. 完整项目**
- 客户分群
- 数据标准化
- 客户画像

---

## 🎁 明日预告

**明天你将学习：**

```
主题：模型评估和优化

内容：
✓ 准确率、精确率、召回率
✓ 过拟合和欠拟合
✓ 交叉验证
✓ 正则化技术

需要准备：
✓ 复习今天的聚类知识
✓ 了解什么是"泛化能力"
✓ 保持好奇心！
```

---

## 🆘 常见问题

### Q1: K-means 和 KNN 有什么区别？

```
K-means（聚类）:
✓ 无监督学习
✓ 没有标签
✓ 用于分组
✓ K 是类别数

KNN（分类）:
✓ 监督学习
✓ 有标签
✓ 用于预测
✓ K 是邻居数

完全不同！不要混淆！
```

### Q2: 肘部法则不明显怎么办？

```
方法：
1. 试试轮廓系数（后面会学）
2. 试试业务理解（根据实际需求）
3. 多试几个 K 值，选效果好的
4. 换其他算法验证
```

---

## 🌟 鼓励的话

**第五天完成了！** 🎉

```
你已经学会了：
✓ K 近邻（监督）
✓ 决策树 + 随机森林（监督）
✓ SVM（监督）
✓ K-means（无监督）

第一周结束了！
你已经掌握了机器学习的基础算法！

周末休息一下，下周学习深度学习！💪
```

---

## 📞 打卡模板

```
日期：___________
学习时长：_______ 小时
掌握程度：⭐⭐⭐⭐⭐

Week 1 总结：


最大的收获：


遇到的困难：


下周的目标：


```

**第一周完成！给自己一个奖励吧！** ✨

---

## 🔗 相关链接

### 🌐 项目资源
- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **如果觉得有帮助，请给 GitHub 仓库 Star 支持！**

### 📚 学习路径
- [← Day04](../Day04/README.md)
- [→ Day06](../Day06/README.md)

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

![公众号二维码](../../../images/logos/ewm.jpg)

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
