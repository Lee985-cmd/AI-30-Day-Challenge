# 🎯 AI 入门 30 天挑战 - Day 4 真正零基础版

---

## 🌟 完整项目和代码

本教程是 **AI 入门 30 天挑战** 系列的一部分！

- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **欢迎 Star 支持！**

---



> **今天学习支持向量机（SVM）！**  
> **找最优分界线的艺术！**  
> **每个概念都用生活例子解释！**

---

## 📖 先复习一下昨天的内容

### 决策树回顾
```
决策树 = 一系列 if-else 判断

例如猜人游戏：
是男的吗？
├─ 是 → 戴眼镜吗？
│        ├─ 是 → 小明
│        └─ 否 → 小刚
└─ 否 → 长头发吗？
         ├─ 是 → 小红
         └─ 否 → 小丽
```

### 随机森林回顾
```
随机森林 = 很多棵决策树一起投票

一棵树 → 可能看走眼
多棵树 → 集体智慧，更可靠
```

如果这些都记得，我们开始今天的内容！

---

## 🎯 什么是支持向量机（SVM）？

### 故事时间 📚

想象你在**分水果**：

桌子上有两堆水果：
- 左边：苹果（红色的）
- 右边：橙子（橙色的）

你要在中间放一块木板把它们分开：

```
❌ 方法 1：随便放
🍎🍎 | 🍊🍊  ← 可以，但木板太靠近橙子
🍎🍎   | 🍊🍊

❌ 方法 2：太靠近苹果
🍎🍎| 🍊🍊  ← 不行，容易把苹果当成橙子

✅ 方法 3：放在正中间，两边空间都最大
🍎🍎    |    🍊🍊
     ←间隔→
     
这就是 SVM 的思想！
```

### SVM 的核心思想

```
支持向量机 = Support Vector Machine

目标：找到一条"最好"的分界线

什么是"最好"？
→ 让两边的间隔（margin）最大化！
→ 就像马路越宽越好走

关键概念：
1. 超平面 = 分界线（在高维空间）
2. 间隔 = 分界线到最近数据点的距离
3. 支持向量 = 离分界线最近的点（决定分界线位置）
```

---

## 💻 SVM 代码实现

### 第 1 步：准备环境

**在命令行输入：**

```bash
pip install scikit-learn matplotlib
```

---

### 第 2 步：第一个 SVM 模型

**打开 Jupyter Notebook，新建笔记本，输入：**

```python
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

print("=" * 50)
print("🎯 我的第一个 SVM 模型！")
print("=" * 50)

# 1. 创建数据集（两类容易分开的数据）
X, y = make_blobs(
    n_samples=100,      # 100 个样本
    centers=2,          # 2 个中心（两类）
    n_features=2,       # 2 个特征（方便画图）
    cluster_std=1.0,    # 每个类的分散程度
    random_state=42
)

print(f"\n数据集信息：")
print(f"样本数：{len(X)}")
print(f"特征数：{X.shape[1]}")
print(f"类别数：{len(np.unique(y))}")

# 2. 可视化数据
plt.figure(figsize=(8, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='类别 0', alpha=0.6)
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='类别 1', alpha=0.6)
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.title('原始数据分布')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 3. 创建 SVM 模型（线性核）
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)

# 4. 训练模型
print("\n正在训练线性 SVM...")
svm_linear.fit(X, y)
print("✅ 训练完成！")

# 5. 评估
accuracy = svm_linear.score(X, y)
print(f"\n训练集准确率：{accuracy*100:.2f}%")

# 6. 可视化决策边界
print("\n正在绘制决策边界...")

# 创建网格
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测整个网格
Z = svm_linear.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 画图
plt.figure(figsize=(10, 8))

# 画背景（决策区域）
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# 画数据点
plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='类别 0', edgecolors='k', s=100)
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='类别 1', edgecolors='k', s=100)

# 画决策边界
contour = plt.contour(xx, yy, Z, colors='black', linewidths=2)

# 标记支持向量（最关键的点）
support_vectors = svm_linear.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
           c='yellow', s=200, edgecolors='k', 
           marker='*', label='支持向量', zorder=10)

plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.title('SVM 线性分类 - 决策边界和支持向量')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n支持向量数量：{len(support_vectors)}")
print("看到黄色的星星了吗？这些就是支持向量！")
print("它们决定了分界线的位置！")

print("\n" + "=" * 50)
print("🎊 恭喜！你学会了 SVM！")
print("=" * 50)
```

**按 Shift + Enter 运行！**

---

### 逐行解释

**创建模型部分：**

```python
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
# svm = 模型的名字
# SVC() = Support Vector Classifier（支持向量分类器）
# kernel='linear' = 使用线性核（直线分界）
#   - 还有其他核：'rbf'（曲线分界）、'poly'（多项式）
# C=1.0 = 惩罚参数
#   - C 大 → 不允许错判（硬间隔）
#   - C 小 → 允许一些错误（软间隔）
# random_state=42 = 随机种子（保证结果可重复）
```

**支持向量：**

```python
support_vectors = svm_linear.support_vectors_
# support_vectors_ = 支持向量的坐标
# 这些是离分界线最近的点
# 它们"支撑"着分界线的位置
# 其他点不重要，只有支持向量重要！
```

---

## ✨ SVM 的魔法：核技巧

### 问题：非线性数据怎么办？

**如果数据长这样：**

```
    ● ● ●
  ●       ●
●    ○○○    ●
  ●       ●
    ● ● ●

你能用一条直线分开吗？
❌ 不能！

怎么办？
→ 升维！把数据投射到高维空间！
→ 在三维空间里，可以用一个平面分开
```

### 核技巧的原理

```
原始空间（2D）:
无法用直线分开
        ↓ 核函数（魔法映射）
高维空间（3D+）:
可以用平面分开
        ↓
回到原空间
→ 变成了曲线边界！
```

### 常见的核函数

**在 Jupyter 里试试不同的核函数：**

```python
from sklearn.datasets import make_circles

print("=" * 50)
print("🎨 不同核函数对比实验")
print("=" * 50)

# 创建非线性数据（同心圆）
X_circles, y_circles = make_circles(
    n_samples=400,
    noise=0.1,
    factor=0.5,
    random_state=42
)

# 不同的核函数
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for i, kernel in enumerate(kernels):
    # 训练 SVM
    if kernel == 'poly':
        svm = SVC(kernel=kernel, degree=3, C=10, random_state=42)
    else:
        svm = SVC(kernel=kernel, C=10, random_state=42)
    
    svm.fit(X_circles, y_circles)
    
    # 创建网格
    x_min, x_max = X_circles[:, 0].min() - 0.3, X_circles[:, 0].max() + 0.3
    y_min, y_max = X_circles[:, 1].min() - 0.3, X_circles[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # 画图
    axes[i].contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    axes[i].scatter(X_circles[y_circles==0, 0], X_circles[y_circles==0, 1], 
                   c='red', edgecolors='k', s=30)
    axes[i].scatter(X_circles[y_circles==1, 0], X_circles[y_circles==1, 1], 
                   c='blue', edgecolors='k', s=30)
    
    acc = svm.score(X_circles, y_circles)
    axes[i].set_title(f'{kernel} 核\n准确率 = {acc*100:.1f}%')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n核函数对比：")
print("1. linear（线性核）:")
print("   - 直接用直线分开")
print("   - 适合线性可分数据")
print("   - 速度快")
print()
print("2. poly（多项式核）:")
print("   - 用多项式曲线分开")
print("   - 适合复杂边界")
print("   - 参数多，难调")
print()
print("3. rbf（径向基核，最常用！）:")
print("   - 高斯函数，万能核")
print("   - 适合大多数情况")
print("   - 默认推荐！")
print()
print("4. sigmoid（双曲正切核）:")
print("   - 类似神经网络")
print("   - 用得比较少")
```

---

## ✍️ 实战：手写数字识别

### 完整的 SVM 项目

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("=" * 50)
print("✍️ SVM 实战：手写数字识别")
print("=" * 50)

# 1. 加载数据
digits = load_digits()

print(f"\n数据集信息：")
print(f"样本数：{len(digits.data)}")
print(f"特征数：{digits.data.shape[1]} (每个像素是一个特征)")
print(f"类别数：{len(digits.target_names)} (0-9)")

# 显示一些样本
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.ravel()

for i in range(10):
    axes[i].imshow(digits.images[i], cmap='gray')
    axes[i].set_title(f'标签：{digits.target[i]}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=42
)

print(f"\n训练集：{len(X_train)} 个样本")
print(f"测试集：{len(X_test)} 个样本")

# 3. 创建 SVM 模型（用 RBF 核）
print("\n正在训练 SVM 模型...")
svm_digits = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_digits.fit(X_train, y_train)
print("✅ 训练完成！")

# 4. 评估
train_acc = svm_digits.score(X_train, y_train)
test_acc = svm_digits.score(X_test, y_test)

print(f"\n📊 模型性能：")
print(f"训练集准确率：{train_acc*100:.2f}%")
print(f"测试集准确率：{test_acc*100:.2f}%")

# 5. 详细评估
y_pred = svm_digits.predict(X_test)

print("\n" + "=" * 50)
print("详细分类报告：")
print("=" * 50)
print(classification_report(y_test, y_pred))

# 6. 混淆矩阵
print("\n混淆矩阵：")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()

# 7. 实际预测几个数字
print("\n" + "=" * 50)
print("🔮 实际预测示例")
print("=" * 50)

test_indices = [0, 50, 100, 150, 200]

fig, axes = plt.subplots(1, 5, figsize=(12, 3))
if len(test_indices) == 1:
    axes = [axes]

for i, idx in enumerate(test_indices):
    if idx < len(X_test):
        sample = X_test[idx].reshape(8, 8)
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        
        axes[i].imshow(sample, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'真:{true_label} 预:{pred_label}', 
                         color=color, fontsize=12)
        axes[i].axis('off')

plt.tight_layout()
plt.show()

print(f"\n支持向量分析：")
print(f"支持向量总数：{len(svm_digits.support_)}")
print(f"占训练集比例：{len(svm_digits.support_)/len(X_train)*100:.1f}%")

print("\n" + "=" * 50)
print("🎊 恭喜！你完成了手写数字识别项目！")
print("=" * 50)
```

---

## 📝 今日总结

### ✅ 你今天学到了：

**1. SVM 的基本原理**
- 找间隔最大的分界线
- 支持向量决定分界线位置

**2. 核技巧**
- 线性核 → 直线分界
- RBF 核 → 曲线分界（推荐）
- 升维打击，解决非线性问题

**3. 完整项目**
- 手写数字识别
- 混淆矩阵评估
- 支持向量分析

---

## 🎁 明日预告

**明天你将学习：**

```
主题：K-means 聚类（无监督学习）

内容：
✓ 物以类聚的思想
✓ 不需要标签的学习
✓ 肘部法则选 K 值
✓ 实战：客户分群

需要准备：
✓ 复习今天的 SVM
✓ 了解什么是"相似性"
✓ 保持好奇心！
```

---

## 🆘 常见问题

### Q1: SVM 和随机森林选哪个？

```
SVM:
✓ 适合小数据集（<1000 样本）
✓ 高维数据表现好
✓ 内存消耗大
✗ 大数据集慢

随机森林:
✓ 适合大数据集
✓ 速度快
✓ 不容易过拟合
✗ 需要更多样本

建议：
小数据集 → SVM
大数据集 → 随机森林
```

### Q2: 怎么选核函数？

```
选择策略：
1. 先试试线性核（快）
2. 如果效果不好 → 用 RBF 核（万能）
3. 特殊数据 → 试试多项式核
4. 不确定 → RBF 准没错
```

---

## 🌟 鼓励的话

**第四天完成了！** 🎉

```
你已经学会了：
✓ K 近邻（Day 2）
✓ 决策树 + 随机森林（Day 3）
✓ SVM（Day 4）

四种不同的算法！
你现在是个真正的机器学习初学者了！

继续加油！明天是无监督学习！💪
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

**坚持就是胜利！明天见！** ✨

---

## 🔗 相关链接

### 🌐 项目资源
- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **如果觉得有帮助，请给 GitHub 仓库 Star 支持！**

### 📚 学习路径
- [← Day03](../Day03/README.md)
- [→ Day05](../Day05/README.md)

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
