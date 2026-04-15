# 📝 AI 入门 30 天挑战 - Day 13 真正零基础版

---

## 🌟 完整项目和代码

本教程是 **AI 入门 30 天挑战** 系列的一部分！

- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **欢迎 Star 支持！**

---



> **今天学习 RNN 和 LSTM！**  
> **处理序列数据的秘密武器！**  
> **每个概念都用生活例子解释！**

---

## 📖 先复习一下昨天的内容

### CNN 架构回顾
```
✓ AlexNet → 深度学习的开始
✓ VGG → 全部用 3×3 卷积
✓ ResNet → 跳跃连接，解决梯度消失
✓ 迁移学习 → 站在巨人肩膀上
```

如果准备好了，我们开始今天的序列数据之旅！

---

## 🤔 什么是序列数据？

### 故事时间 📚

**生活中的序列数据：**

```
例子 1：看电影
你看《复仇者联盟》:
第 1 幕：钢铁侠出现
第 2 幕：美国队长出现
第 3 幕：他们打起来了
...

如果你只看第 3 幕：
❌ 看不懂他们在干嘛

必须按顺序看：
✅ 才能理解剧情发展

这就是序列！有先后顺序！
```

```
例子 2：读句子
"我 喜欢 吃 苹果"

如果打乱顺序：
"果 欢 吃 我 喜 苹" ❌ 看不懂了

按顺序读：
✅ 才能明白意思
```

### 常见的序列数据

```
1. 文字（自然语言）
   - 句子、文章、对话
   - 词有前后关系
   
2. 时间序列
   - 股票价格（每天的价格）
   - 天气数据（每小时的气温）
   - 心跳图（每秒的心率）
   
3. 语音
   - 声音波形
   - 音节有先后顺序
   
4. 视频
   - 一帧一帧的画面
   - 动作有连续性
```

---

## 🔄 RNN（循环神经网络）

### RNN 的核心思想

**普通神经网络 vs RNN：**

```
普通神经网络（如 CNN）:
输入 → [网络] → 输出

问题：
- 每个输入独立处理
- 不记得之前的输入
- 不适合序列数据

RNN:
输入 1 → [网络] → 输出 1
            ↓
          记忆
            ↓
输入 2 → [网络] → 输出 2
            ↓
          记忆
            ↓
输入 3 → [网络] → 输出 3

特点：
✓ 有"记忆"能力
✓ 当前输出取决于：
  - 当前输入
  - 之前的记忆
✓ 适合序列数据
```

### RNN 的工作原理

**生活中的例子：背电话号码**

```
你要背一个电话号码：13812345678

方法 1：一次全记住（普通网络）
❌ 太难了，记不住

方法 2：一个一个数字背（RNN）
第 1 步：记住 "1"
         ↓ (记忆)
第 2 步：记住 "13"（基于之前的记忆 + 新数字）
         ↓ (记忆)
第 3 步：记住 "138"
         ↓
...重复直到背完

这就是 RNN 的工作方式！
```

### RNN 的结构

```
标准 RNN 单元:

      输入 xₜ
        ↓
    ┌─────────┐
    │  RNN    │ ← 记忆 hₜ₋₁（来自上一步）
    │  单元   │
    └─────────┘
        ↓
    输出 hₜ
        ↓
    新的记忆（传给下一步）

计算公式:
hₜ = tanh(W·[hₜ₋₁, xₜ] + b)

解释:
- hₜ₋₁ = 上一步的记忆
- xₜ = 当前的输入
- W = 权重矩阵
- tanh = 激活函数
- hₜ = 新的输出 + 新的记忆
```

---

## 💻 RNN 代码实现

### 第 1 步：用 PyTorch 实现简单 RNN

**打开 Jupyter Notebook，输入：**

```python
import torch
import torch.nn as nn
import numpy as np

print("=" * 50)
print("📝 RNN 基础详解")
print("=" * 50)

# 1. 创建一个简单的 RNN
print("\n【1. 创建 RNN 模型】")

input_size = 10    # 输入特征维度
hidden_size = 20   # 隐藏层维度（记忆大小）
num_layers = 1     # RNN 层数
seq_length = 5     # 序列长度（比如 5 个词）
batch_size = 3     # 一次处理 3 个样本

# 创建 RNN
rnn = nn.RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True  # 输入格式：(batch, seq, feature)
)

print(f"✓ RNN 创建完成")
print(f"  输入维度：{input_size}")
print(f"  隐藏层维度：{hidden_size}")
print(f"  序列长度：{seq_length}")
print(f"  批次大小：{batch_size}")

# 2. 创建一些假数据
print(f"\n{'='*50}")
print("【2. 准备测试数据】")
print(f"{'='*50}")

# 随机生成输入数据
x = torch.randn(batch_size, seq_length, input_size)
print(f"输入形状：{x.shape}")
print(f"  (批次={batch_size}, 序列={seq_length}, 特征={input_size})")

# 3. 前向传播
print(f"\n{'='*50}")
print("【3. 运行 RNN】")
print(f"{'='*50}")

# 初始隐藏状态（可以是 None，会自动初始化为 0）
h0 = None

# 运行 RNN
output, hn = rnn(x, h0)

print(f"输出形状：{output.shape}")
print(f"  (批次={batch_size}, 序列={seq_length}, 隐藏层={hidden_size})")

print(f"\n最后一步的隐藏状态：{hn.shape}")
print(f"  这包含了整个序列的'记忆'")

print(f"\n💡 RNN 的特点:")
print(f"- 每一步处理一个输入")
print(f"- 同时接收上一步的记忆")
print(f"- 产生输出 + 新的记忆")
print(f"- 最后的隐藏状态包含整个序列的信息")
```

**按 Shift + Enter 运行！**

---

## 🧠 LSTM（长短期记忆网络）

### LSTM 解决的问题

**问题：RNN 记不住太长的东西**

```
普通 RNN:
序列短 → 记得住 ✅
序列长 → 前面的忘了 ❌

就像你背书：
背短句 → 没问题
背长文章 → 开头就忘了

这叫"长期依赖问题"
```

### LSTM 的创新

**LSTM 有三个"门"：**

```
门 1：遗忘门（Forget Gate）
作用：决定扔掉什么旧记忆

就像清理房间：
- 没用的东西 → 扔掉
- 重要的东西 → 保留

门 2：输入门（Input Gate）
作用：决定存入什么新记忆

就像记笔记：
- 重要的新知识 → 记下来
- 不重要的 → 忽略

门 3：输出门（Output Gate）
作用：决定输出什么

就像回答问题：
- 根据记忆组织答案
- 只说相关的部分
```

### LSTM 单元结构

```
LSTM 内部结构（简化）:

细胞状态 Cₜ（长期记忆）
    ↓
┌──────────────┐
│  遗忘门 fₜ   │ → 决定忘记什么
│  输入门 iₜ   │ → 决定记住什么
│  输出门 oₜ   │ → 决定输出什么
└──────────────┘
    ↓
隐藏状态 hₜ（短期记忆/输出）

关键：
✓ 细胞状态可以长时间保持
✓ 三个门控制信息流动
✓ 解决了长期依赖问题
```

---

## 💻 LSTM 代码实现

```python
print("=" * 50)
print("🧠 LSTM 详解")
print("=" * 50)

# 1. 创建 LSTM 模型
print("\n【1. 创建 LSTM 模型】")

lstm = nn.LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True
)

print(f"✓ LSTM 创建完成")
print(f"  输入维度：{input_size}")
print(f"  隐藏层维度：{hidden_size}")

# 2. 运行 LSTM
print(f"\n{'='*50}")
print("【2. 运行 LSTM】")
print(f"{'='*50}")

# LSTM 需要两个初始状态
# h0 = 隐藏状态（短期记忆）
# c0 = 细胞状态（长期记忆）
h0 = torch.zeros(num_layers, batch_size, hidden_size)
c0 = torch.zeros(num_layers, batch_size, hidden_size)

# 前向传播
output_lstm, (hn, cn) = lstm(x, (h0, c0))

print(f"输出形状：{output_lstm.shape}")
print(f"最终隐藏状态：{hn.shape}")
print(f"最终细胞状态：{cn.shape}")

print(f"\n💡 LSTM vs RNN:")
print(f"- LSTM 有额外的细胞状态（长期记忆）")
print(f"- 通过门机制控制信息")
print(f"- 能记住更长的序列")
print(f"- 工业界最常用")
```

---

## 🎬 实战：情感分析

### 完整的 LSTM 项目

```python
print("=" * 50)
print("🎬 实战：电影评论情感分析")
print("=" * 50)

print("""
任务：判断电影评论是正面还是负面

例子:
"这部电影太好看了！" → 正面 ✅
"太无聊了，浪费时间" → 负面 ❌

应用:
✓ 电商评价分析
✓ 社交媒体监控
✓ 品牌声誉管理
""")

# 1. 准备数据（模拟）
print("\n【1. 准备数据】")

# 模拟一些评论数据
reviews = [
    "这部电影非常精彩",      # 正面
    "太差了，不好看",        # 负面
    "演员演技很棒",          # 正面
    "剧情很烂",              # 负面
    "强烈推荐大家去看",       # 正面
    "浪费时间和金钱",        # 负面
]

labels = [1, 0, 1, 0, 1, 0]  # 1=正面，0=负面

print(f"✓ 准备了 {len(reviews)} 条评论")
print(f"  正面：{sum(labels)} 条")
print(f"  负面：{len(labels) - sum(labels)} 条")

# 2. 文本预处理
print(f"\n{'='*50}")
print("【2. 文本预处理】")
print(f"{'='*50}")

# 简单的字符级编码
# 实际项目会用分词 + 词向量

# 创建词汇表
char_vocab = set()
for review in reviews:
    char_vocab.update(review)

char_to_idx = {char: idx+1 for idx, char in enumerate(char_vocab)}
# 0 留给 padding

print(f"✓ 词汇表大小：{len(char_to_idx)} 个字符")

# 编码评论
def encode_text(text, max_len=20):
    """将文本转成数字序列"""
    encoded = [char_to_idx.get(char, 0) for char in text]
    # padding 或截断到固定长度
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return encoded

encoded_reviews = [encode_text(review) for review in reviews]
print(f"✓ 编码完成")
print(f"  每条评论长度：20 个字符")

# 转成 tensor
X = torch.tensor(encoded_reviews, dtype=torch.long)
y = torch.tensor(labels, dtype=torch.float32)

print(f"  输入形状：{X.shape}")
print(f"  标签形状：{y.shape}")

# 3. 创建情感分析模型
print(f"\n{'='*50}")
print("【3. 创建情感分析模型】")
print(f"{'='*50}")

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(SentimentRNN, self).__init__()
        
        # 词嵌入层（把数字转成向量）
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM 层
        self.lstm = nn.LSTM(embedding_dim, hidden_size, 
                           batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Sigmoid（输出概率）
        self.sigmoid = nn.Sigmoid()
        
        print(f"✓ 创建了情感分析模型:")
        print(f"  词嵌入：{vocab_size} → {embedding_dim}")
        print(f"  LSTM: {embedding_dim} → {hidden_size}")
        print(f"  全连接：{hidden_size} → {output_size}")
    
    def forward(self, x):
        # 词嵌入
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # 取最后一步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接
        output = self.fc(last_output)
        output = self.sigmoid(output)
        
        return output

# 创建模型
vocab_size = len(char_to_idx) + 1  # +1 for padding
embedding_dim = 32
hidden_size = 64
output_size = 1  # 二分类

model = SentimentRNN(vocab_size, embedding_dim, hidden_size, output_size)

# 4. 训练模型
print(f"\n{'='*50}")
print("【4. 训练模型】")
print(f"{'='*50}")

criterion = nn.BCELoss()  # 二分类损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    
    # 前向传播
    outputs = model(X).squeeze()
    loss = criterion(outputs, y)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 打印进度
    if (epoch + 1) % 20 == 0:
        with torch.no_grad():
            predicted = (outputs > 0.5).float()
            acc = (predicted == y).sum().item() / len(y) * 100
        print(f"第{epoch+1}/{num_epochs}轮 - "
              f"损失：{loss.item():.4f} - "
              f"准确率：{acc:.1f}%")

# 5. 测试模型
print(f"\n{'='*50}")
print("【5. 测试模型】")
print(f"{'='*50}")

model.eval()
with torch.no_grad():
    outputs = model(X).squeeze()
    predicted = (outputs > 0.5).float()
    
    print("\n预测结果:")
    for i, review in enumerate(reviews):
        pred_label = "正面😊" if predicted[i] == 1 else "负面😞"
        true_label = "正面😊" if labels[i] == 1 else "负面😞"
        status = "✅" if predicted[i] == labels[i] else "❌"
        
        print(f"{i+1}. \"{review}\"")
        print(f"   预测：{pred_label}, 真实：{true_label} {status}")

print(f"\n{'='*50}")
print("🎊 恭喜！你用 LSTM 完成了情感分析！")
print(f"{'='*50}")

print("""
总结 RNN/LSTM 的应用:

✓ 文本分类（情感分析）
✓ 机器翻译
✓ 语音识别
✓ 视频字幕生成
✓ 股票预测
✓ ...任何序列数据！
""")
```

---

## 📝 今日总结

### ✅ 你今天学到了：

**1. 序列数据**
- 有前后顺序的数据
- 文字、时间序列、语音、视频

**2. RNN（循环神经网络）**
- 有"记忆"能力
- 适合处理序列数据
- 但记不住太长的东西

**3. LSTM（长短期记忆）**
- 三个门控制信息
- 能记住长期依赖
- 工业界标准选择

**4. 实战应用**
- 情感分析
- 完整的训练流程

---

## 🎁 明日预告

**明天你将学习：**

```
主题：Week 2 综合项目

内容：
✓ Week 2 知识总结
✓ 三选一项目：
  A. 图像分类进阶（CIFAR-10）
  B. 文本生成（唐诗宋词）
  C. 人脸识别

完整流程：
- 问题定义
- 数据准备
- 模型搭建
- 训练优化
- 评估部署

需要准备：
✓ 复习本周所有内容
✓ 选择感兴趣的方向
✓ 准备好做毕业项目！
```

---

## 🆘 常见问题

### Q1: RNN 和 LSTM 选哪个？

```
选择建议：

RNN:
✓ 序列很短（<10 步）
✓ 计算资源有限
✗ 现在很少用了

LSTM:
✓ 大多数情况（首选）
✓ 序列较长
✓ 需要长期记忆
✓ 工业界标准

GRU:
✓ 类似 LSTM
✓ 稍微快一点
✓ 效果差不多

推荐：
默认用 LSTM
准没错！
```

### Q2: 怎么处理文本数据？

```
文本处理流程:

1. 分词
   - 中文：jieba 分词
   - 英文：空格分割
   
2. 建词汇表
   - 词 → 数字 ID
   
3. 词向量
   - Word2Vec
   - GloVe
   - 或用 Embedding 层学习
   
4. 输入 RNN/LSTM
   - 训练或微调

工具：
✓ torchtext（PyTorch 官方）
✓ jieba（中文分词）
✓ transformers（Hugging Face）
```

### Q3: 序列太长怎么办？

```
解决方法:

1. 截断
   - 只取前面 N 个
   - 或只取后面 N 个
   
2. 分层处理
   - 先分段落
   - 每段单独处理
   - 再组合结果
   
3. 注意力机制
   - 只关注重要部分
   - Transformer 架构
   
4. 双向 RNN
   - 从前往后 + 从后往前
   - 看到完整上下文
```

---

## 🌟 鼓励的话

**第十三天完成了！** 🎉

```
你已经学会了：
✓ Week 1: 7 种机器学习算法
✓ Week 2: 神经网络 + PyTorch + CNN + RNN/LSTM

看看你的成就：
从编程零基础
到能处理图像和文字！

这已经是全栈 AI 工程师的技能了！
继续加油！明天做毕业项目！💪✨
```

---

## 📞 打卡模板

```
日期：___________
学习时长：_______ 小时
掌握程度：⭐⭐⭐⭐⭐

对 RNN 的理解：


对 LSTM 的理解：


情感分析实战感受：


明天的项目选择：


```

**继续前进！Week 2 即将完成！** 🚀

---

## 🔗 相关链接

### 🌐 项目资源
- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **如果觉得有帮助，请给 GitHub 仓库 Star 支持！**

### 📚 学习路径
- [← Day12](../Day12/README.md)
- [→ Day14](../Day14/README.md)

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
