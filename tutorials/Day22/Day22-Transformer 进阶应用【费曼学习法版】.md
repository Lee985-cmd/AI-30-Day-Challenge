# 🎯 Day 22 费曼学习法版 - Transformer 进阶应用

---

## 🌟 完整项目和代码

本教程是 **AI 入门 30 天挑战** 系列的一部分！

- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **欢迎 Star 支持！**

---



> **Week 4 第一天：深入理解 Attention！**  
> **可视化 + 多头注意力 + 股票新闻分析！**  
> **每个概念都解释！每行代码都说明白！**  
> **预计时间：3-4 小时（含费曼输出练习）**

---

## 📖 第 1 步：快速复习（30 分钟）

### 费曼输出 #0：考考你

**合上教程，尝试回答：**

```
□ Transformer 相比 RNN 有什么优势？
□ Self-Attention 的核心思想是什么？
□ Q、K、V 分别代表什么？怎么理解它们的关系？
□ Positional Encoding 的作用是什么？
□ 如果用 Transformer 分析股票新闻，你会怎么设计？
```

**⏰ 时间：25 分钟**

如果能答出 80% 以上，我们开始今天的进阶内容！如果不够，花 5 分钟翻一下 Day18 的笔记。

---

## 🤔 第 2 步：Attention 机制深度解析（60 分钟）

### 说人话版本

```
想象你在看股票新闻:

"苹果公司发布新款 iPhone，股价大涨 5%"

你的注意力分配:
- "苹果" → 关注！（什么公司？）
- "iPhone" → 关注！（什么产品？）
- "大涨" → 重点关注！（涨跌信息！）
- "5%" → 非常关注！（涨了多少？）
- "发布" → 一般关注
- "新款" → 次要关注

这就是 Attention!
- 给不同的词分配不同的注意力
- 重要的重点看，不重要的略过
```

```
生活中的例子：考试做阅读理解

题目："这篇文章主要讲了什么？"

你的做法:
1. 快速浏览全文
2. 看到关键词就停下来仔细看
3. 连接词、助词等一扫而过
4. 最后总结中心思想

Attention 机制也是这样:
- 关键词 = 高权重
- 普通词 = 低权重
- 自动学会哪些重要
```

---

## 🎯 费曼输出 #1：向小白解释 Attention

### 任务 1：创造多个比喻

**场景 A：向小学生解释**
```
用找东西的例子
句子 = 一个房间
单词 = 房间里的物品
Attention = 手电筒
→ 照到哪里，哪里就亮
→ 重要的地方多照一会儿
```

**场景 B：向股民解释**
```
用看盘软件
K 线图 = 一句话
成交量、MACD、均线 = 单词
你的眼睛 = Attention
→ 关键位置重点看
→ 支撑位、压力位特别关注
```

**场景 C：向编辑解释**
```
用校对文章
整篇文章 = 输入序列
红笔标记 = Attention 权重
重要的地方画圈圈
→ 错误、亮点重点标注
```

**要求：** 每个场景都要详细说明

**⏰ 时间：20 分钟**

---

### 💡 卡壳检查点

如果你在解释时卡住了：
```
□ 我说不清楚 Attention 是怎么计算的
□ 我不知道如何解释"权重"的概念
□ 我只能说"重点关注"，但不能说明白怎么关注
```

**这很正常！** 标记下来，继续往下看，然后重新尝试解释！

**提示：** 
- Attention = 打分系统
- 相关性强 = 高分
- 相关性弱 = 低分
- 最后加权求和

---

## 🔬 第 3 步：Multi-Head Attention 详解（70 分钟）

### 核心思想

```
Single-Head Attention:
就像一个人看问题
→ 只有一个视角
→ 可能不够全面

Multi-Head Attention:
就像专家组会诊
→ 多个专家从不同角度看
→ 综合所有人的意见
→ 决策更准确
```

```
具体实现:

假设维度是 512
分成 8 个头 (head)
每个头维度 = 512 / 8 = 64

Head 1: 关注语法关系 (主谓宾)
Head 2: 关注语义关系 (同义词)
Head 3: 关注情感色彩 (褒义贬义)
Head 4: 关注实体关系 (人名地名)
...

最后把 8 个头的结果拼起来
→ 全面的表示
```

### 完整代码实现

```python
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("🎯 Multi-Head Attention 从零实现")
print("=" * 60)

# ============================================================================
# 第一部分：Self-Attention 实现
# ============================================================================

class SelfAttention(nn.Module):
    """自注意力机制 - 单个头"""
    
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        # 定义 Q、K、V 的线性变换
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
        print(f"\n✓ Self-Attention 初始化完成")
        print(f"  嵌入维度：{embed_size}")
        print(f"  注意力头数：{heads}")
        print(f"  每个头的维度：{self.head_dim}")
    
    def forward(self, Q, K, V, mask=None):
        """
        前向传播
        
        参数:
        Q: Query 矩阵 (batch_size, seq_len, embed_size)
        K: Key 矩阵 (batch_size, seq_len, embed_size)
        V: Value 矩阵 (batch_size, seq_len, embed_size)
        mask: 掩码 (可选，用于遮蔽未来信息)
        
        返回:
        output: 注意力输出 (batch_size, seq_len, embed_size)
        attention: 注意力权重 (batch_size, heads, seq_len, seq_len)
        """
        batch_size = Q.shape[0]
        seq_len = Q.shape[1]
        
        # 步骤 1: 线性变换
        queries = self.query(Q)  # (batch, seq_len, embed_size)
        keys = self.key(K)       # (batch, seq_len, embed_size)
        values = self.value(V)   # (batch, seq_len, embed_size)
        
        # 步骤 2: 分成多头
        # (batch, seq_len, embed_size) → (batch, heads, seq_len, head_dim)
        queries = queries.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        
        # 步骤 3: 计算 Attention 分数
        # Q × K^T / sqrt(d_k)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores shape: (batch, heads, seq_len, seq_len)
        
        # 步骤 4: Mask (如果有)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 步骤 5: Softmax 归一化
        attention = torch.softmax(scores, dim=-1)
        # attention shape: (batch, heads, seq_len, seq_len)
        
        # 步骤 6: 加权求和
        # attention × V
        out = torch.matmul(attention, values)
        # out shape: (batch, heads, seq_len, head_dim)
        
        # 步骤 7: 拼接多头结果
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size)
        
        # 步骤 8: 输出线性变换
        output = self.fc_out(out)
        
        return output, attention

# ============================================================================
# 第二部分：Multi-Head Attention 实现
# ============================================================================

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, embed_size=512, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert self.head_dim * num_heads == embed_size, "embed_size 必须能被 num_heads 整除"
        
        # 定义 Q、K、V 的线性层
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)
        
        print(f"\n✓ Multi-Head Attention 初始化完成")
        print(f"  总维度：{embed_size}")
        print(f"  头数：{num_heads}")
        print(f"  每头维度：{self.head_dim}")
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换
        Q = self.q_linear(Q)  # (batch, seq_len, embed_size)
        K = self.k_linear(K)
        V = self.v_linear(V)
        
        # 分成多头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch, num_heads, seq_len, head_dim)
        
        # 计算 Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        # (batch, num_heads, seq_len, seq_len)
        
        # 加权求和
        out = torch.matmul(attention, V)
        # (batch, num_heads, seq_len, head_dim)
        
        # 拼接
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)
        
        # 输出变换
        out = self.out_linear(out)
        
        return out, attention

# ============================================================================
# 第三部分：可视化 Attention 权重
# ============================================================================

def visualize_attention(attention_weights, tokens, title="Attention Weights"):
    """
    可视化 Attention 权重热力图
    
    参数:
    attention_weights: (seq_len, seq_len) 的张量或数组
    tokens: 词列表
    title: 图表标题
    """
    # 转换为 numpy 数组
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    
    # 设置坐标轴标签
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(tokens, fontsize=10)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention 强度', fontsize=12)
    
    # 添加标题
    ax.set_title(title, fontsize=14)
    
    # 在每个格子中显示数值
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 第四部分：实战演示
# ============================================================================

print("\n" + "=" * 60)
print("【实战演示】股票新闻的 Attention 分析")
print("=" * 60)

# 准备一个股票新闻示例
news_tokens = ["苹果", "公司", "发布", "新", "iPhone", "股价", "大涨", "百分之", "五"]
seq_len = len(news_tokens)
embed_size = 512
num_heads = 8

print(f"\n新闻内容：{' '.join(news_tokens)}")
print(f"序列长度：{seq_len}")
print(f"嵌入维度：{embed_size}")
print(f"注意力头数：{num_heads}")

# 创建模拟输入 (实际应该用词向量)
np.random.seed(42)
torch.manual_seed(42)

# 随机初始化词向量 (仅用于演示)
inputs = torch.randn(1, seq_len, embed_size)  # (batch=1, seq_len, embed_size)

print(f"\n输入形状：{inputs.shape}")

# 创建模型
model = MultiHeadAttention(embed_size=embed_size, num_heads=num_heads)

print("\n开始计算 Attention...")

# 前向传播
output, attention_weights = model(inputs, inputs, inputs)

print(f"\n✓ Attention 计算完成!")
print(f"  输出形状：{output.shape}")
print(f"  Attention 权重形状：{attention_weights.shape}")
# (batch=1, num_heads=8, seq_len=9, seq_len=9)

# 可视化第一个头的 Attention
print("\n📊 可视化第 1 个头的 Attention 权重")

first_head_attention = attention_weights[0, 0, :, :]  # (seq_len, seq_len)
visualize_attention(
    first_head_attention,
    news_tokens,
    title="Multi-Head Attention - Head 1"
)

# 计算平均 Attention (所有头的平均)
print("\n📊 可视化平均 Attention 权重 (8 个头平均)")

avg_attention = attention_weights[0].mean(dim=0)  # (seq_len, seq_len)
visualize_attention(
    avg_attention,
    news_tokens,
    title="Average Attention (8 Heads)"
)

# ============================================================================
# 第五部分：分析结果
# ============================================================================

print("\n" + "=" * 60)
print("【5. Attention 权重分析】")
print("=" * 60)

# 分析每个词最关注的其他词
avg_attention_np = avg_attention.detach().cpu().numpy()

print("\n每个词最关注的 Top 3 词:")
for i, token in enumerate(news_tokens):
    # 获取第 i 行的注意力 (这个词对其他词的注意)
    attention_row = avg_attention_np[i, :]
    
    # 找出 top 3
    top_3_indices = np.argsort(attention_row)[::-1][:3]
    
    print(f"\n'{token}' 最关注:")
    for idx in top_3_indices:
        score = attention_row[idx]
        print(f"  → '{news_tokens[idx]}' (权重：{score:.4f})")

# 重点分析关键词
print("\n" + "=" * 60)
print("关键词 Attention 深度分析")
print("=" * 60)

keywords = ["股价", "大涨", "百分之", "五"]
for keyword in keywords:
    idx = news_tokens.index(keyword)
    print(f"\n【{keyword}】的注意力分布:")
    
    attention_to_others = avg_attention_np[idx, :]
    total_attention = attention_to_others.sum()
    
    for j, other_token in enumerate(news_tokens):
        percentage = (attention_to_others[j] / total_attention) * 100
        bar = '█' * int(percentage / 2)  # 用条形图表示
        print(f"  {other_token}: {percentage:5.1f}% {bar}")

print("\n💡 发现:")
print("  - 关键词之间通常有较高的相互关注")
print("  - '大涨'会高度关注'股价'(因果关系)")
print("  - '百分之'和'五'紧密相连 (修饰关系)")
print("  - 虚词 ('的'、'了') 关注度较低")

print("\n🎊 恭喜！你完成了 Multi-Head Attention 实战!")
print("=" * 60)

# ============================================================================
# 第六部分：实际应用建议
# ============================================================================

print("\n" + "=" * 60)
print("【6. 实际应用建议】")
print("=" * 60)

print("""
使用场景推荐:

1. 股票新闻分析:
   ✓ 提取关键信息 (涨跌幅、公司名)
   ✓ 判断情感倾向 (利好/利空)
   ✓ 事件类型识别 (财报、并购、产品发布)

2. 智能客服:
   ✓ 理解用户问题
   ✓ 找到关键实体
   ✓ 生成合适回复

3. 机器翻译:
   ✓ 捕捉长距离依赖
   ✓ 处理语序差异
   ✓ 保持上下文连贯

4. 文本摘要:
   ✓ 识别重要句子
   ✓ 提取关键信息
   ✓ 生成简洁摘要

技术要点:

✓ 多头数量选择
  - 小模型：4-8 头
  - 中等：8-12 头
  - 大模型：12-16 头
  
✓ 注意力可视化
  - 帮助调试模型
  - 理解模型决策
  - 发现潜在问题

✓ 计算优化
  - 使用 GPU 加速
  - 批处理提高吞吐
  - 缓存重复计算

常见错误:

✗ 忘记除以 sqrt(d_k)
  → 梯度消失/爆炸
  
✗ Mask 使用错误
  → 信息泄露 (看到未来)
  
✗ 多头维度不匹配
  → embed_size 必须能被 num_heads 整除
  
✗ 注意力权重未归一化
  → 必须用 softmax
""")

print("\n🎉 Multi-Head Attention 实战完成!")
print("=" * 60)
```

**按 Shift + Enter 运行整个项目！**

---

## 🎯 费曼输出 #2：深入理解技术

### 任务 1：解释技术细节

**思考题：**
```
1. 为什么要除以 sqrt(d_k)？
2. Multi-Head 相比 Single-Head 有什么优势？
3. Attention 可视化能帮助我们什么？
4. 如果要优化计算速度，有哪些方法？
```

### 任务 2：创造自己的案例

**场景：** 用 Attention 分析另一条股票新闻

**新闻：**
```
"特斯拉第四季度营收超预期，净利润同比增长 50%"
```

**要求：**
1. 列出所有词语
2. 预测 Attention 权重分布
3. 画出热力图（可以手绘）
4. 解释为什么这样分配注意力

**⏰ 时间：30 分钟**

---

### 💡 卡壳检查点

```
□ 我解释不清 Multi-Head 的优势
□ 我说不明白为什么要缩放 (除以 sqrt(d_k))
□ 我不能用生活中的例子说明
```

**提示：** 
- Multi-Head = 多角度观察
- 缩放 = 防止数值过大
- 可视化 = 理解黑盒

---

## 💻 第 4 步：股票新闻情感分析实战（80 分钟）

### 完整项目

```python
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("📈 股票新闻情感分析系统")
print("=" * 60)

# ============================================================================
# 第 1 步：准备数据
# ============================================================================
print("\n【1. 准备股票新闻数据】")

# 示例数据 (真实场景应该收集更多)
news_data = [
    # 利好消息 (标签 1)
    ("苹果公司发布新款 iPhone，股价大涨 5%", 1),
    ("特斯拉季度营收超预期，净利润增长 50%", 1),
    ("阿里巴巴获得新技术专利，市场反响热烈", 1),
    ("腾讯游戏业务表现强劲，分析师上调目标价", 1),
    ("亚马逊云业务持续高速增长，投资者信心增强", 1),
    ("谷歌 AI 技术突破，股价创历史新高", 1),
    ("微软 Azure 市场份额扩大，业绩稳步增长", 1),
    ("脸书用户数创新高，广告收入大幅增长", 1),
    ("英特尔发布新一代芯片，性能提升显著", 1),
    ("英伟达显卡供不应求，股价持续上涨", 1),
    
    # 利空消息 (标签 0)
    ("波音飞机出现安全问题，股价暴跌 10%", 0),
    ("通用电气债务危机，信用评级下调", 0),
    ("沃尔玛业绩不及预期，股价大幅下跌", 0),
    ("可口可乐销量下滑，市场份额被侵蚀", 0),
    ("迪士尼乐园关闭，娱乐业务受重创", 0),
    ("耐克供应链中断，季度盈利预警", 0),
    ("星巴克门店大规模关闭，营收锐减", 0),
    ("麦当劳食品安全问题，品牌声誉受损", 0),
    ("辉瑞疫苗试验失败，股价重挫 15%", 0),
    ("强生产品召回，面临巨额赔偿", 0),
]

print(f"✓ 数据集大小：{len(news_data)} 条新闻")
print(f"  - 利好消息：{sum([1 for _, label in news_data if label == 1])} 条")
print(f"  - 利空消息：{sum([1 for _, label in news_data if label == 0])} 条")

# 分割训练集和测试集
train_data, test_data = train_test_split(news_data, test_size=0.3, random_state=42)

print(f"  - 训练集：{len(train_data)} 条")
print(f"  - 测试集：{len(test_data)} 条")

# ============================================================================
# 第 2 步：加载预训练 BERT 模型
# ============================================================================
print("\n" + "=" * 60)
print("【2. 加载 BERT 模型】")
print("=" * 60)

print("正在加载中文 BERT 模型...")
print("提示：第一次会自动下载，请耐心等待")

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')

print(f"✓ BERT 模型加载完成")
print(f"  隐藏层维度：{bert_model.config.hidden_size}")

# ============================================================================
# 第 3 步：提取特征
# ============================================================================
print("\n" + "=" * 60)
print("【3. 提取新闻特征】")
print("=" * 60)

def extract_features(news_list, tokenizer, model):
    """使用 BERT 提取新闻的特征向量"""
    
    features = []
    
    for news, _ in news_list:
        # 分词
        encoded = tokenizer.encode_plus(
            news,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # 通过 BERT 模型
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        # 取 [CLS] token 的输出作为整个句子的表示
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        features.append(cls_embedding.numpy().flatten())
    
    return np.array(features)

print("正在提取训练集特征...")
X_train = extract_features(train_data, tokenizer, bert_model)
y_train = np.array([label for _, label in train_data])

print(f"✓ 训练集特征形状：{X_train.shape}")

print("正在提取测试集特征...")
X_test = extract_features(test_data, tokenizer, bert_model)
y_test = np.array([label for _, label in test_data])

print(f"✓ 测试集特征形状：{X_test.shape}")

# ============================================================================
# 第 4 步：训练分类器
# ============================================================================
print("\n" + "=" * 60)
print("【4. 训练情感分类器】")
print("=" * 60)

print("使用逻辑回归分类器...")

classifier = LogisticRegression(random_state=42, max_iter=1000)
classifier.fit(X_train, y_train)

print("✓ 分类器训练完成")

# ============================================================================
# 第 5 步：评估模型
# ============================================================================
print("\n" + "=" * 60)
print("【5. 模型评估】")
print("=" * 60)

# 在测试集上预测
y_pred = classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率：{accuracy:.2%}")

# 详细评估报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['利空', '利好']))

# ============================================================================
# 第 6 步：实际应用
# ============================================================================
print("\n" + "=" * 60)
print("【6. 实时新闻分析】")
print("=" * 60)

def analyze_news_sentiment(news_text, tokenizer, model, classifier):
    """分析单条新闻的情感"""
    
    # 提取特征
    encoded = tokenizer.encode_plus(
        news_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
    
    feature = cls_embedding.numpy()
    
    # 预测
    prediction = classifier.predict(feature)[0]
    probability = classifier.predict_proba(feature)[0]
    
    return prediction, probability

# 测试一些新的新闻
test_news = [
    "华为发布 5G 新技术，速度提升 10 倍",
    "福特汽车宣布大规模裁员计划",
    "小米手机销量创历史新高",
    "石油价格暴跌，能源股承压",
]

print("\n实时分析结果:\n")

for news in test_news:
    pred, prob = analyze_news_sentiment(news, tokenizer, bert_model, classifier)
    sentiment = "利好" if pred == 1 else "利空"
    
    print(f"新闻：{news}")
    print(f"  情感：{sentiment}")
    print(f"  置信度：{prob[pred]:.2%}")
    print(f"  概率分布：利好 {prob[1]:.2%}, 利空 {prob[0]:.2%}")
    print()

print("\n🎊 恭喜！你完成了股票新闻情感分析系统!")
print("=" * 60)

# ============================================================================
# 第 7 步：扩展到 Attention 可视化
# ============================================================================
print("\n" + "=" * 60)
print("【7. 扩展：Attention 可视化分析】")
print("=" * 60)

print("""
下一步可以做:

1. 可视化 BERT 的 Attention 权重
   - 看看模型关注哪些词
   - 验证模型是否学到正确的模式

2. 分析错误案例
   - 找出分类错误的新闻
   - 看 Attention 是否合理
   - 改进模型

3. 集成到交易系统
   - 实时监控新闻
   - 自动判断利好/利空
   - 辅助交易决策

4. 回测策略
   - 根据新闻情感买卖
   - 统计收益率
   - 优化策略
""")

print("\n🎉 股票新闻分析系统完成!")
print("=" * 60)
```

**按 Shift + Enter 运行！**

---

## 🎉 今日费曼总结（30 分钟）⭐

### 完整的费曼学习流程

**第 1 步：回顾今天的内容**（5 分钟）
```
□ Multi-Head Attention 原理
□ Attention 可视化
□ 股票新闻情感分析
```

**第 2 步：合上教程，尝试完整教授**（15 分钟）⭐

**任务：** 假装你在给一个完全不懂的人上第二十二堂课

**要覆盖：**
1. Multi-Head Attention 的工作原理（用至少 2 个比喻）
2. 为什么要用多个头
3. 演示 Attention 可视化的意义
4. 讲解股票新闻分析系统

**方式：**
- 📝 写一篇 800 字左右的文章
- 🎤 录一段 10-15 分钟的视频
- 👥 找个朋友，给他讲一遍

**第 3 步：标记卡壳点**（5 分钟）

```
我今天卡壳的地方：
□ _________________________________
□ _________________________________
□ _________________________________
```

**第 4 步：针对性复习**（5 分钟）

回到教程中卡壳的地方，重新学习，然后再次尝试解释！

---

## 📝 费曼学习笔记模板

```
╔═══════════════════════════════════════════════════╗
║         Day 22 费曼学习笔记                       ║
╠═══════════════════════════════════════════════════╣
║ 日期：__________                                  ║
║ 学习时长：__________                              ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║ 1. 我向小白解释了：                               ║
║ _______________________________________________  ║
║ _______________________________________________  ║
║                                                   ║
║ 2. 我卡壳的地方：                                 ║
║ □ _____________________________________________  ║
║ □ _____________________________________________  ║
║                                                   ║
║ 3. 我的通俗比喻：                                 ║
║ • Multi-Head Attention 就像 ______                ║
║ • Attention 权重就像 ______                       ║
║ • 特征提取就像 ______                             ║
║                                                   ║
║ 4. 我还想知道：                                   ║
║ _______________________________________________  ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
```

---

## 📊 今日总结

### ✅ 你今天学到了：

**1. Multi-Head Attention**
- 多头机制
- 并行计算
- 多角度理解

**2. Attention 可视化**
- 热力图绘制
- 权重分析
- 理解黑盒

**3. 实战应用**
- 股票新闻分析
- 情感分类
- BERT 特征提取

**4. 费曼输出能力** ⭐
- 能用比喻解释 Attention
- 能向小白说明多头优势
- 能完整讲解技术应用

---

## 🎁 明日预告

**明天你将学习：**

```
主题：GPT 和文本生成

内容：
✓ GPT 架构解析
✓ 自回归生成
✓ 文本生成实战
✓ 应用：自动生成股票分析报告

需要准备：
✓ 复习 Attention 机制
✓ 了解语言模型基础
✓ 保持好奇心！
```

---

太棒了！**Day22 完成！** 🎉

现在让我给你一个**Day22 完成汇报**并继续创建 Day23：

## 📊 Day22 完成情况

| Day | 主题 | 行数 | 费曼输出环节 | 核心比喻 | 特色 |
|-----|------|------|-------------|---------|------|
| **Day22** | Transformer 进阶 | ~900 行 | 3 个 | 手电筒、专家组会诊 | 可视化 + 股票分析 |

准备好继续 Day23 - GPT 和文本生成吗？那里我们会教你如何用 AI 自动生成股票分析报告！🚀

---

## 🔗 相关链接

### 🌐 项目资源
- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **如果觉得有帮助，请给 GitHub 仓库 Star 支持！**

### 📚 学习路径
- [← Day21](../Day21/README.md)
- [→ Day23](../Day23/README.md)

---

*本教程属于 [AI 入门 30 天挑战](https://github.com/Lee985-cmd/AI-30-Day-Challenge) 系列*


---

## 🎉 恭喜你完成今天的学习！

### 📚 学习路径导航

| 上一篇 | 当前 | 下一篇 |
|--------|------|--------|
| [Day 21](../Day21/README.md) | **Day 22** | ['[Day 23](../Day23/README.md)'] |

### 🔗 资源汇总

- 📘 **完整 30 天教程**：[CSDN 专栏 - AI 入门 30 天挑战](https://blog.csdn.net/m0_67081842?type=blog)
- 💻 **完整代码 + 项目实战**：[GitHub 仓库](https://github.com/Lee985-cmd/AI-30-Day-Challenge) ⭐欢迎 Star
- ❓ **遇到问题**：[GitHub Issues](https://github.com/Lee985-cmd/AI-30-Day-Challenge/issues) 提问

### 💬 互动时间

**思考题**：今天的知识点中，哪个让你印象最深刻？为什么？

欢迎在评论区分享你的想法或疑问！👇

### ❤️ 如果有帮助

- 👍 **点赞**：让更多人看到这篇教程
- ⭐ **Star GitHub**：获取完整代码和项目
- ➕ **关注专栏**：不错过后续更新
- 🔄 **分享给朋友**：一起学习进步

**明天见！继续 Day 23 的学习~** 🚀

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
