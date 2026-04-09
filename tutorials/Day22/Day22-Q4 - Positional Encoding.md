# Day22-Q4 - Positional Encoding（位置编码）

## 📝 问题描述

Transformer 并行处理所有 token，失去了序列的顺序信息。Positional Encoding（位置编码）通过将位置信息注入到 token embedding 中，让模型知道每个 token 的位置。

**核心问题：**
- 为什么 Transformer 需要位置编码？
- 正弦位置编码的数学原理是什么？
- 为什么使用不同频率的正弦/余弦函数？
- 有哪些替代方案？

---

## 💡 核心答案

**位置编码的核心思想：** 为每个位置生成一个唯一的向量表示，加到 token embedding 上，让模型能够区分不同位置的相同 token。

**标准实现（正弦位置编码）：**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

这个设计有三个关键优势：
1. **唯一性**：每个位置有独特的编码
2. **相对位置**：可以学习相对位置关系
3. **外推性**：可以处理比训练时更长的序列

---

## 🎓 三个版本的解答

### 版本一：初学者比喻版（5 分钟理解）

#### 把位置编码比作"座位号"

**场景：** 电影院观影

```
问题：如果只告诉你"有人坐在第 3 排"，但不知道是哪一排的哪个座位，你能找到他吗？

❌ 不能！需要完整的位置信息："第 3 排第 5 座"
```

---

**Token Embedding 的问题：**

```
句子: "I love AI"

Token Embedding:
"I"    → [0.1, 0.5, 0.3, ...]
"love" → [0.2, 0.6, 0.4, ...]
"AI"   → [0.3, 0.7, 0.5, ...]

问题：如果把顺序打乱变成 "AI love I"
Token Embedding 还是一样！
模型不知道词的顺序
```

---

**位置编码的解决方案：**

```
给每个位置分配一个唯一的"座位号"：

位置 0 ("I"):    PE_0 = [1.0, 0.0, 0.8, ...]
位置 1 ("love"): PE_1 = [0.9, 0.4, 0.6, ...]
位置 2 ("AI"):   PE_2 = [0.7, 0.7, 0.3, ...]

最终输入 = Token Embedding + 位置编码
"I" (位置 0):    [0.1+1.0, 0.5+0.0, 0.3+0.8, ...]
"love" (位置 1): [0.2+0.9, 0.6+0.4, 0.4+0.6, ...]
"AI" (位置 2):   [0.3+0.7, 0.7+0.7, 0.5+0.3, ...]

现在模型知道：
- 第一个向量是位置 0 的 "I"
- 第二个向量是位置 1 的 "love"
- 第三个向量是位置 2 的 "AI"
```

---

**为什么用正弦/余弦？**

**类比：** GPS 定位

```
GPS 使用多个卫星信号定位你的位置：
- 卫星 1: 低频信号 → 粗略定位（大范围）
- 卫星 2: 中频信号 → 中等精度
- 卫星 3: 高频信号 → 精确定位（小范围）

组合多个频率的信号，可以唯一确定位置
```

**位置编码同理：**
```
不同维度使用不同频率：
- 维度 0,1: 低频 → 捕捉大范围位置关系
- 维度 2,3: 中频 → 捕捉中等范围关系
- 维度 4,5: 高频 → 捕捉局部关系

组合起来，每个位置有独特的"指纹"
```

---

### 版本二：学生技术版（深入理解原理）

#### 1. 为什么需要位置编码？

**Transformer 的排列不变性：**

```python
# Self-Attention 对输入顺序不敏感
x1 = [emb("I"), emb("love"), emb("AI")]
x2 = [emb("AI"), emb("love"), emb("I")]

# 如果不加位置编码，输出相同！
output1 = transformer(x1)
output2 = transformer(x2)
# output1 ≈ output2（错误！）
```

**原因：**
- Token Embedding 只包含词义信息
- Self-Attention 计算的是集合内关系，不考虑顺序
- 模型无法区分 "I love AI" 和 "AI love I"

---

#### 2. 正弦位置编码公式

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中：
- pos: 位置（0, 1, 2, ..., seq_len-1）
- i: 维度索引（0, 1, 2, ..., d_model/2-1）
- d_model: 模型维度（如 512）
```

**实现：**
```python
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        
        pos = torch.arange(0, max_seq_len).unsqueeze(1)  # (max_seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )  # (d_model/2,)
        
        pe[:, 0::2] = torch.sin(pos * div_term)  # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(pos * div_term)  # 奇数维度用 cos
        
        # 注册为 buffer（不参与梯度更新）
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_seq_len, d_model)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]
```

---

#### 3. 可视化位置编码

```python
import matplotlib.pyplot as plt

def visualize_positional_encoding(d_model=512, max_seq_len=100):
    pe = PositionalEncoding(d_model, max_seq_len)
    
    # 绘制前 10 个维度
    plt.figure(figsize=(12, 6))
    for i in range(0, 20, 2):
        plt.plot(pe.pe[0, :, i].numpy(), label=f'Dim {i}')
    
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.title('Positional Encoding (First 10 Dimensions)')
    plt.legend()
    plt.grid(True)
    plt.savefig('positional_encoding.png')
    plt.show()

visualize_positional_encoding()
```

**观察：**
- 低频维度：变化缓慢，长周期
- 高频维度：变化快速，短周期
- 每个位置的编码都是唯一的

---

#### 4. 为什么这样设计？

**A. 唯一性**

```
对于任意两个不同位置 pos1 ≠ pos2：
PE(pos1) ≠ PE(pos2)

证明：由于使用了不同频率的正弦/余弦函数，
      每个位置的编码向量都是唯一的
```

---

**B. 相对位置关系**

```
关键性质：PE(pos + k) 可以表示为 PE(pos) 的线性函数

这意味着：模型可以学习到相对位置关系
例如："动词通常在主语后面 1-2 个位置"
```

**数学推导：**
```
PE(pos + k, 2i) = sin((pos + k) / 10000^(2i/d_model))
                = sin(pos/ω_i + k/ω_i)
                = sin(pos/ω_i)cos(k/ω_i) + cos(pos/ω_i)sin(k/ω_i)
                = PE(pos, 2i) * cos(k/ω_i) + PE(pos, 2i+1) * sin(k/ω_i)

其中 ω_i = 10000^(2i/d_model)

结论：PE(pos+k) 是 PE(pos) 的线性组合
      系数只依赖于相对距离 k，与绝对位置 pos 无关
```

---

**C. 外推性**

```
训练时：最大序列长度 512
推理时：可以处理长度 1024 的序列

原因：正弦/余弦函数是周期性的，可以自然扩展到更长序列
```

---

#### 5. 替代方案

**A. 可学习位置编码**

```python
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_seq_len, d_model))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```

**优点：**
- ✅ 模型可以自由学习最优的位置表示
- ✅ 可能更适合特定任务

**缺点：**
- ❌ 无法外推到更长序列
- ❌ 需要更多训练数据

---

**B. 相对位置编码（RoPE）**

```python
# Rotary Positional Embedding (RoPE)
# GPTNeoX、LLaMA 使用

def apply_rope(q, k, pos):
    """应用旋转位置编码"""
    # 将位置信息编码为旋转矩阵
    # q, k: (seq_len, d_model)
    # pos: 位置
    
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2) / d_model))
    theta = pos * freqs
    
    # 旋转
    q_rotated = rotate(q, theta)
    k_rotated = rotate(k, theta)
    
    return q_rotated, k_rotated
```

**优点：**
- ✅ 更好的相对位置建模
- ✅ 适用于长序列

---

**C. ALiBi (Attention with Linear Biases)**

```python
# 在 attention scores 中添加线性偏置
scores = Q @ K.T / sqrt(d_k)

# 添加基于距离的偏置
for i in range(seq_len):
    for j in range(seq_len):
        scores[i][j] += -m * abs(i - j)  # m 是斜率
```

**优点：**
- ✅ 无需修改 embedding
- ✅ 天然支持任意长度序列

---

## ⚠️ 常见错误与避坑指南

### 错误 1：忘记缩放 Token Embedding

**❌ 错误：**
```python
x = embedding(tokens)
x = x + positional_encoding
```

**✅ 正确：**
```python
x = embedding(tokens) * math.sqrt(d_model)  # 缩放
x = x + positional_encoding
```

**原因：** 平衡 token embedding 和 positional encoding 的量级。

---

### 错误 2：位置编码维度不匹配

**❌ 错误：**
```python
pe = torch.randn(seq_len, d_model//2)  # 维度错误
x = x + pe
```

**✅ 正确：**
```python
pe = torch.randn(seq_len, d_model)  # 维度匹配
x = x + pe
```

---

## ✍️ 自我检测练习

### 练习 1：手动计算

**任务：** 计算 pos=2, i=0, d_model=4 时的 PE 值。

**参考答案：**
```python
pos = 2
i = 0
d_model = 4

# 偶数维度 (2i = 0)
PE(2, 0) = sin(2 / 10000^(0/4))
         = sin(2 / 10000^0)
         = sin(2 / 1)
         = sin(2)
         ≈ 0.909

# 奇数维度 (2i+1 = 1)
PE(2, 1) = cos(2 / 10000^(0/4))
         = cos(2)
         ≈ -0.416
```

---

### 练习 2：代码实现

**任务：** 实现 RoPE 位置编码。

**参考答案：**
```python
def apply_rotary_pos_emb(q, k, cos, sin):
    """应用旋转位置编码"""
    # q, k: (batch, heads, seq_len, head_dim)
    # cos, sin: (seq_len, head_dim)
    
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed
```

---

## 📝 本章小结

### 位置编码要点

✅ **必要性**：Transformer 并行处理，需要显式注入位置信息  
✅ **正弦编码**：使用不同频率的正弦/余弦函数  
✅ **相对位置**：可以学习位置间的相对关系  
✅ **外推性**：能处理比训练时更长的序列  

---

**📚 相关文档：**
- [Day22-Q3 - Encoder-Decoder 架构](./Day22-Q3%20-%20Encoder-Decoder%20架构.md)
- [Day22-Q5 - Transformer 完整实现](./Day22-Q5%20-%20Transformer%20完整实现.md)（待创建）
