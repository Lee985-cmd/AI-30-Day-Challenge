# Day22-Q3 - Encoder-Decoder 架构

## 📝 问题描述

Transformer 采用 Encoder-Decoder 架构，这是处理序列到序列任务（如机器翻译）的经典范式。理解这个架构对于掌握 Transformer 至关重要。

**核心问题：**
- Encoder 和 Decoder 各自的作用是什么？
- 为什么 Decoder 需要 Masked Attention？
- Cross-Attention 如何工作？
- 完整的 Transformer 架构是怎样的？

---

## 💡 核心答案

**Encoder-Decoder 架构的核心思想：**

- **Encoder**：理解输入序列，提取特征表示
- **Decoder**：基于编码信息和已生成的部分，逐步生成输出序列

**类比：**
- Encoder = 阅读理解（读懂原文）
- Decoder = 写作表达（写出译文）

---

## 🎓 三个版本的解答

### 版本一：初学者比喻版（5 分钟理解）

#### 把 Encoder-Decoder 比作"翻译官"

**场景：** 中英翻译

```
输入: "I love AI"
输出: "我爱人工智能"
```

---

**Encoder（理解者）：**

```
任务：读懂英文句子

过程：
1. 看到 "I" → 理解：第一人称代词
2. 看到 "love" → 理解：动词，表达喜爱
3. 看到 "AI" → 理解：名词，人工智能

Self-Attention 帮助理解：
- "love" 关注 "I"（谁爱？）
- "love" 关注 "AI"（爱什么？）

最终输出：一个包含完整语义的向量表示
```

**类比：** 就像你读英文时，在脑海中形成对句子的理解。

---

**Decoder（表达者）：**

```
任务：用中文表达同样的意思

过程（自回归生成）：

第 1 步：生成 "我"
  - 看到 Encoder 的输出（英文理解）
  - 开始生成中文

第 2 步：生成 "爱"
  - 看到 Encoder 的输出
  - 看到已生成的 "我"
  - 决定下一个词是 "爱"

第 3 步：生成 "人工"
  - 看到 Encoder 的输出
  - 看到已生成的 "我爱"
  - 决定下一个词是 "人工"

第 4 步：生成 "智能"
  - 看到 Encoder 的输出
  - 看到已生成的 "我爱人工"
  - 决定下一个词是 "智能"

第 5 步：生成 "<END>"
  - 句子完成
```

**类比：** 就像你根据对英文的理解，逐字写出中文翻译。

---

**Masked Attention 的重要性：**

```
问题：生成第 2 个词时，能不能看到第 3、4、5 个词？

❌ 如果能看：
  - 作弊！还没生成就知道答案
  - 训练时有效，推理时无效（因为推理时要逐个生成）

✅ 如果不能看（Masked）：
  - 只能看到已生成的部分
  - 训练和推理一致
  - 模型学会真正的生成能力
```

**类比：** 考试时不能偷看后面的题目答案。

---

### 版本二：学生技术版（深入理解原理）

#### 1. Encoder 结构详解

**Encoder 由 N 层相同的模块堆叠而成：**

```
Input Embeddings + Positional Encoding
         ↓
┌─────────────────────┐
│   Encoder Layer 1   │
│  ┌───────────────┐  │
│  │ Multi-Head    │  │
│  │ Self-Attention│  │
│  └───────┬───────┘  │
│  ┌───────▼───────┐  │
│  │ Add & Norm    │  │
│  └───────┬───────┘  │
│  ┌───────▼───────┐  │
│  │ Feed Forward  │  │
│  └───────┬───────┘  │
│  ┌───────▼───────┐  │
│  │ Add & Norm    │  │
│  └───────────────┘  │
├─────────────────────┤
│   Encoder Layer 2   │
│  (same structure)   │
├─────────────────────┤
│         ...         │
├─────────────────────┤
│   Encoder Layer N   │
└─────────────────────┘
         ↓
   Encoder Output
```

**每层的两个子模块：**

1. **Multi-Head Self-Attention**
   ```python
   # 每个位置关注所有位置
   output = MultiHeadAttention(x, x, x)
   # Q=K/V=x，因为是 self-attention
   ```

2. **Feed Forward Network**
   ```python
   # 两层全连接网络
   FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
   # 通常 d_ff = 4 * d_model
   ```

**残差连接和层归一化：**
```python
# Sublayer 后应用
output = LayerNorm(x + Sublayer(x))
```

---

#### 2. Decoder 结构详解

**Decoder 也由 N 层相同模块堆叠：**

```
Output Embeddings + Positional Encoding
         ↓
┌─────────────────────┐
│   Decoder Layer 1   │
│  ┌───────────────┐  │
│  │ Masked Multi- │  │
│  │ Head Self-    │  │
│  │ Attention     │  │
│  └───────┬───────┘  │
│  ┌───────▼───────┐  │
│  │ Add & Norm    │  │
│  └───────┬───────┘  │
│  ┌───────▼───────┐  │
│  │ Multi-Head    │  │
│  │ Cross-        │  │
│  │ Attention     │  │
│  └───────┬───────┘  │
│  ┌───────▼───────┘  │
│  │ Add & Norm      │
│  └───────┬─────────┘
│  ┌───────▼───────┐  │
│  │ Feed Forward  │  │
│  └───────┬───────┘  │
│  ┌───────▼───────┐  │
│  │ Add & Norm    │  │
│  └───────────────┘  │
├─────────────────────┤
│   Decoder Layer 2   │
│  (same structure)   │
├─────────────────────┤
│         ...         │
└─────────────────────┘
         ↓
   Linear + Softmax
         ↓
   Output Probabilities
```

**每层的三个子模块：**

1. **Masked Multi-Head Self-Attention**
   ```python
   # 只能看到当前位置及之前的位置
   mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
   output = MultiHeadAttention(x, x, x, mask=mask)
   ```

2. **Multi-Head Cross-Attention**
   ```python
   # Query 来自 Decoder，Key/Value 来自 Encoder
   output = MultiHeadAttention(
       Q=decoder_output,  # 当前生成的部分
       K=encoder_output,  # 输入的编码
       V=encoder_output
   )
   ```

3. **Feed Forward Network**
   ```python
   # 与 Encoder 相同
   FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
   ```

---

#### 3. Cross-Attention 机制

**Cross-Attention vs Self-Attention：**

```
Self-Attention:
  Q, K, V 都来自同一个序列
  用途：建模序列内部关系

Cross-Attention:
  Q 来自 Decoder，K, V 来自 Encoder
  用途：让 Decoder 关注 Encoder 的相关部分
```

**具体例子：机器翻译**

```
输入（Encoder）: "I love AI"
输出（Decoder）: "我爱人工智能"

当 Decoder 生成 "人工" 时：

Q("人工") 关注 Encoder 的各个部分：
  - "I": 0.1
  - "love": 0.2
  - "AI": 0.7  ← 最高！

结论：生成 "人工" 时，主要关注 "AI"
```

**实现：**
```python
class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, decoder_x, encoder_x):
        # Q 来自 Decoder
        Q = self.W_Q(decoder_x)
        
        # K, V 来自 Encoder
        K = self.W_K(encoder_x)
        V = self.W_V(encoder_x)
        
        # 计算 attention
        scores = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(d_model)
        weights = torch.softmax(scores, dim=-1)
        output = torch.bmm(weights, V)
        
        return self.W_O(output)
```

---

#### 4. 完整的前向传播流程

**机器翻译示例：**

```python
def translate(source_sentence, target_sentence_start):
    """
    source_sentence: "I love AI"
    target_sentence_start: "<START>"
    """
    
    # ===== Encoder 阶段 =====
    
    # 1. Tokenize 和 Embedding
    src_tokens = tokenize(source_sentence)  # ["I", "love", "AI"]
    src_embeds = embedding(src_tokens) + positional_encoding
    
    # 2. Encoder 前向传播
    encoder_output = encoder(src_embeds)
    # encoder_output: (seq_len_src, d_model)
    
    # ===== Decoder 阶段 =====
    
    # 3. 初始化 Decoder 输入
    tgt_tokens = [target_sentence_start]  # ["<START>"]
    
    # 4. 自回归生成
    for _ in range(max_length):
        # Embedding
        tgt_embeds = embedding(tgt_tokens) + positional_encoding
        
        # Decoder 前向传播
        decoder_output = decoder(tgt_embeds, encoder_output)
        # 注意：decoder 接收 encoder_output 用于 cross-attention
        
        # 预测下一个 token
        logits = linear(decoder_output[-1])  # 只取最后一个位置
        probs = softmax(logits)
        next_token = argmax(probs)
        
        # 添加到序列
        tgt_tokens.append(next_token)
        
        # 如果生成 <END>，停止
        if next_token == "<END>":
            break
    
    return detokenize(tgt_tokens[1:-1])  # 去掉 <START> 和 <END>
```

---

### 版本三：工程师实践版（生产级实现）

#### 1. 完整 Transformer 实现

```python
import torch
import torch.nn as nn
import math

class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 d_model=512, 
                 n_heads=8, 
                 n_layers=6, 
                 d_ff=2048, 
                 dropout=0.1, 
                 max_seq_len=512):
        super().__init__()
        
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
        # 参数初始化
        self._init_weights()
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode
        enc_output = self.encoder(src, src_mask)
        
        # Decode
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        
        # Generate probabilities
        logits = self.generator(dec_output)
        
        return logits
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_seq_len):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Embedding + Positional Encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Encoder Layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_seq_len):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # Embedding + Positional Encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Decoder Layers
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        
        return self.norm(x)
```

---

#### 2. 训练技巧

**A. Label Smoothing**

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        # 平滑标签
        confidence = 1.0 - self.smoothing
        smooth_target = torch.full_like(pred, self.smoothing / (pred.size(-1) - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), confidence)
        
        # 交叉熵
        loss = -torch.sum(smooth_target * torch.log_softmax(pred, dim=-1))
        return loss
```

**效果：** 防止模型过于自信，提升泛化能力。

---

**B. Learning Rate Scheduling**

```python
class TransformerLR Scheduler:
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        return lr

# 使用
scheduler = TransformerLRScheduler(d_model=512)
for epoch in range(num_epochs):
    lr = scheduler.step()
    optimizer.param_groups[0]['lr'] = lr
```

**特点：** 先线性增长，再按平方根衰减。

---

**C. Gradient Clipping**

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## ⚠️ 常见错误与避坑指南

### 错误 1：Decoder 中忘记 Mask

**❌ 错误：**
```python
# Decoder self-attention 没有 mask
output = multi_head_attention(x, x, x)
```

**✅ 正确：**
```python
# 应用 causal mask
mask = generate_causal_mask(seq_len)
output = multi_head_attention(x, x, x, mask=mask)
```

---

### 错误 2：Cross-Attention 输入混淆

**❌ 错误：**
```python
# Q, K, V 都来自 Decoder
output = cross_attention(decoder_x, decoder_x, decoder_x)
```

**✅ 正确：**
```python
# Q 来自 Decoder，K, V 来自 Encoder
output = cross_attention(
    Q=decoder_x,
    K=encoder_output,
    V=encoder_output
)
```

---

## ✍️ 自我检测练习

### 练习 1：架构理解

**问题：** 为什么 Encoder 不需要 Mask，而 Decoder 需要？

**参考答案：**
```
Encoder:
- 处理完整输入序列
- 可以看到所有位置
- 无需 mask

Decoder:
- 自回归生成（逐个生成）
- 生成位置 t 时，只能看到 1~t-1
- 需要 mask 防止看到未来
- 保证训练和推理一致
```

---

### 练习 2：代码实现

**任务：** 实现 Causal Mask 生成函数。

**参考答案：**
```python
def generate_causal_mask(seq_len):
    """生成因果掩码"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    # 上三角为 True（需要 mask），下三角和对角线为 False
    return mask

# 测试
mask = generate_causal_mask(4)
print(mask)
# tensor([[False,  True,  True,  True],
#         [False, False,  True,  True],
#         [False, False, False,  True],
#         [False, False, False, False]])
```

---

## 📝 本章小结

### Encoder-Decoder 架构要点

✅ **Encoder**：理解输入，提取特征  
✅ **Decoder**：基于编码和已生成部分，逐步生成输出  
✅ **Masked Attention**：防止 Decoder 看到未来  
✅ **Cross-Attention**：让 Decoder 关注 Encoder 的相关部分  

---

**📚 相关文档：**
- [Day22-Q2 - Self-Attention 机制详解](./Day22-Q2%20-%20Self-Attention%20机制详解.md)
- [Day22-Q4 - Positional Encoding](./Day22-Q4%20-%20Positional%20Encoding.md)（待创建）
