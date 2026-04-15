# 🤖 Day22: Transformer 基础 - NLP 的革命性架构【真正零基础版】

---

## 🌟 完整项目和代码

本教程是 **AI 入门 30 天挑战** 系列的一部分！

- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **欢迎 Star 支持！**

---



> **ChatGPT、BERT 都基于它！理解 Transformer 是进入 NLP 的钥匙!**  
> **本教程：超详细图解 + 完整代码实现 + 手把手教学**

---

## 📚 目录

1. [为什么需要 Transformer?](#为什么需要-transformer)
2. [Attention 机制 - Transformer 的核心](#attention 机制 -transformer 的核心)
3. [Transformer 架构详解](#transformer 架构详解)
4. [实战：英译中翻译系统](#实战：英译中翻译系统)
5. [常见问题和调试技巧](#常见问题和调试技巧)

---

## 🤔 为什么需要 Transformer?

### RNN 的痛苦你经历过吗？

想象一下这个场景:

```
句子："我出生在中国，... (中间省略 1000 字) ...,所以我会说中文"

RNN 处理:
一个字一个字读:
我 → 出 → 生 → 在 → 中 → 国 → ... (读了 1000 个字) → 所 → 以 → 我 → 会 → 说 → 中 → 文

问题:
读到"所以"时，已经忘了开头的"中国"!
就像你读长文章，读到后面忘了前面...
```

**这就是 RNN 的梯度消失问题!**

- 句子太长，前面的信息传不到后面
- 记不住长距离的依赖关系

### Transformer 的解决方案

```
Transformer:
一眼看完整个句子!
"我出生在中国，...,所以我会说中文"
        ↑                        ↑
        └─────── 直接关联 ───────┘

用 Attention 机制:
- 每个词都能看到其他所有词
- 不管多远，直接建立联系
- 并行计算，速度飞快
```

**关键优势:**

| 特性 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 长距离依赖 | ❌ 困难 | ✅ 轻松 |
| 并行计算 | ❌ 不能 | ✅ 可以 |
| 训练速度 | 🐌 慢 | 🚀 快 |
| 效果 | 👌 还行 | 🏆 SOTA |

---

## 🎯 Attention 机制 - Transformer 的核心

### 说人话版本

什么是 Attention?

```
看这句话:
"The animal didn't cross the street because it was too tired"

问题："it"指的是什么？

人的注意力:
- 看到"it",你会想：这指谁？
- 往前找："animal"(动物) 最相关!
- 于是你知道：it = animal

这就是 Attention!
- 给不同的词分配不同的"注意力"
- 找到重要的词，忽略不重要的
```

### Self-Attention(自注意力)详解

```python
"""
Self-Attention 做什么？

输入：一个句子的所有单词
输出：每个单词的新表示 (融合了其他单词的信息)

怎么做的？
三个关键角色:
1. Query(Q) - "我在找什么？"
2. Key(K)    - "我有什么信息？"
3. Value(V)  - "我的实际内容"

说人话:
- Q: 问题 (比如：谁是主语？)
- K: 标签 (比如：这是名词)
- V: 内容 (比如："animal"的意思)

匹配过程:
1. 用 Q 去匹配所有的 K
2. 算出相似度分数
3. 根据分数，加权求和 V
4. 得到新的表示
"""
```

### Attention 计算公式 (不用怕，很简单!)

```python
import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(Q, K, V):
    """
    缩放点积注意力
    
    参数:
    Q: Query 矩阵 (batch_size, seq_len, d_k)
    K: Key 矩阵 (batch_size, seq_len, d_k)
    V: Value 矩阵 (batch_size, seq_len, d_v)
    
    返回:
    output: 注意力输出 (batch_size, seq_len, d_v)
    attention_weights: 注意力权重 (batch_size, seq_len, seq_len)
    """
    
    # 步骤 1: Q 乘以 K 的转置，计算相似度
    # (batch, seq_len, d_k) × (batch, d_k, seq_len) = (batch, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # 步骤 2: 缩放 (除以根号 d_k)
    # 为什么？防止数值太大，softmax 梯度消失
    d_k = Q.size(-1)
    scores /= math.sqrt(d_k)
    
    # 步骤 3: Softmax 归一化 (变概率)
    # 每行的和为 1，表示注意力权重
    attention_weights = nn.functional.softmax(scores, dim=-1)
    
    # 步骤 4: 用权重乘以 V，加权求和
    # (batch, seq_len, seq_len) × (batch, seq_len, d_v) 
    # = (batch, seq_len, d_v)
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

"""
例子演示:

句子："I love AI"

Q, K, V 矩阵 (简化版，假设维度是 4):

Q (我在找什么):
I:     [0.8, 0.2, 0.1, 0.9]  # 我在找主语
love:  [0.6, 0.8, 0.3, 0.5]  # 我在找对象
AI:    [0.7, 0.3, 0.9, 0.2]  # 我在找修饰词

K (我有什么):
I:     [0.9, 0.1, 0.2, 0.8]  # 我是代词
love:  [0.5, 0.9, 0.4, 0.6]  # 我是动词
AI:    [0.8, 0.2, 0.8, 0.3]  # 我是名词

V (实际内容):
I:     [ Embedding of "I" ]
love:  [ Embedding of "love" ]
AI:    [ Embedding of "AI" ]

计算 attention("love"):
1. love 的 Q 分别和 I、love、AI 的 K 算相似度
2. softmax 变成权重：[0.3, 0.5, 0.2]
3. 加权求和 V: 0.3*V(I) + 0.5*V(love) + 0.2*V(AI)

结果:
"love"的新表示融合了"I"和"AI"的信息!
"""
```

---

## 🏗️ Transformer 架构详解

### 整体结构

```
Transformer = Encoder(编码器) + Decoder(解码器)

Encoder(理解输入):
输入："I love AI"
↓
Self-Attention(自己理解自己)
↓
Feed Forward(加深理解)
↓
输出：编码后的表示

Decoder(生成输出):
输入："<start>" (开始标记)
↓
Masked Attention(只能看前面)
↓
Cross Attention(看 Encoder 的输出)
↓
Feed Forward
↓
输出："我"
↓
循环...直到生成"<end>"
```

### 完整代码实现

让我们从零实现一个简化版 Transformer:

```python
# ============================================================================
# 第一部分：导入库
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

print("=" * 60)
print("Transformer 从零实现 - 英译中翻译系统")
print("=" * 60)

# ============================================================================
# 第二部分：准备数据 (简化版)
# ============================================================================

"""
真实场景应该用大规模语料
这里为了演示，用一个小例子
"""

# 示例数据 (英文 - 中文对照)
sentences = [
    ("I love you", "我爱你"),
    ("He loves me", "他爱我"),
    ("She loves him", "她爱他"),
    ("We love them", "我们爱他们"),
    ("They love us", "他们爱我们"),
    ("You love me", "你爱我"),
    ("I hate him", "我讨厌他"),
    ("She hates her", "她讨厌她"),
]

# 构建词汇表
class Vocabulary:
    """词汇表类"""
    
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<sos>", 3: "<eos>"}
        self.word_count = {}
        self.n_words = 4  # 初始 4 个特殊 token
    
    def add_sentence(self, sentence):
        """添加句子到词汇表"""
        for word in sentence.split():
            if word not in self.word2idx:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1
    
    def sentence_to_indices(self, sentence):
        """句子转索引序列"""
        return [self.word2idx.get(word, self.word2idx["<unk>"]) 
                for word in sentence.split()]
    
    def indices_to_sentence(self, indices):
        """索引序列转句子"""
        words = []
        for idx in indices:
            if idx in [2, 3]:  # <sos>, <eos>
                continue
            if idx == 0:  # <pad>
                break
            words.append(self.idx2word.get(idx, "<unk>"))
        return " ".join(words)

# 创建词汇表
input_vocab = Vocabulary()  # 英文词汇表
output_vocab = Vocabulary()  # 中文词汇表

for en_sent, zh_sent in sentences:
    input_vocab.add_sentence(en_sent)
    output_vocab.add_sentence(zh_sent)

print(f"\n✓ 词汇表创建完成!")
print(f"  - 英文词汇量：{input_vocab.n_words}")
print(f"  - 中文词汇量：{output_vocab.n_words}")

# 准备训练数据
def prepare_data(sentences, input_vocab, output_vocab, max_len=20):
    """准备训练数据"""
    
    src_sentences = []  # 源语言 (英文)
    trg_sentences = []  # 目标语言 (中文)
    
    for en_sent, zh_sent in sentences:
        # 添加特殊 token
        en_sent = "<sos> " + en_sent + " <eos>"
        zh_sent = "<sos> " + zh_sent + " <eos>"
        
        # 转索引
        src_indices = input_vocab.sentence_to_indices(en_sent)
        trg_indices = output_vocab.sentence_to_indices(zh_sent)
        
        src_sentences.append(src_indices)
        trg_sentences.append(trg_indices)
    
    return src_sentences, trg_sentences

src_data, trg_data = prepare_data(sentences, input_vocab, output_vocab)

print(f"  - 训练样本数：{len(src_data)}")
print(f"\n示例:")
print(f"  英文：{sentences[0][0]} → 索引：{src_data[0]}")
print(f"  中文：{sentences[0][1]} → 索引：{trg_data[0]}")

# ============================================================================
# 第三部分：位置编码 (Positional Encoding)
# ============================================================================

"""
为什么需要位置编码？

RNN/LSTM:
- 按顺序处理，天然有位置信息

Transformer:
- 同时处理所有词，没有顺序概念!
- 必须手动添加位置信息

怎么添加？
用正弦和余弦函数:
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

说人话:
- 每个位置有一个独特的编码
- 相邻位置的编码相似
- 远距离位置的编码也能区分
"""

class PositionalEncoding(nn.Module):
    """位置编码层"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建一个足够大的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        
        # 计算位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # 偶数维用 sin，奇数维用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 调整形状 (max_len, 1, d_model)
        pe = pe.unsqueeze(1)
        
        # 注册为 buffer(不参与梯度更新)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        参数:
        x: (seq_len, batch_size, d_model)
        
        返回:
        x + pe: 加上位置编码的表示
        """
        return x + self.pe[:x.size(0), :]

# 测试位置编码
print("\n【测试位置编码】")
d_model = 8  # 用 8 维做演示
pe = PositionalEncoding(d_model, max_len=10)

test_input = torch.ones(5, 1, d_model)  # 5 个词，1 个 batch，8 维
output = pe(test_input)

print(f"输入形状：{test_input.shape}")
print(f"输出形状：{output.shape}")
print(f"\n前 3 个位置的编码 (部分维度):")
for i in range(3):
    print(f"位置{i}: {output[i, 0, :4].numpy()}")  # 只显示前 4 维

# ============================================================================
# 第四部分：Multi-Head Attention(多头注意力)
# ============================================================================

"""
为什么要多头？

单头 Attention:
- 只从一个角度理解句子
- 可能漏掉重要信息

Multi-Head Attention:
- 多个头，每个头关注不同的东西
- 有的头关注语法，有的关注语义
- 有的关注长距离，有的关注短距离

就像:
- 一个人看问题，可能片面
- 一群人讨论，看得更全面

实现:
把 d_model 分成 h 个头，每个头 d_k = d_model / h
"""

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 定义线性变换
        self.W_q = nn.Linear(d_model, d_model)  # Query
        self.W_k = nn.Linear(d_model, d_model)  # Key
        self.W_v = nn.Linear(d_model, d_model)  # Value
        self.W_o = nn.Linear(d_model, d_model)  # Output
        
        self.attention = None  # 保存注意力权重 (可视化用)
    
    def forward(self, Q, K, V, mask=None):
        """
        参数:
        Q, K, V: (batch_size, seq_len, d_model)
        mask: 掩码 (可选)
        
        返回:
        output: (batch_size, seq_len, d_model)
        """
        
        batch_size = Q.size(0)
        
        # 步骤 1: 线性变换并分头
        # (batch, seq_len, d_model) → (batch, seq_len, d_model)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 现在形状：(batch, num_heads, seq_len, d_k)
        
        # 步骤 2: 计算每个头的注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 如果有掩码 (Decoder 用)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 掩码的位置设为负无穷
        
        # Softmax
        attention_weights = nn.functional.softmax(scores, dim=-1)
        self.attention = attention_weights  # 保存用于可视化
        
        # 加权求和
        head_output = torch.matmul(attention_weights, V)
        # (batch, num_heads, seq_len, d_k)
        
        # 步骤 3: 合并多头
        head_output = head_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        # (batch, seq_len, d_model)
        
        # 步骤 4: 输出线性变换
        output = self.W_o(head_output)
        
        return output

# 测试多头注意力
print("\n【测试多头注意力】")
d_model = 512
num_heads = 8
mha = MultiHeadAttention(d_model, num_heads)

batch_size = 2
seq_len = 10
Q = K = V = torch.randn(batch_size, seq_len, d_model)

output = mha(Q, K, V)
print(f"输入形状：{Q.shape}")
print(f"输出形状：{output.shape}")
print(f"头数：{num_heads}, 每头维度：{d_model//num_heads}")

# ============================================================================
# 第五部分：Feed Forward Network(前馈神经网络)
# ============================================================================

"""
FFN 是什么？

就是两个全连接层:
Linear → ReLU → Linear

作用:
- 对每个位置的表示进行非线性变换
- 加深网络，增强表达能力
"""

class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# ============================================================================
# 第六部分：Encoder Layer(编码器层)
# ============================================================================

"""
Encoder Layer 包含:
1. Multi-Head Self-Attention
2. Feed Forward Network
3. Layer Normalization (每层都有)
4. Residual Connection (残差连接)

Residual Connection:
输入 → 层 → 输出
 ↓____________↑
      相加

作用:
- 防止梯度消失
- 让信息直接传递
"""

class EncoderLayer(nn.Module):
    """编码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)  # 归一化 1
        self.norm2 = nn.LayerNorm(d_model)  # 归一化 2
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        参数:
        x: (batch_size, seq_len, d_model)
        mask: 可选
        """
        
        # Self-Attention + 残差连接
        attn_output = self.attention(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)  # 归一化
        
        # FFN + 残差连接
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.norm2(x + ffn_output)  # 归一化
        
        return x

# ============================================================================
# 第七部分：Decoder Layer(解码器层)
# ============================================================================

"""
Decoder Layer 包含:
1. Masked Multi-Head Attention (只能看前面)
2. Cross Attention (看 Encoder 的输出)
3. Feed Forward Network
4. Layer Norm + Residual

Masked Attention:
- 预测第 t 个词时，只能看到 1~t-1 的词
- 防止偷看未来
"""

class DecoderLayer(nn.Module):
    """解码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # Masked Self-Attention
        self.attention1 = MultiHeadAttention(d_model, num_heads)
        
        # Cross Attention (连接 Encoder)
        self.attention2 = MultiHeadAttention(d_model, num_heads)
        
        # FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, trg_mask=None):
        """
        参数:
        x: Decoder 输入
        enc_output: Encoder 输出
        src_mask: 源语言掩码
        trg_mask: 目标语言掩码
        """
        
        # Masked Self-Attention
        attn1 = self.attention1(x, x, x, trg_mask)
        attn1 = self.dropout(attn1)
        x = self.norm1(x + attn1)
        
        # Cross Attention
        attn2 = self.attention2(x, enc_output, enc_output, src_mask)
        attn2 = self.dropout(attn2)
        x = self.norm2(x + attn2)
        
        # FFN
        ffn = self.ffn(x)
        ffn = self.dropout(ffn)
        x = self.norm3(x + ffn)
        
        return x

# ============================================================================
# 第八部分：完整的 Transformer
# ============================================================================

class Transformer(nn.Module):
    """完整的 Transformer 模型"""
    
    def __init__(self, src_vocab_size, trg_vocab_size, d_model=512, 
                 num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 d_ff=2048, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        
        # 词嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.pos_decoder = PositionalEncoding(d_model, max_len)
        
        # Encoder 层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder 层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
    
    def generate_mask(self, seq_len, batch_size):
        """生成掩码 (下三角矩阵)"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    def forward(self, src, trg):
        """
        参数:
        src: 源句子 (batch_size, src_len)
        trg: 目标句子 (batch_size, trg_len)
        """
        
        batch_size, src_len = src.shape
        _, trg_len = trg.shape
        
        # 生成掩码
        src_mask = None  # Encoder 不需要掩码
        trg_mask = self.generate_mask(trg_len, batch_size).to(src.device)
        
        # 词嵌入 + 位置编码
        src_emb = self.dropout(self.pos_encoder(self.src_embedding(src)))
        trg_emb = self.dropout(self.pos_decoder(self.trg_embedding(trg)))
        
        # Encoder
        enc_output = src_emb
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        # Decoder
        dec_output = trg_emb
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, trg_mask)
        
        # 输出层
        output = self.fc_out(dec_output)
        
        return output

# ============================================================================
# 第九部分：训练模型
# ============================================================================

print("\n" + "=" * 60)
print("开始训练 Transformer!")
print("=" * 60)

# 超参数
SRC_VOCAB_SIZE = input_vocab.n_words
TRG_VOCAB_SIZE = output_vocab.n_words
D_MODEL = 256  # 用小一点的模型方便演示
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
D_FF = 512
DROPOUT = 0.1
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
BATCH_SIZE = 2

# 创建模型
model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    trg_vocab_size=TRG_VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT
)

print(f"\n✓ 模型创建成功!")
print(f"  - 参数量：{sum(p.numel() for p in model.parameters()):,}")
print(f"  - 词嵌入维度：{D_MODEL}")
print(f"  - 注意力头数：{NUM_HEADS}")
print(f"  - Encoder 层数：{NUM_ENCODER_LAYERS}")
print(f"  - Decoder 层数：{NUM_DECODER_LAYERS}")

# 损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 padding
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 准备批次数据 (简化版，不做 padding)
src_tensor = torch.LongTensor(src_data)
trg_tensor = torch.LongTensor(trg_data)

print(f"\n开始训练...")
print(f"训练集大小：{len(src_data)} 句子")

# 训练循环
losses = []
for epoch in range(NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    # src: (batch, src_len)
    # trg_input: (batch, trg_len-1) 去掉最后一个
    # trg_target: (batch, trg_len-1) 去掉第一个
    src_input = src_tensor[:, :-1]  # 去掉 <eos>
    trg_input = trg_tensor[:, :-1]  # 去掉 <eos>
    trg_target = trg_tensor[:, 1:]  # 去掉 <sos>
    
    # 模型输出
    output = model(src_input, trg_input)
    
    # 计算损失
    # output: (batch, trg_len-1, vocab_size)
    # trg_target: (batch, trg_len-1)
    loss = criterion(output.view(-1, output.size(-1)), trg_target.reshape(-1))
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

# ============================================================================
# 第十部分：测试和可视化
# ============================================================================

print("\n" + "=" * 60)
print("训练完成！开始测试...")
print("=" * 60)

# 绘制损失曲线
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(losses, linewidth=2, color='blue')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('训练损失变化', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('transformer_loss.png', dpi=150)
plt.show()

# 测试翻译
model.eval()

print("\n【翻译测试】")
with torch.no_grad():
    for i, (en_sent, _) in enumerate(sentences[:3]):
        # 准备输入
        src_indices = input_vocab.sentence_to_indices("<sos> " + en_sent + " <eos>")
        src_tensor = torch.LongTensor([src_indices])
        
        # 贪心解码 (简化版)
        trg_indices = [output_vocab.word2idx["<sos>"]]
        
        for _ in range(10):  # 最多生成 10 个词
            trg_tensor = torch.LongTensor([trg_indices])
            
            output = model(src_tensor, trg_tensor)
            
            # 取最后一个位置的预测
            next_word_prob = output[0, -1, :]
            next_word = torch.argmax(next_word_prob).item()
            
            if next_word == output_vocab.word2idx["<eos>"]:
                break
            
            trg_indices.append(next_word)
        
        # 转成句子
        zh_sent = output_vocab.indices_to_sentence(trg_indices)
        
        print(f"英文：{en_sent}")
        print(f"翻译：{zh_sent}")
        print()

print("\n🎉 恭喜你完成了 Transformer 实现!")
print("\n下一步学习:")
print("  1. 用更大的数据集训练")
print("  2. 学习 Beam Search 解码")
print("  3. 尝试预训练的 BERT/GPT")
print("  4. 做自己的 NLP 项目")

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
