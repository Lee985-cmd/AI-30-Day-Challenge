"""
Day18 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day18_examples.py

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
print("Day18 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

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
    output: 注意力输出
    attention_weights: 注意力权重
    """
    
    # 步骤 1: Q 乘以 K 的转置，计算相似度
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # 步骤 2: 缩放 (除以根号 d_k)
    # 为什么？防止数值太大，softmax 梯度消失
    d_k = Q.size(-1)
    scores /= math.sqrt(d_k)
    
    # 步骤 3: Softmax 归一化 (变概率)
    attention_weights = nn.functional.softmax(scores, dim=-1)
    
    # 步骤 4: 用权重乘以 V，加权求和
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

"""
例子演示:

句子："I love AI"

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

# ===== 代码块 2 =====

import torch
import torch.nn as nn
import math

print("=" * 60)
print("🤖 Transformer 从零实现")
print("=" * 60)

# ============================================================================
# 第 1 步：定义 Self-Attention
# ============================================================================
print("\n【1. Self-Attention 机制】")

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "embed_size 必须是 heads 的倍数"
        
        # 定义 Q、K、V 的线性变换
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # 最后的线性变换
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # 分成多头
        # (batch, seq_len, embed_size) -> (batch, heads, seq_len, head_dim)
        queries = self.query(Q).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        keys = self.key(K).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        values = self.value(V).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        
        # 计算 Attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, values)
        
        # 合并多头
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)
        out = self.fc_out(out)
        
        return out, attention

print("✓ Self-Attention 定义完成")

# ============================================================================
# 第 2 步：定义 Transformer Block
# ============================================================================
print("\n【2. Transformer Block】")

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        
        # Feed Forward 网络
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )
        
        # LayerNorm 和 Dropout
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        # Attention + Residual + LayerNorm
        attention_out, attention_weights = self.attention(Q, K, V, mask)
        x = self.layer_norm1(Q + self.dropout(attention_out))
        
        # Feed Forward + Residual + LayerNorm
        ff_out = self.feed_forward(x)
        out = self.layer_norm2(x + self.dropout(ff_out))
        
        return out, attention_weights

print("✓ Transformer Block 定义完成")

# ============================================================================
# 第 3 步：测试模型
# ============================================================================
print("\n【3. 测试模型】")

# 参数设置
embed_size = 512  # 嵌入维度
heads = 8  # 多头数
seq_len = 10  # 序列长度
batch_size = 4  # 批次大小

# 创建模型
transformer_block = TransformerBlock(embed_size, heads)

# 创建测试数据
Q = torch.randn(batch_size, seq_len, embed_size)
K = torch.randn(batch_size, seq_len, embed_size)
V = torch.randn(batch_size, seq_len, embed_size)

# 运行模型
output, attention_weights = transformer_block(Q, K, V)

print(f"输入形状：{Q.shape}")
print(f"输出形状：{output.shape}")
print(f"Attention 权重形状：{attention_weights.shape}")
print(f"  (batch={batch_size}, heads={heads}, seq={seq_len})")

# 可视化 Attention 权重
fig, axes = plt.subplots(1, heads, figsize=(20, 4))
if heads == 1:
    axes = [axes]

for i in range(heads):
    im = axes[i].imshow(attention_weights[0, i].detach().numpy(), cmap='Blues')
    axes[i].set_title(f'Head {i+1}', fontsize=12)
    axes[i].set_xlabel('Keys', fontsize=10)
    axes[i].set_ylabel('Queries', fontsize=10)
    plt.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.show()

print("\n✅ Transformer Block 测试完成！")
print("=" * 60)