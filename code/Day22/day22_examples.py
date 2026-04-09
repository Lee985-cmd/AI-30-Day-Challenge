"""
Day22 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day22_examples.py

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
print("Day22 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

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

# ===== 代码块 2 =====

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