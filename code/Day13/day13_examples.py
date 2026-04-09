"""
Day13 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day13_examples.py

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
print("Day13 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

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

# ===== 代码块 2 =====

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

# ===== 代码块 3 =====

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