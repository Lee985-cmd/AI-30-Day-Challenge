# Day22-Q5 - Transformer 完整实现

## 📝 问题描述

理解了 Transformer 的各个组件后，现在需要将它们组合成一个完整的、可运行的 Transformer 模型。

**核心问题：**
- 如何从零实现完整的 Transformer？
- 训练时需要注意什么？
- 如何调试和验证实现是否正确？
- 如何在真实任务上应用？

---

## 💡 核心答案

完整的 Transformer 实现包括：
1. **Embedding 层**：Token + Positional Encoding
2. **Encoder**：N 层 Self-Attention + FFN
3. **Decoder**：N 层 Masked Self-Attention + Cross-Attention + FFN
4. **输出层**：Linear + Softmax

**关键技巧：**
- 残差连接和 LayerNorm
- 正确的 Mask 应用
- 学习率调度
- Label Smoothing

---

## 🎓 完整代码实现

### 1. 基础组件

```python
import torch
import torch.nn as nn
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分头
        Q = self.W_Q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # 加权求和
        output = torch.matmul(weights, V)
        
        # 合并头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_O(output)
        
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

---

### 2. Encoder 和 Decoder

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-Attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # Masked Self-Attention
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-Attention
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        
        return self.norm(x)
```

---

### 3. 完整 Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        output = self.generator(dec_output)
        return output
    
    def generate(self, src, max_len=50, start_symbol=0):
        """自回归生成"""
        self.eval()
        
        with torch.no_grad():
            enc_output = self.encoder(src)
            
            # 初始化目标序列
            tgt = torch.full((src.size(0), 1), start_symbol, dtype=torch.long, device=src.device)
            
            for _ in range(max_len - 1):
                # 生成 causal mask
                tgt_mask = self._generate_causal_mask(tgt.size(1)).to(src.device)
                
                # Decoder 前向传播
                dec_output = self.decoder(tgt, enc_output, tgt_mask)
                
                # 预测下一个 token
                logits = self.generator(dec_output[:, -1, :])
                next_token = logits.argmax(dim=-1, keepdim=True)
                
                # 添加到序列
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # 如果生成结束符，停止
                if next_token.item() == 1:  # 假设 1 是 <END>
                    break
        
        return tgt
    
    def _generate_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
```

---

### 4. 训练代码

```python
def train_transformer(model, train_loader, val_loader, num_epochs=20, lr=0.0001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 padding
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            src, tgt = batch
            
            # 前向传播
            output = model(src, tgt[:, :-1])  # Teacher forcing
            
            # 计算损失
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = evaluate(model, val_loader, criterion)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer.pth')
        
        scheduler.step()


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            src, tgt = batch
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    
    return total_loss / len(data_loader)
```

---

## ⚠️ 常见错误与调试

### 错误 1：维度不匹配

**调试方法：**
```python
def check_dimensions(model, src, tgt):
    print(f"Input shapes: src={src.shape}, tgt={tgt.shape}")
    
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")
    
    expected_shape = (tgt.size(0), tgt.size(1), tgt_vocab_size)
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
```

---

### 错误 2：梯度爆炸/消失

**检查方法：**
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 100:
            print(f"Gradient explosion in {name}: {grad_norm}")
        elif grad_norm < 1e-6:
            print(f"Vanishing gradient in {name}: {grad_norm}")
```

---

## ✍️ 自我检测练习

### 练习：在机器翻译任务上训练

**任务：** 使用上述代码训练一个英德翻译模型。

**步骤：**
1. 准备数据集（如 WMT14 En-De）
2. 创建 DataLoader
3. 初始化模型
4. 训练 20 个 epoch
5. 评估 BLEU 分数

**预期结果：**
- BLEU 分数：~25-28
- 训练时间：~1-2 天（单 GPU）

---

## 📝 本章小结

### Transformer 实现要点

✅ **模块化设计**：每个组件独立实现  
✅ **正确应用 Mask**：Decoder self-attention 需要 causal mask  
✅ **残差连接**：x + Sublayer(x)  
✅ **LayerNorm**：稳定训练  
✅ **梯度裁剪**：防止梯度爆炸  
✅ **学习率调度**：warmup + decay  

---

**📚 相关文档：**
- [Day22-Q4 - Positional Encoding](./Day22-Q4%20-%20Positional%20Encoding.md)
- [Day22-Q6 - Transformer 应用场景](./Day22-Q6%20-%20Transformer%20应用场景.md)（待创建）

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
