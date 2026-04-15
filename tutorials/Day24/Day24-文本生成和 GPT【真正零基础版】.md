# ✍️ Day24: 文本生成和 GPT - 让 AI 写文章【真正零基础版】

---

## 🌟 完整项目和代码

本教程是 **AI 入门 30 天挑战** 系列的一部分！

- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **欢迎 Star 支持！**

---



> **ChatGPT、文心一言的核心技术！从零理解文本生成的奥秘!**  
> **本教程：完整代码 + 详细讲解 + ChatGPT API 实战**

---

## 📚 目录

1. [文本生成是什么？](#文本生成是什么)
2. [从规则到深度学习](#从规则到深度学习)
3. [GPT 系列架构详解](#gpt 系列架构详解)
4. [文本生成策略](#文本生成策略)
5. [Prompt Engineering](#prompt-engineering)
6. [实战：写诗机器人](#实战：写诗机器人)
7. [常见问题](#常见问题)

---

## 🤔 文本生成是什么？

### 说人话版本

想象一下这个场景:

```
你给 AI 一个开头:
"床前明月光"

AI 接着写:
"疑是地上霜。举头望明月，低头思故乡。"

这就是文本生成!
- 输入：一些提示 (prompt)
- 输出：连贯的文章/诗歌/故事

就像接龙游戏:
你说前半句，AI 接后半句
而且接得合情合理、文采飞扬
```

**文本生成能做什么？**

1. **创意写作**
   - 写诗、写小说
   - 写歌词、写剧本
   - 广告文案

2. **实用写作**
   - 邮件回复
   - 工作报告
   - 新闻稿

3. **代码生成**
   - GitHub Copilot
   - 根据注释写代码
   - 自动补全

4. **对话系统**
   - 智能客服
   - 聊天机器人
   - 心理陪伴

5. **内容创作**
   - 自媒体文章
   - 营销文案
   - 产品描述

---

## 📜 从规则到深度学习

### 第一代：基于规则 (1980s)

```python
"""
思路：
- 专家编写大量规则
- "如果...那么..."的逻辑

例子：天气报告生成

if 温度 > 30:
    输出："今天很热"
elif 温度 > 20:
    输出："今天温暖"
else:
    输出："今天凉爽"

缺点:
❌ 需要人工编写规则 (累)
❌ 覆盖不了所有情况 (死板)
❌ 生成的文本生硬 (不自然)
"""
```

### 第二代：统计模型 (1990s-2000s)

```python
"""
思路:
- 从大量文本中学习规律
- N-gram 模型最常见

N-gram 是什么？
看前 N-1 个词，预测下一个词

例子 (Bigram, N=2):
看到"人工" → 预测"智能"(概率 80%)
看到"机器" → 预测"学习"(概率 70%)

计算:
P(智能 | 人工) = count("人工智能") / count("人工")

缺点:
❌ 只能看很近的上下文 (短视)
❌ 数据稀疏问题 (没见过的组合就懵了)
❌ 生成质量一般般
"""
```

### 第三代：深度学习 (2010s-现在)

```python
"""
思路:
- 用神经网络学习语言表示
- RNN/LSTM → Transformer → GPT

RNN/LSTM:
✓ 能处理长句子
✓ 学习语法和语义
❌ 训练慢 (要按顺序处理)
❌ 长距离依赖还是困难

Transformer(GPT):
✓ 并行计算，训练飞快
✓ Attention 机制，长距离依赖轻松
✓ 预训练 + 微调，效果 SOTA

现在的水平:
✓ 能写高质量文章
✓ 能写代码
✓ 能对话
✓ 几乎以假乱真
"""
```

---

## 🤖 GPT 系列架构详解

### GPT 是什么？

```
GPT = Generative Pre-trained Transformer
生成式预训练 Transformer

核心特点:
1. Generative(生成式)
   - 能创造新文本
   - 不是分类/预测

2. Pre-trained(预训练)
   - 在海量数据上自学
   - 学会语言规律

3. Transformer
   - 基于 Transformer 架构
   - Decoder-only 版本
```

### GPT vs BERT

```python
"""
BERT(Encoder-only):
输入："我 [MASK] 你"
输出："爱" (填空)

擅长:
✓ 理解类任务
✓ 情感分析
✓ 文本分类
✓ 问答

GPT(Decoder-only):
输入："我爱你"
输出："，就像老鼠爱大米" (续写)

擅长:
✓ 生成类任务
✓ 写文章
✓ 对话
✓ 翻译
"""
```

### GPT 的自回归生成

```python
"""
自回归 (Autoregressive) 是什么？

说人话：
一个字一个字地生成，每次只看前面生成的内容

过程演示:

第 1 步:
输入：<start>
模型预测："春"(概率最高)

第 2 步:
输入：<start> 春
模型预测："眠"(概率最高)

第 3 步:
输入：<start> 春眠
模型预测："不"(概率最高)

第 4 步:
输入：<start> 春眠不
模型预测："觉"(概率最高)

...

直到生成 <end> 或达到最大长度

优点:
✓ 简单直接
✓ 能控制生成方向
✓ 质量高

缺点:
❌ 慢 (要一步步来)
❌ 不能回头修改
❌ 一错到底
"""
```

### GPT 架构细节 (简化版)

```python
import torch
import torch.nn as nn
import math

class SimpleGPT(nn.Module):
    """简化版 GPT"""
    
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, 
                 num_layers=4, max_len=512):
        super(SimpleGPT, self).__init__()
        
        # 1. 词嵌入层 (把词索引转成向量)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. 位置编码 (告诉模型词的顺序)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        
        # 3. Transformer Decoder 层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # 4. 输出层 (预测下一个词)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
        self.embed_dim = embed_dim
        self.max_len = max_len
    
    def forward(self, x):
        """
        参数:
        x: 输入序列 (batch_size, seq_len)
        """
        batch_size, seq_len = x.shape
        
        # 词嵌入
        token_emb = self.token_embedding(x)  # (batch, seq_len, embed_dim)
        
        # 位置编码
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)  # (batch, seq_len, embed_dim)
        
        # 合并
        x = token_emb + pos_emb  # (batch, seq_len, embed_dim)
        
        # Decoder (带 causal mask)
        # causal_mask: 防止偷看未来
        causal_mask = self.generate_causal_mask(seq_len, x.device)
        output = self.transformer_decoder(x, memory=None, tgt_mask=causal_mask)
        
        # 输出层
        logits = self.fc_out(output)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def generate_causal_mask(self, size, device):
        """生成因果掩码 (下三角矩阵)"""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

# 测试一下
print("=" * 60)
print("简化版 GPT 测试")
print("=" * 60)

vocab_size = 1000  # 词汇量
model = SimpleGPT(vocab_size)

# 模拟输入
batch_size = 2
seq_len = 10
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

output = model(input_ids)
print(f"输入形状：{input_ids.shape}")
print(f"输出形状：{output.shape}")
print(f"✓ GPT 前向传播成功!")

"""
GPT 的关键设计:

1. Causal Mask(因果掩码)
   - 防止偷看未来的词
   - 保证只能根据前面的词预测

2. 自回归训练
   - 输入：前 t 个词
   - 目标：第 t+1 个词
   
3. 预训练目标
   - 最大化似然估计
   - 让正确词的概率最大
"""
```

---

## 🎲 文本生成策略

### 策略 1: Greedy Search(贪心搜索)

```python
"""
思路:
每次选概率最大的词

例子:
预测下一个词:
- "天": 60%
- "空": 25%
- "气": 10%
- 其他：5%

贪心选择："天"(概率最大)

优点:
✓ 简单快速
✓ 确定性 (同样的输入总是同样的输出)

缺点:
❌ 容易陷入局部最优
❌ 生成结果单一
❌ 可能重复循环

适用场景:
- 需要确定性结果
- 对创造性要求不高
"""

def greedy_search(model, input_ids, max_length=50):
    """贪心搜索生成"""
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_length):
            # 模型预测
            logits = model(input_ids)
            
            # 取最后一个位置的预测
            next_token_logits = logits[:, -1, :]
            
            # 选概率最大的
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            
            # 添加到序列
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(1)], dim=1)
            
            # 遇到结束符就停
            if next_token_id.item() == 3:  # 假设 3 是<eos>
                break
    
    return input_ids
```

### 策略 2: Beam Search(束搜索)

```python
"""
思路:
保留多个候选序列 (beam),每个都继续扩展

例子 (beam_size=2):

第 1 步:
预测：[A:60%, B:25%, C:10%, ...]
保留：A 和 B

第 2 步:
从 A 扩展：[AA:40%, AB:30%, AC:20%, ...]
从 B 扩展：[BA:50%, BB:30%, BC:10%, ...]

计算联合概率:
- AA: 0.6 × 0.4 = 0.24
- AB: 0.6 × 0.3 = 0.18
- BA: 0.25 × 0.5 = 0.125
- BB: 0.25 × 0.3 = 0.075

保留概率最高的两个：AA 和 AB

优点:
✓ 考虑更多可能性
✓ 比贪心更优
✓ 全局视角

缺点:
❌ 计算量大 (beam_size 倍)
❌ 占用内存多
❌ 生成结果还是较保守

适用场景:
- 机器翻译
- 摘要生成
- 需要高质量的场景
"""

def beam_search(model, input_ids, beam_size=3, max_length=50):
    """束搜索生成"""
    model.eval()
    
    # 初始化 beams
    # 每个 beam: (序列，累积概率)
    beams = [(input_ids, 1.0)]
    completed_beams = []
    
    with torch.no_grad():
        for step in range(max_length):
            new_beams = []
            
            for seq, cum_prob in beams:
                # 模型预测
                logits = model(seq)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                
                # 取 top-k
                top_probs, top_ids = torch.topk(probs[0], beam_size)
                
                for prob, token_id in zip(top_probs, top_ids):
                    new_seq = torch.cat([seq, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_cum_prob = cum_prob * prob.item()
                    
                    # 检查是否结束
                    if token_id.item() == 3:  # <eos>
                        completed_beams.append((new_seq, new_cum_prob))
                    else:
                        new_beams.append((new_seq, new_cum_prob))
            
            # 保留 top-k 个 beams
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]
            
            if not beams:
                break
    
    # 返回最好的完成序列
    if completed_beams:
        completed_beams.sort(key=lambda x: x[1], reverse=True)
        return completed_beams[0][0]
    else:
        return beams[0][0]
```

### 策略 3: Sampling(采样)

```python
"""
思路:
按概率随机采样，不是每次都选最大的

为什么采样更好？
贪心/Beam Search:
- 总是选最常见的词
- 生成结果无聊、可预测

Sampling:
- 有时选次常见的词
- 更有创造性、更丰富

就像写作文:
学渣：总是用"很好""不错"(贪心)
学霸：用各种成语、名言 (采样)
"""

# Temperature 采样
def temperature_sampling(logits, temperature=1.0):
    """
    温度采样
    
    temperature:
    - < 1: 更保守 (放大差异)
    - = 1: 原始分布
    - > 1: 更随机 (抹平差异)
    """
    
    # 除以 temperature
    scaled_logits = logits / temperature
    
    # softmax 变概率
    probs = torch.softmax(scaled_logits, dim=-1)
    
    # 多项式采样
    next_token_id = torch.multinomial(probs, num_samples=1)
    
    return next_token_id

"""
Temperature 的效果:

temperature = 0.1(很保守):
输入："今天天气"
输出："真好""很好""不错"(反复这些词)

temperature = 0.7(适中):
输入："今天天气"
输出："真不错，阳光明媚""挺好的，适合出去玩"

temperature = 1.5(很随机):
输入："今天天气"
输出:"简直太棒了，让人心情愉悦""意外的好，蓝天白云"

建议:
- 创意写作：0.7-1.2
- 对话：0.7-0.9
- 正式文档：0.2-0.5
"""

# Top-k 采样
def top_k_sampling(logits, top_k=50):
    """
    Top-k 采样
    
    思路:
    只在概率最高的 k 个词里采样
    排除那些明显不靠谱的词
    """
    
    # 移除 top-k 之外的词 (设概率为 0)
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    
    # softmax + 采样
    probs = torch.softmax(logits, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1)
    
    return next_token_id

"""
Top-k 的效果:

top_k = 5(很保守):
只在最常见的 5 个词里选
→ 安全，但可能无聊

top_k = 50(适中):
在常见的 50 个词里选
→ 平衡质量和多样性

top_k = 500(很自由):
在 500 个词里选
→ 很有创意，但可能跑题

建议:
- top_k = 40-60 常用
- 配合 temperature 使用
"""

# Top-p 采样 (Nucleus Sampling)
def top_p_sampling(logits, top_p=0.95):
    """
    Top-p 采样 (核采样)
    
    思路:
    选一组词，让它们的累积概率达到 p
    在这组词里采样
    
    优势:
    - 动态调整候选词数量
    - 概率分布尖锐时，候选词少
    - 概率分布平坦时，候选词多
    """
    
    # 排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # 累积概率
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 找到累积概率超过 top_p 的位置
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # 至少保留一个词
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # 移除后面的词
    indices_to_remove = sorted_indices_to_remove.scatter(
        0, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float('-inf')
    
    # 采样
    probs = torch.softmax(logits, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1)
    
    return next_token_id

"""
Top-p vs Top-k:

Top-k:
- 固定数量 (比如 50 个)
- 不管概率分布如何

Top-p:
- 动态数量
- 概率集中时，选少数几个
- 概率分散时，选很多个

建议:
- top_p = 0.9-0.95 常用
- 可以只使用 top-p，或者 top-k + top-p 结合
"""
```

---

## 💬 Prompt Engineering

### 什么是 Prompt Engineering?

```
说人话：
就是"问问题的技巧"

同样的问题，不同的问法:
问法 1: "法国首都是哪？"
回答："巴黎"

问法 2: "请用法语回答：法国的首都是哪个城市？"
回答："Paris"

问法 3: "你是一个地理学家，请用专业术语解释法国首都的历史沿革"
回答：(一大段详细的介绍)

看！同样的知识，不同的问法得到不同的回答!
```

### 技巧 1: Zero-shot(零样本)

```python
"""
Zero-shot:
不给例子，直接问

例子:
Prompt: "将以下英文翻译成中文：I love you"
Output: "我爱你"

Prompt: "这篇文章的情感是正面还是负面？这个产品很好用"
Output: "正面"

适用场景:
- 模型已经知道任务
- 简单的指令遵循
"""

zero_shot_examples = [
    ("翻译", "将以下英文翻译成中文：Hello, how are you?"),
    ("情感分析", "判断这句话的情感：这部电影太精彩了！"),
    ("问答", "中国的首都是哪里？"),
    ("总结", "用一句话总结：人工智能是..."),
]

for task, prompt in zero_shot_examples:
    print(f"{task}: {prompt}")
    print()
```

### 技巧 2: Few-shot(少样本)

```python
"""
Few-shot:
给几个例子，让模型模仿

例子 (教模型做类比):

Prompt:
苹果→水果
胡萝卜→蔬菜
猪肉→肉类
西瓜→？

Model: 水果

为什么有效？
- 模型从例子中学会了模式
- "A 是 B 的一种"的关系

Few-shot 的威力:
GPT-3 论文显示:
- Zero-shot: 60% 准确率
- Few-shot: 85% 准确率!
"""

few_shot_prompt = """
请完成类比推理:

例子 1:
苹果→水果
胡萝卜→蔬菜

例子 2:
椅子→家具
T 恤→服装

例子 3:
铅笔→文具
手术刀→？

答案：医疗器械

现在请你做下面的:
桌子→家具
牙刷→？
"""

print("Few-shot 示例:")
print(few_shot_prompt)
```

### 技巧 3: Chain of Thought(思维链)

```python
"""
Chain of Thought(CoT):
让模型一步一步推理

普通问法:
Q: "小明有 5 个苹果，给了小红 2 个，又买了 3 个，现在有几个？"
A: "6 个"(可能答错)

CoT 问法:
Q: "小明有 5 个苹果，给了小红 2 个，又买了 3 个，现在有几个？
请逐步思考:
1. 一开始有几个？
2. 给了小红后剩几个？
3. 又买了几个？
4. 最后总共几个？"

A: 
"1. 一开始有 5 个
2. 给了小红 2 个，剩下 5-2=3 个
3. 又买了 3 个，变成 3+3=6 个
4. 最后总共 6 个"

答案正确率大幅提升!
"""

cot_prompt = """
问题：一个农场有鸡和兔子共 35 个头，94 只脚，问鸡和兔子各有多少只？

请逐步思考:
1. 设鸡有 x 只，兔子有 y 只
2. 根据头的数量列方程：x + y = 35
3. 根据脚的数量列方程：2x + 4y = 94
4. 解方程组:
   - 从方程 1 得：x = 35 - y
   - 代入方程 2: 2(35-y) + 4y = 94
   - 70 - 2y + 4y = 94
   - 2y = 24
   - y = 12
   - x = 35 - 12 = 23
5. 答案：鸡有 23 只，兔子有 12 只

现在请你解这道题:
小明买了一些书和笔，共花了 100 元。书每本 15 元，笔每支 5 元。他一共买了 12 件商品。
问：书和笔各买了多少？

请逐步思考:
"""

print("思维链示例:")
print(cot_prompt)
```

### 技巧 4: Role Playing(角色扮演)

```python
"""
Role Playing:
给模型设定一个角色

效果:
- 回答更符合角色身份
- 语气和专业度提升
- 更有针对性

例子:
"""

role_prompts = [
    ("老师", "你是一个耐心的小学老师，请用简单易懂的语言解释什么是重力"),
    ("医生", "你是一名经验丰富的医生，请给出健康饮食的建议"),
    ("程序员", "你是资深 Python 工程师，请解释什么是装饰器"),
    ("导游", "你是北京导游，请介绍故宫的主要景点"),
]

for role, prompt in role_prompts:
    print(f"角色：{role}")
    print(f"Prompt: {prompt}")
    print()
```

---

## 🎨 实战：写诗机器人

让我们用 Hugging Face 的 transformers 库实现一个完整的写诗机器人:

```python
# ============================================================================
# 第一部分：导入库
# ============================================================================

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random

print("=" * 60)
print("GPT 写诗机器人 - 从零开始")
print("=" * 60)

# ============================================================================
# 第二部分：加载预训练模型
# ============================================================================

"""
选择哪个模型？

GPT-2:
- OpenAI 的第二代 GPT
- 1.5B 参数 (中文版)
- 适合文本生成
- 可以本地运行

GPT-2 中文:
- uer/gpt2-chinese-cluecorpussmall
- 在中文语料上训练
- 更适合写诗

注意:
第一次运行会自动下载 (约 500MB)
"""

print("\n正在加载 GPT-2 模型...")
print("提示：第一次运行需要下载，请耐心等待")

# 模型名称 (中文 GPT-2)
model_name = 'uer/gpt2-chinese-cluecorpussmall'

try:
    # 加载分词器
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # 加载模型
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    print(f"✓ 模型加载成功!")
    print(f"  - 模型：{model_name}")
    print(f"  - 参数量：{model.num_parameters():,}")
    
except Exception as e:
    print(f"加载失败：{e}")
    print("使用备用方案：创建一个随机初始化的模型")
    
    # 备用方案
    from transformers import GPT2Config
    config = GPT2Config(
        vocab_size=21128,  # 中文 GPT-2 词汇量
        n_positions=512,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel(config)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"  - 运行设备：{device}")

# ============================================================================
# 第三部分：定义生成函数
# ============================================================================

def generate_poem(prompt, max_length=100, temperature=0.8, 
                  top_k=50, top_p=0.95, num_return_sequences=1):
    """
    生成诗歌
    
    参数:
    prompt: 提示词/开头
    max_length: 最大长度
    temperature: 温度 (创造性)
    top_k: Top-k 采样
    top_p: Top-p 采样
    num_return_sequences: 返回几个版本
    """
    
    # 1. 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # 2. 生成
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,          # 启用采样
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  # 防止重复
            no_repeat_ngram_size=3   # 不允许 3 元语法重复
        )
    
    # 3. 解码
    poems = []
    for i in range(num_return_sequences):
        generated_text = tokenizer.decode(output[i], skip_special_tokens=True)
        poems.append(generated_text)
    
    return poems

# ============================================================================
# 第四部分：测试不同风格
# ============================================================================

print("\n" + "=" * 60)
print("开始写诗!")
print("=" * 60)

# 测试不同的开头
prompts = [
    "床前明月光",
    "春眠不觉晓",
    "白日依山尽",
    "红豆生南国",
]

print("\n【经典诗词续写】\n")

for prompt in prompts:
    print(f"开头：{prompt}")
    print("-" * 40)
    
    poems = generate_poem(
        prompt=prompt,
        max_length=60,
        temperature=0.7,
        top_k=40,
        top_p=0.92,
        num_return_sequences=2
    )
    
    for i, poem in enumerate(poems, 1):
        # 提取从 prompt 开始的部分
        if prompt in poem:
            poem_content = poem.split(prompt)[-1][:100]
            print(f"版本{i}: {prompt}{poem_content}")
        else:
            print(f"版本{i}: {poem[:100]}")
    
    print()

# ============================================================================
# 第五部分：不同温度的效果对比
# ============================================================================

print("\n【不同 Temperature 对比】\n")

prompt = "春风又绿江南岸"
temperatures = [0.3, 0.7, 1.2]

print(f"开头：{prompt}\n")

for temp in temperatures:
    print(f"Temperature = {temp}:")
    print("-" * 40)
    
    poems = generate_poem(
        prompt=prompt,
        max_length=50,
        temperature=temp,
        top_k=50,
        num_return_sequences=1
    )
    
    poem = poems[0]
    if prompt in poem:
        poem_content = poem.split(prompt)[-1][:80]
        print(f"{prompt}{poem_content}")
    else:
        print(poem[:80])
    
    print()

# ============================================================================
# 第六部分：创作完整诗歌
# ============================================================================

print("\n【创作完整诗歌】\n")

themes = [
    ("春天", ["春", "花", "风", "暖"]),
    ("秋天", ["秋", "月", "叶", "凉"]),
    ("思念", ["思", "念", "远", "梦"]),
    ("山水", ["山", "水", "云", "静"]),
]

for theme_name, keywords in themes:
    print(f"主题：{theme_name}")
    print("-" * 40)
    
    # 用关键词作为提示
    prompt = random.choice(keywords)
    
    poems = generate_poem(
        prompt=prompt,
        max_length=80,
        temperature=0.75,
        top_k=45,
        top_p=0.93,
        num_return_sequences=1
    )
    
    print(f"《{theme_name}》")
    print(poems[0][:150])
    print()

# ============================================================================
# 第七部分：填词游戏
# ============================================================================

print("\n【填词游戏】\n")

# 给出上句，让 AI 对下句
pairs = [
    "白日依山尽",
    "举头望明月",
    "春眠不觉晓",
    "千山鸟飞绝",
]

print("对对联:\n")

for upper_line in pairs:
    print(f"上联：{upper_line}")
    
    lower_lines = generate_poem(
        prompt=upper_line,
        max_length=30,
        temperature=0.6,
        top_k=30,
        num_return_sequences=2
    )
    
    for i, line in enumerate(lower_lines, 1):
        # 尝试提取下联
        if upper_line in line:
            rest = line.split(upper_line)[-1].strip()[:20]
            # 取第一句
            if '。' in rest:
                lower = rest.split('。')[0]
            elif ',' in rest:
                lower = rest.split(',')[0]
            else:
                lower = rest[:10]
            print(f"下联{i}: {lower}")
    
    print()

# ============================================================================
# 第八部分：保存和导出
# ============================================================================

print("\n" + "=" * 60)
print("导出诗歌")
print("=" * 60)

# 生成几首完整的诗并保存
output_poems = []

for i in range(3):
    prompt = random.choice(["春", "月", "山", "水", "风", "花"])
    
    poems = generate_poem(
        prompt=prompt,
        max_length=100,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
    
    poem_text = f"《无题·其{i+1}》\n\n{poems[0]}"
    output_poems.append(poem_text)

# 保存到文件
with open('generated_poems.txt', 'w', encoding='utf-8') as f:
    for poem in output_poems:
        f.write(poem + "\n\n" + "-"*40 + "\n\n")

print(f"✓ 已生成 {len(output_poems)} 首诗歌")
print(f"✓ 已保存到 'generated_poems.txt'")

print("\n生成的诗歌预览:")
for poem in output_poems[:2]:
    print(poem[:200])
    print("...\n")

# ============================================================================
# 第九部分：交互式写诗
# ============================================================================

print("\n" + "=" * 60)
print("交互式写诗模式")
print("=" * 60)
print("""
现在你可以自己输入开头，让 AI 帮你写诗!

提示:
- 输入一个词或一句话
- AI 会续写成完整的诗
- 按 q 退出

示例:
输入：春花
输出：《春花》
      春花烂漫开满园，
      芬芳香气扑鼻来。
      ...
""")

# 取消下面注释即可启用交互模式
"""
while True:
    user_input = input("\n请输入开头 (或输入 q 退出): ").strip()
    
    if user_input.lower() == 'q':
        print("再见!")
        break
    
    if not user_input:
        print("请输入内容!")
        continue
    
    print("\n正在创作...\n")
    
    poems = generate_poem(
        prompt=user_input,
        max_length=120,
        temperature=0.75,
        top_k=50,
        top_p=0.93,
        num_return_sequences=3
    )
    
    print(f"《{user_input}》")
    for i, poem in enumerate(poems, 1):
        print(f"\n版本{i}:")
        print(poem[:150])
        print("...")
"""

print("\n🎉 恭喜你完成了写诗机器人!")
print("\n下一步可以尝试:")
print("  1. 调整参数看效果变化")
print("  2. 用 ChatGPT API 获得更好效果")
print("  3. 训练自己的诗歌模型")
print("  4. 做成 APP 或小程序")

# ============================================================================
# 第十部分：用 ChatGPT API (可选)
# ============================================================================

print("\n" + "=" * 60)
print("使用 ChatGPT API (高级)")
print("=" * 60)
print("""
如果你想获得更好的效果，可以用 OpenAI 的 ChatGPT API:

安装:
pip install openai

使用示例:
""")

code_example = """
import openai

# 设置 API key
openai.api_key = '你的 API_KEY'

# 调用 GPT-3.5
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "你是一位诗人"},
        {"role": "user", "content": "请以春天为主题写一首诗"}
    ]
)

print(response.choices[0].message.content)
"""

print(code_example)

print("""
优点:
✓ 效果更好 (GPT-4 级别)
✓ 更智能
✓ 理解能力更强

缺点:
❌ 需要付费
❌ 需要网络
❌ 速度较慢

建议:
- 学习阶段用本地 GPT-2
- 生产环境用 ChatGPT API
""")

---

## 🔗 相关链接

### 🌐 项目资源
- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **如果觉得有帮助，请给 GitHub 仓库 Star 支持！**

### 📚 学习路径
- [← Day23](../Day23/README.md)
- [→ Day25](../Day25/README.md)

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
