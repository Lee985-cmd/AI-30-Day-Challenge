# 📝 Day 23 费曼学习法版 - GPT 和文本生成

---

## 🌟 完整项目和代码

本教程是 **AI 入门 30 天挑战** 系列的一部分！

- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **欢迎 Star 支持！**

---



> **Week 4 第二天：AI 如何写文章！**  
> **GPT 架构解析 + 自动生成股票分析报告！**  
> **每个概念都解释！每行代码都说明白！**  
> **预计时间：3-4 小时（含费曼输出练习）**

---

## 📖 第 1 步：快速复习昨天的内容（30 分钟）

### 费曼输出 #0：考考你

**合上教程，尝试回答：**

```
□ Multi-Head Attention 为什么比 Single-Head 好？
□ Q、K、V 在 Attention 中各起什么作用？
□ 为什么要对 Attention 权重进行缩放 (除以 sqrt(d_k))？
□ Attention 可视化能帮助我们理解什么？
□ 如果用 GPT 生成股票评论，你会怎么设计？
```

**⏰ 时间：25 分钟**

如果能答出 80% 以上，我们开始今天的 GPT 之旅！如果不够，花 5 分钟翻一下 Day22 的笔记。

---

## 🤔 第 2 步：GPT 到底是什么？（60 分钟）

### 说人话版本

想象一个作家在写小说：

传统写作：先想好整个故事大纲，然后一章一章写，可以跳着写。

GPT 写作（自回归）：
- 只写第一个字："我"
- 根据"我"预测下一个字："爱"
- 现在有"我爱"，再预测下一个："你"
- 就这样一个字一个字写下去...

特点：从左到右，顺序生成；每个字只能看到前面的内容；不能偷看后面的字。

生活中的例子：接龙游戏

你和朋友玩词语接龙：
- 你："人工智..."
- 朋友："能！"
- 你："能力..."
- 朋友："力..."

GPT 就是这样：输入"今天天气"→GPT："真"→输入"今天天气真"→GPT："好"→一直生成到结束标记。

### GPT vs BERT 对比

BERT（学霸型）：
- 双向注意力（能看到全文）
- 擅长理解（完形填空高手）
- 适合：分类、问答、情感分析

GPT（作家型）：
- 单向注意力（只能看前面）
- 擅长创作（续写专家）
- 适合：生成、翻译、对话

举个例子：完成句子"我喜欢吃___"

BERT 的做法：两边一起看，"我 ___ 吃苹果"，根据整句判断填"喜欢"

GPT 的做法：从左到右看，"我"→"喜欢"→"吃"→"苹果"，根据上文预测下文

---

## 🎯 费曼输出 #1：向小白解释 GPT

### 任务 1：创造多个比喻

场景 A：向小学生解释
GPT = 故事大王，你开个头"从前有座山..."，它接着讲"山里有个庙..."，就这样一句接一句，故事越来越长。

场景 B：向股民解释
用写股评的方式，给个开头"今日大盘..."，GPT 续写"震荡上行..."，继续"成交量放大..."，最后"建议关注科技股..."，一篇股评就完成了。

场景 C：向厨师解释
GPT = 自动炒菜机，你放食材（输入提示词），机器翻炒（计算概率），出锅（生成文本），一道菜（一篇文章）完成了。

要求：每个场景都要详细说明

⏰ 时间：20 分钟

---

### 💡 卡壳检查点

如果你在解释时卡住了：
- 我说不清楚"自回归"是什么意思
- 我不知道如何解释"只能看前面"的限制
- 我只能说"AI 写作"，但不能说明白怎么写的

这很正常！标记下来，继续往下看，然后重新尝试解释！

提示：
- 自回归 = 自己预测自己的下一步
- 单向注意 = 只能回顾过去，不能预知未来
- 概率采样 = 从最可能的词中选一个

---

## 🔬 第 3 步：GPT 的核心机制详解（70 分钟）

### 机制 1：自回归生成

自回归 (Autoregressive) 是什么？

公式：P(句子) = P(w1) × P(w2|w1) × P(w3|w1,w2) × ...

说人话：
生成句子的概率 = 
  第一个字的概率 ×
  第二个字的概率 (给定第一个字) ×
  第三个字的概率 (给定前两个字) ×
  ...

例子：生成"我爱你"

步骤 1: 预测第一个字，P("我") = 0.8

步骤 2: 给定"我"，预测第二个字，P("爱" | "我") = 0.9

步骤 3: 给定"我爱"，预测第三个字，P("你" | "我爱") = 0.85

总概率 = 0.8 × 0.9 × 0.85 = 0.612

### 机制 2：Masked Attention

Masked Attention = 不许作弊的注意力

问题：如果生成时能看到未来的词，就会作弊！就像考试提前知道答案

解决方案：用 Mask 遮住后面的词，每个位置只能看到它之前的内容

演示：

句子："我 爱 你"

生成"我"时：我 [被遮住的] [被遮住的]，只能看到自己

生成"爱"时：我 爱 [被遮住的]，能看到"我"，但看不到后面

生成"你"时：我 爱 你，能看到前面所有

这样保证公平！

### 完整代码实现

```python
import torch
import torch.nn as nn
import math
import numpy as np

print("=" * 60)
print("📝 GPT 文本生成从零实现")
print("=" * 60)

class MaskedSelfAttention(nn.Module):
    """带掩码的自注意力 - GPT 的核心"""
    
    def __init__(self, embed_size, heads):
        super(MaskedSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "embed_size 必须能被 heads 整除"
        
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
        self.register_buffer("mask", None)
        
        print(f"\n✓ Masked Self-Attention 初始化完成")
        print(f"  嵌入维度：{embed_size}")
        print(f"  注意力头数：{heads}")
    
    def forward(self, x):
        batch_size, seq_len, embed_size = x.shape
        
        if self.mask is None or self.mask.shape[1] != seq_len:
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
            self.mask = mask.to(x.device)
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_size)
        output = self.fc_out(out)
        
        return output, attention

class SimpleGPT(nn.Module):
    """简化的 GPT 模型"""
    
    def __init__(self, vocab_size, embed_size=512, num_layers=6, heads=8, max_len=512):
        super(SimpleGPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_encoding = nn.Parameter(torch.randn(1, max_len, embed_size))
        self.layers = nn.ModuleList([MaskedSelfAttention(embed_size, heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, vocab_size)
        
        print(f"\n✓ SimpleGPT 模型初始化完成")
        print(f"  词表大小：{vocab_size}")
        print(f"  Transformer 层数：{num_layers}")
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.token_embedding(x) + self.position_encoding[:, :seq_len, :]
        for layer in self.layers:
            x, _ = layer(x)
        logits = self.fc_out(x)
        return logits
    
    def generate(self, start_tokens, max_len=50, temperature=1.0):
        if isinstance(start_tokens, list):
            current = torch.tensor([start_tokens], dtype=torch.long)
        else:
            current = start_tokens
        
        generated = current.clone()
        
        with torch.no_grad():
            for _ in range(max_len):
                logits = self.forward(current)
                next_token_logits = logits[:, -1, :]
                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                current = generated
                if next_token.item() == 0:
                    break
        
        return generated

print("\n" + "=" * 60)
print("【演示】股票评论自动生成")
print("=" * 60)

vocab = {'<pad>': 0, '我': 1, '觉得': 2, '这个': 3, '股票': 4, '会': 5, '上涨': 6, '下跌': 7, '因为': 8, '公司': 9, '业绩': 10, '很好': 11, '不好': 12, '市场': 13, '前景': 14, '广阔': 15, '担忧': 16, '推荐': 17, '买入': 18, '卖出': 19, '持有': 20}

vocab_size = len(vocab)
idx_to_word = {idx: word for word, idx in vocab.items()}

model = SimpleGPT(vocab_size=vocab_size, embed_size=128, num_layers=2, heads=4)

test_prompts = [[1, 2, 3, 4], [21, 22], [9, 10]]
prompt_texts = ["我觉得这个股票", "今天大盘", "公司业绩"]

for i, (prompt, prompt_text) in enumerate(zip(test_prompts, prompt_texts), 1):
    print(f"\n【生成 {i}】提示词：{prompt_text}")
    generated = model.generate(prompt, max_len=15, temperature=0.8)
    generated_words = [idx_to_word[idx.item()] for idx in generated[0]]
    print(f"生成结果：{''.join(generated_words)}")

print("\n💡 说明：由于模型未训练，生成内容是随机的")
print("真实场景需要用大量文本训练")

print("\n🎊 GPT 文本生成演示完成!")
```

**按 Shift + Enter 运行！**

---

## 🎯 费曼输出 #2：深入理解技术

### 任务 1：解释技术细节

思考题：
1. 为什么 GPT 只能用 Masked Attention？
2. 温度系数 (temperature) 的作用是什么？
3. Top-p 采样相比贪婪搜索有什么优势？
4. GPT 适合做什么类型的任务？不适合做什么？

### 任务 2：设计应用场景

场景：你要用 GPT 帮助股民写日报

要求：
1. 设计输入格式（提供哪些信息）
2. 设计输出格式（报告包含哪些部分）
3. 选择合适的温度系数并说明理由
4. 如何保证生成内容的准确性

⏰ 时间：30 分钟

---

### 💡 卡壳检查点

- 我解释不清 Masked Attention 的必要性
- 我说不明白温度系数的作用
- 我不能设计实际的应用方案

提示：
- Masked = 不许作弊（不能看未来）
- 温度 = 控制随机性
- Top-p = 从靠谱的词里选

---

## 💻 第 4 步：实战项目（80 分钟）

### 项目：自动生成股票晨会纪要

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("=" * 60)
print("🌅 股票晨会纪要自动生成系统")
print("=" * 60)

try:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ 模型就绪")
    
    def generate_morning_briefing(news_items, focus_points):
        prompt = "Stock Morning Briefing:\n\n"
        prompt += "Overnight News:\n"
        for i, news in enumerate(news_items, 1):
            prompt += f"{i}. {news}\n"
        
        prompt += "\nToday's Focus:\n"
        for i, point in enumerate(focus_points, 1):
            prompt += f"{i}. {point}\n"
        
        prompt += "\nMarket Outlook and Trading Strategy:\n"
        
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        outputs = model.generate(inputs, max_length=400, temperature=0.7, top_p=0.9, repetition_penalty=1.2, do_sample=True)
        briefing = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\n" + "=" * 60)
        print(briefing)
        print("=" * 60)
        
        return briefing
    
    news = ["Fed announces interest rate decision", "Tesla reports Q4 earnings beat", "Oil prices surge 5%", "Tech stocks rally"]
    focus = ["Market reaction to Fed", "EV sector outlook", "Energy sector opportunities", "Key technical levels"]
    
    briefing = generate_morning_briefing(news, focus)
    print("\n✅ 晨会纪要生成成功!")
    print("\n💡 使用建议:")
    print("  1. 人工校对关键数据和观点")
    print("  2. 补充具体个股推荐")
    print("  3. 添加风险提示")
    
except Exception as e:
    print(f"模型加载失败：{e}")
    print("重点理解思路即可")

print("\n🎊 晨会纪要系统完成!")
```

---

## 🎉 今日费曼总结（30 分钟）⭐

### 完整的费曼学习流程

第 1 步：回顾今天的内容（5 分钟）
- GPT 的自回归生成机制
- Masked Attention 原理
- 文本生成温度和采样
- 股票报告自动生成

第 2 步：合上教程，尝试完整教授（15 分钟）⭐

任务：假装你在给一个完全不懂的人上第二十三堂课

要覆盖：
1. GPT 是怎么写文章的（用至少 2 个比喻）
2. 为什么 GPT 只能用单向 Attention
3. 温度和采样的作用
4. 演示股票报告自动生成

方式：写一篇 800 字左右的文章，或录一段 10-15 分钟的视频

第 3 步：标记卡壳点（5 分钟）

我今天卡壳的地方：
□ _________________________________
□ _________________________________

第 4 步：针对性复习（5 分钟）

回到教程中卡壳的地方，重新学习，然后再次尝试解释！

---

## 📝 费曼学习笔记模板

```
╔═══════════════════════════════════════════════════╗
║         Day 23 费曼学习笔记                       ║
╠═══════════════════════════════════════════════════╣
║ 日期：__________                                  ║
║ 学习时长：__________                              ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║ 1. 我向小白解释了：                               ║
║ _______________________________________________  ║
║                                                   ║
║ 2. 我卡壳的地方：                                 ║
║ □ _____________________________________________  ║
║                                                   ║
║ 3. 我的通俗比喻：                                 ║
║ • GPT 写作就像 ______                             ║
║ • Masked Attention 就像 ______                    ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
```

---

## 📊 今日总结

### ✅ 你今天学到了：

1. GPT 核心原理
   - 自回归生成
   - Masked Attention
   - 从左到右的顺序

2. 文本生成技术
   - 温度系数控制
   - Top-p 采样
   - Repetition Penalty

3. 实战应用
   - 股票报告生成
   - 晨会纪要自动化

4. 费曼输出能力 ⭐
   - 能用比喻解释 GPT
   - 能向小白说明生成过程

---

## 🎁 明日预告

明天你将学习：情感分析实战

内容：
- BERT 情感分析深入
- 股票论坛情绪指标
- 舆情监控系统
- 实战：雪球评论分析

准备好继续 Day24 吗？那里我们会教你如何用 AI 监控股票论坛的情绪，发现投资机会！🚀

---

## 🔗 相关链接

### 🌐 项目资源
- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **如果觉得有帮助，请给 GitHub 仓库 Star 支持！**

### 📚 学习路径
- [← Day22](../Day22/README.md)
- [→ Day24](../Day24/README.md)

---

*本教程属于 [AI 入门 30 天挑战](https://github.com/Lee985-cmd/AI-30-Day-Challenge) 系列*
