# Day24-Q3 - 文本生成策略

## 🎲 为什么需要不同的生成策略？

### 问题背景

```python
"""
GPT 生成文本时，每一步都要从词汇表中选择一个词。

词汇表大小: 50,257 个 token

问题:
- 如何选择下一个词?
- 每次都选概率最高的? (太 deterministic)
- 完全随机选择? (太 chaotic)
- 如何平衡质量和多样性?

答案:
使用不同的解码策略 (Decoding Strategies)
"""
```

### 生成过程的本质

```
每一步生成:
1. 模型输出所有 token 的概率分布
   P(token_1) = 0.01
   P(token_2) = 0.05
   P(token_3) = 0.60  ← 最高
   P(token_4) = 0.02
   ...

2. 根据某种策略选择一个 token

3. 将选中的 token 添加到序列中

4. 重复直到结束条件
```

## 🔍 主要生成策略

### 1. Greedy Search (贪心搜索)

**原理：**
```
每一步都选择概率最高的 token

示例:
步骤1: "今天" → 选择 "天气" (P=0.6)
步骤2: "今天天气" → 选择 "很好" (P=0.7)
步骤3: "今天天气很好" → 选择 "，" (P=0.8)
...
```

**代码实现：**
```python
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def greedy_search(prompt, max_length=50):
    """贪心搜索生成"""
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        # greedy_search=True 就是贪心搜索
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=False,  # 不采样，直接选最高概率
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 使用
result = greedy_search("今天天气")
print(result)
```

**优点：**
✅ 简单快速
✅ 每次都是最优选择
✅ 结果可复现

**缺点：**
❌ 容易陷入重复循环
❌ 缺乏多样性
❌ 可能错过更好的全局序列

**示例问题：**
```
输入: "我喜欢的颜色是"

Greedy 输出:
"我喜欢的颜色是蓝色。我喜欢的颜色是蓝色。我喜欢的颜色是蓝色..."

原因:
- "蓝色" 后面最可能还是"蓝色"
- 陷入局部最优
- 无法跳出循环
```

### 2. Beam Search (束搜索)

**原理：**
```
同时维护多个候选序列 (beam)
每一步扩展所有候选
保留概率最高的 k 个

示例 (beam_size=3):

步骤1:
候选1: "今天天气" (P=0.6)
候选2: "今天心情" (P=0.3)
候选3: "今天我们" (P=0.1)

步骤2: 每个候选扩展
候选1a: "今天天气很好" (P=0.6×0.7=0.42)
候选1b: "今天天气不错" (P=0.6×0.2=0.12)
候选2a: "今天心情很好" (P=0.3×0.5=0.15)
...

步骤3: 保留 top-3
候选1a: "今天天气很好" (P=0.42)
候选2a: "今天心情很好" (P=0.15)
候选1b: "今天天气不错" (P=0.12)
```

**代码实现：**
```python
def beam_search(prompt, max_length=50, num_beams=5):
    """束搜索生成"""
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,      # beam 数量
            early_stopping=True,       # 提前停止
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 使用
result = beam_search("今天天气", num_beams=5)
print(result)
```

**优点：**
✅ 考虑多个候选
✅ 比 greedy 更好
✅ 找到更优的全局序列

**缺点：**
❌ 计算量大 (beam 越大越慢)
❌ 仍然可能重复
❌ 缺乏创造性

**参数选择：**
```
num_beams 的影响:

beam=1: 等价于 greedy search
beam=3-5: 常用范围，平衡质量和速度
beam=10+: 质量更高，但速度慢
beam=50+: 边际效益递减

建议:
- 翻译任务: beam=5-10
- 摘要任务: beam=4-6
- 创意写作: 不用 beam search
```

### 3. Top-k Sampling

**原理：**
```
只从概率最高的 k 个 token 中随机采样

示例 (k=50):

原始分布:
token_1: P=0.30  ← top-50
token_2: P=0.25  ← top-50
token_3: P=0.20  ← top-50
...
token_50: P=0.001  ← top-50
token_51: P=0.0005  ✗ 被过滤
token_52: P=0.0003  ✗ 被过滤

重新归一化后采样:
从 top-50 中按概率随机选择
```

**代码实现：**
```python
def top_k_sampling(prompt, max_length=50, top_k=50, temperature=1.0):
    """Top-k 采样"""
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,           # 启用采样
            top_k=top_k,              # top-k 值
            temperature=temperature,  # 温度
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 不同 k 值的对比
print("k=10:", top_k_sampling("故事开头", top_k=10))
print("k=50:", top_k_sampling("故事开头", top_k=50))
print("k=100:", top_k_sampling("故事开头", top_k=100))
```

**优点：**
✅ 避免低概率的荒谬词
✅ 保持一定多样性
✅ 可控性强

**缺点：**
❌ k 值固定，不够灵活
❌ 可能截断有用的词
❌ 对 k 值敏感

**k 值选择指南：**
```
k=1: 等价于 greedy search (无多样性)
k=10: 非常保守，接近确定性
k=40-50: 推荐默认值，平衡质量和多样性
k=100: 更多样，但可能降低质量
k=1000+: 几乎等于无限制采样

应用场景:
- 事实性内容: k=10-20 (保守)
- 创意写作: k=50-100 (多样)
- 对话系统: k=40-60 (平衡)
```

### 4. Top-p (Nucleus) Sampling ⭐推荐

**原理：**
```
从累积概率达到 p 的最小 token 集合中采样

示例 (p=0.9):

排序后的概率:
token_1: P=0.30  → 累积: 0.30
token_2: P=0.25  → 累积: 0.55
token_3: P=0.20  → 累积: 0.75
token_4: P=0.15  → 累积: 0.90  ← 达到 0.9，停止
token_5: P=0.05  ✗ 被过滤
token_6: P=0.03  ✗ 被过滤

从 {token_1, token_2, token_3, token_4} 中采样
```

**代码实现：**
```python
def top_p_sampling(prompt, max_length=50, top_p=0.9, temperature=1.0):
    """Top-p (核) 采样"""
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_p=top_p,              # top-p 值
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 不同 p 值的对比
print("p=0.7:", top_p_sampling("写一首诗", top_p=0.7))
print("p=0.9:", top_p_sampling("写一首诗", top_p=0.9))
print("p=0.95:", top_p_sampling("写一首诗", top_p=0.95))
```

**优点：**
✅ 动态调整候选集大小
✅ 比 top-k 更灵活
✅ 适应不同的概率分布
✅ OpenAI 推荐使用

**缺点：**
❌ p 值需要调优
❌ 极端情况下仍可能不理想

**p 值选择指南：**
```
p=0.1-0.5: 非常保守，高质量但缺乏多样性
p=0.7-0.8: 保守，适合事实性内容
p=0.9: 推荐默认值，良好平衡
p=0.95: 更多样，适合创意任务
p=0.99: 几乎无限制

优势 vs Top-k:
- Top-k: 固定数量，不考虑概率分布
- Top-p: 动态数量，适应分布形状

例子:
平坦分布 (多个词概率相近):
  Top-k=50: 选50个
  Top-p=0.9: 可能选100个 (因为每个概率都不高)

尖锐分布 (少数词概率很高):
  Top-k=50: 选50个 (包括很多低概率词)
  Top-p=0.9: 可能只选5个 (因为前几个就够0.9了)
```

### 5. Temperature Scaling (温度调节)

**原理：**
```
通过温度参数控制概率分布的平滑程度

公式:
P'(token_i) = exp(log(P(token_i)) / T) / Σ exp(log(P(token_j)) / T)

T < 1: 分布更尖锐 (更 deterministic)
T = 1: 原始分布
T > 1: 分布更平坦 (更 random)
```

**可视化：**
```
原始概率:
A: 0.5, B: 0.3, C: 0.15, D: 0.05

T=0.5 (低温):
A: 0.75, B: 0.20, C: 0.04, D: 0.01
→ 更倾向于选 A

T=1.0 (正常):
A: 0.5, B: 0.3, C: 0.15, D: 0.05
→ 原始分布

T=2.0 (高温):
A: 0.38, B: 0.28, C: 0.21, D: 0.13
→ 更均匀，更多样
```

**代码实现：**
```python
def temperature_sampling(prompt, max_length=50, temperature=1.0):
    """温度采样"""
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 不同温度的效果
print("T=0.2:", temperature_sampling("故事", temperature=0.2))
print("T=0.7:", temperature_sampling("故事", temperature=0.7))
print("T=1.0:", temperature_sampling("故事", temperature=1.0))
print("T=1.5:", temperature_sampling("故事", temperature=1.5))
```

**温度选择指南：**
```
T=0.1-0.3: 非常确定，几乎总是选最高概率
  适用: 代码生成、数学计算、事实问答

T=0.5-0.7: 较为确定，少量变化
  适用: 技术文档、商务邮件、新闻报道

T=0.8-1.0: 平衡，推荐默认值
  适用: 一般对话、文章写作

T=1.2-1.5: 更多样，更有创意
  适用: 创意写作、诗歌、头脑风暴

T=1.5-2.0: 高度随机，可能不连贯
  适用: 艺术实验、抽象创作

注意:
- T=0: 等价于 greedy search
- T→∞: 等价于均匀随机
- 通常与 top-k/top-p 结合使用
```

## 🎯 组合策略

### 最佳实践

```python
"""
推荐的组合:

1. 高质量生成 (默认)
top_p=0.9, temperature=0.7

2. 创意写作
top_p=0.95, temperature=1.0

3. 事实性内容
top_p=0.8, temperature=0.5

4. 代码生成
top_k=50, temperature=0.2

5. 对话系统
top_p=0.9, temperature=0.8
"""

# 示例: 组合使用
def high_quality_generation(prompt, max_length=100):
    """高质量生成配置"""
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_p=0.9,                # top-p 采样
            temperature=0.7,          # 适中温度
            repetition_penalty=1.2,   # 重复惩罚
            no_repeat_ngram_size=3,   # 禁止3-gram重复
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)
```

## 🛠️ 高级控制技术

### 1. Repetition Penalty (重复惩罚)

**问题：**
```
模型容易重复相同的短语:

"我喜欢编程。我喜欢编程。我喜欢编程..."
```

**解决方案：**
```python
output = model.generate(
    input_ids,
    repetition_penalty=1.2,  # >1 表示惩罚重复
    # 1.0: 无惩罚
    # 1.2: 适度惩罚 (推荐)
    # 1.5: 强烈惩罚
    # <1.0: 鼓励重复 (很少用)
)
```

**原理：**
```
如果 token 已经出现过:
P_new(token) = P_original(token) / penalty

例如:
原始概率: "编程" P=0.3
如果"编程"已出现:
P_new = 0.3 / 1.2 = 0.25

降低了重复的可能性
```

### 2. No Repeat N-gram

**原理：**
```
禁止生成重复的 n-gram

设置 no_repeat_ngram_size=3:
- 禁止重复任何 3-gram
- "我喜欢编程" 出现后
- 不能再生成"我喜欢编程"
```

**代码：**
```python
output = model.generate(
    input_ids,
    no_repeat_ngram_size=3,  # 禁止3-gram重复
)
```

### 3. Length Penalty (长度惩罚)

**用途：**
控制生成文本的长度

```python
output = model.generate(
    input_ids,
    length_penalty=1.0,  # 默认值
    # <1.0: 鼓励更短
    # >1.0: 鼓励更长
)
```

**应用：**
```
摘要生成: length_penalty=0.8 (偏好简短)
文章写作: length_penalty=1.2 (偏好详细)
```

### 4. Forced Tokens (强制 token)

**用途：**
确保某些词出现在输出中

```python
output = model.generate(
    input_ids,
    forced_bos_token_id=tokenizer.bos_token_id,
    forced_eos_token_id=tokenizer.eos_token_id,
)
```

## 📊 策略对比总结

| 策略 | 多样性 | 质量 | 速度 | 适用场景 |
|------|--------|------|------|----------|
| Greedy | 低 | 中 | 快 | 代码、事实 |
| Beam Search | 低 | 高 | 慢 | 翻译、摘要 |
| Top-k | 中 | 中 | 快 | 通用 |
| Top-p | 中 | 高 | 快 | **推荐默认** |
| Temperature | 可调 | 可调 | 快 | 配合其他策略 |

## 💡 选择指南

### 根据任务选择

```
1. 代码生成
   策略: top_k=50, temperature=0.2
   原因: 需要准确性，不需要太多创意

2. 翻译
   策略: beam_search(num_beams=5)
   原因: 需要高质量的对应翻译

3. 摘要
   策略: top_p=0.8, temperature=0.7
   原因: 平衡准确性和简洁性

4. 创意写作
   策略: top_p=0.95, temperature=1.0
   原因: 需要多样性和创造力

5. 对话系统
   策略: top_p=0.9, temperature=0.8
   原因: 自然流畅，有一定变化

6. 事实问答
   策略: top_p=0.8, temperature=0.5
   原因: 优先准确性
```

### 调试技巧

```python
"""
如何找到最佳参数?

1. 从小范围开始
   temperature: 0.7-1.0
   top_p: 0.8-0.95

2. 观察生成结果
   - 太重复? → 增加 temperature 或 repetition_penalty
   - 太随机? → 降低 temperature 或 top_p
   - 质量差? → 降低 temperature

3. 多次采样比较
   for i in range(5):
       result = generate(temperature=0.8)
       print(f"Sample {i}: {result}")

4. A/B 测试
   - 对不同用户使用不同参数
   - 收集反馈
   - 选择最佳配置
"""
```

## 🎓 学习要点总结

### 核心策略

1. **Greedy Search**: 简单快速，但缺乏多样性
2. **Beam Search**: 质量高，但计算量大
3. **Top-k Sampling**: 固定候选集，可控
4. **Top-p Sampling**: 动态候选集，推荐默认
5. **Temperature**: 调节随机性，配合使用

### 最佳实践

1. **默认配置**: top_p=0.9, temperature=0.7
2. **重复控制**: repetition_penalty=1.2
3. **根据任务调整**: 代码保守，创意开放
4. **多次采样**: 选择最佳结果

### 高级技巧

1. **组合策略**: top-p + temperature + repetition penalty
2. **长度控制**: length_penalty
3. **N-gram 限制**: no_repeat_ngram_size
4. **调试方法**: 多次采样，A/B 测试

## 🚀 下一步

现在我们掌握了文本生成策略，接下来让我们深入学习 Prompt Engineering 的技巧。

---

**下一步：** [Day24-Q4 - Prompt Engineering](./Day24-Q4%20-%20Prompt%20Engineering.md)
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
