# Day24-Q6 - 常见问题和最佳实践

## ❓ 常见问题解答

### Q1: 生成的文本重复怎么办？

**问题：**
```
AI 总是重复相同的内容:
"我喜欢编程。我喜欢编程。我喜欢编程..."
```

**解决方案：**

```python
# 方案 1: 增加 repetition_penalty
output = model.generate(
    input_ids,
    repetition_penalty=1.2,  # 从 1.0 增加到 1.2-1.5
)

# 方案 2: 使用 no_repeat_ngram_size
output = model.generate(
    input_ids,
    no_repeat_ngram_size=3,  # 禁止 3-gram 重复
)

# 方案 3: 降低 temperature
output = model.generate(
    input_ids,
    temperature=0.5,  # 更确定性
)

# 方案 4: 组合使用（推荐）
output = model.generate(
    input_ids,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    temperature=0.7,
    top_p=0.9,
)
```

### Q2: 如何让生成结果更稳定？

**问题：**
```
同样的 prompt，每次运行结果差异很大
```

**解决方案：**

```python
# 方案 1: 设置随机种子
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 方案 2: 使用 greedy search 或 beam search
output = model.generate(
    input_ids,
    do_sample=False,  # 不采样，完全确定性
)

# 方案 3: 降低 temperature
output = model.generate(
    input_ids,
    temperature=0.1,  # 接近确定性
)

# 方案 4: 多次采样取最佳
results = []
for _ in range(5):
    result = generate_with_sampling()
    results.append(result)

# 选择最一致的或评分最高的
best = select_best(results)
```

### Q3: 生成的内容不符合事实怎么办？

**问题：**
```
AI 编造事实 (幻觉问题):
问: "谁在 2025 年获得了诺贝尔奖?"
答: "张三获得了诺贝尔物理学奖" (完全是编的)
```

**解决方案：**

```python
"""
方案 1: RAG (检索增强生成)

步骤:
1. 从可靠来源检索相关信息
2. 将信息作为上下文提供给 AI
3. 要求 AI 基于上下文回答

实现:
"""

from transformers import GPT2Tokenizer, GPT2LMHeadModel

def rag_qa(question, context):
    """基于上下文的问答"""
    
    prompt = f"""
基于以下上下文回答问题。如果上下文中没有答案，请说"我不知道"。

上下文:
{context}

问题: {question}

答案:
"""
    
    # 生成答案
    # ... (省略具体实现)
    
    return answer

"""
方案 2: 引用来源

Prompt:
"回答问题时，请引用信息来源。如果不确定，请说明。"

方案 3: 事实核查

后处理步骤:
1. 提取关键事实
2. 与知识库对比
3. 标记可疑内容
4. 人工审核

方案 4: 限制范围

Prompt:
"只回答你确定知道的事实。不要猜测或编造。"
"""
```

### Q4: 如何控制生成长度？

**问题：**
```
有时太短，有时太长，难以控制
```

**解决方案：**

```python
# 方案 1: 设置 max_length 和 min_length
output = model.generate(
    input_ids,
    max_length=100,  # 最大长度
    min_length=50,   # 最小长度
)

# 方案 2: 使用 length_penalty
output = model.generate(
    input_ids,
    length_penalty=0.8,  # <1 鼓励更短
    # length_penalty=1.2,  # >1 鼓励更长
)

# 方案 3: 在 prompt 中明确指定
prompt = "用不超过 100 字总结以下内容:"

# 方案 4: 后处理截断
text = generate_text()
words = text.split()[:100]  # 保留前 100 个词
truncated = ' '.join(words)
```

### Q5: 如何提高生成速度？

**问题：**
```
生成速度慢，影响用户体验
```

**解决方案：**

```python
"""
方案 1: 使用更小的模型

# GPT-2 Small (124M 参数) vs GPT-2 XL (1.5B 参数)
model = GPT2LMHeadModel.from_pretrained('gpt2')  # Small
# vs
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')  # XL

速度提升: 5-10 倍

方案 2: 批量处理

# 一次生成多个序列
batch_input = tokenizer.batch_encode_plus(
    ["prompt1", "prompt2", "prompt3"],
    padding=True,
    return_tensors='pt'
)
outputs = model.generate(**batch_input)

方案 3: 量化加速

from transformers import quantization
quantized_model = quantization.quantize_model(
    model,
    load_in_8bit=True  # 8-bit 量化
)

速度提升: 2-3 倍
内存减少: 50%

方案 4: 使用 KV Cache

# GPT-2 默认启用 KV cache
# 避免重复计算前面的 token

方案 5: GPU 加速

# 确保使用 GPU
model.to('cuda')
input_ids = input_ids.to('cuda')

方案 6: 优化生成参数

output = model.generate(
    input_ids,
    max_length=50,      # 减小最大长度
    num_beams=3,        # 减少 beam 数量
    early_stopping=True, # 提前停止
)
"""
```

### Q6: 如何处理敏感内容？

**问题：**
```
AI 可能生成不当、有害或偏见的内容
```

**解决方案：**

```python
"""
方案 1: 内容过滤

使用开源过滤工具:
"""

from detoxify import Detoxify

def filter_content(text):
    """检测有毒内容"""
    model = Detoxify('original')
    results = model.predict(text)
    
    # 检查各类毒性
    toxic_threshold = 0.5
    if any(v > toxic_threshold for v in results.values()):
        return False, results
    
    return True, results

"""
方案 2: Prompt 约束

prompt = '''
请遵守以下准则:
- 不使用歧视性语言
- 不生成暴力内容
- 尊重所有群体
- 保持专业和礼貌

任务: {task}
'''

方案 3: 后处理审核

步骤:
1. 关键词过滤
2. 情感分析
3. 毒性检测
4. 人工审核 (高风险内容)

方案 4: RLHF 对齐

- 使用经过人类反馈优化的模型
- 如: ChatGPT, Claude
- 已经过安全对齐

方案 5: 多层防护

架构:
用户输入 → 输入过滤 → AI 生成 → 输出过滤 → 人工审核 → 最终输出
"""
```

## 🎯 最佳实践总结

### 1. Prompt 设计最佳实践

```python
"""
✅ DO:

1. 清晰具体
prompt = "写一篇 300 字的科技新闻摘要"

2. 提供示例
prompt = """
示例:
输入: xxx
输出: yyy

现在处理:
输入: zzz
"""

3. 设定角色
prompt = "你是一位经验丰富的医生..."

4. 指定格式
prompt = "用 JSON 格式输出..."

5. 迭代优化
- 测试不同版本
- 收集反馈
- 持续改进

❌ DON'T:

1. 模糊不清
prompt = "写点什么"

2. 假设 AI 知道上下文
prompt = "那个怎么做?"

3. 一次性给太多指令
prompt = "做 A, B, C, D, E..." (分解任务)

4. 忽略边界情况
- 测试各种输入
- 考虑异常情况

5. 一成不变
- 根据结果调整
- 适应新需求
"""
```

### 2. 生成策略最佳实践

```python
"""
默认配置 (适用于大多数场景):

output = model.generate(
    input_ids,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    repetition_penalty=1.2,
    max_length=100,
)

特定场景调整:

1. 代码生成
top_k=50, temperature=0.2

2. 创意写作
top_p=0.95, temperature=1.0

3. 事实问答
top_p=0.8, temperature=0.5

4. 对话系统
top_p=0.9, temperature=0.8

5. 翻译
beam_search(num_beams=5)
"""
```

### 3. 性能优化最佳实践

```python
"""
开发阶段:
- 使用小模型快速迭代
- GPT-2 Small (124M)

测试阶段:
- 中等模型验证效果
- GPT-2 Medium (355M)

生产环境:
- 大模型保证质量
- GPT-2 XL (1.5B) 或 GPT-3

优化技巧:

1. 缓存常用结果
cache = {}
def cached_generate(prompt):
    if prompt in cache:
        return cache[prompt]
    result = generate(prompt)
    cache[prompt] = result
    return result

2. 异步处理
import asyncio
async def async_generate(prompts):
    tasks = [generate(p) for p in prompts]
    return await asyncio.gather(*tasks)

3. 负载均衡
# 多个模型实例
# 分散请求压力

4. 监控和日志
- 记录生成时间
- 跟踪 token 使用
- 监控错误率
"""
```

### 4. 质量控制最佳实践

```python
"""
多层次质量控制:

第 1 层: Prompt 设计
- 清晰的指令
- 适当的约束
- 有效的示例

第 2 层: 生成策略
- 合适的解码方法
- 优化的参数
- 重复控制

第 3 层: 自动评估
- 困惑度检查
- 毒性检测
- 事实核查

第 4 层: 人工审核
- 抽样检查
- 高风险内容必审
- 持续改进

评估指标:

1. 准确性
- 事实正确性
- 逻辑一致性

2. 相关性
- 与 prompt 的相关度
- 主题一致性

3. 流畅性
- 语法正确
- 表达自然

4. 安全性
- 无有害内容
- 无偏见
- 符合伦理

5. 有用性
- 满足用户需求
- 实际价值
"""
```

## 🔧 调试工具箱

### 1. 生成过程可视化

```python
def visualize_generation(prompt, max_length=20):
    """可视化生成过程"""
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    print(f"Prompt: {prompt}\n")
    print("生成过程:")
    print("-" * 60)
    
    current_ids = input_ids
    for step in range(max_length):
        with torch.no_grad():
            outputs = model(current_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # 获取概率分布
            probs = torch.softmax(next_token_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs[0], 5)
            
            # 显示 top-5 候选
            print(f"Step {step + 1}:")
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                token = tokenizer.decode([idx])
                marker = " ← 选中" if i == 0 else ""
                print(f"  {i+1}. '{token}' (P={prob:.3f}){marker}")
            
            # 选择最高概率的 token
            next_token = top_indices[0].unsqueeze(0).unsqueeze(0)
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            
            # 显示当前生成的文本
            current_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
            print(f"  当前文本: {current_text}")
            print()
    
    final_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
    print("=" * 60)
    print(f"最终结果:\n{final_text}")

# 使用
visualize_generation("今天天气")
```

### 2. 参数敏感性分析

```python
def parameter_sensitivity_analysis(prompt):
    """分析参数对生成的影响"""
    
    temperatures = [0.2, 0.5, 0.8, 1.0, 1.5]
    top_ps = [0.7, 0.9, 0.95]
    
    print("参数敏感性分析")
    print("=" * 80)
    
    for temp in temperatures:
        for top_p in top_ps:
            output = model.generate(
                tokenizer.encode(prompt, return_tensors='pt'),
                max_length=50,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id
            )
            
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            print(f"\nT={temp}, P={top_p}:")
            print(text[:100] + "...")
    
    print("\n" + "=" * 80)
    print("观察不同参数组合的效果，选择最适合的配置")

# 使用
parameter_sensitivity_analysis("写一个故事")
```

### 3. 批量测试框架

```python
class BatchTester:
    """批量测试框架"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def run_tests(self, test_cases, config):
        """运行批量测试"""
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            prompt = test_case['prompt']
            expected = test_case.get('expected', None)
            
            # 生成
            output = self.model.generate(
                self.tokenizer.encode(prompt, return_tensors='pt'),
                **config
            )
            
            generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # 评估
            score = self.evaluate(generated, expected)
            
            results.append({
                'id': i,
                'prompt': prompt,
                'generated': generated,
                'expected': expected,
                'score': score
            })
            
            print(f"Test {i+1}/{len(test_cases)}: Score={score:.2f}")
        
        # 统计
        avg_score = sum(r['score'] for r in results) / len(results)
        print(f"\n平均分数: {avg_score:.2f}")
        
        return results
    
    def evaluate(self, generated, expected):
        """简单的评估函数"""
        if expected is None:
            return 1.0  # 无期望值，默认满分
        
        # 简单的重叠度计算
        gen_words = set(generated.lower().split())
        exp_words = set(expected.lower().split())
        
        if not exp_words:
            return 0.0
        
        overlap = len(gen_words.intersection(exp_words))
        return overlap / len(exp_words)


# 使用示例
test_cases = [
    {'prompt': '1+1=', 'expected': '2'},
    {'prompt': '法国的首都是', 'expected': '巴黎'},
    # ... 更多测试用例
]

config = {
    'max_length': 50,
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 0.9,
}

tester = BatchTester(model, tokenizer)
results = tester.run_tests(test_cases, config)
```

## 📚 学习资源

### 官方文档

1. **Hugging Face Transformers**
   - https://huggingface.co/transformers/
   - 完整的 API 文档
   - 示例代码

2. **OpenAI API**
   - https://platform.openai.com/docs
   - API 参考
   - 最佳实践

3. **GPT 技术报告**
   - GPT-2: Language Models are Unsupervised Multitask Learners
   - GPT-3: Language Models are Few-Shot Learners

### 社区资源

1. **Awesome Prompt Engineering**
   - GitHub 上的精选资源列表
   - Prompt 模板集合

2. **Learn Prompting**
   - https://learnprompting.org/
   - 免费的 Prompt Engineering 课程

3. **Hugging Face Course**
   - https://huggingface.co/course
   - 免费的 NLP 课程

### 研究论文

1. **"Attention Is All You Need"**
   - Transformer 原论文

2. **"Language Models are Few-Shot Learners"**
   - GPT-3 论文

3. **"Chain-of-Thought Prompting Elicits Reasoning"**
   - CoT 提示论文

## 🎓 总结

通过 Day24 的学习，我们掌握了：

### 核心知识

1. **文本生成原理**
   - 从规则到深度学习
   - GPT 架构详解
   - 生成策略对比

2. **Prompt Engineering**
   - 设计原则
   - 高级技巧
   - 模板库

3. **实战应用**
   - 写诗机器人
   - 质量控制
   - 性能优化

### 实用技能

1. **选择合适的生成策略**
   - 根据任务调整参数
   - 平衡质量和速度

2. **设计有效的 Prompt**
   - 明确具体
   - 提供上下文
   - 迭代优化

3. **解决常见问题**
   - 重复内容
   - 事实幻觉
   - 性能优化

4. **质量保证**
   - 自动评估
   - 人工审核
   - 持续改进

### 下一步方向

1. **深入学习**
   - 研究最新论文
   - 关注技术发展
   - 参与社区讨论

2. **实践项目**
   - 构建实际应用
   - 解决真实问题
   - 积累项目经验

3. **专业发展**
   - NLP 工程师
   - AI 产品经理
   - AI 研究员

---

**恭喜完成 Day24 的学习！**

🎉 🎉 🎉

**下一步：** [🎉 Day24 全部完成](./🎉%20Day24%20全部完成.md)
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
