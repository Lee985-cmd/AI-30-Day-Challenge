# Day24-Q5 - 实战：写诗机器人

## 🎯 项目目标

构建一个能够创作诗歌的 AI 机器人，支持多种诗歌风格和主题。

## 📦 完整代码实现

### 1. 基础写诗机器人

```python
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class PoetryBot:
    """写诗机器人"""
    
    def __init__(self, model_name='gpt2'):
        """初始化模型"""
        print("正在加载模型...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # 设置 pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✓ 模型加载完成!")
    
    def generate_poem(self, theme, style='modern', max_length=100):
        """
        生成诗歌
        
        参数:
        theme: 诗歌主题
        style: 诗歌风格 (modern/classical/romantic)
        max_length: 最大长度
        
        返回:
        poem: 生成的诗歌
        """
        
        # 根据风格构建 prompt
        if style == 'modern':
            prompt = f"写一首关于{theme}的现代诗:\n\n"
        elif style == 'classical':
            prompt = f"写一首关于{theme}的古体诗:\n\n"
        elif style == 'romantic':
            prompt = f"写一首关于{theme}的浪漫主义诗歌:\n\n"
        else:
            prompt = f"写一首关于{theme}的诗:\n\n"
        
        # 编码
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # 生成
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        poem = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return poem
    
    def batch_generate(self, themes, style='modern'):
        """批量生成诗歌"""
        poems = []
        for theme in themes:
            print(f"正在创作: {theme}...")
            poem = self.generate_poem(theme, style)
            poems.append(poem)
            print(f"✓ 完成\n")
        return poems


# 使用示例
if __name__ == "__main__":
    bot = PoetryBot()
    
    # 单首诗歌
    poem = bot.generate_poem("春天", style='modern')
    print(poem)
    
    # 批量生成
    themes = ["月亮", "爱情", "大海", "梦想"]
    poems = bot.batch_generate(themes, style='romantic')
```

### 2. 高级写诗机器人（支持更多控制）

```python
class AdvancedPoetryBot:
    """高级写诗机器人"""
    
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_with_structure(self, theme, structure='free', 
                                 mood='neutral', max_length=150):
        """
        生成有结构的诗歌
        
        参数:
        theme: 主题
        structure: 结构 (free/sonnet/haiku)
        mood: 情感基调 (happy/sad/neutral/romantic)
        max_length: 最大长度
        """
        
        # 构建详细的 prompt
        prompts = {
            'free': f"写一首关于{theme}的自由诗，情感基调是{mood}",
            'sonnet': f"写一首关于{theme}的十四行诗，情感基调是{mood}",
            'haiku': f"写一首关于{theme}的俳句（三行，5-7-5音节）",
        }
        
        prompt = prompts.get(structure, prompts['free'])
        prompt += ":\n\n"
        
        # 添加情感引导
        mood_keywords = {
            'happy': ['快乐', '阳光', '美好', '欢笑'],
            'sad': ['忧伤', '孤独', '思念', '泪水'],
            'romantic': ['爱', '温柔', '甜蜜', '永恒'],
            'neutral': ['平静', '自然', '思考', '存在']
        }
        
        keywords = mood_keywords.get(mood, [])
        if keywords:
            prompt += f"可以使用这些词汇: {', '.join(keywords)}\n\n"
        
        # 生成
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                top_p=0.92,
                temperature=0.85,
                repetition_penalty=1.3,
                length_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        poem = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return poem
    
    def interactive_mode(self):
        """交互模式"""
        print("=" * 60)
        print("🎭 欢迎使用 AI 写诗机器人!")
        print("=" * 60)
        print("\n输入格式: 主题 | 风格 | 情感")
        print("例如: 春天 | modern | happy")
        print("输入 'quit' 退出\n")
        
        while True:
            user_input = input("请输入: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见! 🌟")
                break
            
            # 解析输入
            parts = [p.strip() for p in user_input.split('|')]
            theme = parts[0] if len(parts) > 0 else "自然"
            style = parts[1] if len(parts) > 1 else "modern"
            mood = parts[2] if len(parts) > 2 else "neutral"
            
            # 生成诗歌
            print(f"\n正在创作'{theme}'主题的诗歌...\n")
            poem = self.generate_with_structure(theme, style, mood)
            
            # 美化输出
            print("-" * 60)
            print(poem)
            print("-" * 60)
            print()


# 运行交互模式
# bot = AdvancedPoetryBot()
# bot.interactive_mode()
```

### 3. 诗歌评估和优化

```python
class PoetryEvaluator:
    """诗歌评估器"""
    
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    def calculate_perplexity(self, poem):
        """计算困惑度（越低越好）"""
        
        input_ids = self.tokenizer.encode(poem, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            perplexity = torch.exp(outputs.loss)
        
        return perplexity.item()
    
    def check_rhyme(self, poem):
        """简单的押韵检查"""
        
        lines = [line.strip() for line in poem.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return False, "诗歌行数太少"
        
        # 提取每行最后一个字
        last_chars = [line[-1] for line in lines if len(line) > 0]
        
        # 检查是否有重复
        from collections import Counter
        char_counts = Counter(last_chars)
        
        rhyming_chars = [char for char, count in char_counts.items() if count > 1]
        
        if rhyming_chars:
            return True, f"发现押韵: {rhyming_chars}"
        else:
            return False, "未发现明显押韵"
    
    def evaluate_poem(self, poem):
        """综合评估诗歌"""
        
        results = {}
        
        # 1. 困惑度
        perplexity = self.calculate_perplexity(poem)
        results['perplexity'] = perplexity
        results['fluency_score'] = max(0, 100 - perplexity / 10)
        
        # 2. 押韵检查
        has_rhyme, rhyme_info = self.check_rhyme(poem)
        results['has_rhyme'] = has_rhyme
        results['rhyme_info'] = rhyme_info
        
        # 3. 长度分析
        lines = [line for line in poem.split('\n') if line.strip()]
        results['num_lines'] = len(lines)
        results['avg_line_length'] = sum(len(line) for line in lines) / max(len(lines), 1)
        
        # 4. 总体评分
        score = results['fluency_score']
        if has_rhyme:
            score += 10
        if 4 <= len(lines) <= 20:
            score += 5
        
        results['overall_score'] = min(100, score)
        
        return results
    
    def generate_multiple_and_select(self, theme, num_attempts=5):
        """生成多个版本并选择最好的"""
        
        bot = AdvancedPoetryBot()
        
        poems = []
        scores = []
        
        print(f"正在生成 {num_attempts} 个版本...\n")
        
        for i in range(num_attempts):
            poem = bot.generate_with_structure(theme, 'modern', 'neutral')
            evaluation = self.evaluate_poem(poem)
            
            poems.append(poem)
            scores.append(evaluation['overall_score'])
            
            print(f"版本 {i+1}: 评分 {evaluation['overall_score']:.1f}")
        
        # 选择最佳
        best_idx = scores.index(max(scores))
        best_poem = poems[best_idx]
        
        print(f"\n✓ 选择版本 {best_idx + 1} (评分: {scores[best_idx]:.1f})")
        
        return best_poem, poems, scores


# 使用示例
evaluator = PoetryEvaluator()
best_poem, all_poems, scores = evaluator.generate_multiple_and_select("秋天")
print("\n最佳诗歌:")
print(best_poem)
```

## 🎨 实际应用示例

### 示例 1: 节日祝福诗

```python
bot = AdvancedPoetryBot()

# 春节祝福
poem = bot.generate_with_structure(
    theme="春节",
    structure='free',
    mood='happy',
    max_length=120
)

print("春节祝福诗:")
print(poem)
```

### 示例 2: 情诗生成

```python
# 浪漫情诗
poem = bot.generate_with_structure(
    theme="永恒的爱",
    structure='free',
    mood='romantic',
    max_length=150
)

print("浪漫情诗:")
print(poem)
```

### 示例 3: 自然景观诗

```python
# 山水诗
themes = ["山川", "江河", "森林", "星空"]

for theme in themes:
    poem = bot.generate_with_structure(
        theme=theme,
        structure='free',
        mood='neutral',
        max_length=100
    )
    print(f"\n【{theme}】")
    print(poem)
    print("-" * 60)
```

## 💡 优化技巧

### 1. Prompt 优化

```python
"""
好的 Prompt 设计:

1. 提供诗歌示例
prompt = '''
参考以下诗歌的风格:

示例:
春眠不觉晓，
处处闻啼鸟。
夜来风雨声，
花落知多少。

现在写一首关于夏天的诗:
'''

2. 指定韵律要求
prompt = '''
写一首押韵的诗，每行 7 个字，共 4 行
主题: 秋天
'''

3. 使用意象引导
prompt = '''
写一首诗，包含以下意象:
- 落叶
- 秋风
- 夕阳
- 思念
'''
"""
```

### 2. 后处理优化

```python
def polish_poem(poem):
    """诗歌后处理"""
    
    # 1. 清理多余空白
    lines = [line.strip() for line in poem.split('\n') if line.strip()]
    
    # 2. 移除可能的重复行
    unique_lines = []
    for line in lines:
        if line not in unique_lines:
            unique_lines.append(line)
    
    # 3. 格式化
    polished = '\n'.join(unique_lines)
    
    return polished

# 使用
raw_poem = bot.generate_with_structure("冬天", 'modern', 'sad')
polished_poem = polish_poem(raw_poem)
```

### 3. 人工审核流程

```python
"""
人机协作流程:

1. AI 生成初稿 (3-5 个版本)
2. 自动评估筛选
3. 人工选择最佳版本
4. 人工微调和完善
5. 最终定稿

优势:
- AI 提供创意和效率
- 人类保证质量和品味
- 结合两者优势
"""
```

## 🎓 项目总结

通过这个写诗机器人项目，我们学会了：

1. **基础文本生成**
   - 使用 GPT-2 生成诗歌
   - 控制生成长度和质量

2. **Prompt Engineering**
   - 设计有效的提示
   - 引导风格和情感

3. **质量控制**
   - 自动评估指标
   - 多版本选择
   - 后处理优化

4. **实际应用**
   - 节日祝福
   - 情感表达
   - 创意写作

## 🚀 扩展方向

1. **支持更多语言**
   - 中文诗歌
   - 英文十四行诗
   - 日本俳句

2. **更精细的控制**
   - 韵律控制
   - 格律约束
   - 特定格式

3. **个性化风格**
   - 学习用户偏好
   - 模仿特定诗人
   - 风格迁移

4. **交互式创作**
   - 人机协作
   - 实时反馈
   - 迭代优化

---

**下一步：** [Day24-Q6 - 常见问题和最佳实践](./Day24-Q6%20-%20常见问题和最佳实践.md)