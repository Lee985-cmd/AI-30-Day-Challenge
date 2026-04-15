# Day23-Q4 - 实战：情感分析系统

## 🎯 项目目标

让我们用 Hugging Face 的 transformers 库实现一个完整的情感分析系统，从零开始到部署上线。

## 📦 环境准备

### 安装依赖

```bash
pip install torch transformers scikit-learn numpy
```

### 导入库

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np

print("=" * 60)
print("BERT 情感分析系统 - 从零到部署")
print("=" * 60)
```

## 📊 数据准备

### 创建示例数据集

```python
"""
我们用一个简单的示例数据集
真实场景可以用:
- ChnSentiCorp(中文情感语料)
- THUCNews(清华新闻分类)
- 自己收集的数据
"""

# 示例数据 (商品评论)
reviews = [
    # 正面评论 (标签 1)
    ("这个产品太好了，质量很棒!", 1),
    ("非常满意，物流也快", 1),
    ("物超所值，推荐购买", 1),
    ("用了几天才来评价，真的很不错", 1),
    ("包装很好，没有破损", 1),
    ("客服态度好，产品也满意", 1),
    ("第二次购买了，一如既往的好", 1),
    ("比实体店便宜很多，赞!", 1),
    ("颜色正，尺码准，喜欢", 1),
    ("做工精细，材质环保", 1),
    
    # 负面评论 (标签 0)
    ("太差了，用了一次就坏了", 0),
    ("完全不值这个价", 0),
    ("物流慢死了，等了一个月", 0),
    ("和图片差距太大，失望", 0),
    ("客服态度恶劣，不推荐", 0),
    ("质量很差，都是线头", 0),
    ("味道刺鼻，不敢用", 0),
    ("尺寸不对，申请退货", 0),
    ("电池不耐用，半天就没电了", 0),
    ("屏幕有划痕，疑似二手", 0),
]

print(f"\n✓ 数据集大小：{len(reviews)} 条评论")
print(f"  - 正面评论：{sum([1 for _, label in reviews if label == 1])} 条")
print(f"  - 负面评论：{sum([1 for _, label in reviews if label == 0])} 条")

# 分割训练集和测试集
train_data, test_data = train_test_split(reviews, test_size=0.2, random_state=42)

print(f"  - 训练集：{len(train_data)} 条")
print(f"  - 测试集：{len(test_data)} 条")
```

## 🤖 加载预训练模型

### 选择合适的 BERT 模型

```python
"""
选择哪个 BERT?

bert-base-chinese:
- 中文版本
- 12 层，768 隐藏单元，12 头
- 1.1 亿参数
- 适合大多数任务

bert-base-uncased:
- 英文版本
- 小写处理
- 适合英文任务

bert-large:
- 更大更强
- 24 层，1024 隐藏单元，16 头
- 3.4 亿参数
- 需要更多显存
"""

print("\n正在加载 BERT 模型...")
print("提示：第一次运行会自动下载，大约 400MB")

# 分词器 (负责把文字转成 BERT 能懂的格式)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 分类模型 (在 BERT 基础上加了一个分类层)
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2,  # 二分类 (正面/负面)
    output_attentions=False,  # 不输出注意力权重 (省内存)
    output_hidden_states=False  # 不输出隐藏状态
)

print(f"✓ BERT 加载成功!")
print(f"  - 模型类型：bert-base-chinese")
print(f"  - 分类数：2 (正面/负面)")
print(f"  - 参数量：{model.num_parameters():,}")
```

## 🔧 数据预处理

### BERT 输入格式详解

```python
"""
BERT 的输入格式:

input_ids: 词的索引 [CLS] 我 [SEP] 喜 [SEP] 欢 [SEP]
token_type_ids: 句子标识 0 0 0 0 0 0 0 0 0
attention_mask: 注意力掩码 1 1 1 1 1 1 1 1 1 (1 表示看，0 表示不看)

特殊 token:
[CLS]: 句子开始标记，它的输出代表整个句子的表示
[SEP]: 句子结束标记
[PAD]: 填充标记 (长度不够时补零)
"""

MAX_LENGTH = 128  # 最大长度
PADDING = 'max_length'  # 填充到最大长度
TRUNCATION = True  # 超过最大长度就截断

def encode_review(review):
    """编码单条评论"""
    text, label = review
    
    encoding = tokenizer(
        text,
        add_special_tokens=True,      # 添加特殊 token ([CLS], [SEP])
        max_length=MAX_LENGTH,        # 最大长度
        padding=PADDING,              # 填充
        truncation=TRUNCATION,        # 截断
        return_attention_mask=True,   # 返回注意力掩码
        return_tensors='pt',          # 返回 PyTorch Tensor
    )
    
    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'labels': torch.tensor(label, dtype=torch.long)
    }

# 编码所有数据
print("\n正在编码数据...")
encoded_train = [encode_review(review) for review in train_data]
encoded_test = [encode_review(review) for review in test_data]

print(f"✓ 编码完成!")

# 查看一个样本
sample = encoded_train[0]
print(f"\n示例编码:")
print(f"  input_ids 形状：{sample['input_ids'].shape}")
print(f"  attention_mask 形状：{sample['attention_mask'].shape}")
print(f"  labels: {sample['labels']}")

# 解码看看
decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
print(f"  解码后：{decoded_text[:100]}...")
```

## 📚 创建 Dataset 类

```python
class ReviewDataset(torch.utils.data.Dataset):
    """自定义 Dataset 类"""
    
    def __init__(self, encoded_data):
        self.data = encoded_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': item['labels']
        }

# 创建 Dataset 对象
train_dataset = ReviewDataset(encoded_train)
test_dataset = ReviewDataset(encoded_test)

print(f"\n✓ Dataset 创建成功!")
print(f"  - 训练集大小：{len(train_dataset)}")
print(f"  - 测试集大小：{len(test_dataset)}")
```

## ⚙️ 设置训练参数

```python
training_args = TrainingArguments(
    output_dir='./results',              # 输出目录
    num_train_epochs=3,                  # 训练轮数
    per_device_train_batch_size=8,       # 训练批次大小
    per_device_eval_batch_size=16,       # 评估批次大小
    warmup_steps=50,                     # 预热步数
    weight_decay=0.01,                   # 权重衰减 (防止过拟合)
    logging_dir='./logs',                # 日志目录
    logging_steps=10,                    # 每 10 步记录一次
    evaluation_strategy='epoch',         # 每个 epoch 评估一次
    save_strategy='epoch',               # 每个 epoch 保存一次
    load_best_model_at_end=True,         # 加载最佳模型
    learning_rate=2e-5,                  # 学习率 (BERT 通常用小学习率)
)

print(f"\n✓ 训练参数设置完成!")
print(f"  - 训练轮数：{training_args.num_train_epochs}")
print(f"  - 批次大小：{training_args.per_device_train_batch_size}")
print(f"  - 学习率：{training_args.learning_rate}")
```

## 📈 定义评估函数

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    """计算评估指标"""
    
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # 计算准确率、精确率、召回率、F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

## 🚀 开始训练

```python
trainer = Trainer(
    model=model,                         # BERT 模型
    args=training_args,                 # 训练参数
    train_dataset=train_dataset,        # 训练集
    eval_dataset=test_dataset,          # 测试集
    compute_metrics=compute_metrics     # 评估函数
)

print("\n" + "=" * 60)
print("开始训练 BERT!")
print("=" * 60)
print("\n提示:")
print("  - 第一次训练可能需要 5-10 分钟")
print("  - 可以在 logs 目录查看训练过程")
print("  - 按 Ctrl+C 可以随时中断")
print()

# 开始训练!
trainer.train()
```

## 📊 评估模型性能

```python
print("\n" + "=" * 60)
print("训练完成！评估模型性能...")
print("=" * 60)

# 在测试集上评估
results = trainer.evaluate()

print(f"\n【测试结果】")
print(f"  - 准确率：{results['eval_accuracy']:.4f}")
print(f"  - 精确率：{results['eval_precision']:.4f}")
print(f"  - 召回率：{results['eval_recall']:.4f}")
print(f"  - F1 分数：{results['eval_f1']:.4f}")
```

## 🎯 实际使用演示

```python
print("\n" + "=" * 60)
print("现在你可以用训练好的模型做预测了!")
print("=" * 60)

def predict_sentiment(text):
    """
    预测情感
    
    参数:
    text: 文本字符串
    
    返回:
    sentiment: '正面😊' 或 '负面😞'
    confidence: 置信度
    """
    
    # 编码
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )
    
    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        prediction = torch.argmax(probabilities, dim=0).item()
        confidence = probabilities[prediction].item()
    
    # 转成人话
    sentiment = '正面😊' if prediction == 1 else '负面😞'
    
    return sentiment, confidence

# 测试一些新评论
test_reviews = [
    "这个产品非常好，我很满意",
    "太差劲了，再也不会买了",
    "一般般吧，没什么特别的",
    "超级棒！强烈推荐!",
    "浪费钱，别买!"
]

print("\n【情感分析演示】\n")

for review in test_reviews:
    sentiment, confidence = predict_sentiment(review)
    print(f"评论：{review}")
    print(f"情感：{sentiment} (置信度：{confidence:.2%})")
    print()
```

## 💾 保存和加载模型

```python
print("\n" + "=" * 60)
print("保存模型...")
print("=" * 60)

# 保存模型
model.save_pretrained('./my_sentiment_model')
tokenizer.save_pretrained('./my_sentiment_model')

print("✓ 模型已保存到 './my_sentiment_model/'")

# 以后可以这样加载
print("\n【如何加载保存的模型】")
print("""
from transformers import BertForSequenceClassification, BertTokenizer

# 加载
model = BertForSequenceClassification.from_pretrained('./my_sentiment_model')
tokenizer = BertTokenizer.from_pretrained('./my_sentiment_model')

# 直接使用
inputs = tokenizer("你的文本", return_tensors='pt')
outputs = model(**inputs)
""")
```

## 🔄 迁移到其他任务

```python
print("\n" + "=" * 60)
print("BERT 还能做什么？")
print("=" * 60)

print("""
【常见 NLP 任务】

1. 文本分类 (就像我们刚做的)
   - 情感分析 ✓
   - 新闻分类
   - 垃圾邮件检测

2. 命名实体识别 (NER)
   - 提取人名、地名、机构名
   - 医疗：疾病、药品、症状

3. 问答系统
   - 输入：问题 + 文档
   - 输出：答案

4. 自然语言推理
   - 判断两句话的关系
   - 蕴含、矛盾、中立

5. 文本相似度
   - 判断两个句子是否相似
   - 用于搜索引擎、推荐系统

所有这些都只需:
1. 加载预训练的 BERT
2. 加上特定任务的头 (Head)
3. 在任务数据上微调
""")
```

## 🛠️ 调试和优化技巧

### 常见问题解决

```python
"""
问题 1: 显存不足
解决:
- 减小 batch_size
- 使用梯度累积
- 启用混合精度训练

问题 2: 训练不收敛
解决:
- 检查学习率 (通常 2e-5)
- 增加训练轮数
- 检查数据质量

问题 3: 过拟合
解决:
- 减少训练轮数
- 增加 dropout
- 使用早停
- 增加数据量

问题 4: 欠拟合
解决:
- 增加训练轮数
- 使用更大的模型
- 调整学习率
- 检查数据标注质量
"""
```

### 性能优化

```python
"""
加速训练技巧:

1. 混合精度训练
training_args.fp16 = True

2. 梯度累积
training_args.gradient_accumulation_steps = 4

3. 数据加载优化
train_loader = DataLoader(dataset, batch_size=8, num_workers=4)

4. 模型量化
from transformers import quantization
quantized_model = quantization.quantize_model(model)
"""
```

## 📈 结果分析和改进

### 错误分析

```python
def analyze_errors(model, tokenizer, test_data):
    """分析模型的错误预测"""
    
    errors = []
    
    for text, true_label in test_data:
        # 预测
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_label = torch.argmax(outputs.logits, dim=1).item()
        
        # 记录错误
        if pred_label != true_label:
            errors.append({
                'text': text,
                'true_label': true_label,
                'pred_label': pred_label,
                'confidence': torch.softmax(outputs.logits, dim=1)[0][pred_label].item()
            })
    
    return errors

# 分析错误
errors = analyze_errors(model, tokenizer, test_data)
print(f"\n发现 {len(errors)} 个错误预测:")
for error in errors:
    print(f"文本: {error['text']}")
    print(f"真实标签: {error['true_label']}, 预测标签: {error['pred_label']}")
    print(f"置信度: {error['confidence']:.2%}")
    print()
```

### 改进策略

```python
"""
基于错误分析的改进:

1. 数据增强
- 同义词替换
- 随机删除
- 回译

2. 模型集成
- 多个模型投票
- 加权平均

3. 特征工程
- 添加额外特征
- 领域知识融合

4. 主动学习
- 选择最有价值的样本标注
- 迭代优化
"""
```

## 🎉 项目总结

通过这个完整的情感分析项目，我们学会了：

1. **数据准备**: 收集和预处理文本数据
2. **模型选择**: 选择合适的预训练模型
3. **训练调优**: 设置合适的训练参数
4. **评估分析**: 全面评估模型性能
5. **实际应用**: 部署和使用模型
6. **问题解决**: 调试和优化技巧

## 🚀 下一步

现在我们有了实用的情感分析系统，接下来让我们了解大语言模型的发展和应用。

---

**下一步：** [Day23-Q5 - 大语言模型概览](./Day23-Q5%20-%20大语言模型概览.md)
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
