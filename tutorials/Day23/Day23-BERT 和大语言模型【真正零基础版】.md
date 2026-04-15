# 🚀 Day23: BERT 和大语言模型 - 预训练的力量【真正零基础版】

---

## 🌟 完整项目和代码

本教程是 **AI 入门 30 天挑战** 系列的一部分！

- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **欢迎 Star 支持！**

---



> **ChatGPT、文心一言都基于类似原理!理解 BERT 是进入大模型时代的门票!**  
> **本教程：完整代码 + 详细讲解 + 实战项目**

---

## 📚 目录

1. [BERT 到底是什么？](#bert-到底是什么)
2. [为什么需要预训练？](#为什么需要预训练)
3. [BERT 的核心创新](#bert 的核心创新)
4. [实战：情感分析系统](#实战：情感分析系统)
5. [大语言模型概览](#大语言模型概览)
6. [常见问题](#常见问题)

---

## 🤔 BERT 到底是什么？

### 说人话版本

想象一下这个场景:

```
传统 NLP 模型 (从零开始):
就像培养一个大学生
- 从高中知识开始教 (随机初始化)
- 学专业课程 (在特定任务上训练)
- 毕业找工作 (做具体任务)

问题:
- 每个学生都要从头教
- 花大量时间学基础知识
- 效果一般般

BERT (预训练 + 微调):
就像招聘一个博士生
- 已经学了 16 年 (在海量数据上预训练)
- 读了无数书 (看过整个维基百科)
- 只需要简单培训 (微调)就能上岗

优势:
- 基础扎实
- 学习快
- 效果好
```

**这就是 BERT 的范式革命!**

- **预训练**: 在大规模无标注数据上学习通用语言知识
- **微调**: 在小规模有标注数据上学习特定任务

### BERT 能做什么？

**真实应用场景:**

1. **搜索排序**
   - Google 搜索用 BERT 理解你的查询
   - 更准确的结果

2. **情感分析**
   - 商品评论是正面还是负面？
   - 舆情监控

3. **问答系统**
   - 智能客服
   - 知识问答

4. **文本分类**
   - 垃圾邮件识别
   - 新闻分类

5. **命名实体识别**
   - 从文本中提取人名、地名、机构名

---

## 💡 为什么需要预训练？

### 传统方法的痛苦

```python
"""
场景：训练一个情感分析模型

传统方法:
1. 收集 10000 条带标签的评论
2. 从零开始训练模型
3. 发现效果不好...

问题:
- 数据太少，模型学不到东西
- 过拟合严重 (死记硬背)
- 换个领域就不行了 (泛化差)

就像:
- 只做了 100 道题就去高考
- 题目稍微变一下就不会了
"""
```

### BERT 的解决方案

```python
"""
BERT 的做法:

第 1 步：预训练 (自学成才)
- 读遍整个维基百科 (33 亿词)
- 读完所有书籍 (BooksCorpus, 8 亿词)
- 学会语言的基本规律

第 2 步：微调 (岗前培训)
- 用 10000 条评论微调
- 因为基础好，很快学会
- 效果吊打从零开始

关键:
- 预训练用无标注数据 (便宜，量大)
- 微调用有标注数据 (贵，但需要少)
"""
```

---

## 🔬 BERT 的核心创新

### 创新 1: 双向 Transformer

```
传统语言模型 (单向):
"我 ___ 中国"
只能从左到右看 → 猜不出是"爱"还是"恨"

BERT(双向):
"我 ___ 中国"
 ← 两边一起看 → 确定是"爱"

就像完形填空:
看完整个句子再填，准确率更高
```

### 创新 2: Masked Language Model(MLM)

```python
"""
MLM 任务:

输入："我 [MASK] 你" (把"爱"遮住)
输出："爱"

训练方式:
1. 随机遮住 15% 的词
2. 让模型预测被遮住的词
3. 模型必须理解上下文才能猜对

好处:
- 强迫模型学习双向表示
- 不是死记硬背，真正理解
"""
```

### 创新 3: Next Sentence Prediction(NSP)

```python
"""
NSP 任务:

句子 A: "我喜欢看电影"
句子 B: "我经常去电影院"
问题：B 是不是 A 的下一句？答案：是

句子 A: "我喜欢看电影"
句子 B: "今天天气真好"
问题：B 是不是 A 的下一句？答案：不是

好处:
- 学习句子间的关系
- 对问答、推理任务很有用
"""
```

---

## 🎯 实战：情感分析系统

让我们用 Hugging Face 的 transformers 库实现一个完整的情感分析系统:

```python
# ============================================================================
# 第一部分：导入库
# ============================================================================

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np

print("=" * 60)
print("BERT 情感分析系统 - 从零到部署")
print("=" * 60)

# ============================================================================
# 第二部分：准备数据
# ============================================================================

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

# ============================================================================
# 第三部分：加载预训练模型和分词器
# ============================================================================

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

# ============================================================================
# 第四部分：数据预处理
# ============================================================================

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

# ============================================================================
# 第五部分：创建 Dataset 类
# ============================================================================

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

# ============================================================================
# 第六部分：设置训练参数
# ============================================================================

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

# ============================================================================
# 第七部分：定义评估函数
# ============================================================================

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

# ============================================================================
# 第八部分：创建 Trainer 并开始训练!
# ============================================================================

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

# ============================================================================
# 第九部分：评估模型
# ============================================================================

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

# ============================================================================
# 第十部分：实际使用!
# ============================================================================

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

# ============================================================================
# 第十一部分：保存和加载模型
# ============================================================================

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

# ============================================================================
# 第十二部分：迁移到其他任务
# ============================================================================

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

# ============================================================================
# 第十三部分：大语言模型概览
# ============================================================================

print("\n" + "=" * 60)
print("大语言模型家族")
print("=" * 60)

print("""
【BERT 系列】(Encoder-only)
├─ BERT (Google, 2018)
│  └─ 双向 Transformer
│  └─ 擅长：理解类任务 (分类、QA)
│
├─ RoBERTa (Facebook, 2019)
│  └─ BERT 的改进版
│  └─ 训练更久，数据更多
│
└─ ALBERT (Google, 2019)
   └─ 轻量级 BERT
   └─ 参数少，速度快

【GPT 系列】(Decoder-only)
├─ GPT-2 (OpenAI, 2019)
│  └─ 自回归生成
│  └─ 擅长：生成类任务
│
├─ GPT-3 (OpenAI, 2020)
│  └─ 1750 亿参数!
│  └─ Few-shot Learning
│
└─ ChatGPT/GPT-4 (OpenAI, 2022-2023)
   └─ 对话优化
   └─ 多模态能力

【其他重要模型】
├─ T5 (Google, 2019)
│  └─ Encoder-Decoder
│  └─ 统一所有 NLP 任务
│
├─ LLaMA (Meta, 2023)
│  └─ 开源大模型
│  └─ 可本地运行
│
└─ ChatGLM (清华，2023)
   └─ 中文优化
   └─ 高效推理

【选择建议】
- 中文任务：BERT/RoBERTa/ChatGLM
- 英文任务：GPT/BERT
- 资源有限：ALBERT/DistilBERT
- 需要生成：GPT/T5
- 需要理解：BERT/RoBERTa
""")

# ============================================================================
# 第十四部分：实战建议
# ============================================================================

print("\n" + "=" * 60)
print("实战建议和最佳实践")
print("=" * 60)

print("""
【数据准备】
✓ 至少几百条标注数据 (BERT 数据效率高)
✓ 数据要平衡 (正负样本相当)
✓ 清洗数据 (去除噪声)

【模型选择】
✓ 中文：bert-base-chinese
✓ 英文：bert-base-uncased
✓ 资源少：distilbert-base (更快更小)

【超参数调优】
✓ 学习率：1e-5 ~ 5e-5 (通常 2e-5)
✓ 批次大小：8~32 (根据显存调整)
✓ 训练轮数：3~10 (早停防止过拟合)

【常见错误】
✗ 学习率太大 → 不收敛
✗ 批次太小 → 训练不稳定
✗ 训练太久 → 过拟合
✗ 忘记设 dropout → 容易过拟合

【加速训练】
✓ 用 GPU (Colab 免费)
✓ 混合精度训练 (fp16)
✓ 梯度累积 (模拟大批次)

【部署上线】
✓ TorchScript 导出
✓ ONNX 格式转换
✓ 用 FastAPI 封装 API
✓ Docker 容器化
""")

print("\n🎉 恭喜你掌握了 BERT 和情感分析!")
print("\n下一步学习:")
print("  1. 尝试其他 NLP 任务 (NER、QA)")
print("  2. 学习 GPT 和文本生成")
print("  3. 了解大模型的 Fine-tuning")
print("  4. 做自己的 NLP 项目")

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
