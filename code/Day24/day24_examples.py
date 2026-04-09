"""
Day24 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day24_examples.py

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
print("Day24 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("📊 BERT 情感分析系统 - 股票评论监控")
print("=" * 60)

# ============================================================================
# 第一部分：准备数据
# ============================================================================
print("\n【1. 准备股票评论数据】")

# 示例数据（真实场景应该收集更多）
stock_comments = [
    # 正面评论（标签 1）
    ("这个公司业绩太好了，强烈推荐买入！", 1),
    ("财报超预期，股价肯定要涨", 1),
    ("新产品很有竞争力，看好未来", 1),
    ("管理层很给力，战略清晰", 1),
    ("行业龙头，护城河深", 1),
    ("估值合理，值得长期持有", 1),
    ("技术领先，市场前景广阔", 1),
    ("盈利能力强劲，分红大方", 1),
    ("订单爆满，产能跟不上", 1),
    ("分析师上调目标价，信心十足", 1),
    
    # 负面评论（标签 0）
    ("这个公司要完蛋了，赶紧跑！", 0),
    ("财报暴雷，利润大幅下滑", 0),
    ("产品没有竞争力，被淘汰了", 0),
    ("管理层乱来，战略不明", 0),
    ("行业衰退，前景堪忧", 0),
    ("估值太高，泡沫严重", 0),
    ("技术落后，被对手超越", 0),
    ("连年亏损，快要破产了", 0),
    ("订单稀少，产能过剩", 0),
    ("分析师下调评级，建议卖出", 0),
    
    # 中性评论（标签 2）
    ("今天股价波动不大，正常", 2),
    ("消息面平静，观望为主", 2),
    ("业绩符合预期，无功无过", 2),
    ("市场反应平淡，没什么亮点", 2),
    ("维持原有评级，不做调整", 2),
]

print(f"✓ 数据集大小：{len(stock_comments)} 条评论")

# 统计各类别数量
from collections import Counter
labels = [label for _, label in stock_comments]
label_counts = Counter(labels)

print(f"  - 正面评论：{label_counts[1]} 条")
print(f"  - 负面评论：{label_counts[0]} 条")
print(f"  - 中性评论：{label_counts[2]} 条")

# 分割训练集和测试集
train_data, test_data = train_test_split(stock_comments, test_size=0.3, random_state=42)

print(f"\n  - 训练集：{len(train_data)} 条")
print(f"  - 测试集：{len(test_data)} 条")

# ============================================================================
# 第二部分：加载预训练 BERT 模型
# ============================================================================
print("\n" + "=" * 60)
print("【2. 加载 BERT 模型】")
print("=" * 60)

print("正在加载中文 BERT 模型...")
print("提示：第一次会自动下载，请耐心等待")

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=3  # 3 类：正面、负面、中性
)

print(f"✓ BERT 模型加载完成")
print(f"  隐藏层维度：{model.config.hidden_size}")
print(f"  分类类别数：{model.config.num_labels}")

# ============================================================================
# 第三部分：数据预处理
# ============================================================================
print("\n" + "=" * 60)
print("【3. 数据预处理】")
print("=" * 60)

def preprocess_data(data_list, tokenizer, max_length=128):
    """将文本转换为 BERT 可以接受的格式"""
    
    encodings = []
    labels = []
    
    for text, label in data_list:
        # 分词和编码
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,      # 添加 [CLS] 和 [SEP]
            max_length=max_length,         # 最大长度
            padding='max_length',          # 填充到最大长度
            truncation=True,               # 截断超长文本
            return_tensors='pt'            # 返回 PyTorch tensor
        )
        
        encodings.append(encoding)
        labels.append(label)
    
    return encodings, labels

print("正在处理训练集...")
train_encodings, train_labels = preprocess_data(train_data, tokenizer)

print(f"✓ 训练集处理完成")
print(f"  输入形状：{train_encodings[0]['input_ids'].shape}")
print(f"  标签数量：{len(train_labels)}")

# ============================================================================
# 第四部分：创建 Dataset 类
# ============================================================================
print("\n" + "=" * 60)
print("【4. 创建 Dataset】")
print("=" * 60)

class StockCommentDataset(torch.utils.data.Dataset):
    """股票评论数据集"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val.squeeze(0) for key, val in self.encodings[idx].items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = StockCommentDataset(train_encodings, train_labels)

print(f"✓ Dataset 创建完成")
print(f"  样本数量：{len(train_dataset)}")

# ============================================================================
# 第五部分：定义评估函数
# ============================================================================
print("\n" + "=" * 60)
print("【5. 定义评估指标】")
print("=" * 60)

def compute_metrics(pred):
    """计算准确率等评估指标"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
    }

print("✓ 评估指标定义完成")
print("  - 准确率 (accuracy)")

# ============================================================================
# 第六部分：训练模型
# ============================================================================
print("\n" + "=" * 60)
print("【6. 训练模型】")
print("=" * 60)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    warmup_steps=10,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

print("训练参数:")
print(f"  - 训练轮数：{training_args.num_train_epochs}")
print(f"  - 批次大小：{training_args.per_device_train_batch_size}")
print(f"  - 学习率：{training_args.learning_rate}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
)

print("\n开始训练...")
print("提示：由于数据量小，训练很快完成")

trainer.train()

print("\n✓ 模型训练完成")

# ============================================================================
# 第七部分：模型评估
# ============================================================================
print("\n" + "=" * 60)
print("【7. 模型评估】")
print("=" * 60)

# 处理测试集
test_encodings, test_labels = preprocess_data(test_data, tokenizer)
test_dataset = StockCommentDataset(test_encodings, test_labels)

print("在测试集上评估...")

eval_results = trainer.evaluate(test_dataset)

print(f"\n测试结果:")
print(f"  - 准确率：{eval_results['eval_accuracy']:.2%}")

# ============================================================================
# 第八部分：实际预测
# ============================================================================
print("\n" + "=" * 60)
print("【8. 实际预测】")
print("=" * 60)

def predict_sentiment(comment_text, model, tokenizer):
    """预测单条评论的情感"""
    
    # 编码
    inputs = tokenizer.encode_plus(
        comment_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)
        probs = predictions[0].numpy()
        label = torch.argmax(predictions, dim=-1).item()
    
    # 标签映射
    label_map = {0: "负面", 1: "正面", 2: "中性"}
    
    return label_map[label], probs[label]

# 测试一些新的评论
test_comments = [
    "这家公司的前景非常好，值得投资",
    "完了完了，股价要崩盘了",
    "今天股价波动不大，正常震荡",
    "业绩大幅增长，超出市场预期",
    "产品销量下滑，面临激烈竞争",
]

print("\n实时预测结果:\n")

for comment in test_comments:
    label, confidence = predict_sentiment(comment, model, tokenizer)
    
    # 显示表情符号
    if label == "正面":
        emoji = "🟢"
    elif label == "负面":
        emoji = "🔴"
    else:
        emoji = "⚪"
    
    print(f"{emoji} {comment}")
    print(f"   情感：{label} (置信度：{confidence:.2%})")
    print()

print("\n🎊 情感分析系统完成!")
print("=" * 60)

# ============================================================================
# 第九部分：可视化结果
# ============================================================================
print("\n" + "=" * 60)
print("【9. 可视化分析】")
print("=" * 60)

# 绘制混淆矩阵（如果有测试集预测结果）
try:
    predictions_output = trainer.predict(test_dataset)
    preds = predictions_output.predictions.argmax(-1)
    
    cm = confusion_matrix(test_labels, preds)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    
    classes = ['负面', '正面', '中性']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # 在每个格子中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.show()
    
except:
    print("混淆矩阵生成失败（数据量太小），跳过此步骤")

print("\n💡 实际应用建议:")
print("""
使用场景推荐:

1. 股票论坛舆情监控:
   ✓ 实时抓取雪球、东方财富评论
   ✓ 自动判断情感倾向
   ✓ 发现情绪异常波动
   ✓ 辅助投资决策

2. 新闻媒体分析:
   ✓ 监控财经新闻情感
   ✓ 统计利好/利空消息比例
   ✓ 跟踪舆情变化趋势
   ✓ 提前发现风险

3. 智能客服质检:
   ✓ 分析用户反馈情感
   ✓ 发现不满意客户
   ✓ 及时介入处理
   ✓ 提升服务质量

4. 产品口碑监控:
   ✓ 收集电商评论
   ✓ 分析用户满意度
   ✓ 发现产品问题
   ✓ 改进产品质量

技术要点:

✓ 数据质量很重要
  - 标注准确的训练数据
  - 多样化的样本
  - 平衡的类别分布

✓ 模型选择
  - BERT: 效果好，速度慢
  - DistilBERT: 速度快，效果略差
  - RoBERTa: 效果更好，需要更多数据

✓ 阈值设置
  - 高置信度才采纳
  - 低置信度的转人工
  - 平衡准确率和覆盖率

常见问题:

✗ 讽刺和反语识别不了
  → "真是太好了（其实是反话）"
  → 需要更复杂的模型
  
✗ 领域差异
  → 医疗、法律等专业领域
  → 需要领域特定的训练数据
  
✗ 多义词问题
  → "这个股票很火"（好）
  → "着火了"（坏）
  → 需要上下文理解
""")

print("\n🎉 情感分析实战完成!")
print("=" * 60)

# ===== 代码块 2 =====

"""
项目背景:
雪球是中国最大的投资者社区
每天有数百万条评论
用 AI 实时监控情绪变化
发现投资机会和风险

功能:
1. 实时抓取热门股票评论
2. 自动分析情感倾向
3. 统计多空比例
4. 发现情绪拐点
"""

import time
from datetime import datetime

print("=" * 60)
print("📈 雪球论坛情感监控系统")
print("=" * 60)

# 模拟实时数据流
mock_comments_stream = [
    ("贵州茅台", "茅台的业绩太稳了，长期持有没问题", 1),
    ("贵州茅台", "估值太高了，随时可能回调", 0),
    ("贵州茅台", "今天股价波动不大，正常", 2),
    ("宁德时代", "新能源前景广阔，坚定看好", 1),
    ("宁德时代", "竞争太激烈，利润会被压缩", 0),
    ("比亚迪", "销量创新高，股价要起飞", 1),
    ("比亚迪", "补贴退坡，影响很大", 0),
    ("腾讯控股", "游戏业务回暖，推荐买入", 1),
    ("腾讯控股", "政策监管风险大，谨慎", 0),
    ("阿里巴巴", "组织架构调整，观望", 2),
]

# 情感统计
sentiment_stats = {
    'positive': 0,
    'negative': 0,
    'neutral': 0,
}

print("\n开始实时监控...\n")

for stock, comment, sentiment in mock_comments_stream:
    # 模拟处理延迟
    time.sleep(0.5)
    
    # 更新统计
    if sentiment == 1:
        sentiment_stats['positive'] += 1
        emoji = "🟢"
        label = "正面"
    elif sentiment == 0:
        sentiment_stats['negative'] += 1
        emoji = "🔴"
        label = "负面"
    else:
        sentiment_stats['neutral'] += 1
        emoji = "⚪"
        label = "中性"
    
    # 计算比例
    total = sum(sentiment_stats.values())
    positive_ratio = sentiment_stats['positive'] / total * 100
    negative_ratio = sentiment_stats['negative'] / total * 100
    
    # 显示结果
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {emoji} {stock}: {comment}")
    print(f"        情感：{label}")
    print(f"        多空比：🟢{positive_ratio:.1f}% : 🔴{negative_ratio:.1f}%")
    print()
    
    # 检测情绪异常
    if total >= 5:
        if positive_ratio > 80:
            print("  ⚠️  警报：过度乐观，注意风险！")
        elif negative_ratio > 80:
            print("  ⚠️  警报：过度悲观，可能是机会！")
        print()

# 最终统计
print("\n" + "=" * 60)
print("最终统计结果:")
print("=" * 60)
print(f"总评论数：{total} 条")
print(f"正面：{sentiment_stats['positive']} 条 ({positive_ratio:.1f}%)")
print(f"负面：{sentiment_stats['negative']} 条 ({negative_ratio:.1f}%)")
print(f"中性：{sentiment_stats['neutral']} 条 ({sentiment_stats['neutral']/total*100:.1f}%)")

if positive_ratio > negative_ratio:
    print(f"\n整体情绪：🟢 偏向乐观")
elif negative_ratio > positive_ratio:
    print(f"\n整体情绪：🔴 偏向悲观")
else:
    print(f"\n整体情绪：⚪ 中性")

print("\n💡 使用建议:")
print("  1. 结合技术指标一起看")
print("  2. 不要盲目跟风")
print("  3. 警惕过度一致的情绪")
print("  4. 逆向思维：别人贪婪我恐惧")

print("\n🎊 监控系统演示完成!")
print("=" * 60)