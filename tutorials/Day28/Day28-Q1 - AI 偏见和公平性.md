# Day28-Q1 - AI 偏见和公平性

## ⚖️ 当 AI 学会"歧视"

### 问题背景

你训练了一个招聘筛选 AI,准确率 95%,看起来很完美。但上线后发现:

**问题:** AI 倾向于拒绝女性求职者!

- 男性简历通过率: 80%
- 女性简历通过率: 45%

**为什么?** 因为训练数据中,历史招聘记录本身就偏向男性(科技行业男性多)。

这就是 **AI 偏见 (AI Bias)** - 算法不公平地对待某些群体。

---

## 一、什么是 AI 偏见?

### 大白话解释

**AI 偏见 = AI 学会了人类的偏见**

就像:
- **人类偏见**: "女生不适合学理科" (刻板印象)
- **AI 偏见**: 从历史数据中学到这个"规律",然后歧视女生

**关键点:**
- AI 本身没有价值观
- AI 只是反映训练数据中的模式
- 如果数据有偏见,AI 就会有偏见

### 技术定义

AI 偏见是指机器学习系统对某些群体(种族、性别、年龄等)产生系统性不公平的结果。

---

## 二、偏见的来源

### 来源1: 数据偏见

**例子: 面部识别系统**

```
训练数据:
- 白人男性: 70%
- 白人女性: 15%
- 黑人男性: 10%
- 黑人女性: 5%

结果:
- 白人男性识别准确率: 99%
- 黑人女性识别准确率: 65%  ← 差了 34%!
```

**真实案例:** 
- 2018年 MIT 研究发现,商业面部识别系统对深色皮肤女性的错误率高达 35%
- Amazon 的招聘 AI 歧视女性 (2018)
- Google Photos 把黑人标记为"大猩猩" (2015)

### 来源2: 算法偏见

**例子: 贷款审批**

```python
# 特征工程时无意中引入偏见
features = [
    'income',           # 收入
    'credit_score',     # 信用分
    'zip_code',         # 邮政编码 ← 可能关联种族!
    'education',        # 学历
]

# 邮政编码可能间接反映种族/社会经济地位
# 导致算法对某些社区的人不利
```

### 来源3: 评估偏见

**问题:** 只用整体准确率评估,掩盖了群体差异

```python
# ❌ 只看整体准确率
overall_accuracy = 95%  # 看起来很好

# ✅ 分群体看准确率
accuracy_by_group = {
    'group_A': 98%,
    'group_B': 92%,
    'group_C': 75%,  ← 这个群体被忽视了!
}
```

### 来源4: 部署偏见

**例子: 语音助手**

- 在安静办公室测试: 识别率 95%
- 在嘈杂街道使用: 识别率 60%
- 对不同口音: 识别率差异巨大

**原因:** 测试环境和用户群体不匹配

---

## 三、偏见的类型

### 类型1: 历史偏见

**定义:** 反映现实世界的不平等

**例子:**
- 工资数据中男性普遍高于女性 → AI 学会给男性更高工资预测
- 犯罪数据中某些族裔被捕率高 → AI 认为他们更危险

**问题:** AI 强化了现有的不平等

### 类型2: 表示偏见

**定义:** 某些群体在数据中代表性不足

**例子:**
- 医疗 AI 主要用白人数据训练 → 对其他人种效果差
- 自动驾驶主要在晴天测试 → 雨天表现差

### 类型3: 评估偏见

**定义:** 评估指标不能反映所有群体的表现

**例子:**
- 只看平均准确率,忽略少数群体
- 测试集分布与真实用户不匹配

### 类型4: 聚合偏见

**定义:** 对不同群体使用同一模型,但他们需求不同

**例子:**
- 同一健康风险评估模型用于所有年龄段
- 但年轻人和老年人的风险因素完全不同

---

## 四、检测和量化偏见

### 指标1: 统计parity (统计均等)

**定义:** 不同群体获得正面结果的比例应该相近

```python
def statistical_parity(predictions, protected_attribute):
    """
    计算统计parity
    
    predictions: 预测结果 (0或1)
    protected_attribute: 保护属性 (如性别: 0=女, 1=男)
    """
    
    groups = np.unique(protected_attribute)
    rates = {}
    
    for group in groups:
        mask = protected_attribute == group
        rate = np.mean(predictions[mask])
        rates[group] = rate
    
    # 计算差异
    max_diff = max(rates.values()) - min(rates.values())
    
    return rates, max_diff

# 使用
rates, diff = statistical_parity(predictions, gender)
print(f"男性通过率: {rates[1]:.2%}")
print(f"女性通过率: {rates[0]:.2%}")
print(f"差异: {diff:.2%}")

# 理想情况: 差异 < 5%
```

### 指标2: Equal Opportunity (机会均等)

**定义:** 不同群体中,真正符合条件的人被正确识别的比例应该相近

```python
from sklearn.metrics import recall_score

# 分别计算每个群体的召回率
recall_male = recall_score(y_true[male_mask], y_pred[male_mask])
recall_female = recall_score(y_true[female_mask], y_pred[female_mask])

print(f"男性召回率: {recall_male:.2%}")
print(f"女性召回率: {recall_female:.2%}")
print(f"差异: {abs(recall_male - recall_female):.2%}")
```

### 指标3: Predictive Parity (预测均等)

**定义:** 不同群体中,预测为正的人实际为正的比例应该相近

```python
from sklearn.metrics import precision_score

# 分别计算每个群体的精确率
precision_male = precision_score(y_true[male_mask], y_pred[male_mask])
precision_female = precision_score(y_true[female_mask], y_pred[female_mask])

print(f"男性精确率: {precision_male:.2%}")
print(f"女性精确率: {precision_female:.2%}")
```

### 工具: AIF360 (IBM)

```python
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

# 创建数据集
dataset = BinaryLabelDataset(
    df=df,
    label_names=['hired'],
    protected_attribute_names=['gender']
)

# 计算偏见指标
metric = BinaryLabelDatasetMetric(dataset)

print(f"Statistical parity difference: {metric.statistical_parity_difference()}")
print(f"Disparate impact: {metric.disparate_impact()}")
print(f"Consistency: {metric.consistency()}")
```

---

## 五、减轻偏见的方法

### 方法1: 数据层面

#### 1.1 平衡数据集

```python
# ❌ 不平衡数据
# 男性: 9000条, 女性: 1000条

# ✅ 过采样少数群体
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 现在男女比例接近 1:1
```

#### 1.2 数据增强

```python
# 为少数群体生成合成数据
def augment_minority_data(data, factor=2):
    """为少数群体数据增强"""
    augmented = []
    for _ in range(factor):
        for sample in data:
            # 添加噪声生成新样本
            noise = np.random.normal(0, 0.1, sample.shape)
            new_sample = sample + noise
            augmented.append(new_sample)
    return np.array(augmented)
```

#### 1.3 去除敏感属性

```python
# ❌ 包含敏感属性
features = ['age', 'gender', 'race', 'income', 'education']

# ✅ 去除敏感属性
features = ['income', 'education', 'experience', 'skills']

# ⚠️ 注意: 其他特征可能与敏感属性相关,仍可能有偏见
```

### 方法2: 算法层面

#### 2.1 公平约束

```python
import torch
import torch.nn as nn

class FairClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.classifier(x)
    
    def fair_loss(self, predictions, labels, protected_attr, lambda_fair=0.5):
        """
        公平性损失
        
        同时优化准确率和公平性
        """
        # 分类损失
        classification_loss = nn.BCELoss()(predictions, labels)
        
        # 公平性损失: 不同群体的预测均值差异
        group_0_mean = predictions[protected_attr == 0].mean()
        group_1_mean = predictions[protected_attr == 1].mean()
        fairness_loss = torch.abs(group_0_mean - group_1_mean)
        
        # 总损失
        total_loss = classification_loss + lambda_fair * fairness_loss
        
        return total_loss
```

#### 2.2 对抗去偏见

```python
class AdversarialDebiasing(nn.Module):
    """
    对抗去偏见
    
    主分类器: 预测目标
    对抗分类器: 尝试从主分类器的表示中预测敏感属性
    目标: 让对抗分类器失败 (表示中不包含敏感信息)
    """
    
    def __init__(self, input_dim):
        super().__init__()
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 主分类器
        self.main_classifier = nn.Linear(32, 1)
        
        # 对抗分类器 (预测敏感属性)
        self.adversary = nn.Linear(32, 1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        main_output = torch.sigmoid(self.main_classifier(features))
        adv_output = torch.sigmoid(self.adversary(features))
        return main_output, adv_output, features
```

### 方法3: 后处理层面

#### 3.1 阈值调整

```python
def adjust_thresholds_by_group(y_scores, protected_attr, target_rate=0.5):
    """
    为不同群体调整决策阈值,使通过率相同
    """
    groups = np.unique(protected_attr)
    thresholds = {}
    
    for group in groups:
        mask = protected_attr == group
        scores = y_scores[mask]
        
        # 找到使通过率等于 target_rate 的阈值
        threshold = np.percentile(scores, (1 - target_rate) * 100)
        thresholds[group] = threshold
    
    return thresholds

# 使用
thresholds = adjust_thresholds_by_group(scores, gender, target_rate=0.5)

# 预测时使用不同阈值
predictions = np.zeros_like(scores)
for group in np.unique(gender):
    mask = gender == group
    predictions[mask] = (scores[mask] >= thresholds[group]).astype(int)
```

#### 3.2 重新校准

```python
from sklearn.calibration import CalibratedClassifierCV

# 为每个群体单独校准
calibrated_models = {}
for group in np.unique(protected_attr):
    mask = protected_attr == group
    X_group = X[mask]
    y_group = y[mask]
    
    cal_model = CalibratedClassifierCV(base_model, cv=5)
    cal_model.fit(X_group, y_group)
    calibrated_models[group] = cal_model
```

---

## 六、实际案例

### 案例1: Amazon 招聘 AI

**问题:**
- 2014年开始开发
- 用10年历史招聘数据训练
- 发现系统歧视女性简历

**原因:**
- 科技行业历史招聘以男性为主
- AI 学会了"男性=好候选人"的模式
- 甚至惩罚包含"women's"的词 (如 "women's chess club")

**结果:**
- 2018年项目被废弃
- 从未真正投入使用

**教训:**
- 历史数据可能包含偏见
- 需要主动检测和纠正

### 案例2: COMPAS 再犯预测

**问题:**
- 美国法院使用的再犯风险评估系统
- ProPublica 调查发现对黑人有偏见

**发现:**
- 黑人被错误标记为"高风险"的概率是白人的2倍
- 白人罪犯被错误标记为"低风险"的概率更高

**争议:**
- 开发商 Northpointe 否认偏见
- 说系统在不同群体中预测准确性相同

**启示:**
- "公平"有多种定义,可能互相冲突
- 需要透明度和独立审计

### 案例3: 医疗 AI

**问题:**
- 2019年 Science 论文发现
- 医疗资源分配算法对黑人有偏见

**原因:**
- 用"医疗费用"作为"健康状况"的代理
- 黑人由于经济原因就医少,费用低
- AI 误认为他们更健康,分配更少资源

**影响:**
- 数百万黑人患者受到影响
- 需要重新设计算法

---

## 七、最佳实践

### 实践1: 多样化团队

**为什么重要:**
- 不同背景的人能发现不同的偏见
- 避免盲点

**建议:**
- 团队包含不同性别、种族、年龄
- 包括领域专家和社会科学家
- 听取受影响群体的声音

### 实践2: 持续监控

```python
class BiasMonitor:
    """偏见监控系统"""
    
    def __init__(self, model, protected_attributes):
        self.model = model
        self.protected_attributes = protected_attributes
        self.history = []
    
    def monitor(self, X, y_true, y_pred, metadata):
        """监控预测中的偏见"""
        
        metrics = {}
        
        for attr_name in self.protected_attributes:
            attr_values = metadata[attr_name]
            
            # 计算各群体指标
            for value in np.unique(attr_values):
                mask = attr_values == value
                
                metrics[f'{attr_name}_{value}_accuracy'] = accuracy_score(
                    y_true[mask], y_pred[mask]
                )
                metrics[f'{attr_name}_{value}_precision'] = precision_score(
                    y_true[mask], y_pred[mask], zero_division=0
                )
                metrics[f'{attr_name}_{value}_recall'] = recall_score(
                    y_true[mask], y_pred[mask], zero_division=0
                )
        
        # 检查是否有显著差异
        if self.has_significant_bias(metrics):
            self.send_alert(metrics)
        
        self.history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        return metrics
    
    def has_significant_bias(self, metrics, threshold=0.1):
        """检查是否有显著偏见"""
        # 简化实现
        accuracies = [v for k, v in metrics.items() if 'accuracy' in k]
        if len(accuracies) >= 2:
            return max(accuracies) - min(accuracies) > threshold
        return False
    
    def send_alert(self, metrics):
        """发送警报"""
        print("⚠️ 检测到潜在偏见!")
        print(metrics)
```

### 实践3: 文档和透明度

**Model Cards (模型卡片):**

```markdown
# Model Card: Face Recognition System

## Model Details
- Developer: Company X
- Date: 2024-01-01
- Version: 1.0

## Intended Use
- Primary: Security access control
- Out-of-scope: Law enforcement, surveillance

## Training Data
- Dataset: CelebA + Custom
- Size: 100,000 images
- Demographics:
  - Gender: 50% male, 50% female
  - Race: 40% White, 30% Asian, 20% Black, 10% Other
  - Age: 18-65 years

## Performance Metrics
- Overall accuracy: 95%
- By gender:
  - Male: 96%
  - Female: 94%
- By race:
  - White: 97%
  - Asian: 95%
  - Black: 92%  ← 需要改进
  - Other: 93%

## Ethical Considerations
- Potential bias against darker skin tones
- Regular audits recommended
- Human oversight required for critical decisions
```

### 实践4: 用户反馈循环

```python
@app.post("/feedback")
def collect_feedback(prediction_id: str, was_correct: bool, user_comment: str = None):
    """收集用户对预测的反馈"""
    
    feedback = {
        'prediction_id': prediction_id,
        'was_correct': was_correct,
        'user_comment': user_comment,
        'timestamp': datetime.now()
    }
    
    # 存储到数据库
    db.feedback.insert_one(feedback)
    
    # 如果大量负面反馈,触发重新评估
    if is_anomaly_detected(prediction_id):
        trigger_model_review()
    
    return {"status": "received"}
```

---

## 八、本章小结

### 核心要点

✅ **偏见来源:**
- 数据偏见 (最常见)
- 算法偏见
- 评估偏见
- 部署偏见

✅ **检测方法:**
- Statistical parity
- Equal opportunity
- Predictive parity
- 工具: AIF360, Fairlearn

✅ **减轻方法:**
- 数据: 平衡、增强、去敏感
- 算法: 公平约束、对抗学习
- 后处理: 阈值调整、重新校准

✅ **最佳实践:**
- 多样化团队
- 持续监控
- 透明文档
- 用户反馈

### 重要认知

⚠️ **没有完美的公平:**
- 不同公平定义可能冲突
- 需要在准确性和公平性之间权衡
- 公平是持续的过程,不是一次性的修复

⚠️ **技术不是万能的:**
- 需要政策、法规、社会共识
- 需要跨学科合作
- 需要持续的对话和反思

---

## 🎯 下一步

理解了 AI 偏见,接下来学习其他伦理问题:

- [Q2](./Day28-Q2%20-%20隐私保护和数据安全.md): 隐私和数据安全
- [Q3](./Day28-Q3%20-%20透明度和可解释性.md): AI 黑盒问题
- [Q4](./Day28-Q4%20-%20责任和安全.md): AI 出错了谁负责
- [Q5](./Day28-Q5%20-%20法律法规和监管.md): 各国 AI 法规
- [Q6](./Day28-Q6%20-%20AI 从业者的责任.md): 职业道德和行为准则

**思考:** 技术能力越大,责任越大。我们该如何负责任地使用 AI? 🤔

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
