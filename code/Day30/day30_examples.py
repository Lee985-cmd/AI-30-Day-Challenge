"""
Day30 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day30_examples.py

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
print("Day30 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

"""
Week 1: 机器学习基础
├─ Python 和 NumPy 基础 ✅
├─ KNN（监督学习入门）✅
├─ 决策树和随机森林 ✅
├─ SVM（支持向量机）✅
├─ K-means（无监督学习）✅
├─ 模型评估和优化 ✅
└─ Week1 综合项目 ✅

Week 2: 深度学习入门
├─ 神经网络原理 ✅
├─ PyTorch 框架 ✅
├─ CNN 基础 ✅
├─ 经典 CNN 架构 ✅
├─ RNN 和 LSTM ✅
└─ Week2 综合项目 ✅

Week 3: 进阶深度学习
├─ 目标检测 ✅
├─ 图像分割 ✅
├─ GAN 生成对抗网络 ✅
├─ Transformer 架构 ✅
├─ BERT 和大语言模型 ✅
├─ 语音识别 ✅
└─ Week3 综合项目 ✅

Week 4: 综合应用
├─ Transformer 进阶 ✅
├─ GPT 和文本生成 ✅
├─ 情感分析 ✅
├─ 强化学习 ✅
├─ 模型部署 ✅
├─ AI 伦理 ✅
├─ 前沿技术 ✅
├─ 面试准备 ✅
└─ 毕业项目 ⏳【今天】

总计:
✓ 30 篇教程
✓ 115+ 个费曼输出环节
✓ 11 个综合项目
✓ 约 30,000 行内容
✓ 完整的 AI 知识体系
"""

# ===== 代码块 2 =====

"""
项目目标:
综合运用 30 天学到的所有知识
开发一个完整的股票分析系统

功能需求:
1. 数据获取和可视化
2. 技术指标计算
3. 情感分析（新闻/评论）
4. 价格预测（LSTM/Transformer）
5. 交易信号生成
6. 回测和评估
7. Web 界面展示

技术栈:
✓ Python 基础
✓ NumPy/Pandas数据处理
✓ Matplotlib可视化
✓ 机器学习（特征工程）
✓ 深度学习（LSTM/Transformer）
✓ NLP（情感分析）
✓ Flask（Web 部署）
✓ 综合应用

这是你的毕业设计!
展示你 30 天的学习成果!
"""

# ===== 代码块 3 =====

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertForSequenceClassification

print("=" * 60)
print("🎓 毕业项目：智能股票分析系统")
print("=" * 60)

# ============================================================================
# 第一部分：数据获取和预处理
# ============================================================================
print("\n【1. 生成模拟股票数据】")

def generate_stock_data(days=365):
    """生成模拟股票数据"""
    np.random.seed(42)
    
    # 日期
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()
    
    # 股价（随机游走 + 趋势）
    price = 100
    prices = []
    for i in range(days):
        change = np.random.randn() * 2 + 0.05  # 趋势向上
        price += change
        price = max(price, 50)  # 不低于 50
        prices.append(price)
    
    # 成交量
    volumes = np.random.randint(1000000, 5000000, days)
    
    # 创建 DataFrame
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': volumes
    })
    
    # 计算技术指标
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(window=20).std()
    
    return df

df = generate_stock_data()
print(f"✓ 数据生成完成")
print(f"  数据量：{len(df)} 天")
print(f"  最新收盘价：{df['close'].iloc[-1]:.2f}")
print(f"  最高价：{df['close'].max():.2f}")
print(f"  最低价：{df['close'].min():.2f}")

# 可视化
plt.figure(figsize=(14, 6))
plt.plot(df['date'], df['close'], label='收盘价', linewidth=2)
plt.plot(df['date'], df['ma5'], label='MA5', alpha=0.7)
plt.plot(df['date'], df['ma20'], label='MA20', alpha=0.7)
plt.xlabel('日期')
plt.ylabel('股价')
plt.title('股票价格走势')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# 第二部分：LSTM 价格预测
# ============================================================================
print("\n" + "=" * 60)
print("【2. LSTM 价格预测模型】")
print("=" * 60)

class StockPredictor(nn.Module):
    """股票预测 LSTM 模型"""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(StockPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        
        return out

# 准备数据
print("\n准备训练数据...")
data = df['close'].values.reshape(-1, 1)

# 归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 创建序列
seq_length = 60
X, y = [], []
for i in range(len(data_scaled) - seq_length):
    X.append(data_scaled[i:i+seq_length])
    y.append(data_scaled[i+seq_length])

X = np.array(X)
y = np.array(y)

# 划分训练集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"✓ 数据准备完成")
print(f"  训练集大小：{len(X_train)}")
print(f"  测试集大小：{len(X_test)}")

# 转换为 Tensor
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)

# 创建模型
model = StockPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n✓ LSTM 模型创建完成")
print(f"  输入维度：1")
print(f"  隐藏层维度：50")
print(f"  LSTM 层数：2")

# 训练模型
print("\n开始训练...")
num_epochs = 50

for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n✓ 模型训练完成")

# 预测
with torch.no_grad():
    train_predict = model(X_train_tensor).numpy()
    train_predict = scaler.inverse_transform(train_predict)
    
    X_test_tensor = torch.FloatTensor(X_test)
    test_predict = model(X_test_tensor).numpy()
    test_predict = scaler.inverse_transform(test_predict)

# 实际值
train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 可视化预测结果
plt.figure(figsize=(14, 6))
plt.plot(train_actual, label='实际值（训练集）', linewidth=2)
plt.plot(train_predict, label='预测值（训练集）', alpha=0.7)
plt.xlabel('时间')
plt.ylabel('股价')
plt.title('LSTM 股价预测 - 训练集')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(test_actual, label='实际值（测试集）', linewidth=2)
plt.plot(test_predict, label='预测值（测试集）', alpha=0.7)
plt.xlabel('时间')
plt.ylabel('股价')
plt.title('LSTM 股价预测 - 测试集')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# 第三部分：情感分析模块
# ============================================================================
print("\n" + "=" * 60)
print("【3. 新闻情感分析】")
print("=" * 60)

# 模拟新闻数据
news_samples = [
    {"date": datetime.now() - timedelta(days=5), "text": "公司业绩大幅增长，超出市场预期"},
    {"date": datetime.now() - timedelta(days=4), "text": "行业政策利好，发展前景广阔"},
    {"date": datetime.now() - timedelta(days=3), "text": "竞争对手推出新产品，市场竞争加剧"},
    {"date": datetime.now() - timedelta(days=2), "text": "公司高管减持股份，市场信心受挫"},
    {"date": datetime.now() - timedelta(days=1), "text": "签订重大合同，未来收入有保障"},
]

# 简单的情感分析（基于关键词）
positive_words = ['增长', '利好', '广阔', '保障', '超出', '重大']
negative_words = ['下跌', '风险', '受挫', '加剧', '减持', '亏损']

print("\n新闻情感分析结果:\n")

for news in news_samples:
    text = news['text']
    
    pos_count = sum([1 for word in positive_words if word in text])
    neg_count = sum([1 for word in negative_words if word in text])
    
    if pos_count > neg_count:
        sentiment = "正面 😊"
        score = pos_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0.5
    elif neg_count > pos_count:
        sentiment = "负面 😠"
        score = neg_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0.5
    else:
        sentiment = "中性 😐"
        score = 0.5
    
    print(f"{news['date'].strftime('%Y-%m-%d')}: {sentiment} (置信度：{score:.2f})")
    print(f"  新闻：{text}\n")

# ============================================================================
# 第四部分：交易信号生成
# ============================================================================
print("\n" + "=" * 60)
print("【4. 交易信号生成】")
print("=" * 60)

def generate_trading_signals(df, predictions):
    """生成交易信号"""
    
    signals = []
    
    for i in range(len(predictions)):
        actual_price = df['close'].iloc[seq_length + i]
        pred_price = predictions[i][0]
        
        # 计算预期涨跌幅
        expected_change = (pred_price - actual_price) / actual_price
        
        # 生成信号
        if expected_change > 0.02:  # 预期涨 2% 以上
            signal = "买入 🔴"
        elif expected_change < -0.02:  # 预期跌 2% 以上
            signal = "卖出 🔵"
        else:
            signal = "持有 ⚪"
        
        signals.append({
            'date': df['date'].iloc[seq_length + i],
            'actual_price': actual_price,
            'predicted_price': pred_price,
            'expected_change': f"{expected_change*100:.2f}%",
            'signal': signal
        })
    
    return signals

# 生成最近 10 天的信号
recent_signals = generate_trading_signals(df, test_predict[-10:])

print("\n最近 10 天交易信号:\n")
print(f"{'日期':<12} {'实际价':<10} {'预测价':<10} {'预期涨跌':<10} {'信号':<10}")
print("-" * 60)

for s in recent_signals:
    date_str = s['date'].strftime('%Y-%m-%d')
    print(f"{date_str:<12} {s['actual_price']:<10.2f} {s['predicted_price']:<10.2f} {s['expected_change']:<10} {s['signal']:<10}")

# ============================================================================
# 第五部分：系统整合
# ============================================================================
print("\n" + "=" * 60)
print("【5. 系统整合与总结】")
print("=" * 60)

print("""
╔═══════════════════════════════════════════════════╗
║         智能股票分析系统 - 功能清单               ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║ ✓ 数据获取和预处理                                ║
║   - 历史股价数据                                  ║
║   - 技术指标计算（MA5, MA20）                     ║
║   - 收益率和波动率                                ║
║                                                   ║
║ ✓ 价格预测                                        ║
║   - LSTM 深度学习模型                             ║
║   - 序列预测                                      ║
║   - 可视化对比                                    ║
║                                                   ║
║ ✓ 情感分析                                        ║
║   - 新闻情感判断                                  ║
║   - 正面/负面/中性分类                            ║
║   - 置信度评分                                    ║
║                                                   ║
║ ✓ 交易信号                                        ║
║   - 基于预测生成买卖信号                          ║
║   - 买入/卖出/持有建议                            ║
║   - 预期涨跌幅                                    ║
║                                                   ║
║ 用到的技术:                                       ║
║ ✓ Python 基础（Day 1）                            ║
║ ✓ NumPy/Pandas（Day 1）                           ║
║ ✓ Matplotlib 可视化（Day 3）                      ║
║ ✓ LSTM 神经网络（Day 13）                         ║
║ ✓ PyTorch 框架（Day 10）                          ║
║ ✓ NLP 情感分析（Day 24）                          ║
║ ✓ 模型部署思路（Day 26）                          ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
""")

print("\n💡 可扩展方向:")
print("  1. 接入真实 API 获取实时数据")
print("  2. 用 Transformer 替代 LSTM")
print("  3. 加入更多技术指标")
print("  4. 用强化学习优化交易策略")
print("  5. 开发 Web 界面（Flask）")
print("  6. 添加回测功能")
print("  7. 集成到交易平台")

print("\n🎊 毕业项目完成!")
print("=" * 60)

# ===== 代码块 4 =====

career_paths = """
方向 1：AI 算法工程师

工作内容:
✓ 设计和实现 AI 模型
✓ 优化模型性能
✓ 解决实际问题

技能要求:
✓ 扎实的机器学习基础
✓ 熟练使用 PyTorch/TensorFlow
✓ 良好的编程能力
✓ 论文阅读能力

发展方向:
→ 初级算法工程师
→ 中级算法工程师
→ 高级算法工程师
→ 算法专家/技术总监

薪资范围（应届）:
一线城市：20-40w/年
二线城市：15-30w/年
"""

print("=" * 60)
print("💼 AI 职业发展方向")
print("=" * 60)

print(career_paths)

direction_2 = """
方向 2：AI 应用开发工程师

工作内容:
✓ 将 AI 模型应用到产品
✓ 开发 AI 应用系统
✓ 前后端开发

技能要求:
✓ AI 模型理解
✓ 全栈开发能力
✓ 系统设计能力

发展方向:
→ 应用开发工程师
→ 高级工程师
→ 技术架构师
→ 技术总监
"""

print(direction_2)

direction_3 = """
方向 3：AI 产品经理

工作内容:
✓ 定义 AI 产品功能
✓ 协调技术和业务
✓ 推动产品落地

技能要求:
✓ AI 技术理解
✓ 产品思维
✓ 沟通能力

发展方向:
→ 产品助理
→ 产品经理
→ 高级产品经理
→ 产品总监
"""

print(direction_3)

direction_4 = """
方向 4：AI 研究员（需要博士）

工作内容:
✓ 前沿技术研究
✓ 发表论文
✓ 申请专利

技能要求:
✓ 深厚的理论基础
✓ 创新能力
✓ 英文写作能力

发展方向:
→ 博士后
→ 研究员
→ 首席科学家
"""

print(direction_4)

# ===== 代码块 5 =====

learning_plan = """
╔═══════════════════════════════════════════════════╗
║         毕业后继续学习计划                        ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║ 短期（1-3 个月）:                                 ║
║ ✓ 巩固基础知识                                    ║
║ ✓ 深入一个方向（CV/NLP/推荐）                    ║
║ ✓ 做 2-3 个完整项目                               ║
║ ✓ 准备简历和面试                                  ║
║                                                   ║
║ 中期（3-12 个月）:                                ║
║ ✓ 找到第一份 AI 工作                              ║
║ ✓ 在工作中学习                                    ║
║ ✓ 建立技术博客                                    ║
║ ✓ 参加技术社区                                    ║
║                                                   ║
║ 长期（1-3 年）:                                   ║
║ ✓ 成为团队骨干                                    ║
║ ✓ 带新人                                          ║
║ ✓ 技术分享                                        ║
║ ✓ 建立个人品牌                                    ║
║                                                   ║
║ 推荐资源:                                         ║
║ ✓ 书籍：《深度学习》《机器学习》                  ║
║ ✓ 课程：吴恩达、李宏毅                            ║
║ ✓ 论文：Arxiv、顶会                               ║
║ ✓ 社区：GitHub、知乎、Stack Overflow              ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
"""

print(learning_plan)