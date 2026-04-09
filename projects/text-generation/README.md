# 文本生成项目

## 📖 项目简介

使用 LSTM 生成唐诗宋词或创意文本。

## 🎯 学习目标

- 掌握序列数据处理
- 理解 LSTM 的工作原理
- 学会文本生成技术
- 能够调整生成质量

## 📂 项目结构

```
text-generation/
├── main.py              # 主程序
├── dataset.py           # 数据加载和预处理
├── model.py             # LSTM 模型
├── train.py             # 训练脚本
├── generate.py          # 文本生成
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 前置要求

- Python 3.7+
- 推荐：GPU（可加速训练 5-10 倍）
- 磁盘空间：至少 1GB

### 1. 克隆项目

```bash
git clone https://github.com/Lee985-cmd/AI-30-Day-Challenge.git
cd AI-30-Day-Challenge/projects/text-generation
```

### 2. 创建虚拟环境（推荐）

**Windows:**
```bash
python -m venv text-env
text-env\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv text-env
source text-env/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> 💡 **国内用户加速：**
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 4. 准备数据

项目包含示例数据集，也可以用自己的文本：

```python
# 在 data/ 目录下放置文本文件
# 每行一首诗或一段文字
# 例如：data/poems.txt
```

### 5. 训练模型

```bash
python main.py --mode train --epochs 50
```

> ⏱️ **训练时间预估：**
> - CPU: 1-2 小时
> - GPU: 10-20 分钟

### 4. 生成文本

```bash
# 随机生成
python main.py --mode generate

# 给定开头生成
python main.py --mode generate --prompt "床前明月光"
```

## 📊 预期结果

### 训练时间
- **CPU**: 1-2 小时
- **GPU**: 10-20 分钟

### 生成质量
- 经过充分训练后能生成通顺的诗句
- 温度参数可调（0.2-1.5）

### 模型大小
- 约 10-20 MB

### 生成文件
- `model.pth` - 训练好的模型权重
- `generated_samples.txt` - 生成的文本示例

## 🔧 可调参数

```python
# 模型参数
EMBEDDING_DIM = 128     # 词嵌入维度
HIDDEN_DIM = 256        # LSTM 隐藏层维度
NUM_LAYERS = 2          # LSTM 层数
DROPOUT = 0.2           # Dropout 比例

# 训练参数
BATCH_SIZE = 64
SEQ_LENGTH = 50         # 序列长度
LEARNING_RATE = 0.001

# 生成参数
TEMPERATURE = 0.8       # 温度（控制创造性）
# 低温度 (0.2-0.5): 更保守、更可预测
# 中温度 (0.6-0.9): 平衡
# 高温度 (1.0-1.5): 更有创意但可能不通顺
```

## 💡 改进建议

### 1. 使用更大的数据集

```python
# 收集更多诗词
# 或使用小说、文章等
```

### 2. 尝试 Transformer

```python
# 使用 GPT-2 或其他预训练模型
from transformers import GPT2LMHeadModel, GPT2Tokenizer
```

### 3. 调整温度参数

```python
# 探索不同温度的效果
for temp in [0.5, 0.7, 0.9, 1.2]:
    generated = generate(prompt, temperature=temp)
    print(f"Temperature {temp}: {generated}")
```

### 4. Beam Search

```python
# 使用束搜索提高生成质量
# 而不是简单的贪婪解码或采样
```

## 🐛 常见问题

### Q: 生成的文本不通顺

**A:** 
- 增加训练轮数
- 增大数据集
- 调整温度参数（降低）
- 检查数据预处理

### Q: 训练很慢

**A:**
- 使用 GPU
- 减小序列长度
- 增大批次大小
- 使用更小的模型

### Q: 内存不足

**A:**
```python
# 减小批次大小
BATCH_SIZE = 32

# 减小序列长度
SEQ_LENGTH = 30

# 使用梯度累积
```

## 📚 相关资源

- [LSTM 论文](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [PyTorch RNN 教程](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
- [Day 13 教程](../../Day13/)

## 📄 许可证

MIT License
