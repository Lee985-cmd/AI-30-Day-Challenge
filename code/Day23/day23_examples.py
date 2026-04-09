"""
Day23 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day23_examples.py

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
print("Day23 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

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

# ===== 代码块 2 =====

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

# ===== 代码块 3 =====

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

# ===== 代码块 4 =====

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