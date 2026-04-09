"""
Day28 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day28_examples.py

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
print("Day28 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

"""
偏见从哪里来？

1. 数据偏见 (最常见)
   - 历史数据本身有偏见
   - 样本不均衡
   - 标注者主观偏见

例子:
- 医生照片大多是男性 → AI 认为医生=男性
- 护士照片大多是女性 → AI 认为护士=女性


2. 算法偏见
   - 目标函数设计不当
   - 特征选择有问题
   - 优化过程有偏差

例子:
- 用"点击率"优化新闻推荐
  → 标题党内容泛滥
  → 质量下降


3. 使用偏见
   - 应用场景不合适
   - 用户误解结果
   - 缺乏监督机制

例子:
- AI 面试系统用于创意岗位
  → 可能错过特立独行的人才
"""