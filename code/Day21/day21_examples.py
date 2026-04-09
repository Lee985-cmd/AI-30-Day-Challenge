"""
Day21 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day21_examples.py

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
print("Day21 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

"""
场景:
商场、公司、小区都需要监控
传统监控:
- 只能录像
- 出事了才回看
- 被动响应

智能监控:
✓ 实时识别人脸
✓ 统计人流量
✓ 发现异常行为
✓ 主动预警

这个项目能做进简历:
✓ 计算机视觉综合应用
✓ 有实际价值
✓ 效果直观
"""

# ===== 代码块 2 =====

"""
核心功能:

1. 人脸检测和识别
   - 检测画面中的人脸
   - 识别是不是熟人
   - 标记陌生人

2. 人数统计
   - 实时统计在场人数
   - 记录进出人流
   - 生成统计报表

3. 轨迹追踪
   - 跟踪每个人的移动路径
   - 分析活动区域
   - 发现异常行为

4. 告警系统
   - 发现黑名单人员报警
   - 区域入侵检测
   - 推送通知
"""

# ===== 代码块 3 =====

"""
技术栈:

【人脸检测】
- MTCNN: 高精度人脸检测
- RetinaFace: 更快的人脸检测

【人脸识别】
- FaceNet: 128 维特征向量
- ArcFace: 更准确的识别

【目标跟踪】
- DeepSORT: 多目标跟踪算法
- ByteTrack: 最新的跟踪方法

【系统集成】
- OpenCV: 视频处理
- Flask/FastAPI: Web 界面
- SQLite: 数据存储
"""