"""
Day15 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day15_examples.py

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
print("Day15 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

import torch
import torchvision
from torchvision.ops import nms, box_iou
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

print("=" * 50)
print("👁️ 目标检测基础：边界框详解")
print("=" * 50)

# 1. 边界框的两种表示方法
print("\n【1. 边界框表示方法】")

# 方法 1：[x_min, y_min, x_max, y_max]
box1 = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
print(f"方法 1：[xmin, ymin, xmax, ymax]")
print(f"  例子：{box1}")
print(f"  含义：左上角 (50,50), 右下角 (150,150)")

# 方法 2：[center_x, center_y, width, height]
box2 = torch.tensor([[100, 100, 100, 100]], dtype=torch.float32)
print(f"\n方法 2：[center_x, center_y, width, height]")
print(f"  例子：{box2}")
print(f"  含义：中心点 (100,100), 宽 100, 高 100")

print(f"\n💡 两种方法可以互相转换:")
print(f"  PyTorch 默认用方法 1")
print(f"  YOLO 等模型用方法 2")

# 可视化
fig, ax = plt.subplots(1, figsize=(6, 6))
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# 画法 1
rect1 = patches.Rectangle(
    (box1[0, 0], box1[0, 1]),
    box1[0, 2] - box1[0, 0],
    box1[0, 3] - box1[0, 1],
    linewidth=2, edgecolor='r', facecolor='none',
    label='[xmin,ymin,xmax,ymax]'
)
ax.add_patch(rect1)

# 画法 2（转换后）
x_min = box2[0, 0] - box2[0, 2] / 2
y_min = box2[0, 1] - box2[0, 3] / 2
rect2 = patches.Rectangle(
    (x_min, y_min),
    box2[0, 2],
    box2[0, 3],
    linewidth=2, edgecolor='b', facecolor='none',
    linestyle='--',
    label='[cx,cy,w,h]'
)
ax.add_patch(rect2)

ax.legend()
ax.set_title('边界框的两种表示方法')
plt.show()

# ===== 代码块 2 =====

print("=" * 50)
print("【2. 计算 IoU（交并比）】")
print("=" * 50)

# 创建两个边界框
box_a = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)  # 真实框
box_b = torch.tensor([[60, 60, 160, 160]], dtype=torch.float32)  # 预测框

print(f"真实框：{box_a}")
print(f"预测框：{box_b}")

# 计算 IoU
iou = box_iou(box_a, box_b)
print(f"\nIoU = {iou.item():.4f}")

# 可视化
fig, ax = plt.subplots(1, figsize=(6, 6))
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# 画真实框（绿色）
rect_gt = patches.Rectangle(
    (box_a[0, 0], box_a[0, 1]),
    box_a[0, 2] - box_a[0, 0],
    box_a[0, 3] - box_a[0, 1],
    linewidth=2, edgecolor='g', facecolor='green',
    alpha=0.3, label='真实框'
)
ax.add_patch(rect_gt)

# 画预测框（红色）
rect_pred = patches.Rectangle(
    (box_b[0, 0], box_b[0, 1]),
    box_b[0, 2] - box_b[0, 0],
    box_b[0, 3] - box_b[0, 1],
    linewidth=2, edgecolor='r', facecolor='red',
    alpha=0.3, label='预测框'
)
ax.add_patch(rect_pred)

ax.legend()
ax.set_title(f'IoU = {iou.item():.4f}')
plt.show()

print(f"\n💡 IoU 解读:")
print(f"- IoU = 1.0 → 完美重合")
print(f"- IoU > 0.5 → 检测成功")
print(f"- IoU < 0.3 → 需要改进")

# ===== 代码块 3 =====

print("=" * 50)
print("【3. 非极大值抑制（NMS）】")
print("=" * 50)

# 模拟多个检测结果
# 格式：[x_min, y_min, x_max, y_max]
boxes = torch.tensor([
    [50, 50, 150, 150],   # 猫 1（置信度 0.95）
    [55, 55, 155, 155],   # 猫 2（置信度 0.93）- 和猫 1 重叠
    [60, 60, 160, 160],   # 猫 3（置信度 0.85）- 和猫 1 重叠
    [300, 300, 400, 400], # 狗 1（置信度 0.90）
    [305, 305, 405, 405], # 狗 2（置信度 0.80）- 和狗 1 重叠
], dtype=torch.float32)

scores = torch.tensor([0.95, 0.93, 0.85, 0.90, 0.80])

print("NMS 之前的检测结果:")
for i in range(len(boxes)):
    print(f"  {i}: 框={boxes[i].tolist()}, 置信度={scores[i]:.2f}")

# 应用 NMS
keep_indices = nms(boxes, scores, iou_threshold=0.5)

print(f"\n应用 NMS (IoU 阈值=0.5):")
print(f"保留的索引：{keep_indices}")

print(f"\nNMS 之后的检测结果:")
for idx in keep_indices:
    print(f"  框={boxes[idx].tolist()}, 置信度={scores[idx]:.2f}")

# 可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# NMS 之前
ax1.set_xlim(0, 500)
ax1.set_ylim(0, 500)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('NMS 之前（多个框重叠）')

colors = ['red', 'orange', 'yellow', 'blue', 'purple']
for i, box in enumerate(boxes):
    rect = patches.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        linewidth=2, edgecolor=colors[i],
        facecolor=colors[i], alpha=0.3,
        label=f'检测{i+1} ({scores[i]:.2f})'
    )
    ax1.add_patch(rect)

ax1.legend(fontsize=8)

# NMS 之后
ax2.set_xlim(0, 500)
ax2.set_ylim(0, 500)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_title('NMS 之后（只保留最好的框）')

for i, idx in enumerate(keep_indices):
    rect = patches.Rectangle(
        (boxes[idx][0], boxes[idx][1]),
        boxes[idx][2] - boxes[idx][0],
        boxes[idx][3] - boxes[idx][1],
        linewidth=2, edgecolor=colors[idx],
        facecolor=colors[idx], alpha=0.5,
        label=f'保留{i+1} ({scores[idx]:.2f})'
    )
    ax2.add_patch(rect)

ax2.legend(fontsize=8)
plt.tight_layout()
plt.show()

print(f"\n💡 NMS 的作用:")
print(f"- 去除重复检测")
print(f"- 只保留置信度最高的框")
print(f"- 让结果更干净、准确")

# ===== 代码块 4 =====

print("=" * 50)
print("🎬 实战：人脸检测")
print("=" * 50)

# 使用 MTCNN 或 Haar Cascade 进行人脸检测
# 这里用简单的示例演示

import cv2

print("""
实际项目中常用的人脸检测方法:

1. OpenCV Haar Cascades
   ✓ 快速
   ✓ 简单
   ✗ 准确度一般

2. MTCNN
   ✓ 准确度高
   ✓ 能检测关键点（眼睛、鼻子）
   ✗ 稍慢

3. RetinaFace
   ✓ 非常准确
   ✓ 工业级
   ✗ 需要 GPU

今天我们用 OpenCV 演示！
""")

# 注意：这里用模拟数据演示
# 实际项目需要安装 opencv-python 和下载模型

print("\n【模拟人脸检测流程】")

# 假设有这些检测结果
face_boxes = torch.tensor([
    [100, 100, 200, 200],  # 人脸 1
    [110, 110, 210, 210],  # 人脸 1（重复）
    [300, 150, 400, 250],  # 人脸 2
    [350, 400, 450, 500],  # 人脸 3
], dtype=torch.float32)

face_scores = torch.tensor([0.98, 0.95, 0.92, 0.88])

# 应用 NMS
keep_faces = nms(face_boxes, face_scores, iou_threshold=0.4)

print(f"检测到 {len(keep_faces)} 张人脸")

for i, idx in enumerate(keep_faces):
    print(f"  人脸{i+1}: 位置={face_boxes[idx].tolist()}, "
          f"置信度={face_scores[idx]:.2f}")

print(f"\n{'='*50}")
print("完整的目标检测流程:")
print(f"{'='*50}")
print("""
1. 输入图片
2. 提取特征（用 CNN）
3. 生成候选框（可能几千个）
4. 对每个框预测类别和置信度
5. 应用 NMS 去除重复
6. 输出最终结果（几个到几十个框）

这就是目标检测的核心流程！
""")