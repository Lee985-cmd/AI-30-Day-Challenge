"""
Day16 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day16_examples.py

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
print("Day16 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

print("=" * 50)
print("⚡ YOLO 实战：实时目标检测")
print("=" * 50)

print("""
YOLO 版本演进:
✓ YOLO v1 (2016) - 开山之作
✓ YOLO v2 (2017) - 更快更准
✓ YOLO v3 (2018) - 经典版本
✓ YOLO v4/v5 (2020) - 性能提升
✓ YOLO v7/v8 (2022-23) - SOTA

今天我们用 YOLO v5！
""")

# 安装 ultralytics（YOLO v5）
# pip install ultralytics

print("\n【演示 YOLO 检测流程】")

# 模拟检测结果
# 实际项目中使用：from ultralytics import YOLO

detection_results = [
    {
        'box': [100, 150, 300, 400],  # [x_min, y_min, x_max, y_max]
        'class': 'person',
        'confidence': 0.95
    },
    {
        'box': [350, 200, 500, 350],
        'class': 'car',
        'confidence': 0.92
    },
    {
        'box': [50, 300, 150, 450],
        'class': 'dog',
        'confidence': 0.88
    }
]

print(f"检测结果:")
for det in detection_results:
    print(f"  - {det['class']}: {det['confidence']:.2f} "
          f"位置={det['box']}")

# 可视化
fig, ax = plt.subplots(1, figsize=(10, 8))

# 创建假图片背景
image = plt.zeros((500, 600, 3))
ax.imshow(image)

# 画边界框
colors = {'person': 'red', 'car': 'blue', 'dog': 'green'}

for i, det in enumerate(detection_results):
    x_min, y_min, x_max, y_max = det['box']
    width = x_max - x_min
    height = y_max - y_min
    
    rect = patches.Rectangle(
        (x_min, y_min),
        width,
        height,
        linewidth=3,
        edgecolor=colors[det['class']],
        facecolor='none',
        label=f"{det['class']} {det['confidence']:.2f}"
    )
    ax.add_patch(rect)
    
    # 添加标签
    ax.text(x_min, y_min - 10,
            f"{det['class']} {det['confidence']:.2f}",
            color=colors[det['class']],
            fontsize=12,
            fontweight='bold')

ax.set_xlim(0, 600)
ax.set_ylim(500, 0)
ax.set_title('YOLO 检测结果')
ax.legend()
plt.tight_layout()
plt.show()

print(f"\n💡 YOLO 的优势:")
print(f"- 速度快（实时 30+ FPS）")
print(f"- 端到端训练")
print(f"- 全局理解图像")
print(f"- 适合嵌入式设备")

# ===== 代码块 2 =====

print("=" * 50)
print("🎬 实战：交通标志检测")
print("=" * 50)

print("""
项目目标：
检测路上的交通标志
- 限速标志
- 停车标志
- 禁止通行
- ...

应用：
✓ 自动驾驶汽车
✓ 驾驶辅助系统
✓ 地图数据采集
""")

# 1. 准备数据
print("\n【1. 数据准备】")

# 实际项目中需要：
# - 收集交通标志图片
# - 标注边界框和类别
# - 划分训练集/测试集

print("数据集结构:")
print("""
traffic_signs/
├── images/
│   ├── train/  (训练图片)
│   └── test/   (测试图片)
└── labels/
    ├── train/  (训练标注)
    └── test/   (测试标注)
""")

# 2. 配置 YOLO
print("\n【2. 配置 YOLO 模型】")

config = """
# YOLO v5 配置示例
model: yolov5s.pt  # 小型版本
image_size: 640
classes: 10        # 10 类交通标志

training:
  batch_size: 16
  epochs: 100
  lr: 0.01
  
augmentation:
  mosaic: true     # 马赛克增强
  mixup: true      # 混合增强
"""

print(config)

# 3. 训练命令
print("\n【3. 训练命令】")
print("""
# 使用 YOLO v5 官方仓库
!git clone https://github.com/ultralytics/yolov5
!cd yolov5
!pip install -r requirements.txt

# 训练
!python train.py \\
  --img 640 \\
  --batch 16 \\
  --epochs 100 \\
  --data traffic_signs.yaml \\
  --weights yolov5s.pt
""")

# 4. 推理示例
print("\n【4. 推理示例】")

inference_code = """
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')

# 检测单张图片
results = model('test_image.jpg')

# 显示结果
results[0].show()

# 保存结果
results[0].save('output.jpg')

# 获取检测数据
boxes = results[0].boxes
for box in boxes:
    cls = box.cls      # 类别
    conf = box.conf    # 置信度
    xyxy = box.xyxy    # 边界框
    print(f'{cls}: {conf:.2f} at {xyxy}')
"""

print(inference_code)

print(f"\n{'='*50}")
print("🎊 恭喜！你了解了 YOLO 的完整流程！")
print(f"{'='*50}")

print("""
总结 YOLO 的特点:

✓ 快 - 实时检测
✓ 准 - 准确率高
✓ 简单 - 端到端训练
✓ 实用 - 工业界首选

这就是为什么 YOLO 这么流行！
""")