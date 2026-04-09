"""
Day17 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day17_examples.py

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
print("Day17 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

print("=" * 50)
print("🎯 Faster R-CNN 详解")
print("=" * 50)

# 1. 加载预训练模型
print("\n【1. 加载 Faster R-CNN 模型】")

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # 切换到评估模式

print("✓ Faster R-CNN 加载完成")
print("  Backbone: ResNet-50")
print("  RPN: Region Proposal Network")
print("  Head: 分类器 + 边界框回归")
print("  预训练：COCO 数据集（80 类）")

# COCO 类别名称
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

print(f"\n可以识别 {len(COCO_CLASSES)-1} 类物体")

# ===== 代码块 2 =====

print("=" * 50)
print("【2. 测试 Faster R-CNN】")
print("=" * 50)

# 创建假图片测试
# 实际项目中用：image = Image.open('test.jpg')

# 模拟一张图片（白色背景上画些形状）
image_np = np.ones((600, 800, 3), dtype=np.uint8) * 255

# 画个红色矩形（模拟人）
cv2.rectangle(image_np, (100, 150), (250, 450), (255, 0, 0), -1)

# 画个蓝色矩形（模拟车）
cv2.rectangle(image_np, (400, 200), (700, 350), (0, 0, 255), -1)

# 转成 PIL Image
image_pil = Image.fromarray(image_np)

# 转成 Tensor
image_tensor = F.to_tensor(image_pil).unsqueeze(0)

print(f"测试图片：{image_pil.size}")
print(f"Tensor 形状：{image_tensor.shape}")

# 预测
with torch.no_grad():
    prediction = model(image_tensor)

print(f"\n预测结果:")
print(f"  检测到 {len(prediction[0]['boxes'])} 个物体")

# 显示检测结果
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(image_pil)

for i, (box, score, label) in enumerate(zip(
    prediction[0]['boxes'],
    prediction[0]['scores'],
    prediction[0]['labels']
)):
    if score > 0.5:  # 只显示置信度>50%的
        x_min, y_min, x_max, y_max = box
        
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor='g',
            facecolor='none',
            label=f"{COCO_CLASSES[label.item()]}: {score:.2f}"
        )
        ax.add_patch(rect)
        
        ax.text(x_min, y_min - 5,
                f"{COCO_CLASSES[label.item()]} {score:.2f}",
                color='green', fontsize=10, fontweight='bold')

ax.set_title('Faster R-CNN 检测结果')
ax.legend()
plt.tight_layout()
plt.show()

print(f"\n💡 Faster R-CNN 特点:")
print(f"- 两阶段检测（先粗筛后精挑）")
print(f"- 精度高（mAP 优于 YOLO）")
print(f"- 速度较慢（0.5-2 FPS）")
print(f"- 适合精度要求高的场景")

# ===== 代码块 3 =====

print("=" * 50)
print("🎬 实战：猫狗宠物检测")
print("=" * 50)

print("""
项目目标：
检测图片中的猫和狗

应用：
✓ 宠物监控摄像头
✓ 宠物行为分析
✓ 智能喂食器
""")

# 1. 准备数据
print("\n【1. 数据准备】")

dataset_structure = """
pets_dataset/
├── images/
│   ├── train/      (训练图片)
│   │   ├── cat_001.jpg
│   │   ├── cat_002.jpg
│   │   ├── dog_001.jpg
│   │   └── ...
│   └── test/       (测试图片)
│       ├── cat_101.jpg
│       ├── dog_101.jpg
│       └── ...
└── annotations/
    ├── train.json  (训练标注)
    └── test.json   (测试标注)
"""

print("数据集结构:")
print(dataset_structure)

print("\n标注格式（COCO 格式）:")
annotation_example = """
{
  "images": [
    {"id": 1, "file_name": "cat_001.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,  // 1=猫，2=狗
      "bbox": [x, y, width, height],
      "area": width*height,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "cat"},
    {"id": 2, "name": "dog"}
  ]
}
"""
print(annotation_example)

# 2. 自定义数据集类
print("\n【2. 自定义 Dataset 类】")

dataset_code = """
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import os

class PetsDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        # 加载标注
        with open(os.path.join(root, 'annotations/train.json')) as f:
            self.annotations = json.load(f)
        
        self.images = self.annotations['images']
        self.anns = self.annotations['annotations']
    
    def __getitem__(self, idx):
        # 加载图片
        img_info = self.images[idx]
        img_path = os.path.join(self.root, 'images/train', img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        
        # 获取这个图片的所有标注
        ann_ids = [ann['id'] for ann in self.anns if ann['image_id'] == img_info['id']]
        anns = [ann for ann in self.anns if ann['id'] in ann_ids]
        
        # 转成 tensor
        boxes = torch.tensor([ann['bbox'] for ann in anns], dtype=torch.float32)
        labels = torch.tensor([ann['category_id'] for ann in anns], dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target
    
    def __len__(self):
        return len(self.images)
"""

print(dataset_code)

# 3. 训练配置
print("\n【3. 训练配置】")

training_config = """
# Faster R-CNN 配置
model: fasterrcnn_resnet50_fpn
pretrained: true  # 迁移学习
num_classes: 3    # background + 猫 + 狗

# 训练参数
batch_size: 4      # 显存不够就设小点
epochs: 50
lr: 0.005
momentum: 0.9
weight_decay: 0.0005

# 学习率调度
lr_scheduler:
  type: StepLR
  step_size: 10    # 每 10 轮 lr×0.1
  gamma: 0.1
"""

print(training_config)

# 4. 训练代码
print("\n【4. 训练代码框架】")

training_code = """
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

# 1. 加载预训练模型
model = fasterrcnn_resnet50_fpn(pretrained=True)

# 2. 修改分类器（改成 2 类：猫 + 狗）
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=3)

# 3. 创建数据集
dataset = PetsDataset(root='./pets_dataset')
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# 4. 配置优化器
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 5. 训练循环
for epoch in range(50):
    model.train()
    epoch_loss = 0
    
    for images, targets in data_loader:
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        epoch_loss += losses.item()
    
    if lr_scheduler is not None:
        lr_scheduler.step()
    
    print(f"Epoch {epoch+1}/50, Loss: {epoch_loss/len(data_loader):.4f}")

# 6. 保存模型
torch.save(model.state_dict(), 'faster_rcnn_pets.pth')
"""

print(training_code)

print(f"\n{'='*50}")
print("🎊 恭喜！你了解了 Faster R-CNN 的完整流程！")
print(f"{'='*50}")

print("""
总结 Faster R-CNN vs YOLO:

Faster R-CNN:
✓ 精度高（mAP 更好）
✓ 适合小物体
✓ 适合医疗、工业等高精度场景
✗ 速度慢（0.5-2 FPS）

YOLO:
✓ 速度快（30+ FPS）
✓ 实时检测
✓ 适合视频流、自动驾驶
✗ 精度略低

选择建议：
- 要速度 → YOLO
- 要精度 → Faster R-CNN
- 都要 → YOLO v7/v8 或试试两者 ensemble
""")