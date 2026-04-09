# 目标检测项目

## 📖 项目简介

使用 YOLOv8 进行实时物体检测。

## 🎯 学习目标

- 理解目标检测的基本概念
- 掌握 YOLO 算法的使用
- 学会标注和准备数据集
- 能够部署实时检测应用

## 📂 项目结构

```
object-detection/
├── detect.py            # 检测脚本
├── train.py             # 训练脚本
├── dataset.yaml         # 数据集配置
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 前置要求

- Python 3.7+
- 推荐：GPU（实时检测需要）
- 磁盘空间：至少 500MB
- 摄像头（可选，用于实时检测）

### 1. 克隆项目

```bash
git clone https://github.com/Lee985-cmd/AI-30-Day-Challenge.git
cd AI-30-Day-Challenge/projects/object-detection
```

### 2. 创建虚拟环境（推荐）

**Windows:**
```bash
python -m venv detect-env
detect-env\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv detect-env
source detect-env/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> 💡 **国内用户加速：**
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

> ⚠️ **注意：** ultralytics 包较大（约 200MB），首次安装可能需要几分钟。

### 2. 使用预训练模型检测

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 检测图片
results = model('image.jpg')

# 显示结果
results[0].show()
```

### 3. 实时摄像头检测

```python
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    annotated_frame = results[0].plot()
    
    cv2.imshow('Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 📊 预期结果

### 推理速度
- **GPU**: 30+ FPS（实时）
- **CPU**: 5-10 FPS

### 检测精度
- **mAP@0.5**: > 0.5（预训练模型）
- **支持类别**: 80 个 COCO 类别（人、车、动物等）

### 模型大小
- YOLOv8n: 6.2 MB（最快）
- YOLOv8s: 21.5 MB
- YOLOv8m: 49.7 MB

## 🔧 自定义训练

### 1. 准备数据集

```yaml
# dataset.yaml
path: ./dataset
train: images/train
val: images/val

nc: 3  # 类别数
names: ['cat', 'dog', 'bird']  # 类别名称
```

### 2. 训练模型

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0  # GPU
)
```

### 3. 评估模型

```python
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
```

## 💡 改进建议

### 1. 选择合适大小的模型

```python
# 根据需求选择
YOLO('yolov8n.pt')   # Nano - 最快，精度较低
YOLO('yolov8s.pt')   # Small - 平衡
YOLO('yolov8m.pt')   # Medium - 较慢，精度高
YOLO('yolov8l.pt')   # Large - 慢，精度很高
YOLO('yolov8x.pt')   # XLarge - 最慢，精度最高
```

### 2. 数据增强

```python
model.train(
    data='dataset.yaml',
    augment=True,      # 启用数据增强
    hsv_h=0.015,       # HSV-Hue 增强
    hsv_s=0.7,         # HSV-Saturation 增强
    hsv_v=0.4,         # HSV-Value 增强
    flipud=0.0,        # 垂直翻转概率
    fliplr=0.5,        # 水平翻转概率
    mosaic=1.0,        # Mosaic 增强概率
)
```

### 3. 超参数调优

```python
model.train(
    lr0=0.01,          # 初始学习率
    lrf=0.01,          # 最终学习率 (lr0 * lrf)
    momentum=0.937,    # SGD momentum
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
)
```

## 🐛 常见问题

### Q: 检测速度慢

**A:**
```python
# 使用更小的模型
model = YOLO('yolov8n.pt')

# 减小输入尺寸
results = model(image, imgsz=320)

# 使用 GPU
model.to('cuda')
```

### Q: 检测精度低

**A:**
- 增加训练数据
- 使用更大的模型
- 调整置信度阈值
- 检查标注质量

### Q: 漏检或误检

**A:**
```python
# 调整置信度阈值
results = model(image, conf=0.25)  # 默认 0.25

# 调整 NMS IoU 阈值
results = model(image, iou=0.7)    # 默认 0.7

# 过滤特定类别
results = model(image, classes=[0, 2, 5])  # 只检测某些类别
```

## 📚 相关资源

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [COCO 数据集](https://cocodataset.org/)
- [LabelImg 标注工具](https://github.com/heartexlabs/labelImg)
- [Day 15 教程](../../Day15/)

## 📄 许可证

MIT License
