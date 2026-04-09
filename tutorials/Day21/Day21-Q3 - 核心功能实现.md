# Day21-Q3 - 核心功能实现

## 📝 问题描述

完成了需求分析和架构设计后，终于到了最激动人心的环节——写代码！但如何高效地实现核心功能？如何保证代码质量？如何避免常见的实现陷阱？

**核心问题：**
- 如何组织项目代码结构？
- 如何实现照片上传和目标检测？
- 如何集成语音搜索功能？
- 如何实现风格迁移？
- 如何处理错误和边界情况？

---

## 💡 核心答案

核心功能实现的黄金法则：

1. **从 MVP 开始**：先实现最简单的可用版本
2. **模块化设计**：每个功能独立成模块
3. **充分测试**：每完成一个功能就测试
4. **逐步优化**：先让它工作，再让它快，最后让它美

我们将按照以下顺序实现：
1. 项目结构和基础配置
2. 照片上传和管理
3. YOLOv5 目标检测
4. Whisper 语音搜索
5. CycleGAN 风格迁移
6. Streamlit UI 集成

---

## 🎓 三个版本的解答

### 版本一：初学者比喻版（5 分钟理解）

#### 把代码实现比作"组装乐高"

想象你要用乐高积木搭建一个城堡，你不会把所有积木倒在地上然后随机拼接。你会：

**步骤 1：分类整理（项目结构）**

```
乐高盒子
├── 地基积木（基础配置）
├── 墙壁积木（核心功能）
├── 屋顶积木（UI 界面）
└── 装饰积木（优化美化）
```

**类比到代码：**
```
smart-album/
├── config.py          # 地基：配置文件
├── database.py        # 地基：数据库操作
├── models/            # 墙壁：AI 模型
│   ├── yolo_detector.py
│   ├── whisper_asr.py
│   └── cyclegan_style.py
├── services/          # 墙壁：业务逻辑
│   ├── photo_service.py
│   └── search_service.py
├── app.py             # 屋顶：主应用
└── uploads/           # 仓库：存储文件
```

---

**步骤 2：先搭框架（MVP）**

不要一开始就建完整的城堡，先搭一个简单的方盒子：

```python
# MVP：最简单的可用版本
import streamlit as st

st.title("智能相册")

uploaded_file = st.file_uploader("上传照片", type=['jpg', 'png'])

if uploaded_file:
    st.image(uploaded_file)
    st.write("✅ 照片上传成功！")
```

**这就够了！** 虽然功能简单，但它能工作。

---

**步骤 3：逐个添加房间（模块化）**

**房间 1：目标检测**
```python
def detect_objects(image):
    """检测照片中的物体"""
    # 加载模型
    model = load_yolo_model()
    
    # 推理
    results = model(image)
    
    return results
```

**房间 2：语音搜索**
```python
def search_by_voice(audio):
    """通过语音搜索照片"""
    # 转录语音
    text = whisper_transcribe(audio)
    
    # 搜索数据库
    photos = search_database(text)
    
    return photos
```

**房间 3：风格迁移**
```python
def apply_style(image, style):
    """应用艺术风格"""
    # 加载风格模型
    model = load_cyclegan_model(style)
    
    # 转换风格
    styled_image = model(image)
    
    return styled_image
```

---

**步骤 4：装修美化（优化）**

```python
# 添加进度条
with st.spinner("处理中..."):
    result = detect_objects(image)

# 添加成功提示
st.success("✅ 检测完成！")

# 添加错误处理
try:
    result = detect_objects(image)
except Exception as e:
    st.error(f"❌ 处理失败：{e}")
```

---

### 版本二：学生技术版（深入理解实现）

#### 1. 项目结构

**完整的项目目录：**
```
smart-album/
├── app.py                      # Streamlit 主应用
├── config.py                   # 配置文件
├── requirements.txt            # Python 依赖
├── database.py                 # 数据库操作
│
├── models/                     # AI 模型封装
│   ├── __init__.py
│   ├── yolo_detector.py        # YOLOv5 目标检测
│   ├── whisper_asr.py          # Whisper 语音识别
│   └── cyclegan_style.py       # CycleGAN 风格迁移
│
├── services/                   # 业务逻辑层
│   ├── __init__.py
│   ├── photo_service.py        # 照片管理服务
│   ├── detection_service.py    # 检测服务
│   └── search_service.py       # 搜索服务
│
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── image_utils.py          # 图像处理工具
│   └── audio_utils.py          # 音频处理工具
│
├── uploads/                    # 上传文件存储
│   ├── original/               # 原图
│   └── processed/              # 处理后图片
│
├── tests/                      # 单元测试
│   ├── test_detection.py
│   └── test_search.py
│
└── README.md                   # 项目说明
```

---

#### 2. 配置文件（config.py）

```python
"""
项目配置文件
"""
import os
from pathlib import Path

# 路径配置
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
ORIGINAL_DIR = UPLOAD_DIR / "original"
PROCESSED_DIR = UPLOAD_DIR / "processed"

# 确保目录存在
ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 数据库配置
DATABASE_URL = "sqlite:///./photos.db"

# 模型配置
YOLO_MODEL_SIZE = "yolov5s"  # tiny, s, m, l, x
WHISPER_MODEL_SIZE = "base"  # tiny, base, small, medium, large
CYCLEGAN_STYLE = "horse2zebra"  # 预训练风格

# 文件限制
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 性能配置
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

#### 3. 数据库操作（database.py）

```python
"""
数据库操作模块
"""
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import config

class DatabaseManager:
    def __init__(self, db_path: str = "photos.db"):
        self.db_path = db_path
        self.init_db()
    
    def get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 返回字典格式
        return conn
    
    def init_db(self):
        """初始化数据库表"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 创建照片表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                thumbnail_path TEXT,
                width INTEGER,
                height INTEGER,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建检测结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_w INTEGER,
                bbox_h INTEGER,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
            )
        ''')
        
        # 创建标签表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)')
        
        conn.commit()
        conn.close()
    
    def add_photo(self, file_path: str, width: int, height: int) -> int:
        """添加照片记录"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO photos (file_path, width, height) VALUES (?, ?, ?)',
            (file_path, width, height)
        )
        
        photo_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return photo_id
    
    def add_detection(self, photo_id: int, class_name: str, 
                     confidence: float, bbox: Dict[str, int]):
        """添加检测结果"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            '''INSERT INTO detections 
               (photo_id, class_name, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (photo_id, class_name, confidence, 
             bbox['x'], bbox['y'], bbox['w'], bbox['h'])
        )
        
        conn.commit()
        conn.close()
    
    def add_tag(self, photo_id: int, tag: str):
        """添加标签"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO tags (photo_id, tag) VALUES (?, ?)',
            (photo_id, tag)
        )
        
        conn.commit()
        conn.close()
    
    def search_by_tag(self, tag: str) -> List[Dict]:
        """根据标签搜索照片"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT p.* 
            FROM photos p
            JOIN tags t ON p.id = t.photo_id
            WHERE t.tag LIKE ?
            ORDER BY p.upload_time DESC
        ''', (f'%{tag}%',))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def get_photo_detections(self, photo_id: int) -> List[Dict]:
        """获取照片的检测结果"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM detections WHERE photo_id = ?',
            (photo_id,)
        )
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def delete_photo(self, photo_id: int):
        """删除照片及其相关数据"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 级联删除会自动处理 detections 和 tags
        cursor.execute('DELETE FROM photos WHERE id = ?', (photo_id,))
        
        conn.commit()
        conn.close()
```

---

#### 4. YOLOv5 目标检测（models/yolo_detector.py）

```python
"""
YOLOv5 目标检测模块
"""
import torch
from PIL import Image
import numpy as np
from typing import List, Dict
import config

class YOLODetector:
    def __init__(self, model_size: str = "yolov5s"):
        """
        初始化 YOLOv5 检测器
        
        Args:
            model_size: 模型大小 (tiny, s, m, l, x)
        """
        print(f"Loading YOLOv5 {model_size}...")
        
        # 从 torch hub 加载模型
        self.model = torch.hub.load(
            'ultralytics/yolov5',
            model_size,
            pretrained=True
        )
        
        # 设置设备
        self.device = config.DEVICE
        self.model.to(self.device)
        
        # 置信度阈值
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        
        print(f"✓ YOLOv5 loaded on {self.device}")
    
    def detect(self, image: Image.Image) -> List[Dict]:
        """
        检测图像中的物体
        
        Args:
            image: PIL Image 对象
        
        Returns:
            检测结果列表，每个元素包含：
            - class: 类别名称
            - confidence: 置信度
            - bbox: 边界框 {x, y, w, h}
        """
        # 转换为 numpy 数组
        image_np = np.array(image)
        
        # 推理
        results = self.model(image_np)
        
        # 解析结果
        detections = []
        
        # results.pandas().xyxy[0] 返回 DataFrame
        df = results.pandas().xyxy[0]
        
        for _, row in df.iterrows():
            # 过滤低置信度检测
            if row['confidence'] < self.conf_threshold:
                continue
            
            detection = {
                'class': row['name'],
                'confidence': float(row['confidence']),
                'bbox': {
                    'x': int(row['xmin']),
                    'y': int(row['ymin']),
                    'w': int(row['xmax'] - row['xmin']),
                    'h': int(row['ymax'] - row['ymin'])
                }
            }
            detections.append(detection)
        
        return detections
    
    def detect_batch(self, images: List[Image.Image]) -> List[List[Dict]]:
        """
        批量检测
        
        Args:
            images: PIL Image 列表
        
        Returns:
            每张图像的检测结果列表
        """
        all_detections = []
        
        for image in images:
            detections = self.detect(image)
            all_detections.append(detections)
        
        return all_detections
    
    def visualize(self, image: Image.Image, detections: List[Dict]) -> Image.Image:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果
        
        Returns:
            标注后的图像
        """
        from PIL import ImageDraw, ImageFont
        
        # 创建绘图对象
        draw = ImageDraw.Draw(image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # 绘制每个检测框
        for det in detections:
            bbox = det['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            
            # 绘制矩形框
            draw.rectangle([x, y, x+w, y+h], outline='red', width=3)
            
            # 绘制标签
            label = f"{det['class']} {det['confidence']:.2f}"
            draw.text((x, y-25), label, fill='red', font=font)
        
        return image


# 单例模式：全局共享检测器实例
_detector_instance = None

def get_detector() -> YOLODetector:
    """获取检测器单例"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YOLODetector(config.YOLO_MODEL_SIZE)
    return _detector_instance
```

---

#### 5. Whisper 语音识别（models/whisper_asr.py）

```python
"""
Whisper 语音识别模块
"""
import whisper
import tempfile
import numpy as np
from typing import Optional
import config

class WhisperASR:
    def __init__(self, model_size: str = "base"):
        """
        初始化 Whisper 模型
        
        Args:
            model_size: 模型大小 (tiny, base, small, medium, large)
        """
        print(f"Loading Whisper {model_size}...")
        
        self.model = whisper.load_model(model_size)
        self.model_size = model_size
        
        print(f"✓ Whisper loaded")
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        转录音频文件
        
        Args:
            audio_path: 音频文件路径
            language: 语言代码（可选，如 'zh' 表示中文）
        
        Returns:
            转录文本
        """
        # 转录
        result = self.model.transcribe(
            audio_path,
            language=language
        )
        
        return result['text'].strip()
    
    def transcribe_from_bytes(self, audio_bytes: bytes, 
                             sample_rate: int = 16000) -> str:
        """
        从字节数据转录音频
        
        Args:
            audio_bytes: 音频字节数据
            sample_rate: 采样率
        
        Returns:
            转录文本
        """
        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            # 转录
            text = self.transcribe(temp_path)
        finally:
            # 清理临时文件
            import os
            os.unlink(temp_path)
        
        return text
    
    def extract_keywords(self, text: str) -> list:
        """
        从文本中提取关键词（简单实现）
        
        Args:
            text: 转录文本
        
        Returns:
            关键词列表
        """
        # 简单的中文分词（实际项目应使用 jieba）
        import jieba
        
        # 分词
        words = jieba.cut(text)
        
        # 过滤停用词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        return keywords


# 单例模式
_asr_instance = None

def get_asr() -> WhisperASR:
    """获取 ASR 单例"""
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = WhisperASR(config.WHISPER_MODEL_SIZE)
    return _asr_instance
```

---

#### 6. CycleGAN 风格迁移（models/cyclegan_style.py）

```python
"""
CycleGAN 风格迁移模块
"""
import torch
from PIL import Image
import numpy as np
from typing import List
import config

class CycleGANStyler:
    def __init__(self, style: str = "horse2zebra"):
        """
        初始化 CycleGAN 风格迁移模型
        
        Args:
            style: 风格类型（horse2zebra, apple2orange, summer2winter 等）
        """
        print(f"Loading CycleGAN style: {style}...")
        
        # 这里使用简化的实现
        # 实际项目中应加载预训练的 CycleGAN 模型
        self.style = style
        self.device = config.DEVICE
        
        # 模拟加载（实际需要下载预训练权重）
        self.model = self._load_pretrained_model(style)
        
        print(f"✓ CycleGAN loaded")
    
    def _load_pretrained_model(self, style: str):
        """加载预训练模型（简化版）"""
        # 实际实现需要：
        # 1. 定义 CycleGAN 架构
        # 2. 加载预训练权重
        # 3. 设置为评估模式
        
        # 这里返回 None，实际使用时替换为真实模型
        return None
    
    def transfer_style(self, image: Image.Image) -> Image.Image:
        """
        应用风格迁移
        
        Args:
            image: 输入图像
        
        Returns:
            风格化后的图像
        """
        # 简化实现：使用 PIL 滤镜模拟风格效果
        # 实际应使用真实的 CycleGAN 模型
        
        if self.style == "horse2zebra":
            # 模拟斑马风格：增加对比度和饱和度
            from PIL import ImageEnhance
            
            enhanced = ImageEnhance.Contrast(image).enhance(1.5)
            enhanced = ImageEnhance.Color(enhanced).enhance(1.3)
            
            return enhanced
        
        elif self.style == "summer2winter":
            # 模拟冬季风格：降低饱和度，增加蓝色调
            from PIL import ImageEnhance, ImageOps
            
            desaturated = ImageEnhance.Color(image).enhance(0.5)
            cooled = ImageOps.colorize(desaturated, black="blue", white="white")
            
            return cooled
        
        else:
            # 默认：返回原图
            return image
    
    def apply_style_batch(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        批量应用风格迁移
        
        Args:
            images: 图像列表
        
        Returns:
            风格化后的图像列表
        """
        return [self.transfer_style(img) for img in images]


# 单例模式
_styler_instance = None

def get_styler(style: str = "horse2zebra") -> CycleGANStyler:
    """获取风格迁移器单例"""
    global _styler_instance
    if _styler_instance is None or _styler_instance.style != style:
        _styler_instance = CycleGANStyler(style)
    return _styler_instance
```

---

#### 7. 业务逻辑层（services/photo_service.py）

```python
"""
照片管理服务
"""
from PIL import Image
import uuid
from pathlib import Path
from typing import Dict, List
import config
from database import DatabaseManager
from models.yolo_detector import get_detector

class PhotoService:
    def __init__(self):
        self.db = DatabaseManager()
        self.detector = get_detector()
    
    def upload_photo(self, file) -> Dict:
        """
        上传照片并自动检测
        
        Args:
            file: Streamlit UploadedFile 对象
        
        Returns:
            上传结果，包含 photo_id 和检测结果
        """
        # 验证文件类型
        if not self._is_valid_image(file):
            raise ValueError(f"Invalid file type. Allowed: {config.ALLOWED_EXTENSIONS}")
        
        # 验证文件大小
        if len(file.getvalue()) > config.MAX_FILE_SIZE:
            raise ValueError(f"File too large. Max size: {config.MAX_FILE_SIZE / 1024 / 1024}MB")
        
        # 生成唯一文件名
        file_extension = file.name.split('.')[-1]
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        file_path = config.ORIGINAL_DIR / unique_filename
        
        # 保存文件
        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())
        
        # 打开图像
        image = Image.open(file_path)
        width, height = image.size
        
        # 添加到数据库
        photo_id = self.db.add_photo(str(file_path), width, height)
        
        # 自动检测
        detections = self.detector.detect(image)
        
        # 保存检测结果
        for det in detections:
            self.db.add_detection(photo_id, det['class'], det['confidence'], det['bbox'])
        
        # 自动生成标签
        tags = list(set([det['class'] for det in detections]))
        for tag in tags:
            self.db.add_tag(photo_id, tag)
        
        # 可视化检测结果
        visualized_image = self.detector.visualize(image.copy(), detections)
        visualized_path = config.PROCESSED_DIR / f"{unique_filename}_detected.jpg"
        visualized_image.save(visualized_path)
        
        return {
            'photo_id': photo_id,
            'file_path': str(file_path),
            'width': width,
            'height': height,
            'detections': detections,
            'tags': tags,
            'visualized_path': str(visualized_path)
        }
    
    def get_photo_info(self, photo_id: int) -> Dict:
        """获取照片信息"""
        # 实际应从数据库查询
        # 这里简化处理
        return {'id': photo_id}
    
    def delete_photo(self, photo_id: int):
        """删除照片"""
        self.db.delete_photo(photo_id)
    
    def _is_valid_image(self, file) -> bool:
        """验证是否为有效图片"""
        extension = file.name.split('.')[-1].lower()
        return extension in config.ALLOWED_EXTENSIONS
```

---

#### 8. Streamlit 主应用（app.py）

```python
"""
智能相册管理系统 - Streamlit 应用
"""
import streamlit as st
from PIL import Image
import config
from services.photo_service import PhotoService
from models.whisper_asr import get_asr
from models.cyclegan_style import get_styler

# 页面配置
st.set_page_config(
    page_title="智能相册管理系统",
    page_icon="📸",
    layout="wide"
)

# 标题
st.title("📸 智能相册管理系统")
st.markdown("---")

# 侧边栏
st.sidebar.header("功能选择")
page = st.sidebar.radio(
    "选择功能",
    ["📤 上传照片", "🔍 语音搜索", "🎨 风格迁移", "📊 相册浏览"]
)

# 初始化服务
@st.cache_resource
def init_services():
    return {
        'photo_service': PhotoService(),
        'asr': get_asr(),
        'styler': get_styler()
    }

services = init_services()

# ==================== 页面 1: 上传照片 ====================
if page == "📤 上传照片":
    st.header("上传照片并自动检测")
    
    uploaded_file = st.file_uploader(
        "选择照片",
        type=['jpg', 'jpeg', 'png'],
        help="支持 JPG、PNG 格式，最大 10MB"
    )
    
    if uploaded_file:
        # 显示原图
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("原图")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # 处理按钮
        if st.button("🚀 开始检测", type="primary"):
            with st.spinner("正在分析照片..."):
                try:
                    # 上传并检测
                    result = services['photo_service'].upload_photo(uploaded_file)
                    
                    # 显示检测结果
                    with col2:
                        st.subheader("检测结果")
                        
                        # 显示标注图
                        if result['visualized_path']:
                            detected_image = Image.open(result['visualized_path'])
                            st.image(detected_image, use_column_width=True)
                        
                        # 显示检测到的物体
                        st.write(f"**检测到 {len(result['detections'])} 个物体：**")
                        
                        for i, det in enumerate(result['detections'], 1):
                            st.write(
                                f"{i}. **{det['class']}** "
                                f"(置信度: {det['confidence']:.2%})"
                            )
                        
                        # 显示标签
                        st.write(f"**自动标签：** {', '.join(result['tags'])}")
                    
                    st.success("✅ 检测完成！")
                
                except Exception as e:
                    st.error(f"❌ 处理失败：{str(e)}")

# ==================== 页面 2: 语音搜索 ====================
elif page == "🔍 语音搜索":
    st.header("语音搜索照片")
    
    st.info("💡 提示：点击麦克风按钮，说出你想搜索的内容，例如'猫'、'汽车'、'风景'")
    
    # 语音输入
    audio_value = st.audio_input("录制语音")
    
    if audio_value:
        with st.spinner("正在识别语音..."):
            try:
                # 转录语音
                asr = services['asr']
                
                # 保存临时文件
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(audio_value.read())
                    temp_path = f.name
                
                # 转录
                text = asr.transcribe(temp_path)
                
                # 提取关键词
                keywords = asr.extract_keywords(text)
                
                st.write(f"**识别结果：** {text}")
                st.write(f"**关键词：** {', '.join(keywords)}")
                
                # 搜索照片
                if keywords:
                    with st.spinner("正在搜索照片..."):
                        results = []
                        for keyword in keywords:
                            photos = services['photo_service'].db.search_by_tag(keyword)
                            results.extend(photos)
                        
                        # 去重
                        seen_ids = set()
                        unique_results = []
                        for photo in results:
                            if photo['id'] not in seen_ids:
                                seen_ids.add(photo['id'])
                                unique_results.append(photo)
                        
                        st.write(f"**找到 {len(unique_results)} 张照片：**")
                        
                        # 显示搜索结果
                        cols = st.columns(3)
                        for i, photo in enumerate(unique_results[:9]):  # 最多显示 9 张
                            with cols[i % 3]:
                                if photo.get('file_path'):
                                    img = Image.open(photo['file_path'])
                                    st.image(img, use_column_width=True)
                                    st.caption(f"ID: {photo['id']}")
                
                # 清理临时文件
                import os
                os.unlink(temp_path)
            
            except Exception as e:
                st.error(f"❌ 搜索失败：{str(e)}")

# ==================== 页面 3: 风格迁移 ====================
elif page == "🎨 风格迁移":
    st.header("艺术风格迁移")
    
    uploaded_file = st.file_uploader(
        "上传照片",
        type=['jpg', 'jpeg', 'png'],
        key="style_upload"
    )
    
    # 风格选择
    style = st.selectbox(
        "选择风格",
        ["horse2zebra", "summer2winter", "apple2orange"],
        format_func=lambda x: {
            "horse2zebra": "🦓 马 → 斑马",
            "summer2winter": "❄️ 夏季 → 冬季",
            "apple2orange": "🍊 苹果 → 橙子"
        }[x]
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("原图")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        if st.button("✨ 应用风格", type="primary"):
            with st.spinner("正在转换风格..."):
                try:
                    styler = get_styler(style)
                    styled_image = styler.transfer_style(image)
                    
                    with col2:
                        st.subheader("风格化后")
                        st.image(styled_image, use_column_width=True)
                    
                    st.success("✅ 风格转换完成！")
                
                except Exception as e:
                    st.error(f"❌ 转换失败：{str(e)}")

# ==================== 页面 4: 相册浏览 ====================
elif page == "📊 相册浏览":
    st.header("相册浏览")
    
    # 这里应该从数据库加载所有照片
    # 简化实现：显示提示信息
    st.info("📷 此功能将在后续版本中实现")
    st.write("将显示所有上传的照片，支持按标签、日期筛选")

# 页脚
st.markdown("---")
st.markdown(
    "Made with ❤️ using Streamlit, YOLOv5, Whisper, and CycleGAN"
)
```

---

### 版本三：工程师实践版（生产级实现）

#### 性能优化技巧

**1. 模型懒加载**

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_yolo_model():
    """缓存模型实例，避免重复加载"""
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')
```

**2. 异步处理**

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def process_image_async(image_path: str):
    """异步处理图像"""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        process_image_sync,  # 同步函数
        image_path
    )
    return result
```

**3. 批量处理**

```python
def process_batch(image_paths: List[str], batch_size: int = 8):
    """批量处理图像"""
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        batch_results = model(batch)  # 批量推理
        results.extend(batch_results)
    
    return results
```

**4. 缓存策略**

```python
import hashlib
import pickle
from pathlib import Path

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

def cache_result(func):
    """结果缓存装饰器"""
    def wrapper(*args, **kwargs):
        # 生成缓存键
        cache_key = hashlib.md5(
            pickle.dumps((args, kwargs))
        ).hexdigest()
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        
        # 检查缓存
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 保存缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
    
    return wrapper

@cache_result
def detect_objects(image_path: str):
    # 检测结果会被缓存
    pass
```

---

## ⚠️ 常见错误与避坑指南

### 错误 1：内存泄漏

**❌ 错误做法：**
```python
# 每次请求都加载模型
def handle_request():
    model = load_model()  # 内存泄漏！
    result = model.predict(image)
```

**✅ 正确做法：**
```python
# 全局共享模型实例
model = load_model()

def handle_request():
    result = model.predict(image)
```

---

### 错误 2：不验证用户输入

**❌ 错误做法：**
```python
def upload(file):
    save(file)  # 危险！
```

**✅ 正确做法：**
```python
def upload(file):
    # 验证文件类型
    if not is_valid_image(file):
        raise ValueError("Invalid file type")
    
    # 验证文件大小
    if file.size > MAX_SIZE:
        raise ValueError("File too large")
    
    # 验证文件内容
    try:
        Image.open(file)
    except:
        raise ValueError("Corrupted file")
    
    save(file)
```

---

### 错误 3：忽视异常处理

**❌ 错误做法：**
```python
result = model.predict(image)  # 可能崩溃
```

**✅ 正确做法：**
```python
try:
    result = model.predict(image)
except torch.cuda.OutOfMemoryError:
    logger.error("GPU out of memory")
    return {"error": "Server busy, please try again"}
except Exception as e:
    logger.error(f"Prediction failed: {e}", exc_info=True)
    return {"error": "Processing failed"}
```

---

## ✍️ 自我检测练习

### 练习 1：代码重构

**任务：** 将 monolithic 代码拆分为模块化结构。

**参考答案：** 见上方的项目结构设计。

---

### 练习 2：添加单元测试

**任务：** 为 `detect_objects` 函数编写测试。

**参考答案：**
```python
import pytest
from PIL import Image
import numpy as np

def test_detect_objects():
    # 创建测试图像
    image = Image.new('RGB', (640, 480), color='red')
    
    # 检测
    detector = YOLODetector()
    results = detector.detect(image)
    
    # 验证
    assert isinstance(results, list)
    for det in results:
        assert 'class' in det
        assert 'confidence' in det
        assert 'bbox' in det

def test_invalid_image():
    detector = YOLODetector()
    
    with pytest.raises(Exception):
        detector.detect(None)
```

---

## 📝 本章小结

### 核心功能实现要点

✅ **模块化设计**：每个功能独立成模块

✅ **单例模式**：模型全局共享，避免重复加载

✅ **异常处理**：捕获并优雅处理错误

✅ **输入验证**：严格验证用户输入

✅ **性能优化**：缓存、批处理、异步

---

**📚 相关文档：**
- [Day21-Q2 - 技术架构与选型](./Day21-Q2%20-%20技术架构与选型.md)
- [Day21-Q4 - 多模态集成](./Day21-Q4%20-%20多模态集成.md)（待创建）

**💡 提示：** 代码实现是一个迭代过程，先让它工作，再优化性能和可读性。
