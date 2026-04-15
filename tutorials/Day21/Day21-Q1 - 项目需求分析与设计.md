# Day21-Q1 - 项目需求分析与设计

## 📝 问题描述

Week3 的综合项目是整个教程的里程碑，需要将前 20 天学到的计算机视觉、生成式 AI、语音识别等技术整合到一个完整的项目中。但在写代码之前，我们必须先做好需求分析和系统设计。

**核心问题：**
- 如何从模糊的想法转化为清晰的项目需求？
- 如何设计一个可扩展、可维护的系统架构？
- 如何平衡功能完整性和开发时间？
- 如何避免常见的项目规划陷阱？

---

## 💡 核心答案

项目需求分析不是"写文档"，而是**澄清思路、降低风险、提高成功率**的关键步骤。一个好的需求分析应该回答三个问题：

1. **做什么？**（功能范围）
2. **为什么做？**（用户价值）
3. **怎么做？**（技术方案）

我们将通过一个实际案例——**"智能相册管理系统"**，来演示完整的需求分析与设计流程。

---

## 🎓 三个版本的解答

### 版本一：初学者比喻版（5 分钟理解）

#### 把项目设计比作"盖房子"

想象你要盖一栋房子，你不能直接拿起砖头就开始砌墙。你需要先：

**步骤 1：明确需求（想清楚要什么样的房子）**

❌ **错误做法：** "我要盖个房子！"（太模糊）

✅ **正确做法：**
- 给谁住？（单身公寓 vs 家庭住宅）
- 多大面积？（50 平米 vs 200 平米）
- 有什么功能？（卧室、厨房、卫生间、书房）
- 预算多少？（10 万 vs 100 万）
- 什么时候完工？（3 个月 vs 1 年）

**类比到 AI 项目：**
- 给谁用？（普通用户 vs 专业人士）
- 处理什么数据？（照片、视频、音频）
- 有什么功能？（分类、搜索、编辑、分享）
- 计算资源？（手机 vs 服务器）
- 开发周期？（1 周 vs 3 个月）

---

**步骤 2：画蓝图（设计房屋结构）**

在盖房子之前，建筑师会画详细的蓝图：

```
房屋蓝图
├── 地基（基础设施）
│   ├── 水电管道（数据流）
│   └── 承重墙（核心模块）
│
├── 一楼（基础功能）
│   ├── 客厅（主界面）
│   ├── 厨房（数据处理）
│   └── 卫生间（存储管理）
│
├── 二楼（高级功能）
│   ├── 卧室（用户个性化）
│   └── 书房（智能分析）
│
└── 屋顶（用户体验）
    ├── 外观美化（UI 设计）
    └── 防水隔热（性能优化）
```

**类比到 AI 项目：**
```
系统架构蓝图
├── 数据层（地基）
│   ├── 数据采集（上传照片）
│   ├── 数据存储（数据库）
│   └── 数据预处理（ resizing, normalization）
│
├── 模型层（承重墙）
│   ├── CV 模型（目标检测、图像分割）
│   ├── GAN 模型（风格迁移、数据增强）
│   └── ASR 模型（语音识别）
│
├── 业务层（房间功能）
│   ├── 相册管理（增删改查）
│   ├── 智能搜索（语音 + 图像检索）
│   └── 自动标签（AI 标注）
│
└── 展示层（装修）
    ├── Web 界面（Streamlit）
    ├── 移动端适配
    └── 性能优化（缓存、压缩）
```

---

**步骤 3：分阶段施工（MVP 思维）**

你不会一次性把所有房间都装修完美，而是：

**第一阶段：毛坯房（MVP - 最小可行产品）**
- ✅ 能住人（核心功能可用）
- ✅ 有水有电（基本数据流打通）
- ❌ 没有家具（缺少高级功能）
- ❌ 没有装修（UI 简陋）

**第二阶段：简装房（V1.0）**
- ✅ 基本家具（常用功能完善）
- ✅ 简单装修（UI 美化）
- ❌ 没有智能家居（缺少 AI 功能）

**第三阶段：精装房（V2.0）**
- ✅ 智能家居（AI 自动化）
- ✅ 豪华装修（极致用户体验）
- ✅ 花园泳池（附加功能）

**类比到 AI 项目：**

**MVP（1 周）：**
- ✅ 能上传照片
- ✅ 能用 YOLO 检测物体
- ✅ 能显示检测结果
- ❌ 没有语音交互
- ❌ 没有风格迁移

**V1.0（2 周）：**
- ✅ 增加语音搜索
- ✅ 增加自动标签
- ✅ UI 美化
- ❌ 没有实时处理

**V2.0（4 周）：**
- ✅ 实时视频分析
- ✅ GAN 风格迁移
- ✅ 多用户支持
- ✅ 云端部署

---

### 版本二：学生技术版（深入理解方法论）

#### 需求分析的标准流程

##### 1. 利益相关者分析（Stakeholder Analysis）

**谁是项目的受益者？**

| 角色 | 需求 | 优先级 |
|------|------|--------|
| 最终用户 | 易用、快速、准确 | P0 |
| 开发者 | 可维护、可扩展 | P1 |
| 运维人员 | 稳定、可监控 | P1 |
| 产品经理 | 按时交付、符合预期 | P0 |

**用户画像（User Persona）：**

```
用户画像：小明，25 岁，摄影爱好者

痛点：
- 拍了 10000+ 张照片，找不到想要的
- 手动整理太耗时
- 想快速找到"去年在海边拍的照片"

期望：
- 上传照片后自动分类
- 可以用语音搜索："显示所有猫的照片"
- 一键美化照片

技术能力：
- 会用智能手机
- 不懂编程
- 对 AI 好奇但不了解
```

---

##### 2. 功能需求 vs 非功能需求

**功能需求（Functional Requirements）：系统"做什么"**

```
FR1: 用户可以上传照片/视频
FR2: 系统自动检测照片中的物体
FR3: 用户可以通过语音搜索照片
FR4: 系统自动生成照片标签
FR5: 用户可以应用风格迁移滤镜
FR6: 用户可以导出处理结果
```

**非功能需求（Non-Functional Requirements）：系统"做得怎么样"**

```
NFR1: 性能
  - 单张照片处理时间 < 2 秒
  - 语音识别延迟 < 500ms
  - 支持并发用户数 ≥ 100

NFR2: 可用性
  - 系统可用性 ≥ 99%
  - 平均故障恢复时间 < 5 分钟

NFR3: 可扩展性
  - 支持水平扩展（增加服务器即可提升容量）
  - 模块化设计，便于添加新功能

NFR4: 安全性
  - 用户数据加密存储
  - 防止 SQL 注入、XSS 攻击

NFR5: 兼容性
  - 支持 Chrome、Firefox、Safari
  - 支持 iOS、Android 移动端浏览器
```

---

##### 3. 用例图（Use Case Diagram）

```
                    ┌─────────────┐
                    │   用户      │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │ 上传照片   │   │ 语音搜索   │   │ 应用滤镜   │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
          │                │                │
          │         ┌──────▼──────┐        │
          │         │ 查看结果     │        │
          │         └──────┬──────┘        │
          │                │                │
          └────────────────┼────────────────┘
                           │
                    ┌──────▼──────┐
                    │  AI 引擎    │
                    │ (CV+GAN+ASR)│
                    └─────────────┘
```

---

##### 4. 数据流设计（Data Flow）

**照片处理流程：**

```
用户上传照片
    ↓
[前端] 验证文件格式、大小
    ↓
[后端] 保存到临时目录
    ↓
[预处理] Resize → Normalize → Augmentation
    ↓
[CV 模型] YOLOv5 目标检测
    ↓
[后处理] 解析检测结果 → 生成标签
    ↓
[存储] 保存原图 + 标注信息到数据库
    ↓
[返回] 向用户展示检测结果
```

**语音搜索流程：**

```
用户说话
    ↓
[前端] 录制音频（WebRTC）
    ↓
[ASR 模型] Whisper 转录为文本
    ↓
[NLP] 提取关键词（"猫"、"海边"）
    ↓
[数据库查询] SELECT * FROM photos WHERE tags LIKE '%猫%'
    ↓
[返回] 展示匹配的照片
```

---

##### 5. 技术选型决策矩阵

| 需求 | 选项 A | 选项 B | 选项 C | 选择 |
|------|--------|--------|--------|------|
| 目标检测 | YOLOv5 | Faster R-CNN | SSD | YOLOv5（速度快） |
| 语音识别 | Whisper | Wav2Vec 2.0 | DeepSpeech | Whisper（多语言） |
| 风格迁移 | CycleGAN | StyleGAN2 | FastNeuralStyle | CycleGAN（效果好） |
| Web 框架 | Streamlit | Flask + React | Django | Streamlit（快速原型） |
| 数据库 | SQLite | PostgreSQL | MongoDB | SQLite（简单够用） |
| 部署 | 本地运行 | Docker | Kubernetes | Docker（易部署） |

**选型理由：**
- **YOLOv5**：实时检测，社区活跃，文档丰富
- **Whisper**：开箱即用，支持中文，精度高
- **CycleGAN**：无需配对数据，风格迁移效果好
- **Streamlit**：Python 原生，50 行代码搭建 UI
- **SQLite**：零配置，适合小规模项目
- **Docker**：一次构建，到处运行

---

### 版本三：工程师实践版（生产级设计）

#### 系统架构设计

##### 1. 分层架构（Layered Architecture）

```
┌─────────────────────────────────────────┐
│         Presentation Layer              │
│  (Streamlit UI / REST API / WebSocket)  │
└────────────────┬────────────────────────┘
                 │ HTTP/WebSocket
┌────────────────▼────────────────────────┐
│         Application Layer               │
│  (Business Logic / Workflow Engine)     │
└────────────────┬────────────────────────┘
                 │ Function Calls
┌────────────────▼────────────────────────┐
│         Service Layer                   │
│  (CV Service / ASR Service / GAN Service)│
└────────────────┬────────────────────────┘
                 │ Model Inference
┌────────────────▼────────────────────────┐
│         Model Layer                     │
│  (YOLOv5 / Whisper / CycleGAN)          │
└────────────────┬────────────────────────┘
                 │ I/O Operations
┌────────────────▼────────────────────────┐
│         Data Layer                      │
│  (SQLite / File System / Cache)         │
└─────────────────────────────────────────┘
```

**每层的职责：**

**Presentation Layer（展示层）：**
- 处理用户输入
- 展示处理结果
- 管理会话状态

**Application Layer（应用层）：**
- 编排业务流程
- 异常处理
- 日志记录

**Service Layer（服务层）：**
- 封装模型调用
- 结果后处理
- 缓存管理

**Model Layer（模型层）：**
- 加载预训练模型
- 执行推理
- 模型版本管理

**Data Layer（数据层）：**
- 持久化存储
- 缓存加速
- 文件管理

---

##### 2. 微服务 vs 单体架构

**对于本项目，选择单体架构（Monolithic）：**

**理由：**
- ✅ 团队规模小（1-3 人）
- ✅ 功能相对简单
- ✅ 快速迭代
- ✅ 部署简单

**何时考虑微服务：**
- ❌ 用户量 > 10 万
- ❌ 需要独立扩展某个模块
- ❌ 多个团队并行开发
- ❌ 需要高可用性（99.99%+）

---

##### 3. 数据库设计

**ER 图（Entity-Relationship）：**

```
┌──────────────┐       ┌──────────────────┐
│   Users      │       │    Photos        │
├──────────────┤       ├──────────────────┤
│ id (PK)      │───┐   │ id (PK)          │
│ username     │   └──→│ user_id (FK)     │
│ email        │       │ file_path        │
│ created_at   │       │ upload_time      │
└──────────────┘       │ width            │
                       │ height           │
                       └────────┬─────────┘
                                │
                       ┌────────▼─────────┐
                       │  Detections      │
                       ├──────────────────┤
                       │ id (PK)          │
                       │ photo_id (FK)    │
                       │ class_name       │
                       │ confidence       │
                       │ bbox (x,y,w,h)   │
                       └──────────────────┘
```

**SQL 建表语句：**

```sql
-- 用户表
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 照片表
CREATE TABLE photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    thumbnail_path VARCHAR(500),
    width INTEGER,
    height INTEGER,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 检测结果表
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL,
    class_name VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_w INTEGER,
    bbox_h INTEGER,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
);

-- 索引优化
CREATE INDEX idx_photos_user_id ON photos(user_id);
CREATE INDEX idx_detections_photo_id ON detections(photo_id);
CREATE INDEX idx_detections_class ON detections(class_name);
```

---

##### 4. API 设计（RESTful）

**端点列表：**

```python
# 用户认证
POST   /api/auth/register      # 注册
POST   /api/auth/login         # 登录
POST   /api/auth/logout        # 登出

# 照片管理
GET    /api/photos             # 获取照片列表
POST   /api/photos             # 上传照片
GET    /api/photos/{id}        # 获取单张照片
DELETE /api/photos/{id}        # 删除照片

# AI 处理
POST   /api/photos/{id}/detect     # 目标检测
POST   /api/photos/{id}/style      # 风格迁移
GET    /api/photos/{id}/tags       # 获取标签

# 语音搜索
POST   /api/search/voice       # 语音搜索
GET    /api/search/text        # 文本搜索
```

**请求/响应示例：**

```python
# 上传照片
POST /api/photos
Content-Type: multipart/form-data

{
    "file": <binary_data>,
    "description": "我的宠物猫"
}

# 响应
{
    "status": "success",
    "data": {
        "id": 123,
        "file_path": "/uploads/2024/01/photo_123.jpg",
        "upload_time": "2024-01-15T10:30:00Z"
    }
}

# 目标检测
POST /api/photos/123/detect

# 响应
{
    "status": "success",
    "data": {
        "detections": [
            {
                "class": "cat",
                "confidence": 0.95,
                "bbox": {"x": 100, "y": 150, "w": 200, "h": 180}
            },
            {
                "class": "sofa",
                "confidence": 0.87,
                "bbox": {"x": 0, "y": 300, "w": 640, "h": 200}
            }
        ],
        "processing_time_ms": 1250
    }
}
```

---

##### 5. 错误处理策略

**全局异常处理器：**

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exc)}
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "File not found", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    # 记录日志
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )
```

**错误码规范：**

| 错误码 | 含义 | 示例 |
|--------|------|------|
| 400 | 请求参数错误 | 文件格式不支持 |
| 401 | 未授权 | Token 过期 |
| 403 | 禁止访问 | 无权操作他人照片 |
| 404 | 资源不存在 | 照片 ID 不存在 |
| 413 | 文件过大 | 超过 10MB 限制 |
| 422 | 模型推理失败 | 图片损坏无法处理 |
| 429 | 请求过于频繁 | 触发限流 |
| 500 | 服务器内部错误 | 未知异常 |

---

##### 6. 性能优化策略

**缓存策略：**

```python
import redis
import hashlib

class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def get_detection_result(self, image_hash):
        """从缓存获取检测结果"""
        cached = self.redis_client.get(f"detection:{image_hash}")
        if cached:
            return json.loads(cached)
        return None
    
    def set_detection_result(self, image_hash, result, ttl=3600):
        """缓存检测结果（1 小时过期）"""
        self.redis_client.setex(
            f"detection:{image_hash}",
            ttl,
            json.dumps(result)
        )
    
    def compute_image_hash(self, image_bytes):
        """计算图片哈希值"""
        return hashlib.md5(image_bytes).hexdigest()

# 使用示例
cache = CacheManager()
image_hash = cache.compute_image_hash(image_bytes)

# 先查缓存
cached_result = cache.get_detection_result(image_hash)
if cached_result:
    return cached_result

# 缓存未命中，执行推理
result = model.detect(image)

# 写入缓存
cache.set_detection_result(image_hash, result)
```

**异步处理：**

```python
from celery import Celery

celery_app = Celery('tasks', broker='redis://localhost:6379/0')

@celery_app.task(bind=True, max_retries=3)
def process_photo_task(self, photo_id):
    """异步处理照片（避免阻塞主线程）"""
    try:
        # 加载照片
        photo = load_photo(photo_id)
        
        # 目标检测
        detections = yolo_model.detect(photo)
        
        # 保存结果
        save_detections(photo_id, detections)
        
        return {"status": "success", "detections_count": len(detections)}
    
    except Exception as exc:
        # 重试机制
        raise self.retry(exc=exc, countdown=60)

# 提交任务
task = process_photo_task.delay(photo_id=123)

# 查询任务状态
result = task.get(timeout=30)
```

---

## ⚠️ 常见错误与避坑指南

### 错误 1：需求蔓延（Scope Creep）

**❌ 错误做法：**
```
初始需求：做一个相册管理系统

第 1 周：增加人脸识别
第 2 周：增加视频编辑
第 3 周：增加社交分享
第 4 周：增加 AI 修图
...
结果：项目永远做不完
```

**✅ 正确做法：**
```
MVP（第 1 周）：
- ✅ 上传照片
- ✅ 目标检测
- ✅ 显示结果

V1.0（第 2 周）：
- ✅ 语音搜索
- ✅ 自动标签

V2.0（第 3-4 周）：
- ✅ 风格迁移
- ✅ 批量处理

原则：每个版本只增加 2-3 个核心功能
```

---

### 错误 2：过度设计（Over-Engineering）

**❌ 错误做法：**
```python
# 为一个小型项目引入微服务、Kubernetes、消息队列
services/
├── auth-service/
├── photo-service/
├── ai-service/
├── search-service/
└── notification-service/

docker-compose.yml (500 行)
kubernetes/
├── deployment.yaml
├── service.yaml
└── ingress.yaml
```

**✅ 正确做法：**
```python
# 单体应用，简单清晰
app/
├── main.py
├── models/
├── services/
└── database.py

requirements.txt
Dockerfile (50 行)
```

**原则：** YAGNI（You Ain't Gonna Need It）——不要提前优化

---

### 错误 3：忽视边界情况

**❌ 错误做法：**
```python
def process_image(image_path):
    image = cv2.imread(image_path)
    result = model.detect(image)
    return result
```

**问题：**
- 如果文件不存在怎么办？
- 如果图片损坏怎么办？
- 如果图片太大（1GB）怎么办？
- 如果模型推理超时怎么办？

**✅ 正确做法：**
```python
def process_image(image_path, max_size=4096, timeout=30):
    # 检查文件存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    
    # 检查文件大小
    file_size = os.path.getsize(image_path)
    if file_size > 100 * 1024 * 1024:  # 100MB
        raise ValueError("File too large (>100MB)")
    
    # 读取图片
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to read image (corrupted?)")
    except Exception as e:
        raise IOError(f"Error reading image: {e}")
    
    # 调整大小
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w*scale), int(h*scale)))
    
    # 推理（带超时）
    try:
        result = model.detect(image, timeout=timeout)
    except TimeoutError:
        raise TimeoutError(f"Model inference timeout (> {timeout}s)")
    
    return result
```

---

### 错误 4：没有考虑数据安全

**❌ 错误做法：**
```python
# 明文存储密码
user.password = "123456"

# 直接拼接 SQL（SQL 注入风险）
query = f"SELECT * FROM users WHERE username='{username}'"

# 未验证文件类型
uploaded_file.save("/uploads/" + filename)
```

**✅ 正确做法：**
```python
import bcrypt
from werkzeug.utils import secure_filename

# 密码哈希
hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

# 参数化查询（防 SQL 注入）
cursor.execute("SELECT * FROM users WHERE username=?", (username,))

# 文件类型验证
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
filename = secure_filename(file.filename)
if '.' not in filename or filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
    raise ValueError("Invalid file type")
```

---

## 📊 需求文档模板

### 项目需求规格说明书（SRS）

```markdown
# 项目名称：智能相册管理系统

## 1. 引言

### 1.1 目的
开发一个基于 AI 的智能相册管理系统，帮助用户快速整理和搜索照片。

### 1.2 范围
- 支持照片上传和管理
- 自动目标检测和标签生成
- 语音搜索功能
- 风格迁移滤镜

### 1.3 目标用户
- 摄影爱好者
- 普通手机用户
- 社交媒体创作者

## 2. 功能需求

### 2.1 用户认证
- FR1.1: 用户注册
- FR1.2: 用户登录/登出
- FR1.3: 密码重置

### 2.2 照片管理
- FR2.1: 上传照片（支持 JPG、PNG）
- FR2.2: 查看照片列表
- FR2.3: 删除照片
- FR2.4: 批量操作

### 2.3 AI 功能
- FR3.1: 自动目标检测（YOLOv5）
- FR3.2: 语音搜索（Whisper）
- FR3.3: 风格迁移（CycleGAN）
- FR3.4: 自动生成标签

### 2.4 搜索功能
- FR4.1: 按标签搜索
- FR4.2: 按日期搜索
- FR4.3: 语音搜索

## 3. 非功能需求

### 3.1 性能
- NFR1: 单张照片处理时间 < 2 秒
- NFR2: 语音识别延迟 < 500ms
- NFR3: 支持 100 并发用户

### 3.2 可用性
- NFR4: 系统可用性 ≥ 99%
- NFR5: 提供详细错误提示

### 3.3 安全性
- NFR6: 密码加密存储
- NFR7: 防止 SQL 注入
- NFR8: 文件上传验证

### 3.4 兼容性
- NFR9: 支持主流浏览器
- NFR10: 移动端响应式设计

## 4. 技术栈

- 前端：Streamlit
- 后端：FastAPI
- 数据库：SQLite
- AI 模型：YOLOv5, Whisper, CycleGAN
- 部署：Docker

## 5. 项目计划

### 5.1 里程碑
- Week 1: MVP（上传 + 检测）
- Week 2: V1.0（语音搜索 + 标签）
- Week 3: V2.0（风格迁移 + 优化）
- Week 4: 测试与部署

### 5.2 风险
- 风险 1: 模型推理速度慢
  - 缓解：使用 GPU、模型量化
- 风险 2: 内存不足
  - 缓解：分批处理、限制并发
- 风险 3: 用户需求变更
  - 缓解：敏捷开发、定期沟通
```

---

## ✍️ 自我检测练习

### 练习 1：需求优先级排序

**场景：** 你有以下功能想法，但只能实现 3 个，如何选择？

1. 人脸美颜
2. 目标检测
3. 语音搜索
4. 社交分享
5. 自动标签
6. 视频编辑
7. 云同步
8. AI 修图

**你的选择：** ______、______、______

**理由：** _______________________________________

**参考答案：**
```
选择：2（目标检测）、5（自动标签）、3（语音搜索）

理由：
- 目标检测是核心 AI 功能，展示技术实力
- 自动标签提升用户体验，解决"找不到照片"痛点
- 语音搜索是差异化功能，体现多模态集成

不选的理由：
- 人脸美颜：复杂度高，非核心需求
- 社交分享：依赖第三方 API，增加复杂度
- 视频编辑：超出范围，工作量巨大
- 云同步：需要云服务，成本高
- AI 修图：可以后续迭代
```

---

### 练习 2：绘制数据流图

**任务：** 画出"用户上传照片并获取检测结果"的完整数据流。

**参考答案：**
```
用户浏览器
    ↓ (HTTP POST /api/photos)
FastAPI 后端
    ↓ (验证文件)
文件系统（保存原图）
    ↓ (读取图片)
预处理模块（Resize + Normalize）
    ↓ (输入张量)
YOLOv5 模型
    ↓ (输出 bounding boxes)
后处理模块（解析结果）
    ↓ (插入数据库)
SQLite 数据库
    ↓ (查询结果)
FastAPI 后端
    ↓ (JSON 响应)
用户浏览器（显示检测结果）
```

---

### 练习 3：API 设计

**任务：** 设计"语音搜索"的 API 接口。

**参考答案：**
```python
# 请求
POST /api/search/voice
Content-Type: multipart/form-data

{
    "audio_file": <binary_data>,
    "user_id": 123,
    "limit": 20  # 返回结果数量
}

# 响应
{
    "status": "success",
    "data": {
        "transcribed_text": "显示所有猫的照片",
        "extracted_tags": ["猫"],
        "results": [
            {
                "photo_id": 456,
                "thumbnail_url": "/thumbnails/456.jpg",
                "tags": ["猫", "宠物", "室内"],
                "match_score": 0.95
            },
            ...
        ],
        "total_count": 15,
        "processing_time_ms": 850
    }
}

# 错误响应
{
    "status": "error",
    "error": "Invalid audio format",
    "detail": "Only WAV and MP3 are supported"
}
```

---

### 练习 4：风险评估

**任务：** 列出项目可能遇到的 3 个最大风险，并给出缓解措施。

**参考答案：**
```
风险 1: 模型推理速度慢
- 概率：高
- 影响：高
- 缓解措施：
  1. 使用 GPU 加速
  2. 模型量化（FP16/INT8）
  3. 异步处理，后台推理
  4. 缓存常见查询结果

风险 2: 内存泄漏
- 概率：中
- 影响：高
- 缓解措施：
  1. 定期重启 worker 进程
  2. 监控内存使用
  3. 及时释放不再使用的张量
  4. 限制并发处理数量

风险 3: 用户上传恶意文件
- 概率：中
- 影响：中
- 缓解措施：
  1. 严格验证文件类型（魔数检查）
  2. 限制文件大小（< 10MB）
  3. 沙箱环境处理文件
  4. 扫描病毒（ClamAV）
```

---

## 🚀 下一步行动清单

### 完成需求分析后，你应该有：

- ✅ 清晰的功能列表（FR1-FRn）
- ✅ 明确的非功能需求（NFR1-NFRn）
- ✅ 系统架构图
- ✅ 数据库设计（ER 图 + SQL）
- ✅ API 接口定义
- ✅ 技术选型理由
- ✅ 风险评估与缓解措施
- ✅ 项目计划（里程碑）

### 接下来进入 Day21-Q2：技术架构与选型

我们将深入讨论：
- 如何选择合适的深度学习框架
- 模型部署的最佳实践
- 前后端分离 vs 一体化
- 容器化部署（Docker）
- 持续集成/持续部署（CI/CD）

---

## 📝 本章小结

### 需求分析的核心原则

✅ **以用户为中心**：始终思考"这解决了什么痛点"

✅ **MVP 思维**：先做核心功能，再迭代优化

✅ **明确边界**：清楚什么不做，避免范围蔓延

✅ **量化指标**：用数字衡量成功（延迟 < 2s，准确率 > 90%）

✅ **风险管理**：提前识别风险，准备应对方案

---

### 关键 takeaway

1. **需求分析不是形式主义**，而是降低项目失败率的关键
2. **好的设计是演化的**，不是一开始就完美的
3. **文档要适度**，够用就好，不要过度文档化
4. **与用户保持沟通**，确保做的是他们真正需要的
5. **技术选型要务实**，选择最适合的，而不是最流行的

---

**📚 相关文档：**
- [Day21-Q0 - 快速复习 Day20](./Day21-Q0%20-%20快速复习%20Day20.md)
- [Day21-Q2 - 技术架构与选型](./Day21-Q2%20-%20技术架构与选型.md)（待创建）

**💡 提示：** 在实际项目中，需求分析可能需要多次迭代。不要追求一次性完美，先写出初稿，然后逐步完善。

---

## 📱 关于作者 & 获取更多资源

本教程由 **Lee（职场宝爸）** 创建，记录从零基础到独立完成 AI 项目的真实历程。

### 关注公众号，获取独家内容

**公众号名称：Lee 的成长日记**

微信搜索关注，获取：
- ✅ **AI 学习路线规划**：零基础如何系统学习 AI
- ✅ **项目实战源码**：完整可运行的项目代码
- ✅ **深度技术解析**：前沿技术原理 + 手写代码实现
- ✅ **职场成长心得**：一个宝爸的 AI 逆袭之路

**关注福利**：
- 回复「**路线**」→ 获取 30 天 AI 学习计划表
- 回复「**项目**」→ 获取 GitHub 项目源码合集
- 回复「**资料**」→ 获取零基础学习资源推荐

**扫码关注公众号**：

![公众号二维码](../../images/logos/ewm.jpg)

### 其他平台

- 📂 **GitHub**：https://github.com/Lee985-cmd/AI-30Days-Challenge
- 📝 **CSDN 博客**：https://blog.csdn.net/m0_67081842
- 💬 **公众号**：微信搜索「Lee 的成长日记」

---

> 💡 **学习建议**
> 
> 如果本篇教程对你有帮助，欢迎：
> 1. **Star GitHub 项目**：https://github.com/Lee985-cmd/AI-30Days-Challenge
> 2. **关注公众号**获取更多独家内容
> 3. **留言交流**你的学习困惑
> 
> **一起学习，一起进步！** 🤝
