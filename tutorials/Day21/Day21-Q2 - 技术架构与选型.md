# Day21-Q2 - 技术架构与选型

## 📝 问题描述

在完成需求分析后，我们需要选择合适的技术栈并设计系统架构。这个决策将影响项目的开发效率、性能、可维护性和扩展性。

**核心问题：**
- 如何在众多技术选项中做出最佳选择？
- 如何设计一个既简单又可扩展的架构？
- 如何平衡新技术的学习成本和项目收益？
- 如何避免"技术债务"累积？

---

## 💡 核心答案

技术选型没有"最好"，只有"最合适"。我们的选择标准是：

1. **适合当前阶段**：不过度设计，满足 MVP 需求
2. **学习曲线平缓**：团队能快速上手
3. **社区活跃**：遇到问题能找到解决方案
4. **生态丰富**：有丰富的库和工具可用
5. **易于维护**：代码清晰，文档完善

对于我们的智能相册项目，最终技术栈是：
- **前端**：Streamlit（快速原型）
- **后端**：FastAPI（高性能 API）
- **数据库**：SQLite（简单够用）
- **AI 模型**：YOLOv5 + Whisper + CycleGAN
- **部署**：Docker（容器化）

---

## 🎓 三个版本的解答

### 版本一：初学者比喻版（5 分钟理解）

#### 把技术选型比作"选交通工具"

想象你要从北京去上海，有多种交通方式可选：

**选项对比：**

| 交通工具 | 速度 | 成本 | 舒适度 | 适用场景 |
|---------|------|------|--------|---------|
| 步行 | 极慢 | 免费 | 累 | 短距离（< 5km） |
| 自行车 | 慢 | 低 | 一般 | 中距离（5-20km） |
| 汽车 | 中 | 中 | 好 | 灵活出行 |
| 高铁 | 快 | 中高 | 很好 | 长途旅行（推荐） |
| 飞机 | 最快 | 高 | 好 | 超长途/紧急 |
| 火箭 | 超快 | 极高 | 复杂 | 太空旅行（过度） |

**类比到技术选型：**

| 技术方案 | 开发速度 | 学习成本 | 性能 | 适用场景 |
|---------|---------|---------|------|---------|
| 纯 Python | 快 | 低 | 慢 | 原型验证 |
| Flask | 快 | 低 | 中 | 小型 Web 应用 |
| FastAPI | 快 | 中 | 快 | 现代 API 服务（推荐） |
| Django | 中 | 中 | 中 | 大型 Web 应用 |
| Microservices | 慢 | 高 | 快 | 超大规模系统 |
| Kubernetes | 极慢 | 极高 | 快 | 云原生企业级（过度） |

**选择原则：**
- ✅ **短途用自行车**：小项目用轻量级工具
- ✅ **长途用高铁**：中型项目用成熟框架
- ❌ **不要用火箭买菜**：不要过度设计

---

#### 把系统架构比作"餐厅运营"

**单体架构（Monolithic）= 小餐馆**

```
┌──────────────────────┐
│     小餐馆            │
│                      │
│  老板 = 厨师 = 服务员 │
│  一个人搞定所有事     │
│                      │
│  ✅ 简单高效          │
│  ✅ 沟通成本低        │
│  ❌ 忙不过来          │
│  ❌ 一人请假就停业    │
└──────────────────────┘
```

**适用场景：**
- 顾客少（用户量 < 1 万）
- 菜单简单（功能 < 10 个）
- 团队小（1-3 人）

---

**微服务架构（Microservices）= 美食广场**

```
┌─────────────────────────────────┐
│       美食广场                   │
│                                 │
│  🍜 面馆   🍣 寿司   🍕 披萨    │
│  (独立运营) (独立运营) (独立运营)│
│                                 │
│  ✅ 各司其职                    │
│  ✅ 一家关门不影响其他           │
│  ❌ 协调复杂                    │
│  ❌ 成本高                      │
└─────────────────────────────────┘
```

**适用场景：**
- 顾客多（用户量 > 10 万）
- 菜品种类多（功能 > 50 个）
- 团队大（> 10 人）
- 需要独立扩展某个业务

---

**我们的选择：小餐馆模式（单体架构）**

**理由：**
1. 我们是小团队（1-3 人）
2. 功能相对简单（上传、检测、搜索）
3. 快速迭代比完美架构更重要
4. 后续可以重构为微服务

**就像开餐馆：**
- 第一阶段：先开个小店，验证商业模式
- 第二阶段：生意好了再扩大
- 第三阶段：连锁经营时才需要标准化流程

---

### 版本二：学生技术版（深入理解原理）

#### 前端技术选型对比

##### Streamlit vs Flask + React vs Django

**Streamlit（我们的选择）**

```python
import streamlit as st
from PIL import Image

st.title("智能相册管理系统")

uploaded_file = st.file_uploader("上传照片", type=['jpg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的照片")
    
    if st.button("开始检测"):
        with st.spinner("处理中..."):
            result = detect_objects(image)
            st.json(result)
            st.success("检测完成！")
```

**优点：**
- ✅ 纯 Python，无需学习 HTML/CSS/JavaScript
- ✅ 50 行代码搭建完整 UI
- ✅ 自动处理状态管理
- ✅ 内置组件丰富（文件上传、图表、表格）
- ✅ 适合数据科学/AI 项目

**缺点：**
- ❌ 定制化能力有限
- ❌ 不适合复杂交互
- ❌ 性能不如专业前端框架

**适用场景：**
- AI/ML 项目原型
- 数据可视化工具
- 内部管理系统
- 快速验证想法

---

**Flask + React**

```python
# Flask 后端
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/detect', methods=['POST'])
def detect():
    file = request.files['image']
    result = model.detect(file)
    return jsonify(result)
```

```javascript
// React 前端
function App() {
  const [result, setResult] = useState(null);
  
  const handleUpload = async (file) => {
    const formData = new FormData();
    formData.append('image', file);
    
    const response = await fetch('/api/detect', {
      method: 'POST',
      body: formData
    });
    
    setResult(await response.json());
  };
  
  return (
    <div>
      <input type="file" onChange={handleUpload} />
      {result && <DetectionResult data={result} />}
    </div>
  );
}
```

**优点：**
- ✅ 完全定制化
- ✅ 前后端分离，职责清晰
- ✅ 适合复杂交互
- ✅ 生态系统庞大

**缺点：**
- ❌ 学习曲线陡峭（需掌握 HTML/CSS/JS/React）
- ❌ 开发周期长（至少 2-3 倍代码量）
- ❌ 需要处理 CORS、状态管理等复杂问题

**适用场景：**
- 商业产品
- 复杂 Web 应用
- 需要精细控制 UI/UX

---

**Django**

```python
# Django View
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def detect_view(request):
    if request.method == 'POST':
        image = request.FILES['image']
        result = model.detect(image)
        return JsonResponse(result)
```

**优点：**
- ✅ "Battery-included"（自带 ORM、Admin、Auth）
- ✅ 适合 CRUD 密集型应用
- ✅ 安全性好（CSRF、SQL 注入防护）
- ✅ 社区成熟

**缺点：**
- ❌ 重量级，启动慢
- ❌ 模板引擎学习成本
- ❌ 对 API 项目不够灵活

**适用场景：**
- 内容管理系统（CMS）
- 电商网站
- 博客平台

---

**决策矩阵：**

| 维度 | Streamlit | Flask+React | Django |
|------|-----------|-------------|--------|
| 开发速度 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| 学习成本 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| 定制化 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 性能 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 适合 AI 项目 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

**结论：** 对于 AI 原型项目，Streamlit 是最佳选择。

---

#### 后端技术选型对比

##### FastAPI vs Flask vs Django REST Framework

**FastAPI（我们的选择）**

```python
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI()

class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: dict

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    image = await file.read()
    result = model.detect(image)
    return result

@app.get("/api/photos/{photo_id}")
async def get_photo(photo_id: int):
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    return photo
```

**优点：**
- ✅ 基于类型提示，自动生成文档（Swagger UI）
- ✅ 异步支持，高性能
- ✅ 数据验证自动化（Pydantic）
- ✅ 现代化设计，符合 Python 最佳实践
- ✅ 性能接近 Go/Node.js

**缺点：**
- ❌ 相对年轻（2018 年发布）
- ❌ 生态不如 Flask/Django 成熟

**性能对比：**
```
框架        | Requests/sec | Latency (ms)
-----------|--------------|-------------
FastAPI    | 50,000       | 2
Flask      | 15,000       | 6
Django     | 10,000       | 10
```

---

**Flask**

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/detect', methods=['POST'])
def detect():
    file = request.files['image']
    result = model.detect(file)
    return jsonify(result)
```

**优点：**
- ✅ 极简主义，易学易用
- ✅ 生态丰富（大量扩展）
- ✅ 社区成熟

**缺点：**
- ❌ 同步阻塞，性能较差
- ❌ 缺少内置数据验证
- ❌ 需要手动配置很多东西

---

**Django REST Framework**

```python
from rest_framework.views import APIView
from rest_framework.response import Response

class DetectView(APIView):
    def post(self, request):
        file = request.FILES['image']
        result = model.detect(file)
        return Response(result)
```

**优点：**
- ✅ 集成 Django ORM、Auth
- ✅ 丰富的序列化器
- ✅ 权限控制完善

**缺点：**
- ❌ 重量级
- ❌ 学习曲线陡
- ❌ 对简单 API 项目过于复杂

---

**决策：** 选择 FastAPI

**理由：**
1. 高性能（异步支持）
2. 自动文档（节省时间）
3. 类型安全（减少 bug）
4. 现代化（未来趋势）

---

#### 数据库选型对比

##### SQLite vs PostgreSQL vs MongoDB

**SQLite（我们的选择）**

```python
import sqlite3

conn = sqlite3.connect('photos.db')
cursor = conn.cursor()

# 创建表
cursor.execute('''
    CREATE TABLE IF NOT EXISTS photos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT NOT NULL,
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

# 插入数据
cursor.execute(
    'INSERT INTO photos (file_path) VALUES (?)',
    ('/uploads/photo_123.jpg',)
)
conn.commit()

# 查询数据
cursor.execute('SELECT * FROM photos WHERE id = ?', (123,))
photo = cursor.fetchone()
```

**优点：**
- ✅ 零配置，无需安装服务器
- ✅ 单文件数据库，便于备份
- ✅ 适合小规模应用（< 100GB）
- ✅ Python 内置支持

**缺点：**
- ❌ 不支持高并发写入
- ❌ 缺少高级功能（分区、复制）
- ❌ 不适合分布式系统

**适用场景：**
- 个人项目
- 原型开发
- 嵌入式系统
- 读多写少的应用

---

**PostgreSQL**

```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="photos",
    user="admin",
    password="secret"
)
```

**优点：**
- ✅ 功能强大（JSONB、全文搜索、GIS）
- ✅ 高并发支持
- ✅ ACID 合规
- ✅ 扩展性强

**缺点：**
- ❌ 需要安装和维护
- ❌ 配置复杂
- ❌ 对小项目过度

---

**MongoDB**

```python
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['photos']
collection = db['images']

# 插入文档
collection.insert_one({
    'file_path': '/uploads/photo.jpg',
    'tags': ['cat', 'pet'],
    'metadata': {'width': 1920, 'height': 1080}
})
```

**优点：**
- ✅ 灵活的 Schema
- ✅ 适合非结构化数据
- ✅ 水平扩展容易

**缺点：**
- ❌ 不支持事务（早期版本）
- ❌ 内存占用高
- ❌ 学习新的查询语言

---

**决策：** 选择 SQLite

**理由：**
1. 项目规模小（预计 < 10GB 数据）
2. 并发不高（< 100 用户）
3. 零配置，快速启动
4. 后续可迁移到 PostgreSQL

---

#### AI 模型选型

##### 目标检测：YOLOv5 vs Faster R-CNN vs SSD

**YOLOv5（我们的选择）**

```python
import torch

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 推理
results = model(image)

# 解析结果
detections = results.pandas().xyxy[0]
print(detections)
#   xmin   ymin   xmax   ymax  confidence  class    name
# 0  100    150    300    350    0.95       0       cat
# 1    0    300    640    500    0.87      56      sofa
```

**性能对比：**

| 模型 | mAP | FPS (GPU) | 模型大小 |
|------|-----|-----------|---------|
| YOLOv5s | 37.4% | 140 | 14 MB |
| YOLOv5m | 45.4% | 80 | 41 MB |
| Faster R-CNN | 42.0% | 20 | 160 MB |
| SSD | 25.1% | 45 | 26 MB |

**优点：**
- ✅ 速度快（实时检测）
- ✅ 精度高
- ✅ 易部署
- ✅ 社区活跃

**缺点：**
- ❌ 对小物体检测稍弱
- ❌ 需要 GPU 才能发挥性能

---

##### 语音识别：Whisper vs Wav2Vec 2.0 vs DeepSpeech

**Whisper（我们的选择）**

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.wav")

print(result['text'])
# "今天天气真好"
```

**对比：**

| 模型 | 多语言 | 精度 | 速度 | 模型大小 |
|------|--------|------|------|---------|
| Whisper base | ✅ 99 种 | 高 | 中 | 142 MB |
| Wav2Vec 2.0 | ❌ 英文为主 | 高 | 快 | 95 MB |
| DeepSpeech | ❌ 英文为主 | 中 | 快 | 50 MB |

**优点：**
- ✅ 开箱即用，无需微调
- ✅ 支持中文
- ✅ 鲁棒性强（噪音环境）

**缺点：**
- ❌ 模型较大
- ❌ 推理较慢（需要 GPU）

---

##### 风格迁移：CycleGAN vs StyleGAN2 vs FastNeuralStyle

**CycleGAN（我们的选择）**

```python
# 使用预训练 CycleGAN
from cycle_gan import CycleGAN

model = CycleGAN.load_pretrained('horse2zebra')
stylized_image = model.transfer_style(original_image)
```

**对比：**

| 模型 | 需要配对数据 | 效果 | 速度 |
|------|------------|------|------|
| CycleGAN | ❌ 不需要 | 好 | 中 |
| StyleGAN2 | ❌ 不需要 | 极好 | 慢 |
| FastNeuralStyle | ✅ 需要 | 一般 | 快 |

**优点：**
- ✅ 无需配对数据
- ✅ 风格多样
- ✅ 效果好

**缺点：**
- ❌ 训练时间长
- ❌ 推理需要 GPU

---

### 版本三：工程师实践版（生产级架构）

#### 完整系统架构图

```
┌──────────────────────────────────────────────────────────┐
│                     Client Layer                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Web Browser │  │ Mobile App   │  │ API Clients    │ │
│  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘ │
└─────────┼────────────────┼──────────────────┼──────────┘
          │ HTTP/WebSocket │                  │
┌─────────▼────────────────▼──────────────────▼──────────┐
│                 API Gateway (Optional)                 │
│  - Rate Limiting  - Authentication  - Load Balancing   │
└────────────────────────┬───────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────┐
│              Application Layer (FastAPI)               │
│                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Auth Module  │  │ Photo Module │  │ Search Module│ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
└─────────┼─────────────────┼─────────────────┼─────────┘
          │                 │                 │
┌─────────▼─────────────────▼─────────────────▼─────────┐
│                Service Layer                           │
│                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ CV Service   │  │ ASR Service  │  │ GAN Service  │ │
│  │ (YOLOv5)     │  │ (Whisper)    │  │ (CycleGAN)   │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
└─────────┼─────────────────┼─────────────────┼─────────┘
          │                 │                 │
┌─────────▼─────────────────▼─────────────────▼─────────┐
│                 Model Layer                            │
│  - Model Loading  - Inference  - Caching  - Monitoring│
└────────────────────────┬───────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────┐
│                 Data Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │
│  │ SQLite   │  │ Redis    │  │ File System (S3/NAS) │ │
│  └──────────┘  └──────────┘  └──────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

---

#### Docker 容器化部署

**项目结构：**
```
smart-album/
├── app/
│   ├── main.py
│   ├── models/
│   ├── services/
│   └── database.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

**Dockerfile：**
```dockerfile
# 使用官方 Python 镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY ./app /app

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml：**
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads
      - ./models:/app/models
    environment:
      - DATABASE_URL=sqlite:///./photos.db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

**启动命令：**
```bash
# 构建并启动
docker-compose up --build

# 后台运行
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

---

#### 监控与日志

**使用 Prometheus + Grafana 监控：**

```python
from prometheus_client import Counter, Histogram, start_http_server

# 定义指标
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')

# 中间件
@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.observe(duration)
    
    return response

# 启动指标服务器
start_http_server(8001)
```

**日志配置：**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 使用
logger.info(f"Processing photo {photo_id}")
logger.error(f"Failed to process photo: {error}", exc_info=True)
```

---

## ⚠️ 常见错误与避坑指南

### 错误 1：盲目追求新技术

**❌ 错误做法：**
```python
# 为了用 Rust 而用 Rust
# 为了用 GraphQL 而用 GraphQL
# 为了用 Kubernetes 而用 Kubernetes
```

**✅ 正确做法：**
```python
# 问自己三个问题：
# 1. 这个技术解决了什么实际问题？
# 2. 团队能否快速上手？
# 3. 是否有更简单的替代方案？

# 如果答案是否定的，就不要用
```

---

### 错误 2：忽视技术债务

**❌ 错误做法：**
```python
# 临时方案变成永久方案
def process_image(path):
    # TODO: 重构这段代码
    # HACK: 这样可以工作
    # FIXME: 这里有 bug
    pass
```

**✅ 正确做法：**
```python
# 记录技术债务
# tech_debt.md
"""
技术债务清单：
1. 图片处理函数需要重构（优先级：高）
2. 添加单元测试（优先级：中）
3. 优化数据库查询（优先级：低）

计划：每个 Sprint 预留 20% 时间还债
"""
```

---

### 错误 3：缺乏文档

**❌ 错误做法：**
```python
def calc(x, y):
    return x * y + 10
```

**✅ 正确做法：**
```python
def calculate_final_score(base_score: float, bonus: float) -> float:
    """
    计算最终得分
    
    Args:
        base_score: 基础分数（0-100）
        bonus: 额外加分
    
    Returns:
        最终得分（可能超过 100）
    
    Example:
        >>> calculate_final_score(80, 10)
        810
    """
    return base_score * bonus + 10
```

---

### 错误 4：不写测试

**❌ 错误做法：**
```python
# 没有测试，手动点击测试
```

**✅ 正确做法：**
```python
import pytest

def test_detect_objects():
    image = load_test_image()
    result = detect_objects(image)
    
    assert len(result) > 0
    assert all('class' in det for det in result)
    assert all('confidence' in det for det in result)

def test_invalid_image():
    with pytest.raises(ValueError):
        detect_objects(None)
```

---

## ✍️ 自我检测练习

### 练习 1：技术选型决策

**场景：** 你要开发一个实时聊天应用，支持 1000 并发用户。

**问题：**
1. 前端选什么？
2. 后端选什么？
3. 数据库选什么？
4. 如何实现实时通信？

**参考答案：**
```
1. 前端：React + WebSocket
   - 理由：需要复杂交互，WebSocket 支持实时更新

2. 后端：Node.js (Socket.io) 或 Go
   - 理由：高并发，异步 I/O

3. 数据库：PostgreSQL + Redis
   - 理由：PG 存历史消息，Redis 缓存在线用户

4. 实时通信：WebSocket
   - 理由：双向通信，低延迟
```

---

### 练习 2：架构设计

**任务：** 画出智能相册系统的组件交互图。

**参考答案：**
```
用户 → Streamlit UI → FastAPI Backend
                          ↓
                    ┌─────┴─────┐
                    ↓           ↓
              YOLOv5 Service  Whisper Service
                    ↓           ↓
              SQLite DB ←──→ Redis Cache
                    ↓
              File Storage
```

---

### 练习 3：Docker 配置

**任务：** 编写一个支持 GPU 的 Dockerfile。

**参考答案：**
```dockerfile
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "main.py"]
```

---

## 📝 本章小结

### 技术选型的核心原则

✅ **合适优于先进**：选择最适合当前阶段的技術

✅ **简单优于复杂**：能用简单方案就不用复杂方案

✅ **社区优于冷门**：选择有活跃社区的技术

✅ **文档优于黑盒**：选择文档完善的技术

✅ **可替换优于绑定**：保持架构灵活性

---

### 我们的技术栈总结

| 层级 | 技术 | 理由 |
|------|------|------|
| 前端 | Streamlit | 快速原型，Python 原生 |
| 后端 | FastAPI | 高性能，自动文档 |
| 数据库 | SQLite | 零配置，简单够用 |
| 缓存 | Redis | 加速热点查询 |
| CV 模型 | YOLOv5 | 实时检测，精度高 |
| ASR 模型 | Whisper | 多语言，开箱即用 |
| GAN 模型 | CycleGAN | 无需配对数据 |
| 部署 | Docker | 容器化，易部署 |

---

**📚 相关文档：**
- [Day21-Q1 - 项目需求分析与设计](./Day21-Q1%20-%20项目需求分析与设计.md)
- [Day21-Q3 - 核心功能实现](./Day21-Q3%20-%20核心功能实现.md)（待创建）

**💡 提示：** 技术选型不是一成不变的，随着项目发展，可能需要调整架构。关键是保持灵活性，避免过早优化。

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

![公众号二维码](../../../images/logos/ewm.jpg)

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
