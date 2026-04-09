# 模型部署示例

## 📖 项目简介

将训练好的 AI 模型部署为 Web API 服务。

## 🎯 学习目标

- 掌握 Flask/FastAPI 框架
- 学会 Docker 容器化
- 理解 RESTful API 设计
- 能够部署到云平台

## 📂 项目结构

```
deployment-example/
├── app.py               # Flask/FastAPI 应用
├── model.py             # 模型加载和推理
├── Dockerfile           # Docker 配置
├── docker-compose.yml   # Docker Compose
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 前置要求

- Python 3.7+
- Docker（可选，用于容器化部署）
- 磁盘空间：至少 1GB

### 方式 1: Flask 部署

#### 1. 克隆项目

```bash
git clone https://github.com/Lee985-cmd/AI-30-Day-Challenge.git
cd AI-30-Day-Challenge/projects/deployment-example
```

#### 2. 创建虚拟环境（推荐）

**Windows:**
```bash
python -m venv deploy-env
deploy-env\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv deploy-env
source deploy-env/bin/activate
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> 💡 **国内用户加速：**
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

#### 4. 运行服务

```bash
python app.py
```

服务将在 `http://localhost:5000` 启动。

#### 3. 测试 API

```bash
# 使用 curl
curl -X POST http://localhost:5000/predict \
  -F "file=@test_image.jpg"

# 或使用 Python
import requests

with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'file': f}
    )
    print(response.json())
```

### 方式 2: FastAPI 部署（推荐）

#### 1. 安装依赖

```bash
pip install fastapi uvicorn python-multipart
```

#### 2. 运行服务

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### 3. 访问文档

浏览器打开：`http://localhost:8000/docs`

FastAPI 自动生成 Swagger UI 文档，可以直接在浏览器中测试 API。

### 方式 3: Docker 部署

#### 1. 安装 Docker

- [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
- [Docker for Mac](https://docs.docker.com/desktop/install/mac-install/)
- [Docker for Linux](https://docs.docker.com/engine/install/)

#### 2. 构建镜像

```bash
docker build -t ai-model-api .
```

> ⏱️ **首次构建可能需要 5-10 分钟**（下载基础镜像和依赖）

#### 3. 运行容器

```bash
docker run -p 8000:8000 ai-model-api
```

或使用 Docker Compose：

```bash
docker-compose up -d
```

服务将在 `http://localhost:8000` 启动。

## 📊 API 端点

### POST /predict

**请求:**
```json
{
  "image": "<base64_encoded_image>"
}
```

或 multipart/form-data:
```
file: <image_file>
```

**响应:**
```json
{
  "success": true,
  "prediction": "cat",
  "confidence": 0.95,
  "all_predictions": [
    {"class": "cat", "confidence": 0.95},
    {"class": "dog", "confidence": 0.03},
    {"class": "bird", "confidence": 0.02}
  ],
  "processing_time_ms": 45
}
```

### GET /health

**响应:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 3600
}
```

### GET /metrics

**响应:**
```json
{
  "total_requests": 1000,
  "average_latency_ms": 50,
  "requests_per_second": 20
}
```

## 🔧 配置选项

### 环境变量

```bash
# .env 文件
MODEL_PATH=./models/best.pth
DEVICE=cuda
PORT=8000
WORKERS=4
LOG_LEVEL=info
MAX_REQUEST_SIZE=10MB
```

### Gunicorn 配置（生产环境）

```bash
# 运行
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

## 💡 最佳实践

### 1. 异步处理

```python
from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    # 异步读取文件
    contents = await file.read()
    
    # 异步推理（如果模型支持）
    result = await asyncio.to_thread(model_predict, contents)
    
    return result
```

### 2. 请求限流

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("10/minute")  # 每分钟最多10次
async def predict(request: Request, file: UploadFile):
    ...
```

### 3. 缓存结果

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_predict(image_hash: str):
    # 缓存推理结果
    ...

# 使用时
image_hash = hashlib.md5(image_bytes).hexdigest()
result = cached_predict(image_hash)
```

### 4. 批量处理

```python
@app.post("/predict/batch")
async def batch_predict(files: List[UploadFile]):
    results = []
    for file in files:
        result = await predict_single(file)
        results.append(result)
    return {"results": results}
```

## 🐛 常见问题

### Q: 内存泄漏

**A:**
```python
# 定期清理
import gc
gc.collect()
torch.cuda.empty_cache()

# 限制请求大小
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
```

### Q: 并发性能差

**A:**
```python
# 使用异步框架
# FastAPI + Uvicorn

# 增加 worker 数量
# gunicorn -w 8 ...

# 使用 GPU 批处理
# 将多个请求合并为一个批次
```

### Q: 模型加载慢

**A:**
```python
# 在应用启动时加载模型
@app.on_event("startup")
def load_model():
    global model
    model = torch.load('model.pth')
    model.eval()
```

## 📚 部署到云平台

### Render

```yaml
# render.yaml
services:
  - type: web
    name: ai-model-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
```

### AWS Elastic Beanstalk

```bash
# 安装 EB CLI
pip install awsebcli

# 初始化
eb init

# 部署
eb create ai-model-env
eb deploy
```

### Google Cloud Run

```bash
# 构建并推送
gcloud builds submit --tag gcr.io/PROJECT-ID/ai-model

# 部署
gcloud run deploy ai-model \
  --image gcr.io/PROJECT-ID/ai-model \
  --platform managed
```

## 📈 监控和日志

### Prometheus 指标

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)

# 访问 /metrics 查看指标
```

### 日志记录

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(file: UploadFile):
    logger.info(f"Received prediction request: {file.filename}")
    # ...
    logger.info(f"Prediction completed in {time_ms}ms")
```

## 📚 相关资源

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [Flask 文档](https://flask.palletsprojects.com/)
- [Docker 文档](https://docs.docker.com/)
- [Day 27 教程](../../Day27/)

## 📄 许可证

MIT License
