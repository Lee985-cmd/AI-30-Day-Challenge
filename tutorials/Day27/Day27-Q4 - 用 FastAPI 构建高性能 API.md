# Day27-Q4 - 用 FastAPI 构建高性能 API

## ⚡ 更快、更现代的 Web 框架

### 问题背景

Flask 很好用,但有个问题:**慢**!

当你的 API 要处理大量并发请求时,Flask 就成了瓶颈。

**FastAPI 的优势:**
- ⚡ **快**: 基于异步,性能接近 Go 和 Node.js
- 📝 **自动文档**: 自动生成 Swagger UI 和 ReDoc
- ✅ **类型检查**: 利用 Python 类型提示,减少 bug
- 🚀 **现代**: 支持 async/await,适合高并发

---

## 一、FastAPI vs Flask

### 对比表

| 特性 | Flask | FastAPI |
|------|-------|---------|
| **性能** | 一般 | 非常快 ⚡ |
| **学习曲线** | 简单 | 中等 |
| **异步支持** | 需要扩展 | 原生支持 |
| **自动文档** | 需要插件 | 内置 ✅ |
| **类型检查** | 无 | 有 ✅ |
| **数据验证** | 手动 | 自动 ✅ |
| **社区成熟度** | 非常成熟 | 快速增长 |

**选择建议:**
- 小项目、快速原型 → Flask
- 高性能需求、生产环境 → FastAPI ⭐

---

## 二、FastAPI 基础

### 安装

```bash
pip install fastapi uvicorn
```

- `fastapi`: Web 框架
- `uvicorn`: ASGI 服务器 (类似 Gunicorn)

### Hello World

```python
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

**运行:**
```bash
uvicorn main:app --reload
```

**访问:**
- API: `http://localhost:8000`
- 文档: `http://localhost:8000/docs` (Swagger UI)
- 备选文档: `http://localhost:8000/redoc` (ReDoc)

**特点:**
- 自动生成交互式文档!
- 可以直接在浏览器里测试 API!

---

## 三、图像分类 API (FastAPI 版)

### 完整实现

```python
# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
import io
from typing import Dict, List
import time

# 导入模型
from model import load_model, CLASS_NAMES

# 创建 FastAPI 应用
app = FastAPI(
    title="图像分类 API",
    description="基于深度学习的图像分类服务",
    version="1.0.0"
)

# 加载模型 (启动时加载一次)
model = load_model('model.pth')

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# 定义响应模型 (用于文档和验证)
class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    class_id: int
    confidence: float
    all_probabilities: Dict[str, float]
    processing_time_ms: float


class ErrorResponse(BaseModel):
    success: bool = False
    error: str


# 预测接口
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    上传图片进行分类
    
    - **file**: 图片文件 (JPG/PNG)
    
    返回分类结果和置信度
    """
    
    start_time = time.time()
    
    # 验证文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="只支持图片文件"
        )
    
    try:
        # 读取图片
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # 预处理
        input_tensor = transform(image).unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 计算处理时间
        processing_time = (time.time() - start_time) * 1000
        
        # 构建响应
        result = PredictionResponse(
            success=True,
            prediction=CLASS_NAMES[predicted_class],
            class_id=predicted_class,
            confidence=round(confidence, 4),
            all_probabilities={
                CLASS_NAMES[i]: round(prob.item(), 4) 
                for i, prob in enumerate(probabilities[0])
            },
            processing_time_ms=round(processing_time, 2)
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 批量预测接口
class BatchRequest(BaseModel):
    texts: List[str]


class BatchResult(BaseModel):
    text: str
    sentiment: str
    confidence: float


class BatchResponse(BaseModel):
    success: bool
    count: int
    results: List[BatchResult]


@app.post("/batch_predict", response_model=BatchResponse)
async def batch_predict(request: BatchRequest):
    """
    批量文本情感分析
    
    - **texts**: 文本列表 (最多 100 条)
    """
    
    if len(request.texts) > 100:
        raise HTTPException(
            status_code=400, 
            detail="一次最多处理 100 条文本"
        )
    
    try:
        results = []
        for text in request.texts:
            result = sentiment_pipeline(text)[0]
            results.append(BatchResult(
                text=text,
                sentiment=result['label'],
                confidence=round(result['score'], 4)
            ))
        
        return BatchResponse(
            success=True,
            count=len(results),
            results=results
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 健康检查
@app.get("/health")
def health_check():
    """检查服务状态"""
    return {
        "status": "healthy",
        "model_loaded": True
    }


# 根路径
@app.get("/")
def root():
    """API 信息"""
    return {
        "name": "图像分类 API",
        "version": "1.0.0",
        "docs": "/docs"
    }
```

**运行:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**测试:**
```python
import requests

# 单张图片预测
with open('cat.jpg', 'rb') as f:
    files = {'file': ('cat.jpg', f, 'image/jpeg')}
    response = requests.post('http://localhost:8000/predict', files=files)
    print(response.json())

# 批量预测
response = requests.post(
    'http://localhost:8000/batch_predict',
    json={'texts': ['I love it!', 'Terrible product']}
)
print(response.json())
```

---

## 四、异步处理

### 为什么需要异步?

**同步 (Flask):**
```
请求1 → 等待模型推理 (100ms) → 返回
请求2 → 等待... (阻塞!)
请求3 → 等待... (阻塞!)
```

**异步 (FastAPI):**
```
请求1 → 开始推理 → 切换
请求2 → 开始推理 → 切换
请求3 → 开始推理 → 切换
... (同时处理多个请求)
```

### 异步示例

```python
from fastapi import FastAPI
import asyncio
import time

app = FastAPI()

# 模拟耗时的 AI 推理
async def ai_inference(text: str):
    """异步推理函数"""
    await asyncio.sleep(0.1)  # 模拟 100ms 延迟
    return f"Result for: {text}"


@app.get("/predict/{text}")
async def predict(text: str):
    """异步预测接口"""
    result = await ai_inference(text)
    return {"result": result}


# 可以同时处理多个请求!
```

### 并行处理多个任务

```python
@app.post("/parallel_predict")
async def parallel_predict(texts: List[str]):
    """并行处理多个文本"""
    
    # 创建多个任务
    tasks = [ai_inference(text) for text in texts]
    
    # 并行执行
    results = await asyncio.gather(*tasks)
    
    return {"results": results}
```

---

## 五、依赖注入

### 什么是依赖注入?

**大白话:** 把共用的逻辑抽出来,自动注入到需要的地方。

**例子:** 数据库连接、用户认证、模型加载

### 实际示例

```python
from fastapi import FastAPI, Depends, HTTPException
from typing import Optional

app = FastAPI()

# 依赖1: 获取当前用户
def get_current_user(token: Optional[str] = None):
    """验证用户 token"""
    if not token or token != "secret_token":
        raise HTTPException(status_code=401, detail="未授权")
    return {"user_id": 123, "username": "test_user"}


# 依赖2: 验证 API Key
def verify_api_key(api_key: str):
    """验证 API Key"""
    valid_keys = ["key1", "key2", "key3"]
    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="无效的 API Key")
    return api_key


# 使用依赖
@app.get("/protected")
def protected_endpoint(user: dict = Depends(get_current_user)):
    """需要认证的接口"""
    return {"message": f"Hello, {user['username']}"}


@app.get("/api/data")
def get_data(api_key: str = Depends(verify_api_key)):
    """需要 API Key 的接口"""
    return {"data": "some data", "api_key": api_key}
```

### 模型加载依赖

```python
from functools import lru_cache

@lru_cache()
def get_model():
    """缓存模型加载 (只加载一次)"""
    print("Loading model...")
    return load_model('model.pth')


@app.post("/predict_with_dependency")
def predict_with_dep(
    file: UploadFile = File(...),
    model = Depends(get_model)  # 自动注入模型
):
    """使用依赖注入的预测接口"""
    # 直接使用注入的模型
    # ...
    return {"prediction": "result"}
```

---

## 六、中间件

### 什么是中间件?

**中间件 = 请求的拦截器**

可以在请求到达视图函数之前或之后执行一些逻辑。

### 常用中间件

```python
from fastapi import FastAPI, Request
import time

app = FastAPI()

# 中间件1: 记录请求时间
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    
    # 执行请求
    response = await call_next(request)
    
    # 添加处理时间到响应头
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# 中间件2: CORS (跨域)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名 (生产环境要限制)
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)


# 中间件3: 请求日志
import logging

logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    logger.info(f"Response: {response.status_code}")
    
    return response
```

---

## 七、后台任务

### 场景: 发送通知邮件

```python
from fastapi import BackgroundTasks
import smtplib

def send_email(email: str, message: str):
    """发送邮件 (耗时操作)"""
    # 模拟发送邮件
    print(f"Sending email to {email}: {message}")
    time.sleep(2)  # 模拟延迟
    print("Email sent!")


@app.post("/notify")
async def notify_user(
    email: str,
    background_tasks: BackgroundTasks
):
    """
    发送通知
    
    立即返回,邮件在后台发送
    """
    
    # 添加后台任务
    background_tasks.add_task(send_email, email, "Your prediction is ready!")
    
    return {
        "message": "Notification will be sent",
        "email": email
    }
```

**好处:** 不用等邮件发送完成,立即返回响应!

---

## 八、文件上传优化

### 大文件处理

```python
from fastapi import FastAPI, UploadFile, File
import aiofiles

app = FastAPI()

@app.post("/upload_large_file")
async def upload_large_file(file: UploadFile = File(...)):
    """处理大文件上传"""
    
    # 流式写入,避免内存溢出
    file_path = f"uploads/{file.filename}"
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        while content := await file.read(1024 * 1024):  # 每次读 1MB
            await out_file.write(content)
    
    return {
        "filename": file.filename,
        "saved_to": file_path
    }
```

---

## 九、性能基准测试

### Flask vs FastAPI 对比

```python
# 测试脚本
import requests
import time

def benchmark(url, num_requests=100):
    """基准测试"""
    start = time.time()
    
    for _ in range(num_requests):
        response = requests.get(url)
    
    elapsed = time.time() - start
    rps = num_requests / elapsed
    
    print(f"{num_requests} requests in {elapsed:.2f}s")
    print(f"Requests per second: {rps:.2f}")
    print(f"Avg response time: {elapsed/num_requests*1000:.2f}ms")


# 测试 Flask
print("Flask:")
benchmark('http://localhost:5000/predict')

# 测试 FastAPI
print("\nFastAPI:")
benchmark('http://localhost:8000/predict')
```

**典型结果:**
```
Flask:
100 requests in 12.50s
Requests per second: 8.00
Avg response time: 125.00ms

FastAPI:
100 requests in 2.30s
Requests per second: 43.48
Avg response time: 23.00ms
```

**FastAPI 快了 5 倍!** ⚡

---

## 十、部署 FastAPI

### 方法1: Uvicorn (开发)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 方法2: Gunicorn + Uvicorn Workers (生产)

```bash
pip install gunicorn

gunicorn main:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    -b 0.0.0.0:8000
```

- `-w 4`: 4 个工作进程
- `-k uvicorn.workers.UvicornWorker`: 使用 Uvicorn worker
- `-b 0.0.0.0:8000`: 绑定地址

### 方法3: Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
```

---

## 十一、完整项目结构

```
fastapi_project/
├── main.py              # FastAPI 主应用
├── model.py             # 模型加载
├── schemas.py           # Pydantic 模型
├── dependencies.py      # 依赖注入
├── middleware.py        # 中间件
├── requirements.txt     # 依赖
├── Dockerfile           # Docker 配置
└── tests/               # 测试
    └── test_main.py
```

**requirements.txt:**
```
fastapi==0.100.0
uvicorn==0.23.0
torch==2.0.0
Pillow==9.5.0
python-multipart==0.0.6
gunicorn==20.1.0
pydantic==2.0.0
```

---

## 十二、本章小结

### FastAPI 核心优势

✅ **性能:** 异步支持,速度快 5-10 倍  
✅ **自动文档:** Swagger UI 和 ReDoc 自动生成  
✅ **类型安全:** Pydantic 数据验证  
✅ **依赖注入:** 代码复用和模块化  
✅ **现代化:** 支持 async/await  

### 关键知识点

✅ **路由定义:**
```python
@app.get("/path")
@app.post("/path")
@app.put("/path")
@app.delete("/path")
```

✅ **请求参数:**
```python
# 路径参数
@app.get("/items/{item_id}")

# 查询参数
@app.get("/items/?skip=0&limit=10")

# 请求体
@app.post("/items/")
def create(item: ItemModel):

# 文件上传
@app.post("/upload/")
def upload(file: UploadFile = File(...)):
```

✅ **响应模型:**
```python
class ResponseModel(BaseModel):
    field1: str
    field2: int

@app.get("/data", response_model=ResponseModel)
```

✅ **依赖注入:**
```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/")
def read_users(db: Session = Depends(get_db)):
```

### Flask vs FastAPI 选择

| 场景 | 推荐 |
|------|------|
| 快速原型、小项目 | Flask |
| 高性能需求 | FastAPI ⭐ |
| 需要自动文档 | FastAPI ⭐ |
| 团队熟悉 Flask | Flask |
| 新项目、微服务 | FastAPI ⭐ |

---

## 🎯 下一步

学会了 FastAPI,接下来学习如何容器化和部署:

- [Q5](./Day27-Q5%20-%20Docker%20容器化部署.md): Docker 详解
- [Q6](./Day27-Q6%20-%20云平台部署实战.md): 部署到 Render/AWS

**继续前进!** 🚀

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
