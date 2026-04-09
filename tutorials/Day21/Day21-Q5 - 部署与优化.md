# Day21-Q5 - 部署与优化

## 📝 问题描述

完成了功能开发后，需要将应用部署到生产环境，让用户能够真正使用。同时，需要优化性能、提升用户体验、确保系统稳定运行。

**核心问题：**
- 如何将本地应用部署到服务器？
- 如何优化模型推理速度？
- 如何处理高并发请求？
- 如何监控系统运行状态？
- 如何保证数据安全？

---

## 💡 核心答案

部署和优化的核心原则：

1. **容器化**：使用 Docker 确保环境一致性
2. **分层优化**：从模型、代码、架构多个层面优化
3. **监控告警**：及时发现问题
4. **安全第一**：保护用户数据和系统安全

我们将按照以下步骤进行：
1. Docker 容器化部署
2. 模型优化（量化、剪枝、蒸馏）
3. 性能优化（缓存、异步、负载均衡）
4. 监控与日志
5. 安全加固

---

## 🎓 三个版本的解答

### 版本一：初学者比喻版（5 分钟理解）

#### 把部署比作"开连锁店"

想象你在北京开了一家成功的餐厅，现在要在全国开设连锁店。

**问题 1：如何保证每家店口味一致？**

❌ **错误做法：** 每家店自己摸索菜谱

✅ **正确做法：** 标准化配方 + 中央厨房

**类比到软件：**
- 标准化配方 = Docker 镜像
- 中央厨房 = CI/CD 流水线
- 每家店 = 不同的服务器

---

**问题 2：如何让上菜更快？**

**优化策略：**

1. **提前备菜（缓存）**
   - 把常用的食材准备好
   - 类比：缓存热点数据

2. **多个厨师并行（并发）**
   - 不要一个厨师做所有菜
   - 类比：多线程/多进程处理

3. **简化菜谱（模型优化）**
   - 去掉不必要的步骤
   - 类比：模型量化、剪枝

4. **优化厨房布局（架构优化）**
   - 冰箱、灶台、出餐口合理摆放
   - 类比：微服务、负载均衡

---

**问题 3：如何知道餐厅运营是否正常？**

**监控指标：**
- 👥 顾客数量（并发用户数）
- ⏱️ 上菜时间（响应延迟）
- 😊 顾客满意度（错误率）
- 💰 营业额（吞吐量）

**类比到软件：**
- QPS（每秒请求数）
- Latency（延迟）
- Error Rate（错误率）
- CPU/Memory Usage（资源使用）

---

### 版本二：学生技术版（深入理解原理）

#### 1. Docker 容器化部署

**完整 Docker 配置：**

**Dockerfile：**
```dockerfile
# 阶段 1: 构建依赖
FROM python:3.9-slim as builder

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 阶段 2: 运行时
FROM python:3.9-slim

WORKDIR /app

# 复制系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 阶段复制安装的包
COPY --from=builder /install /usr/local

# 复制应用代码
COPY ./app /app/app
COPY ./models /app/models

# 创建上传目录
RUN mkdir -p /app/uploads/original /app/uploads/processed

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**docker-compose.yml：**
```yaml
version: '3.8'

services:
  # 主应用
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
      - MODEL_CACHE_DIR=/app/models
    depends_on:
      redis:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    networks:
      - app-network

  # Redis 缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - app-network

  # Nginx 反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web
    restart: unless-stopped
    networks:
      - app-network

  # 监控 (Prometheus)
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped
    networks:
      - app-network

  # 可视化 (Grafana)
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - app-network

volumes:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  app-network:
    driver: bridge
```

**Nginx 配置（nginx.conf）：**
```nginx
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server web:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;

        # 重定向到 HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL 证书
        ssl_certificate /etc/nginx/ssl/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/privkey.pem;

        # 安全头
        add_header X-Frame-Options "SAMEORIGIN";
        add_header X-Content-Type-Options "nosniff";
        add_header X-XSS-Protection "1; mode=block";

        # 最大上传文件大小
        client_max_body_size 10M;

        location / {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket 支持
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # 静态文件
        location /static/ {
            alias /app/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

---

#### 2. 模型优化技术

**A. 模型量化（Quantization）**

```python
import torch
from torch.quantization import quantize_dynamic

class ModelQuantizer:
    """模型量化工具"""
    
    @staticmethod
    def quantize_to_int8(model: torch.nn.Module) -> torch.nn.Module:
        """
        INT8 量化
        
        优点：模型大小减少 75%，速度提升 3-4 倍
        缺点：精度损失 1-2%
        """
        quantized_model = quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM},  # 要量化的层
            dtype=torch.qint8
        )
        
        return quantized_model
    
    @staticmethod
    def convert_to_fp16(model: torch.nn.Module) -> torch.nn.Module:
        """
        FP16 半精度转换
        
        优点：模型大小减少 50%，速度提升 1.5-2 倍
        缺点：精度损失 < 0.5%
        """
        return model.half()
    
    @staticmethod
    def save_quantized_model(model: torch.nn.Module, path: str):
        """保存量化模型"""
        torch.save(model.state_dict(), path)
        print(f"✓ Quantized model saved to {path}")
        
        # 打印模型大小
        import os
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  Model size: {size_mb:.2f} MB")

# 使用示例
model = load_yolo_model()

# FP16 量化
model_fp16 = ModelQuantizer.convert_to_fp16(model)
ModelQuantizer.save_quantized_model(model_fp16, "yolov5_fp16.pt")

# INT8 量化
model_int8 = ModelQuantizer.quantize_to_int8(model)
ModelQuantizer.save_quantized_model(model_int8, "yolov5_int8.pt")
```

**量化效果对比：**

| 精度 | 模型大小 | 推理速度 | 精度损失 | 适用场景 |
|------|---------|---------|---------|---------|
| FP32 | 100% | 1x | 0% | 训练、高精度需求 |
| FP16 | 50% | 1.5-2x | < 0.5% | GPU 推理（推荐） |
| INT8 | 25% | 3-4x | 1-2% | CPU/移动端推理 |

---

**B. 模型剪枝（Pruning）**

```python
import torch.nn.utils.prune as prune

class ModelPruner:
    """模型剪枝工具"""
    
    @staticmethod
    def prune_model(model: torch.nn.Module, amount: float = 0.3):
        """
        非结构化剪枝
        
        Args:
            model: 要剪枝的模型
            amount: 剪枝比例（0.3 = 移除 30% 的参数）
        """
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
        
        print(f"✓ Pruned {amount*100}% of weights")
    
    @staticmethod
    def make_pruning_permanent(model: torch.nn.Module):
        """将剪枝永久化"""
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.remove(module, 'weight')
        
        print("✓ Pruning made permanent")
    
    @staticmethod
    def evaluate_sparsity(model: torch.nn.Module) -> float:
        """评估稀疏度"""
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        sparsity = zero_params / total_params
        return sparsity

# 使用示例
model = load_model()

# 剪枝
ModelPruner.prune_model(model, amount=0.3)

# 检查稀疏度
sparsity = ModelPruner.evaluate_sparsity(model)
print(f"Sparsity: {sparsity:.2%}")

# 永久化剪枝
ModelPruner.make_pruning_permanent(model)
```

---

**C. 知识蒸馏（Knowledge Distillation）**

```python
import torch.nn.functional as F

class KnowledgeDistillation:
    """知识蒸馏"""
    
    def __init__(self, teacher_model, student_model, temperature=4.0):
        """
        Args:
            teacher_model: 大模型（教师）
            student_model: 小模型（学生）
            temperature: 温度参数（软化概率分布）
        """
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
    
    def distillation_loss(self, student_logits, teacher_logits, labels, alpha=0.7):
        """
        蒸馏损失 = α * KL散度 + (1-α) * 交叉熵
        
        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            labels: 真实标签
            alpha: 平衡系数
        """
        # 软化概率分布
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL 散度
        kd_loss = F.kl_div(
            soft_student, 
            soft_teacher, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 交叉熵
        ce_loss = F.cross_entropy(student_logits, labels)
        
        # 总损失
        loss = alpha * kd_loss + (1 - alpha) * ce_loss
        
        return loss
    
    def train_step(self, inputs, labels, optimizer):
        """训练一步"""
        self.teacher.eval()
        self.student.train()
        
        # 前向传播
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        
        student_logits = self.student(inputs)
        
        # 计算损失
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

# 使用示例
teacher = load_large_model()  # YOLOv5x
student = load_small_model()  # YOLOv5s

distiller = KnowledgeDistillation(teacher, student, temperature=4.0)
optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

# 训练
for epoch in range(10):
    for inputs, labels in dataloader:
        loss = distiller.train_step(inputs, labels, optimizer)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

---

#### 3. 性能优化策略

**A. 缓存优化**

```python
import redis
import hashlib
import pickle
from functools import wraps

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, redis_url="redis://localhost:6379/0"):
        self.redis_client = redis.from_url(redis_url)
    
    def cache_result(self, ttl=3600):
        """
        缓存装饰器
        
        Args:
            ttl: 过期时间（秒）
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self._generate_key(func.__name__, args, kwargs)
                
                # 尝试从缓存获取
                cached = self.redis_client.get(cache_key)
                if cached:
                    return pickle.loads(cached)
                
                # 执行函数
                result = func(*args, **kwargs)
                
                # 存入缓存
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    pickle.dumps(result)
                )
                
                return result
            
            return wrapper
        return decorator
    
    def _generate_key(self, func_name: str, args, kwargs) -> str:
        """生成缓存键"""
        key_data = f"{func_name}:{str(args)}:{str(kwargs)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def invalidate_cache(self, pattern: str):
        """使缓存失效"""
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)

# 使用示例
cache = CacheManager()

@cache.cache_result(ttl=3600)  # 缓存 1 小时
def detect_objects(image_path: str):
    # 检测结果会被缓存
    return model.detect(image_path)
```

**缓存命中率监控：**
```python
class CacheMetrics:
    def __init__(self):
        self.hits = 0
        self.misses = 0
    
    def record_hit(self):
        self.hits += 1
    
    def record_miss(self):
        self.misses += 1
    
    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
    
    def get_report(self) -> Dict:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{self.get_hit_rate():.2%}"
        }
```

---

**B. 异步处理**

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import BackgroundTasks

class AsyncProcessor:
    """异步处理器"""
    
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_image_async(self, image_path: str):
        """异步处理图像"""
        loop = asyncio.get_event_loop()
        
        # 在线程池中执行
        result = await loop.run_in_executor(
            self.executor,
            self._process_image_sync,
            image_path
        )
        
        return result
    
    def _process_image_sync(self, image_path: str):
        """同步处理图像（在线程中运行）"""
        # 加载图像
        image = Image.open(image_path)
        
        # 检测
        detections = yolo_model.detect(image)
        
        return detections

# FastAPI 后台任务
@app.post("/api/upload")
async def upload_photo(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    # 立即返回
    photo_id = save_photo(file)
    
    # 后台处理
    background_tasks.add_task(process_photo_async, photo_id)
    
    return {"photo_id": photo_id, "status": "processing"}

async def process_photo_async(photo_id: int):
    """后台处理照片"""
    try:
        # 检测
        detections = await async_processor.process_image_async(
            f"/uploads/{photo_id}.jpg"
        )
        
        # 保存结果
        save_detections(photo_id, detections)
        
        # 发送通知
        send_notification(photo_id, "Processing complete")
    
    except Exception as e:
        logger.error(f"Background processing failed: {e}")
```

---

**C. 负载均衡**

```python
# 使用 Gunicorn + Uvicorn Workers

# gunicorn_config.py
bind = "0.0.0.0:8000"
workers = 4  # CPU 核心数 * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5

# 启动命令
# gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 4
```

**水平扩展：**
```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  web:
    deploy:
      replicas: 3  # 3 个实例
      resources:
        limits:
          cpus: '2'
          memory: 4G

  nginx:
    # Nginx 自动负载均衡
```

---

#### 4. 监控与告警

**Prometheus 配置（prometheus.yml）：**
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fastapi_app'
    static_configs:
      - targets: ['web:8000']
    metrics_path: '/metrics'

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']
```

**FastAPI 集成 Prometheus：**
```python
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# 自动添加指标
Instrumentator().instrument(app).expose(app)

# 自定义指标
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    'custom_requests_total',
    'Total requests',
    ['method', 'endpoint']
)

REQUEST_LATENCY = Histogram(
    'custom_request_latency_seconds',
    'Request latency',
    ['endpoint']
)

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(duration)
    
    return response
```

**Grafana 仪表盘：**

关键指标：
- QPS（每秒请求数）
- P95/P99 延迟
- 错误率
- CPU/内存使用率
- 缓存命中率
- GPU 利用率

---

#### 5. 安全加固

**A. 身份认证**

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict):
    """创建 JWT Token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str = Depends(oauth2_scheme)):
    """验证 Token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

@app.post("/token")
async def login(username: str, password: str):
    """登录获取 Token"""
    # 验证用户名密码
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # 创建 Token
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/photos")
async def get_photos(current_user: str = Depends(verify_token)):
    """需要认证的端点"""
    return get_user_photos(current_user)
```

**B. 速率限制**

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/upload")
@limiter.limit("10/minute")  # 每分钟最多 10 次
async def upload_photo(request: Request, file: UploadFile = File(...)):
    # 处理上传
    pass
```

**C. 输入验证**

```python
from pydantic import BaseModel, validator
import re

class PhotoUpload(BaseModel):
    filename: str
    content_type: str
    
    @validator('filename')
    def validate_filename(cls, v):
        # 只允许字母、数字、下划线、连字符
        if not re.match(r'^[\w\-]+\.(jpg|jpeg|png)$', v, re.IGNORECASE):
            raise ValueError('Invalid filename')
        return v
    
    @validator('content_type')
    def validate_content_type(cls, v):
        allowed = ['image/jpeg', 'image/png']
        if v not in allowed:
            raise ValueError(f'Invalid content type. Allowed: {allowed}')
        return v
```

**D. SQL 注入防护**

```python
# ❌ 危险：SQL 注入
query = f"SELECT * FROM photos WHERE id = {photo_id}"

# ✅ 安全：参数化查询
cursor.execute("SELECT * FROM photos WHERE id = ?", (photo_id,))
```

---

## ⚠️ 常见错误与避坑指南

### 错误 1：忽视资源清理

**❌ 错误做法：**
```python
def process_image(image_path):
    image = Image.open(image_path)
    result = model.detect(image)
    # 忘记关闭文件
    return result
```

**✅ 正确做法：**
```python
def process_image(image_path):
    with Image.open(image_path) as image:
        result = model.detect(image)
    return result
```

---

### 错误 2：硬编码配置

**❌ 错误做法：**
```python
DATABASE_URL = "sqlite:///./photos.db"
REDIS_URL = "redis://localhost:6379/0"
```

**✅ 正确做法：**
```python
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./photos.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SECRET_KEY = os.getenv("SECRET_KEY")  # 必须设置
```

---

### 错误 3：不处理超时

**❌ 错误做法：**
```python
result = model.predict(image)  # 可能永远阻塞
```

**✅ 正确做法：**
```python
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def predict_with_timeout(image, timeout=30):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        result = model.predict(image)
        signal.alarm(0)  # 取消闹钟
        return result
    except TimeoutError:
        logger.error("Prediction timed out")
        return None
```

---

## ✍️ 自我检测练习

### 练习 1：编写 Docker Compose 配置

**任务：** 为智能相册项目编写完整的 docker-compose.yml。

**参考答案：** 见上方的完整配置。

---

### 练习 2：实现缓存策略

**任务：** 为检测结果添加缓存，TTL 为 1 小时。

**参考答案：**
```python
cache = CacheManager()

@cache.cache_result(ttl=3600)
def detect_and_cache(image_hash: str):
    image = load_image_from_hash(image_hash)
    return model.detect(image)
```

---

### 练习 3：性能测试

**任务：** 使用 locust 进行压力测试。

**参考答案：**
```python
# locustfile.py
from locust import HttpUser, task, between

class AlbumUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def upload_photo(self):
        with open("test.jpg", "rb") as f:
            self.client.post(
                "/api/upload",
                files={"file": f}
            )
    
    @task(3)
    def get_photos(self):
        self.client.get("/api/photos")

# 运行：locust -f locustfile.py --host=http://localhost:8000
```

---

## 📝 本章小结

### 部署与优化要点

✅ **容器化**：Docker 确保环境一致性

✅ **模型优化**：量化、剪枝、蒸馏提升性能

✅ **缓存策略**：Redis 加速热点查询

✅ **异步处理**：避免阻塞，提高并发

✅ **监控告警**：及时发现问题

✅ **安全加固**：认证、限流、验证

---

### 性能优化清单

| 优化项 | 预期提升 | 难度 | 优先级 |
|--------|---------|------|--------|
| FP16 量化 | 1.5-2x | 低 | P0 |
| Redis 缓存 | 5-10x | 低 | P0 |
| 异步处理 | 2-3x | 中 | P1 |
| 模型剪枝 | 1.5-2x | 中 | P1 |
| 负载均衡 | 线性扩展 | 中 | P2 |
| CDN 加速 | 显著 | 高 | P2 |

---

**📚 相关文档：**
- [Day21-Q4 - 多模态集成](./Day21-Q4%20-%20多模态集成.md)
- [Day21-Q6 - 项目总结与展示](./Day21-Q6%20-%20项目总结与展示.md)（待创建）

**💡 提示：** 优化是一个持续的过程，先测量性能瓶颈，再针对性优化，不要过早优化。
