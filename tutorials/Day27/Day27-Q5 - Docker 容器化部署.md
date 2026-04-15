# Day27-Q5 - Docker 容器化部署

## 📦 把应用打包成集装箱

### 问题背景

你在本地开发好了 FastAPI 应用,运行完美。但现在要部署到服务器,问题来了:

**"在我电脑上能跑啊!"**

- ❌ 服务器 Python 版本不一样
- ❌ 依赖库版本冲突
- ❌ 环境变量缺失
- ❌ 文件路径不同
- ❌ 操作系统差异

**解决方案:** Docker 容器化!

---

## 一、什么是 Docker?

### 大白话解释

**Docker = 应用的集装箱**

就像海运集装箱:
- **货物** = 你的应用 + 所有依赖
- **集装箱** = Docker 容器
- **货轮** = 任何服务器

**好处:**
- 不管什么船 (服务器),集装箱都能放
- 货物不会损坏 (环境一致)
- 装卸方便 (快速部署)

### 技术定义

Docker 是一个容器化平台,可以将应用及其所有依赖打包成一个独立的、可移植的容器。

**核心概念:**
- **Image (镜像)**: 应用的模板 (只读)
- **Container (容器)**: 镜像的运行实例
- **Dockerfile**: 构建镜像的说明书
- **Registry**: 存储和分享镜像的地方 (如 Docker Hub)

---

## 二、为什么需要 Docker?

### 问题1: 环境不一致

**没有 Docker:**
```
开发者电脑: Python 3.9, PyTorch 2.0
测试服务器: Python 3.8, PyTorch 1.13  ← 出错了!
生产服务器: Python 3.7, PyTorch 1.12  ← 又出错了!
```

**有 Docker:**
```
Docker 镜像: Python 3.9, PyTorch 2.0
  ↓ 在任何地方运行都一样
开发者电脑 ✓
测试服务器 ✓
生产服务器 ✓
```

### 问题2: 依赖冲突

**场景:** 
- 项目 A 需要 Flask 2.0
- 项目 B 需要 Flask 1.0
- 在同一台服务器上怎么办?

**Docker 解决:** 每个项目独立容器,互不影响!

### 问题3: 快速部署

**传统部署:**
```
1. SSH 登录服务器 (5 分钟)
2. 安装 Python (10 分钟)
3. 安装依赖 (15 分钟)
4. 配置环境变量 (5 分钟)
5. 启动应用 (2 分钟)
总计: 37 分钟
```

**Docker 部署:**
```
1. 拉取镜像 (2 分钟)
2. 运行容器 (10 秒)
总计: 2 分 10 秒
```

---

## 三、Docker 基础命令

### 安装 Docker

**Windows/Mac:** 下载 Docker Desktop  
**Linux:** 
```bash
sudo apt-get install docker.io
```

**验证安装:**
```bash
docker --version
docker run hello-world
```

### 常用命令

```bash
# 1. 拉取镜像
docker pull python:3.9-slim

# 2. 查看本地镜像
docker images

# 3. 运行容器
docker run -it python:3.9-slim bash

# 4. 查看运行中的容器
docker ps

# 5. 停止容器
docker stop <container_id>

# 6. 删除容器
docker rm <container_id>

# 7. 删除镜像
docker rmi <image_id>

# 8. 查看容器日志
docker logs <container_id>

# 9. 进入运行中的容器
docker exec -it <container_id> bash
```

---

## 四、Dockerfile 详解

### 什么是 Dockerfile?

**Dockerfile = 构建镜像的食谱**

告诉 Docker 如何一步步构建你的应用镜像。

### 基础示例

```dockerfile
# 1. 基于哪个镜像
FROM python:3.9-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 复制依赖文件
COPY requirements.txt .

# 4. 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制应用代码
COPY . .

# 6. 暴露端口
EXPOSE 8000

# 7. 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 指令说明

| 指令 | 作用 | 示例 |
|------|------|------|
| **FROM** | 基础镜像 | `FROM python:3.9` |
| **WORKDIR** | 工作目录 | `WORKDIR /app` |
| **COPY** | 复制文件 | `COPY . .` |
| **RUN** | 执行命令 | `RUN pip install ...` |
| **EXPOSE** | 暴露端口 | `EXPOSE 8000` |
| **CMD** | 启动命令 | `CMD ["python", "app.py"]` |
| **ENV** | 环境变量 | `ENV API_KEY=xxx` |
| **ARG** | 构建参数 | `ARG VERSION=1.0` |

---

## 五、实战: 容器化 FastAPI 应用

### 项目结构

```
fastapi_app/
├── main.py
├── model.py
├── model.pth
├── requirements.txt
└── Dockerfile
```

### requirements.txt

```
fastapi==0.100.0
uvicorn==0.23.0
torch==2.0.0
Pillow==9.5.0
python-multipart==0.0.6
```

### Dockerfile (优化版)

```dockerfile
# 阶段1: 构建依赖
FROM python:3.9-slim as builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制并安装 Python 依赖
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# 阶段2: 运行环境
FROM python:3.9-slim

WORKDIR /app

# 从 builder 阶段复制安装的包
COPY --from=builder /root/.local /root/.local

# 确保脚本在 PATH 中
ENV PATH=/root/.local/bin:$PATH

# 复制应用代码
COPY main.py .
COPY model.py .
COPY model.pth .

# 创建非 root 用户 (安全最佳实践)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**优化点:**
1. **多阶段构建**: 减小最终镜像大小
2. **非 root 用户**: 提高安全性
3. **健康检查**: 自动监控容器状态
4. **缓存优化**: 先复制 requirements.txt,利用 Docker 缓存

### 构建镜像

```bash
# 构建镜像
docker build -t fastapi-app:latest .

# 查看镜像
docker images

# 应该看到:
# REPOSITORY     TAG       IMAGE ID       CREATED         SIZE
# fastapi-app    latest    abc123def456   2 minutes ago   450MB
```

### 运行容器

```bash
# 基本运行
docker run -p 8000:8000 fastapi-app:latest

# 后台运行
docker run -d -p 8000:8000 --name my-api fastapi-app:latest

# 带环境变量
docker run -d \
    -p 8000:8000 \
    --name my-api \
    -e API_KEY=secret123 \
    -e MODEL_PATH=/app/model.pth \
    fastapi-app:latest

# 挂载卷 (持久化数据)
docker run -d \
    -p 8000:8000 \
    -v $(pwd)/logs:/app/logs \
    --name my-api \
    fastapi-app:latest
```

**参数说明:**
- `-p 8000:8000`: 端口映射 (主机:容器)
- `-d`: 后台运行
- `--name`: 容器名称
- `-e`: 环境变量
- `-v`: 挂载卷

### 测试

```bash
# 访问 API
curl http://localhost:8000/health

# 查看日志
docker logs my-api

# 实时日志
docker logs -f my-api

# 进入容器
docker exec -it my-api bash

# 停止容器
docker stop my-api

# 删除容器
docker rm my-api
```

---

## 六、Docker Compose

### 什么是 Docker Compose?

**Docker Compose = 多容器编排工具**

当你的应用需要多个服务时 (如 API + 数据库 + Redis),用 Compose 一键启动。

### 示例: API + Redis 缓存

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  # API 服务
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  # Redis 缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

**使用:**
```bash
# 启动所有服务
docker-compose up -d

# 查看运行状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止所有服务
docker-compose down

# 停止并删除数据卷
docker-compose down -v
```

---

## 七、镜像优化技巧

### 技巧1: 选择合适的基礎镜像

```dockerfile
# ❌ 太大: 1GB+
FROM python:3.9

# ✅ 较小: 400MB
FROM python:3.9-slim

# ✅ 最小: 150MB (但可能需要额外安装依赖)
FROM python:3.9-alpine
```

### 技巧2: 多阶段构建

```dockerfile
# 构建阶段
FROM python:3.9 as builder
RUN pip install ...

# 运行阶段 (更小)
FROM python:3.9-slim
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
```

**效果:** 镜像大小减少 50-70%

### 技巧3: 利用 .dockerignore

**.dockerignore:**
```
__pycache__
*.pyc
.git
.gitignore
README.md
.env
venv
*.md
tests/
```

**好处:** 不复制不必要的文件,加快构建速度

### 技巧4: 层缓存优化

```dockerfile
# ❌ 不好: 每次代码改动都要重新安装依赖
COPY . .
RUN pip install -r requirements.txt

# ✅ 好: 只有 requirements.txt 改变时才重新安装
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

---

## 八、常见问题和解决

### 问题1: 镜像太大

**检查:**
```bash
docker images
```

**优化:**
- 使用 slim/alpine 基础镜像
- 多阶段构建
- 清理缓存: `RUN apt-get clean && rm -rf /var/lib/apt/lists/*`
- 压缩层: `RUN apt-get update && apt-get install -y xxx && apt-get clean`

### 问题2: 容器启动失败

**调试:**
```bash
# 查看日志
docker logs <container_id>

# 交互式运行
docker run -it <image> bash

# 检查健康状态
docker inspect <container_id> | grep Health
```

### 问题3: 端口冲突

**错误:** `port is already allocated`

**解决:**
```bash
# 查看占用端口的容器
docker ps | grep 8000

# 停止冲突的容器
docker stop <container_id>

# 或使用不同端口
docker run -p 8001:8000 <image>
```

### 问题4: 权限问题

**错误:** `Permission denied`

**解决:**
```dockerfile
# 在 Dockerfile 中设置正确的用户
RUN useradd -m appuser
USER appuser
```

---

## 九、生产环境最佳实践

### 1. 使用标签管理版本

```bash
# 不要只用 latest
docker build -t myapp:1.0.0 .
docker build -t myapp:1.0.1 .

# 同时打多个标签
docker tag myapp:1.0.1 myapp:latest
```

### 2. 资源限制

```bash
docker run -d \
    -p 8000:8000 \
    --memory=512m \
    --cpus=1.0 \
    --name my-api \
    myapp:latest
```

### 3. 日志管理

```bash
# 限制日志大小
docker run --log-opt max-size=10m --log-opt max-file=3 ...
```

### 4. 自动重启

```yaml
# docker-compose.yml
services:
  api:
    restart: unless-stopped  # 除非手动停止,否则总是重启
```

### 5. 安全扫描

```bash
# 安装 trivy
brew install trivy

# 扫描镜像漏洞
trivy image myapp:latest
```

---

## 十、完整部署流程

```bash
# 1. 构建镜像
docker build -t myapp:1.0.0 .

# 2. 测试镜像
docker run -p 8000:8000 myapp:1.0.0

# 3. 推送到 Registry (可选)
docker tag myapp:1.0.0 username/myapp:1.0.0
docker push username/myapp:1.0.0

# 4. 在生产服务器拉取
docker pull username/myapp:1.0.0

# 5. 运行
docker run -d \
    -p 8000:8000 \
    --name production-api \
    --restart unless-stopped \
    -e ENV=production \
    username/myapp:1.0.0

# 6. 监控
docker stats
docker logs -f production-api
```

---

## 十一、本章小结

### Docker 核心概念

✅ **Image (镜像)**: 应用的只读模板  
✅ **Container (容器)**: 镜像的运行实例  
✅ **Dockerfile**: 构建镜像的说明书  
✅ **Registry**: 存储镜像的仓库  

### 关键命令

```bash
# 构建
docker build -t name:tag .

# 运行
docker run -p host_port:container_port image

# 管理
docker ps          # 查看容器
docker logs        # 查看日志
docker exec        # 进入容器
docker stop/rm     # 停止/删除
```

### 最佳实践

✅ 使用 slim/alpine 基础镜像  
✅ 多阶段构建减小体积  
✅ .dockerignore 排除不必要文件  
✅ 利用层缓存优化构建速度  
✅ 非 root 用户运行  
✅ 添加健康检查  
✅ 资源限制  
✅ 版本标签管理  

---

## 🎯 下一步

容器化完成,最后一步: 部署到云平台!

- [Q6](./Day27-Q6%20-%20云平台部署实战.md): 部署到 Render/AWS/GCP

**准备上云!** ☁️🚀

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
