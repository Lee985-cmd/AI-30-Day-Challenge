# Day27-Q6 - 云平台部署实战

## ☁️ 把你的 AI 服务放到云端

### 问题背景

你已经用 Docker 容器化了应用,现在可以在任何地方运行了。但问题是:

**服务器从哪里来?**

- ❌ 自己买服务器? (贵!维护麻烦!)
- ❌ 放在家里? (断电断网就挂了!)
- ✅ 用云平台! (便宜、可靠、易扩展!)

---

## 一、云平台选择

### 主流云服务商

| 平台 | 优点 | 缺点 | 适合场景 |
|------|------|------|---------|
| **Render** | 超简单,免费额度 | 功能有限 | 学习、小项目 ⭐ |
| **Heroku** | 简单易用 | 不再免费 | 快速原型 |
| **AWS** | 服务最全,生态好 | 复杂,可能贵 | 生产环境 ⭐⭐⭐ |
| **GCP** | AI/ML 服务好 | 文档一般 | ML 项目 ⭐⭐ |
| **Azure** | 企业友好 | 学习曲线陡 | 企业应用 |
| **阿里云** | 国内访问快 | 国际生态弱 | 国内用户 |

**推荐路线:**
- 学习阶段 → Render (免费)
- 小项目 → Heroku/Railway ($5-10/月)
- 生产环境 → AWS/GCP ($50+/月)

---

## 二、Render 部署 (最简单)

### 什么是 Render?

Render 是一个现代化的云平台,特点是:
- ✅ 超级简单 (连接 GitHub 自动部署)
- ✅ 有免费额度
- ✅ 支持 Docker
- ✅ 自动 HTTPS
- ❌ 免费版会休眠 (15分钟无请求)

### 步骤1: 准备代码

确保你的项目有:
```
fastapi_app/
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md
```

**推送到 GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/Lee985-cmd/fastapi-app.git
git push -u origin main
```

### 步骤2: 注册 Render

1. 访问 [render.com](https://render.com)
2. 用 GitHub 账号登录
3. 点击 "New +" → "Web Service"

### 步骤3: 配置服务

1. **连接仓库**: 选择你的 GitHub 仓库
2. **配置服务**:
   - Name: `my-fastapi-app`
   - Region: `Oregon` (离你近的)
   - Branch: `main`
   - Root Directory: `/` (如果有子目录就填)
   - Runtime: `Docker`
   - Instance Type: `Free` (学习用)

3. **环境变量** (可选):
   ```
   API_KEY=your_secret_key
   MODEL_PATH=/app/model.pth
   ```

4. 点击 "Create Web Service"

### 步骤4: 等待部署

Render 会自动:
1. 从 GitHub 拉取代码
2. 构建 Docker 镜像
3. 启动容器
4. 分配域名

**大约需要 3-5 分钟**

### 步骤5: 访问服务

部署成功后,你会得到一个 URL:
```
https://my-fastapi-app.onrender.com
```

**测试:**
```python
import requests

# 健康检查
response = requests.get('https://my-fastapi-app.onrender.com/health')
print(response.json())

# 预测
with open('cat.jpg', 'rb') as f:
    files = {'file': ('cat.jpg', f)}
    response = requests.post(
        'https://my-fastapi-app.onrender.com/predict',
        files=files
    )
    print(response.json())
```

### 常见问题

**问题1: 首次访问很慢**
- 原因: 免费版会休眠
- 解决: 升级到付费计划 ($7/月起)

**问题2: 部署失败**
- 检查日志: Render Dashboard → Logs
- 常见原因:
  - Dockerfile 错误
  - 依赖安装失败
  - 端口不对 (必须是 8000 或 $PORT 环境变量)

**修复 Dockerfile:**
```dockerfile
# Render 要求监听 $PORT 环境变量
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# 或者
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## 三、Railway 部署 (也很简单)

### 什么是 Railway?

Railway 是另一个简单的云平台:
- ✅ 简单易用
- ✅ $5 免费额度
- ✅ 自动部署
- ✅ 支持 Docker

### 部署步骤

1. 访问 [railway.app](https://railway.app)
2. 用 GitHub 登录
3. 点击 "New Project" → "Deploy from GitHub repo"
4. 选择仓库
5. 自动检测并部署

**就这么简单!**

---

## 四、AWS 部署 (专业级)

### AWS 服务概览

AWS 有很多服务,我们关注这几个:

| 服务 | 用途 | 难度 |
|------|------|------|
| **EC2** | 虚拟服务器 | ⭐⭐ |
| **ECS** | 容器服务 | ⭐⭐⭐ |
| **Lambda** | 无服务器函数 | ⭐⭐⭐ |
| **SageMaker** | ML 平台 | ⭐⭐⭐⭐ |
| **Elastic Beanstalk** | PaaS | ⭐⭐ |

**推荐:** EC2 (最灵活,最好理解)

### 方法1: EC2 手动部署

#### 步骤1: 创建 EC2 实例

1. 登录 [AWS Console](https://aws.amazon.com/console/)
2. 搜索 "EC2"
3. 点击 "Launch Instance"
4. 配置:
   - Name: `my-api-server`
   - AMI: `Ubuntu Server 22.04 LTS`
   - Instance type: `t2.micro` (免费套餐)
   - Key pair: 创建新的密钥对 (下载 .pem 文件)
   - Security group: 允许 SSH (22) 和 HTTP (8000)
5. 点击 "Launch Instance"

#### 步骤2: 连接到服务器

```bash
# Mac/Linux
chmod 400 your-key.pem
ssh -i "your-key.pem" ubuntu@your-instance-ip

# Windows (PowerShell)
ssh -i "your-key.pem" ubuntu@your-instance-ip
```

#### 步骤3: 安装 Docker

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 Docker
sudo apt install docker.io -y

# 启动 Docker
sudo systemctl start docker
sudo systemctl enable docker

# 添加当前用户到 docker 组 (不用每次 sudo)
sudo usermod -aG docker $USER

# 退出重新登录
exit
ssh -i "your-key.pem" ubuntu@your-instance-ip

# 验证
docker --version
```

#### 步骤4: 部署应用

**方法A: 从 GitHub 拉取**
```bash
# 克隆代码
git clone https://github.com/Lee985-cmd/fastapi-app.git
cd fastapi-app

# 构建镜像
docker build -t myapp:latest .

# 运行容器
docker run -d \
    -p 8000:8000 \
    --name my-api \
    --restart unless-stopped \
    myapp:latest
```

**方法B: 从 Docker Hub 拉取**
```bash
# 先在本地推送镜像到 Docker Hub
docker tag myapp:latest username/myapp:latest
docker push username/myapp:latest

# 在服务器上拉取
docker pull username/myapp:latest
docker run -d -p 8000:8000 --name my-api username/myapp:latest
```

#### 步骤5: 配置防火墙

```bash
# 允许 8000 端口
sudo ufw allow 8000/tcp
sudo ufw enable
```

#### 步骤6: 测试

```bash
# 在本地电脑
curl http://your-instance-ip:8000/health
```

### 方法2: Elastic Beanstalk (更简单)

Elastic Beanstalk 是 AWS 的 PaaS 服务,自动处理基础设施。

#### 步骤1: 安装 EB CLI

```bash
pip install awsebcli
```

#### 步骤2: 初始化

```bash
cd fastapi-app
eb init
```

按提示配置:
- Region: 选择近的
- Application: 创建新应用
- Platform: Docker

#### 步骤3: 创建环境

```bash
eb create my-api-env
```

#### 步骤4: 部署

```bash
eb deploy
```

#### 步骤5: 访问

```bash
eb open
```

会自动打开浏览器!

---

## 五、GCP 部署

### Google Cloud Run (推荐)

Cloud Run 是无服务器容器服务,按使用量付费。

#### 步骤1: 安装 gcloud CLI

```bash
# Mac
brew install google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash
```

#### 步骤2: 登录

```bash
gcloud auth login
gcloud config set project your-project-id
```

#### 步骤3: 构建并推送镜像

```bash
# 使用 Google Container Registry
gcloud builds submit --tag gcr.io/your-project-id/myapp
```

#### 步骤4: 部署

```bash
gcloud run deploy my-api \
    --image gcr.io/your-project-id/myapp \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

#### 步骤5: 访问

命令会返回一个 URL,直接访问即可!

---

## 六、域名和 HTTPS

### 自定义域名

**Render:**
1. Dashboard → Settings → Custom Domain
2. 添加你的域名
3. 按提示配置 DNS (CNAME 记录)

**AWS:**
1. Route 53 购买/管理域名
2. 创建 A 记录指向 EC2 IP
3. 或使用 CloudFront + ACM (HTTPS)

### HTTPS 证书

**Render/Railway:** 自动提供  
**AWS:** 使用 ACM (Amazon Certificate Manager)  
**自己配置:** 使用 Let's Encrypt (免费)

```bash
# 安装 certbot
sudo apt install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d your-domain.com
```

---

## 七、监控和日志

### Render 监控

Dashboard 自带:
- CPU/Memory 使用
- 请求数量
- 响应时间
- 错误率

### AWS CloudWatch

```bash
# 查看日志
aws logs get-log-events \
    --log-group-name /aws/elasticbeanstalk/... \
    --log-stream-name ...

# 设置告警
aws cloudwatch put-metric-alarm ...
```

### 第三方监控

**推荐工具:**
- **Sentry**: 错误追踪
- **Datadog**: 全面监控
- **New Relic**: APM
- **Prometheus + Grafana**: 自建监控

**集成 Sentry:**
```python
# main.py
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0
)
```

---

## 八、成本优化

### Render 成本

| 计划 | 价格 | 适合 |
|------|------|------|
| Free | $0 | 学习、演示 |
| Starter | $7/月 | 小项目 |
| Standard | $25/月 | 生产环境 |

### AWS 成本

**免费套餐 (12个月):**
- t2.micro EC2: 750 小时/月
- 其他服务也有一定免费额度

**优化技巧:**
1. 使用 Spot Instances (便宜 70%)
2. 自动扩缩容
3. 关闭不用的资源
4. 使用 Cost Explorer 监控

### GCP 成本

**免费额度:**
- $300 新用户信用
- Cloud Run: 每月 200 万次请求免费

---

## 九、CI/CD 自动化

### GitHub Actions 自动部署

**.github/workflows/deploy.yml:**
```yaml
name: Deploy to Render

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Deploy to Render
        env:
          RENDER_API_KEY: ${{ secrets.RENDER_API_KEY }}
        run: |
          curl -X POST \
            https://api.render.com/v1/services/srv-xxx/deploys \
            -H "Authorization: Bearer $RENDER_API_KEY"
```

**设置 Secret:**
1. GitHub Repo → Settings → Secrets
2. 添加 `RENDER_API_KEY`

现在每次 push 到 main 分支都会自动部署!

---

## 十、完整部署清单

### 部署前检查

- [ ] 代码已推送到 GitHub
- [ ] Dockerfile 正确
- [ ] requirements.txt 完整
- [ ] 本地测试通过
- [ ] 环境变量配置好
- [ ] .env 文件已添加到 .gitignore

### 部署步骤

1. **选择平台**: Render (简单) / AWS (专业)
2. **创建服务**: 连接 GitHub 或上传代码
3. **配置环境变量**: API Keys,数据库连接等
4. **部署**: 点击部署按钮或 push 代码
5. **测试**: 访问 URL,测试所有接口
6. **配置域名**: (可选) 添加自定义域名
7. **设置监控**: 配置日志和告警
8. **文档更新**: 更新 API 文档

### 部署后监控

- [ ] 检查日志是否有错误
- [ ] 监控 CPU/Memory 使用
- [ ] 测试 API 响应时间
- [ ] 设置告警规则
- [ ] 备份数据

---

## 十一、实战案例

### 案例1: 个人项目 (Render)

**需求:** 部署情感分析 API,给朋友用

**方案:**
```
平台: Render Free
成本: $0
时间: 10 分钟
步骤:
1. Push to GitHub
2. Connect to Render
3. Auto deploy
4. Share URL
```

### 案例2: 创业公司 MVP (Railway)

**需求:** 图像识别服务,1000 DAU

**方案:**
```
平台: Railway
成本: $10/月
时间: 30 分钟
步骤:
1. Dockerize app
2. Deploy to Railway
3. Add custom domain
4. Setup monitoring
```

### 案例3: 企业生产环境 (AWS)

**需求:** 推荐系统,百万用户

**方案:**
```
平台: AWS ECS + RDS + ElastiCache
成本: $500/月
时间: 2-3 天
步骤:
1. Setup VPC and security groups
2. Create ECS cluster
3. Deploy containers
4. Setup RDS database
5. Configure auto-scaling
6. Setup CloudWatch monitoring
7. Configure CI/CD pipeline
```

---

## 十二、本章小结

### 云平台对比

| 平台 | 难度 | 成本 | 适合 |
|------|------|------|------|
| **Render** | ⭐ | 免费-$25/月 | 学习、小项目 |
| **Railway** | ⭐⭐ | $5+/月 | 原型、MVP |
| **AWS** | ⭐⭐⭐⭐ | $50+/月 | 生产环境 |
| **GCP** | ⭐⭐⭐ | $50+/月 | ML 项目 |

### 核心要点

✅ **Render 部署:**
- 连接 GitHub
- 自动构建和部署
- 免费额度够用

✅ **AWS EC2:**
- 完全控制
- 需要手动配置
- 灵活性强

✅ **GCP Cloud Run:**
- 无服务器
- 按使用量付费
- 自动扩缩容

✅ **最佳实践:**
- 从小开始,逐步扩展
- 使用 CI/CD 自动化
- 设置监控和告警
- 注意成本控制
- 定期备份数据

---

## 🎉 Day27 完成!

恭喜你完成了模型部署和工程化的学习!

### 你学会了:

✅ **Q1**: 为什么要部署模型  
✅ **Q2**: 模型序列化和加载  
✅ **Q3**: Flask API 开发  
✅ **Q4**: FastAPI 高性能 API  
✅ **Q5**: Docker 容器化  
✅ **Q6**: 云平台部署  

### 下一步:

**Day28: AI 伦理和安全**
- AI 偏见和公平性
- 隐私保护
- 安全责任
- 法律法规

**或者继续完善部署:**
- 学习 Kubernetes
- 实现蓝绿部署
- 搭建监控系统
- 优化成本和性能

**你已经是一名全栈 AI 工程师了!** 🚀🎊
