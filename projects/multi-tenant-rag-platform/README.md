# 多租户RAG平台

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 项目简介

企业级多租户RAG（检索增强生成）平台，支持1000+企业客户同时使用，提供完整的数据隔离、权限管理和资源调度。

**核心特性：**
- ✅ **混合数据隔离** - Schema隔离 + Row隔离，平衡安全性和性能
- ✅ **RBAC权限管理** - 细粒度的角色和权限控制
- ✅ **JWT认证** - 安全的Token-based认证
- ✅ **配额管理** - 按租户限制资源使用
- ✅ **限流保护** - 防止滥用和DDoS攻击
- ✅ **优先级队列** - VIP租户优先处理
- ✅ **Streamlit管理后台** - 可视化的租户管理
- ✅ **完整的API文档** - OpenAPI/Swagger

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动PostgreSQL数据库

```bash
# 使用Docker
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  postgres:15
```

### 3. 配置环境变量

创建 `.env` 文件：

```bash
# 数据库配置
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db

# JWT配置
JWT_SECRET_KEY=your-secret-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Redis配置（用于缓存和限流）
REDIS_URL=redis://localhost:6379/0

# OpenAI配置
OPENAI_API_KEY=your-openai-api-key
```

### 4. 初始化数据库

```bash
python init_db.py
```

### 5. 启动API服务

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6. 启动管理后台

```bash
streamlit run admin_dashboard.py
```

访问 http://localhost:8501

## 📁 项目结构

```
multi-tenant-rag-platform/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI应用入口
│   ├── config.py               # 配置管理
│   ├── database.py             # 数据库连接
│   ├── models.py               # 数据模型
│   ├── schemas.py              # Pydantic schemas
│   ├── auth.py                 # 认证和授权
│   ├── middleware.py           # 中间件
│   ├── services/
│   │   ├── tenant_service.py   # 租户管理
│   │   ├── document_service.py # 文档管理
│   │   ├── rag_service.py      # RAG查询
│   │   └── quota_service.py    # 配额管理
│   └── api/
│       ├── __init__.py
│       ├── auth.py             # 认证API
│       ├── tenants.py          # 租户API
│       ├── documents.py        # 文档API
│       └── query.py            # 查询API
├── tests/                      # 测试目录
├── init_db.py                  # 数据库初始化脚本
├── admin_dashboard.py          # Streamlit管理后台
├── requirements.txt            # 依赖列表
├── .env.example                # 环境变量示例
├── docker-compose.yml          # Docker编排
└── README.md                   # 本文件
```

## 🔐 数据隔离方案

### 混合隔离策略

- **小租户** (< 1000文档): Row隔离 - 所有租户共用表，通过`tenant_id`区分
- **大租户** (≥ 1000文档): Schema隔离 - 每个租户独立的Schema

```python
# 自动选择隔离策略
def get_isolation_strategy(tenant_id: str) -> str:
    doc_count = get_tenant_document_count(tenant_id)
    if doc_count < 1000:
        return "row"
    else:
        return "schema"
```

## 👥 RBAC权限系统

### 角色定义

| 角色 | 权限 |
|-----|------|
| **Admin** | 所有权限（CRUD + 用户管理） |
| **Editor** | 创建、读取、更新文档 |
| **Viewer** | 仅读取文档 |
| **API User** | 仅通过API读取 |

### 权限验证

```python
@app.post("/api/documents")
@require_permission(Permission.DOCUMENT_CREATE)
async def create_document(doc: DocumentCreate, current_user: User = Depends(get_current_user)):
    # 只有有创建权限的用户才能执行
    ...
```

## ⚖️ 配额管理

### 默认配额

```python
{
    "max_documents": 10000,          # 最大文档数
    "max_queries_per_day": 10000,    # 每日最大查询数
    "max_storage_gb": 10.0,          # 最大存储空间
    "max_concurrent_requests": 10    # 最大并发请求
}
```

### 限流规则

- 普通用户：100次/小时
- VIP用户：1000次/小时
- API用户：根据套餐分级

## 📊 监控指标

项目集成了Prometheus监控：

- 请求QPS
- P95/P99响应时间
- 错误率
- 缓存命中率
- API费用
- 活跃连接数

访问 http://localhost:9090 查看Prometheus

## 🧪 运行测试

```bash
pytest tests/ -v
```

## 🐳 Docker部署

```bash
# 一键启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

## 📝 API文档

启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📜 许可证

MIT License

## 👤 作者

学习论之费曼学习法

- CSDN: https://blog.csdn.net/m0_67081842
- GitHub: https://github.com/Lee985-cmd
