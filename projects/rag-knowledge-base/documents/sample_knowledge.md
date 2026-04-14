# RAG 知识库示例文档

## 1. 产品常见问题 (FAQ)

### Q: 如何重置密码？
A: 请访问登录页面，点击“忘记密码”，输入注册邮箱，我们会发送重置链接。

### Q: 支持哪些支付方式？
A: 我们支持支付宝、微信支付、银联卡以及 PayPal（国际用户）。

### Q: 退款政策是什么？
A: 购买后 7 天内无理由全额退款。超过 7 天但未满 30 天，扣除手续费后退款。

### Q: 如何联系客服？
A: 
- 在线客服：工作日 9:00-18:00
- 邮箱：support@example.com
- 电话：400-123-4567

---

## 2. 技术架构说明

### 系统组成
1. **前端层**：React + TypeScript
2. **后端层**：FastAPI + Python 3.10+
3. **数据库**：PostgreSQL + Redis
4. **向量数据库**：ChromaDB / FAISS
5. **大模型**：OpenAI GPT-4 / 本地 ChatGLM

### 数据流向
```
用户提问 
  → API Gateway 
  → 检索相关文档 (Vector DB) 
  → 组装 Prompt 
  → LLM 生成回答 
  → 返回给用户
```

### 性能指标
- 平均响应时间：< 2 秒
- 并发支持：100 QPS
- 准确率：> 85%（基于内部测试集）

---

## 3. 部署指南

### 环境要求
- Python 3.10+
- Docker & Docker Compose
- 至少 8GB RAM

### 快速部署
```bash
# 1. 克隆代码
git clone https://github.com/your-org/rag-system.git
cd rag-system

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 API Key

# 3. 启动服务
docker-compose up -d

# 4. 验证
curl http://localhost:8000/health
```

### 生产环境建议
- 使用 Kubernetes 编排
- 配置 HTTPS 证书
- 启用监控（Prometheus + Grafana）
- 定期备份向量数据库

---

## 4. API 接口文档

### 基础信息
- Base URL: `https://api.example.com/v1`
- 认证方式: Bearer Token

### 接口列表

#### 1. 上传文档
```http
POST /documents/upload
Content-Type: multipart/form-data

file: <binary>
metadata: {"category": "technical", "tags": ["api", "guide"]}
```

#### 2. 问答接口
```http
POST /chat/ask
Content-Type: application/json
Authorization: Bearer <token>

{
  "question": "如何重置密码？",
  "top_k": 3,
  "stream": false
}
```

**响应示例：**
```json
{
  "answer": "请访问登录页面，点击“忘记密码”...",
  "sources": [
    {
      "document": "FAQ.pdf",
      "page": 2,
      "similarity": 0.92
    }
  ],
  "latency_ms": 1200
}
```

#### 3. 健康检查
```http
GET /health
```

---

## 5. 版本更新日志

### v2.1.0 (2024-03-15)
- ✨ 新增流式输出支持
- 🚀 优化检索速度（提升 40%）
- 🐛 修复中文编码问题

### v2.0.0 (2024-02-01)
- 💥 重构向量存储模块
- ✨ 支持多租户隔离
- 📚 新增 API 文档

### v1.5.0 (2024-01-10)
- ✨ 支持 PDF 解析
- 🔧 优化 Chunk 切分策略

---

## 6. 最佳实践

### 文档准备
1. **格式统一**：优先使用 Markdown 或纯文本
2. **结构清晰**：使用标题、列表等结构化元素
3. **去除噪音**：删除页眉、页脚、广告等无关内容

### 参数调优
```python
# 推荐配置
CHUNK_SIZE = 500       # 适中大小，平衡上下文和精度
CHUNK_OVERLAP = 50     # 10% 重叠，避免信息割裂
TOP_K = 3              # 检索 3 个最相关片段
TEMPERATURE = 0.7      # 平衡创造性和准确性
```

### 安全建议
- 不要在文档中包含敏感信息（密码、密钥）
- 启用访问控制（RBAC）
- 记录所有查询日志（审计用）
- 定期清理过期文档
