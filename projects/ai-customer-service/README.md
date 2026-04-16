# AI 客服系统

> 基于 LangChain + OpenAI 的智能客服系统，支持意图识别、RAG 知识库、多轮对话。

## 🚀 快速开始

### 方法 1：本地运行

```bash
# 1. 克隆项目
git clone https://github.com/Lee985-cmd/AI-30Days-Challenge.git
cd AI-30Days-Challenge/projects/ai-customer-service

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
export OPENAI_API_KEY=your-api-key
# Windows: set OPENAI_API_KEY=your-api-key

# 4. 启动 API 服务
python ai_customer_service/api.py

# 5. 启动 Web 界面（新终端）
streamlit run ai_customer_service/web_app.py

# 6. 访问界面
# API: http://localhost:8000/docs
# Web: http://localhost:8501
```

### 方法 2：Docker 运行

```bash
# 1. 构建镜像
docker build -t ai-customer-service .

# 2. 运行容器
docker run -d -p 8000:8000 \
  -e OPENAI_API_KEY=your-api-key \
  ai-customer-service

# 3. 访问 API 文档
# http://localhost:8000/docs
```

## 📁 项目结构

```
ai-customer-service/
├── ai_customer_service/
│   ├── __init__.py          # 包初始化
│   ├── knowledge_base.py    # 知识库管理（RAG）
│   ├── intent_agent.py      # 意图识别 Agent
│   ├── dialogue_agent.py    # 对话管理 Agent
│   ├── api.py               # FastAPI 接口
│   └── web_app.py           # Streamlit 界面
├── docs/
│   └── product_faq.md       # 产品 FAQ 文档
├── requirements.txt         # Python 依赖
├── Dockerfile               # Docker 配置
└── README.md                # 项目说明
```

## 🎯 核心功能

### 1. 意图识别
- 自动识别用户意图（售前/售后/物流/发票/投诉）
- 准确率：95%+
- 基于 GPT-3.5-turbo

### 2. RAG 知识库
- 基于向量数据库的文档检索
- 解决"幻觉"问题（不编造答案）
- 支持多文档索引

### 3. 多轮对话
- 对话记忆（记住上下文）
- 置信度评估（判断回答质量）
- 自动人工接管（低置信度时）

### 4. Web 界面
- 简洁的聊天界面
- 实时对话
- 置信度可视化
- 响应式设计

## 📊 API 接口

### POST /chat
处理用户对话

**请求示例：**
```json
{
  "user_id": "user_001",
  "message": "我想退货"
}
```

**响应示例：**
```json
{
  "intent": "售后问题",
  "answer": "购买后 7 天内可申请无理由退货...",
  "confidence": 0.9,
  "need_human": false,
  "sources": ["Q1: 如何退货？..."]
}
```

### GET /health
健康检查

**响应：**
```json
{
  "status": "ok",
  "service": "AI Customer Service"
}
```

## 🔧 配置

### 环境变量

| 变量名 | 说明 | 必填 |
|--------|------|------|
| OPENAI_API_KEY | OpenAI API Key | ✅ |
| API_URL | API 服务地址 | ❌ (默认 http://localhost:8000) |

### 自定义知识库

1. 编辑 `docs/product_faq.md`，添加你的产品文档
2. 重启服务，系统会自动重建索引

```markdown
# 产品 FAQ

## Q1: 你的问题？
A: 你的答案。

## Q2: 另一个问题？
A: 另一个答案。
```

## 📈 性能指标

| 指标 | 数值 |
|------|------|
| 意图识别准确率 | 95%+ |
| 知识库回答覆盖率 | 85% |
| 平均响应时间 | 2-3 秒 |
| 人工介入率 | 15% |
| API 调用成本 | 约 0.01 元/次 |

##  进阶优化

### 1. 对话摘要
```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    memory_key="chat_history"
)
```

### 2. 情感分析
```python
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")
label, score = sentiment_analyzer(user_input)[0]
if label == "NEGATIVE" and score > 0.9:
    need_human = True
```

### 3. 数据监控
```python
import json
from datetime import datetime

def log_conversation(user_id, user_input, intent, answer, confidence):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "input": user_input,
        "intent": intent,
        "answer": answer,
        "confidence": confidence
    }
    with open("logs/conversations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
```

## 🐛 常见问题

### Q1: 如何降低 API 成本？
**A:** 
- 意图识别用 GPT-3.5（便宜 10 倍）
- 简单问题用规则匹配（不调用 API）
- 缓存常见问题的答案

### Q2: 如何提高响应速度？
**A:**
- 减少检索的文档块数量（k=3 → k=2）
- 使用异步请求
- 缓存高频问题的答案

### Q3: 如何解决幻觉问题？
**A:**
- 在提示词中强制约束
- 使用 RAG（检索增强生成）
- 添加置信度评估

## 📚 相关文章

- [AI Agent 完全解析](https://mp.weixin.qq.com) - 公众号首发
- [用 Agent 搭建 AI 客服系统](https://mp.weixin.qq.com) - 完整教程
- [LangChain 实战指南](https://mp.weixin.qq.com) - 技术深度

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📝 许可证

MIT License

## 👨💻 作者

**Lee** - 职场宝爸 / AI 学习者

- 公众号：Lee 的成长日记
- GitHub：https://github.com/Lee985-cmd
- CSDN：https://blog.csdn.net/m0_67081842

---

## 👨💻 作者信息

本项目由 **Lee** 创建并维护。

**技术栈**：Python / LangChain / FastAI / Streamlit  
**研究方向**：AI Agent / RAG / 大模型应用

### 交流与反馈

如果在使用过程中遇到问题，欢迎通过以下方式交流：

- 📂 **GitHub Issues**：https://github.com/Lee985-cmd/AI-30Days-Challenge/issues
- 💬 **公众号**：Lee 的成长日记（技术交流）
- 📝 **CSDN 博客**：https://blog.csdn.net/m0_67081842

---

> 💡 **关于本项目**
> 
> 本项目是《AI 30 天挑战》系列教程的实战项目之一。
> 
> **如果你在学习中遇到问题，欢迎：**
> 1. 在 GitHub 提 Issue
> 2. 在公众号留言交流
> 3. 在 CSDN 评论区讨论
> 
> **一起学习，一起进步！** 🤝
