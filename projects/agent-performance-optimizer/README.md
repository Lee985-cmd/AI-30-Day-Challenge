# Agent性能优化工具包

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 项目简介

这是一个完整的Agent性能优化工具包，包含：

- ✅ **多层缓存系统** - 内存缓存、Redis缓存、语义缓存
- ✅ **异步处理框架** - asyncio、多线程、消息队列
- ✅ **批处理器** - 请求合并、流式响应
- ✅ **智能模型选择器** - 分级模型使用
- ✅ **性能监控Dashboard** - 实时监控和可视化
- ✅ **基准测试工具** - 性能对比分析

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动Redis（可选，用于分布式缓存）

```bash
# 使用Docker
docker run -d -p 6379:6379 redis

# 或直接安装Redis
# Windows: https://github.com/microsoftarchive/redis/releases
# Linux: sudo apt-get install redis-server
```

### 3. 运行性能测试

```bash
python benchmark.py
```

### 4. 启动监控面板

```bash
streamlit run dashboard.py
```

## 📁 项目结构

```
agent-performance-optimizer/
├── cache/                  # 缓存模块
│   ├── __init__.py
│   ├── memory_cache.py    # 内存缓存
│   ├── redis_cache.py     # Redis缓存
│   └── semantic_cache.py  # 语义缓存
├── async_processor/        # 异步处理模块
│   ├── __init__.py
│   ├── async_agent.py     # 异步Agent
│   └── batch_processor.py # 批处理器
├── model_selector/         # 模型选择器
│   ├── __init__.py
│   └── smart_selector.py  # 智能选择器
├── monitoring/             # 监控模块
│   ├── __init__.py
│   └── metrics.py         # 指标收集
├── tests/                  # 测试目录
│   └── test_cache.py
├── benchmark.py            # 性能基准测试
├── dashboard.py            # Streamlit监控面板
├── requirements.txt        # 依赖列表
└── README.md              # 本文件
```

## 💡 使用示例

### 缓存系统

```python
from cache import MemoryCache, RedisCache, SemanticCache

# 内存缓存
cache = MemoryCache(ttl=3600)
cache.set("key", "value")
value = cache.get("key")

# Redis缓存
redis_cache = RedisCache(host='localhost', port=6379)
redis_cache.set("key", "value")

# 语义缓存
semantic_cache = SemanticCache(similarity_threshold=0.92)
semantic_cache.add_to_cache("问题", "回答")
result = semantic_cache.search_similar("相似问题")
```

### 异步处理

```python
from async_processor import AsyncAgent

async def main():
    async with AsyncAgent(api_key="your-key") as agent:
        queries = ["问题1", "问题2", "问题3"]
        results = await agent.process_multiple_queries(queries)
        print(results)

import asyncio
asyncio.run(main())
```

### 批处理

```python
from async_processor import BatchProcessor

processor = BatchProcessor(batch_size=10, wait_time=0.2)
asyncio.create_task(processor.start())

result = await processor.submit("用户问题")
```

### 智能模型选择

```python
from model_selector import SmartModelSelector

selector = SmartModelSelector()
model = selector.select_model(user_question, context)
response = call_llm(user_question, model=model)
```

## 📊 性能对比

| 优化策略 | 响应时间降低 | 成本降低 | 实现难度 |
|---------|------------|---------|---------|
| 内存缓存 | 50-80% | 50-80% | ⭐ |
| Redis缓存 | 60-85% | 60-85% | ⭐⭐ |
| 语义缓存 | 70-90% | 70-90% | ⭐⭐⭐ |
| 异步处理 | 60-80% | 0% | ⭐⭐ |
| 批处理 | 40-60% | 50-70% | ⭐⭐⭐ |
| 模型分级 | 30-50% | 60-70% | ⭐⭐ |

**综合优化效果：**
- 平均响应时间：从3500ms降到350ms（**10倍提升**）
- 并发QPS：从10提升到100+（**10倍+提升**）
- API成本：降低**80%**

## 🔧 配置说明

### 环境变量

创建 `.env` 文件：

```bash
# OpenAI API
OPENAI_API_KEY=your-api-key

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# 缓存配置
CACHE_TTL=3600
SEMANTIC_THRESHOLD=0.92

# 模型配置
FAST_MODEL=gpt-3.5-turbo
BALANCED_MODEL=gpt-4o-mini
SMART_MODEL=gpt-4o
```

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_cache.py -v
```

## 📈 监控指标

Dashboard提供以下监控：

- 实时响应时间
- 缓存命中率
- API调用统计
- 成本分析
- 错误率监控
- 并发用户数

访问 http://localhost:8501 查看监控面板。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📜 许可证

MIT License

## 👤 作者

学习论之费曼学习法

- CSDN: https://blog.csdn.net/m0_67081842
- GitHub: https://github.com/Lee985-cmd

## 🙏 致谢

感谢以下开源项目：
- LangChain
- Redis
- Streamlit
- Prometheus
