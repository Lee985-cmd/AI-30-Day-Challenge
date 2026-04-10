# RAG 知识库问答系统

## 📖 项目简介

基于检索增强生成（RAG）技术的智能问答系统，支持上传文档并基于文档内容回答问题。

**核心技术栈：**
- LangChain - LLM 应用框架
- ChromaDB - 向量数据库
- OpenAI/本地模型 - 大语言模型
- Sentence-Transformers - 文本嵌入

## 🎯 技术亮点

- ✅ **向量检索** - 语义相似度搜索，不只是关键词匹配
- ✅ **上下文增强** - 自动检索相关文档片段作为上下文
- ✅ **多文档支持** - 支持 PDF、TXT、Markdown 等多种格式
- ✅ **流式输出** - 实时生成回答，用户体验更好
- ✅ **可替换模型** - 支持 OpenAI、本地模型（ChatGLM、Qwen）

## 📂 项目结构

```
rag-knowledge-base/
├── main.py              # 主程序入口
├── config.py            # 配置文件
├── document_loader.py   # 文档加载和解析
├── vector_store.py      # 向量数据库管理
├── retriever.py         # 检索器
├── qa_chain.py          # 问答链
├── requirements.txt     # 依赖
└── README.md           # 说明文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 国内用户加速：
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 2. 配置 API Key

```bash
# 方式 1：环境变量
export OPENAI_API_KEY="your-api-key"

# 方式 2：修改 config.py
OPENAI_API_KEY = "your-api-key"
```

### 3. 准备文档

在 `documents/` 目录下放置你的文档：

```
documents/
├── 项目文档.pdf
├── 技术手册.md
└── 常见问题.txt
```

### 4. 构建知识库

```bash
python main.py --mode build
```

**这会：**
- 解析所有文档
- 切分成文本块（chunk）
- 生成向量嵌入
- 存储到向量数据库

### 5. 开始问答

```bash
# 交互式问答
python main.py --mode chat

# 单次问答
python main.py --mode ask --question "如何安装这个软件？"
```

## 🔧 核心配置

```python
# config.py

# 文档处理
CHUNK_SIZE = 500          # 文本块大小
CHUNK_OVERLAP = 50        # 块重叠大小

# 检索
TOP_K = 3                 # 检索最相关的 3 个片段
SIMILARITY_THRESHOLD = 0.7  # 相似度阈值

# 模型
MODEL_NAME = "gpt-3.5-turbo"  # 或本地模型
TEMPERATURE = 0.7         # 生成温度
```

## 💡 进阶用法

### 1. 使用本地模型（免费）

```python
# 使用 ChatGLM
from langchain.llms import ChatGLM

llm = ChatGLM(
    endpoint_url="http://localhost:8000",
    max_tokens=2048,
    temperature=0.7
)
```

### 2. 自定义检索策略

```python
# 混合检索：向量 + 关键词
from langchain.retrievers import EnsembleRetriever

vector_retriever = vectorstore.as_retriever()
bm25_retriever = BM25Retriever.from_documents(docs)

ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)
```

### 3. 添加元数据过滤

```python
# 只检索特定来源的文档
retriever = vectorstore.as_retriever(
    search_kwargs={
        "filter": {"source": "技术手册"}
    }
)
```

## 📊 性能优化

### 1. 大批量文档处理

```python
# 使用批量处理
docs = loader.load()
chunks = split_documents(docs, chunk_size=500)

# 分批嵌入
for i in range(0, len(chunks), 100):
    batch = chunks[i:i+100]
    vectorstore.add_documents(batch)
```

### 2. 缓存嵌入结果

```python
from langchain.embeddings import CacheBackedEmbeddings

# 缓存到本地，避免重复计算
embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=OpenAIEmbeddings(),
    document_embedding_cache="./cache"
)
```

### 3. 异步处理

```python
import asyncio

async def process_documents():
    # 并发处理多个文档
    tasks = [load_document(doc) for doc in documents]
    results = await asyncio.gather(*tasks)
```

## 🐛 常见问题

### Q: 回答不准确怎么办？

**A:** 
- 增大 CHUNK_SIZE（800-1000）
- 增加 TOP_K（5-7 个片段）
- 优化 prompt 模板
- 检查文档质量

### Q: 向量数据库太大怎么办？

**A:**
```python
# 使用更小的嵌入模型
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# 384 维 vs 1536 维，体积小 4 倍
```

### Q: 响应速度慢怎么办？

**A:**
- 使用流式输出
- 减少 TOP_K
- 缓存常见问题的回答
- 使用更小的 LLM

## 📚 相关资源

- [LangChain 官方文档](https://python.langchain.com/)
- [RAG 技术详解](https://arxiv.org/abs/2005.11401)
- [ChromaDB 文档](https://docs.trychroma.com/)

## 📄 许可证

MIT License
