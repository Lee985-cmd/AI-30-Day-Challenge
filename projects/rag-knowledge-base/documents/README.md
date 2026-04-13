# RAG 知识库系统 - 示例文档

## 什么是 RAG？

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索和文本生成的技术。

## 核心原理

1. **文档加载** - 读取 PDF、TXT、Markdown 等格式的文档
2. **文本分块** - 将长文档切分成小块（chunk）
3. **向量嵌入** - 使用嵌入模型将文本转换为向量
4. **向量存储** - 将向量存入数据库（ChromaDB/FAISS）
5. **语义检索** - 根据用户问题检索最相关的文档片段
6. **增强生成** - 将检索到的上下文和问题一起发给 LLM 生成回答

## 技术栈

- **LangChain** - LLM 应用框架
- **ChromaDB** - 向量数据库
- **OpenAI Embeddings** - 文本嵌入模型
- **OpenAI GPT** - 大语言模型

## 应用场景

1. **企业知识库问答** - 上传公司文档，员工可以提问
2. **智能客服** - 基于产品手册自动回答用户问题
3. **法律文档检索** - 快速查找相关法律条文
4. **学术论文问答** - 基于论文内容回答问题
5. **技术文档查询** - API 文档、使用手册智能检索

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 OpenAI API Key：

```bash
cp .env.example .env
# 编辑 .env 文件，填入 API Key
```

### 3. 准备文档

将你的文档（PDF/TXT/MD/DOCX）放到 `documents/` 目录下。

### 4. 构建知识库

```bash
python main.py --mode build
```

### 5. 开始问答

```bash
# 交互式问答
python main.py --mode chat

# 单次问答
python main.py --mode ask --question "什么是 RAG？"
```

## 配置说明

在 `config.py` 中可以调整以下参数：

- **CHUNK_SIZE** - 文本块大小（默认 500）
- **TOP_K** - 检索相关片段数量（默认 3）
- **TEMPERATURE** - 生成温度（默认 0.7）

## 常见问题

**Q: 可以使用本地模型吗？**

A: 可以！修改 `config.py` 中的 `USE_LOCAL_MODEL = True`，并配置本地模型服务地址。

**Q: 支持哪些文档格式？**

A: PDF、TXT、Markdown、DOCX

**Q: 回答不准确怎么办？**

A: 尝试调整：
- 增大 CHUNK_SIZE
- 增加 TOP_K
- 优化文档质量
