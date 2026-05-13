# LangChain vs LlamaIndex vs Haystack：三大Agent框架深度对比

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 项目简介

本项目深度对比三大主流AI Agent开发框架：**LangChain**、**LlamaIndex** 和 **Haystack**，帮助开发者选择最适合的工具栈。

**核心功能：**
- ✅ LangChain完整示例（工具调用、记忆、链式编排）
- ✅ LlamaIndex完整示例（RAG、数据索引、查询引擎）
- ✅ Haystack完整示例（Pipeline、DocumentStore、Retriever）
- ✅ 性能基准测试对比
- ✅ 适用场景分析指南
- ✅ 框架迁移示例

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key"
$env:MODEL_NAME="gpt-3.5-turbo"

# Linux/Mac
export OPENAI_API_KEY="your-api-key"
export MODEL_NAME="gpt-3.5-turbo"
```

**注意：** 也可以使用本地模型（如Ollama），修改代码中的LLM配置即可。

### 3. 运行示例

```bash
# LangChain示例
python langchain_example.py

# LlamaIndex示例
python llamaindex_example.py

# Haystack示例
python haystack_example.py

# 性能对比测试
python benchmark.py

# 框架选择助手
python framework_selector.py
```

## 📁 项目结构

```
agent-framework-comparison/
├── langchain_example.py       # LangChain完整示例
├── llamaindex_example.py      # LlamaIndex完整示例
├── haystack_example.py        # Haystack完整示例
├── benchmark.py               # 性能基准测试
├── framework_selector.py      # 框架选择助手
├── requirements.txt           # 依赖列表
└── README.md                  # 本文件
```

## 🔍 框架对比总览

| 特性 | LangChain | LlamaIndex | Haystack |
|-----|-----------|------------|----------|
| **核心定位** | Agent编排 | RAG专精 | 生产级搜索 |
| **学习曲线** | 中等 | 简单 | 较陡 |
| **灵活性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **RAG能力** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Agent能力** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **社区活跃度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **文档质量** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **生产就绪** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🎯 如何选择？

### 选择 LangChain 如果：
- ✅ 需要构建复杂的Agent工作流
- ✅ 需要灵活的工具调用和多步推理
- ✅ 团队熟悉Python生态
- ✅ 需要快速原型开发

### 选择 LlamaIndex 如果：
- ✅ 主要做RAG应用
- ✅ 有大量非结构化数据需要索引
- ✅ 需要高级查询优化
- ✅ 追求简单易用

### 选择 Haystack 如果：
- ✅ 构建企业级搜索系统
- ✅ 需要多语言支持
- ✅ 重视生产环境稳定性
- ✅ 需要REST API服务

## 🔗 相关链接

- [配套文章](https://blog.csdn.net/m0_67081842/article/details/161036578?spm=1011.2415.3001.5331)
- [LangChain官方文档](https://python.langchain.com/)
- [LlamaIndex官方文档](https://docs.llamaindex.ai/)
- [Haystack官方文档](https://haystack.deepset.ai/)
- [Agent专题系列](../)
