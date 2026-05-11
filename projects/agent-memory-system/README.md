# Agent记忆系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 项目简介

Agent记忆系统实现，展示如何让AI拥有短期、长期和工作记忆能力。

**核心功能：**
- ✅ 短期记忆（滑动窗口对话历史）
- ✅ 长期记忆（向量数据库存储）
- ✅ 工作记忆（任务上下文）
- ✅ 知识图谱记忆（结构化关系）
- ✅ 混合记忆系统（生产级实现）
- ✅ 个性化助手示例

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行示例

```bash
# 向量记忆示例
python vector_memory.py

# 知识图谱记忆示例
python knowledge_graph_memory.py

# 混合记忆系统示例
python hybrid_memory.py

# 个性化助手示例
python personalized_assistant.py
```

## 📁 项目结构

```
agent-memory-system/
├── vector_memory.py            # 向量数据库记忆
├── knowledge_graph_memory.py   # 知识图谱记忆
├── hybrid_memory.py            # 混合记忆系统
├── personalized_assistant.py   # 个性化助手
├── requirements.txt            # 依赖列表
└── README.md                   # 本文件
```

## 🔗 相关链接

- [配套文章](https://blog.csdn.net/m0_67081842/article/details/160967551?spm=1011.2415.3001.5331)
- [LangChain Memory文档](https://python.langchain.com/docs/modules/memory/)
- [Chroma向量数据库](https://docs.trychroma.com/)
