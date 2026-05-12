# Agent工具调用系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 项目简介

Agent工具调用系统实现，展示如何让AI通过Function Calling获得超能力。

**核心功能：**
- ✅ 自定义工具定义与注册
- ✅ LangChain Tools集成
- ✅ 智能工具选择与路由
- ✅ 工具结果缓存优化
- ✅ 并行工具调用
- ✅ 智能研究助手示例

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行示例

```bash
# 基础工具调用示例
python basic_tools.py

# 自定义工具类示例
python custom_tools.py

# 智能研究助手
python research_assistant.py

# 工具路由器示例
python tool_router.py
```

## 📁 项目结构

```
agent-tool-calling/
├── basic_tools.py            # 基础工具示例
├── custom_tools.py           # 自定义工具类
├── research_assistant.py     # 智能研究助手
├── tool_router.py            # 工具路由器
├── requirements.txt          # 依赖列表
└── README.md                 # 本文件
```

## 🔗 相关链接

- [配套文章](https://blog.csdn.net/m0_67081842/article/details/161003359?spm=1011.2415.3001.5331)
- [LangChain Tools文档](https://python.langchain.com/docs/modules/agents/tools/)
- [Function Calling指南](https://platform.openai.com/docs/guides/function-calling)
