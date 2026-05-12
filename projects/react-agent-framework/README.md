# ReAct Agent框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 项目简介

ReAct（Reasoning + Acting）框架实现，展示如何让Agent会思考再行动。

**核心功能：**
- ✅ 从零实现ReAct Agent
- ✅ LangChain ReAct集成
- ✅ 智能早期停止策略
- ✅ 并行探索（Tree of Thoughts扩展）
- ✅ 性能基准测试
- ✅ 完整执行轨迹记录

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行示例

```bash
# 基础ReAct实现
python basic_react.py

# LangChain ReAct示例
python langchain_react.py

# 智能ReAct（带早期停止）
python smart_react.py

# 并行ReAct（多路径探索）
python parallel_react.py

# 性能基准测试
python benchmark.py
```

## 📁 项目结构

```
react-agent-framework/
├── basic_react.py            # 基础ReAct实现
├── langchain_react.py        # LangChain ReAct
├── smart_react.py            # 智能ReAct
├── parallel_react.py         # 并行ReAct
├── benchmark.py              # 性能测试
├── requirements.txt          # 依赖列表
└── README.md                 # 本文件
```

## 🔗 相关链接

- [配套文章](https://blog.csdn.net/m0_67081842/article/details/161003867?spm=1011.2415.3001.5331)
- [ReAct论文](https://arxiv.org/abs/2210.03629)
- [LangChain ReAct文档](https://python.langchain.com/docs/modules/agents/agent_types/react)
