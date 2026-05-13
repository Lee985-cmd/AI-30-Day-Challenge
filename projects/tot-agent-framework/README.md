# Tree of Thoughts Agent框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 项目简介

Tree of Thoughts（思维树）框架实现，展示如何让Agent进行多路径探索和智能决策。

**核心功能：**
- ✅ 基础ToT实现（BFS/DFS搜索）
- ✅ 启发式搜索（A*变种）
- ✅ 剪枝优化策略
- ✅ 评估缓存机制
- ✅ 并行探索支持
- ✅ 24点游戏实战案例

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置本地模型（可选）

```bash
# Windows PowerShell
$env:LOCAL_LLM_URL="http://localhost:8000/v1"
$env:LOCAL_LLM_API_KEY="not-needed"
$env:MODEL_NAME="qwen-plus"

# Linux/Mac
export LOCAL_LLM_URL="http://localhost:8000/v1"
export LOCAL_LLM_API_KEY="not-needed"
export MODEL_NAME="qwen-plus"
```

**注意：** 如果不配置环境变量，项目会使用Mock LLM进行演示。

### 3. 运行示例

```bash
# 基础ToT示例
python basic_tot.py

# 24点游戏求解器
python solve_24_game.py

# 创意写作助手
python creative_writer.py

# 性能基准测试
python benchmark.py
```

## 📁 项目结构

```
tot-agent-framework/
├── basic_tot.py              # 基础ToT实现
├── solve_24_game.py          # 24点游戏求解器
├── creative_writer.py        # 创意写作助手
├── benchmark.py              # 性能测试
├── requirements.txt          # 依赖列表
└── README.md                 # 本文件
```

## 🔗 相关链接

- [配套文章](https://blog.csdn.net/m0_67081842/article/details/161035441?spm=1011.2415.3001.5331)
- [ToT论文](https://arxiv.org/abs/2305.10601)
- [ReAct框架项目](../react-agent-framework/)
