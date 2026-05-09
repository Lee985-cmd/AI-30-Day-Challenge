# 多Agent协作系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 项目简介

多Agent协作系统示例，展示如何让多个AI Agent协同工作完成复杂任务。

**包含案例：**
- ✅ 智能投研系统（7个Agent协作）
- ✅ 内容创作团队（4个Agent协作）
- ✅ 客户服务流水线（3个Agent协作）
- ✅ 自定义Agent协调器框架

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

```bash
# 设置环境变量
export OPENAI_API_KEY="your-api-key"
```

### 3. 运行示例

```bash
# 智能投研系统
python investment_research.py --company "特斯拉"

# 内容创作团队
python content_creation.py --topic "AI发展趋势"

# 客户服务流水线
python customer_service.py

# 自定义协调器示例
python custom_coordinator.py
```

## 📁 项目结构

```
multi-agent-systems/
├── investment_research.py    # 智能投研系统
├── content_creation.py       # 内容创作团队
├── customer_service.py       # 客户服务流水线
├── custom_coordinator.py     # 自定义协调器
├── agents/                   # Agent定义模块
│   ├── __init__.py
│   ├── researcher.py
│   ├── analyst.py
│   └── writer.py
├── requirements.txt          # 依赖列表
└── README.md                 # 本文件
```

## 🔗 相关链接

- [配套文章](https://blog.csdn.net/m0_67081842/article/details/160911514?spm=1011.2415.3001.5331)
- [CrewAI文档](https://docs.crewai.com/)
- [LangGraph文档](https://langchain-ai.github.io/langgraph/)
