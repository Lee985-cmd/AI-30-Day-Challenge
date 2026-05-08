# Agent测试框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 项目简介

完整的Agent系统测试框架，包含单元测试、集成测试、质量评估和A/B测试工具。

**核心功能：**
- ✅ 单元测试（Prompt、工具函数、解析器）
- ✅ 集成测试（RAG系统、Agent工作流）
- ✅ 质量评估（RAGAS指标、人工评估）
- ✅ A/B测试框架
- ✅ 自动化测试运行器
- ✅ CI/CD集成示例

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 运行质量评估
python scripts/evaluate_quality.py

# 运行A/B测试
python scripts/ab_test.py
```

## 📁 项目结构

```
agent-testing-framework/
├── tests/
│   ├── unit/                 # 单元测试
│   │   ├── test_prompts.py
│   │   ├── test_tools.py
│   │   └── test_parsers.py
│   ├── integration/          # 集成测试
│   │   ├── test_rag.py
│   │   ├── test_agent_workflow.py
│   │   └── test_multitenant.py
│   └── fixtures/             # 测试数据
├── scripts/
│   ├── evaluate_quality.py   # 质量评估脚本
│   ├── ab_test.py            # A/B测试脚本
│   └── regression_test.py    # 回归测试
├── configs/
│   └── test_config.yml       # 测试配置
├── requirements.txt          # 依赖列表
└── README.md                 # 本文件
```

## 📊 质量指标

| 指标 | 目标值 | 说明 |
|-----|-------|------|
| **Faithfulness** | > 0.8 | 答案忠实于上下文 |
| **Answer Relevancy** | > 0.7 | 答案相关性强 |
| **Context Precision** | > 0.75 | 检索精度高 |
| **Success Rate** | > 95% | 请求成功率 |
| **P95 Response Time** | < 5s | 95%请求在5秒内 |

## 🔗 相关链接

- [配套文章](链接)
- [RAGAS文档](https://docs.ragas.io/)
- [pytest文档](https://docs.pytest.org/)
