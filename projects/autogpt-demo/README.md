# AutoGPT 实战演示项目

## 🎯 项目简介

这是一个简化版的 AutoGPT 实现，展示了 AI Agent 如何自主完成任务的核心理念。

**核心特性：**
- ✅ 自主任务分解
- ✅ 多步骤执行
- ✅ 记忆管理
- ✅ 自我反思和优化

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
export LOCAL_LLM_URL="http://your-server:port/v1"

# 运行演示
python demo_autonomous_agent.py
```

## 📂 项目结构

```
autogpt-demo/
├── autogpt_agent/           # Agent核心模块（可选）
│   ├── __init__.py
│   ├── autonomous_agent.py  # 自主Agent核心
│   ├── task_planner.py      # 任务规划器
│   ├── executor.py          # 任务执行器
│   └── memory.py            # 记忆系统
├── examples/                # 示例文件
│   └── sample_output.md     # 示例输出（参考）
├── outputs/                 # 运行时输出（自动生成，不提交Git）
│   └── *.md                 # 每次运行生成的报告
├── demo_autonomous_agent.py # 主演示脚本
├── requirements.txt         # 依赖配置
├── .gitignore              # Git忽略配置
└── README.md               # 本文件
```

## 💡 使用示例

### 示例 1: 自动写博客

```python
from autogpt_agent import AutonomousAgent

agent = AutonomousAgent()

result = agent.execute(
    goal="写一篇关于 AI Agent 的技术博客",
    max_iterations=5
)

print(result)
```

### 示例 2: 市场调研

```python
result = agent.execute(
    goal="调研 2024 年 AI Agent 市场趋势",
    max_iterations=8
)
```

## ⚠️ 注意事项

1. **需要本地模型** - 配置 `LOCAL_LLM_URL` 环境变量
2. **演示用途** - 这是简化版，展示核心理念
3. **成本考虑** - 真实AutoGPT会消耗大量API调用
4. **输出文件** - `outputs/` 目录在首次运行时会自加创建，生成的报告会保存在此

## 🔗 相关资源

- 完整文章：《AutoGPT 实战：让 AI 自主完成任务》
- GitHub: https://github.com/Lee985-cmd/AI-30-Day-Challenge
