# 📊 智能投研助手 - AI Investment Research Agent

> 基于多智能体协作的自动化投资研究系统

## 🎯 项目简介

智能投研助手是一个基于 LangChain 和多智能体协作的投资研究系统，能够自动完成：

- ✅ **数据收集**：自动收集公司基本面、财务数据、行业信息
- ✅ **深度分析**：多维度分析投资亮点和风险
- ✅ **报告生成**：自动生成专业的投资研究报告
- ✅ **风险评估**：严格的风险控制和合规检查

**效率提升：** 传统投研需要 20+ 小时 → Agent 仅需 15 分钟！

## 🏗️ 系统架构

```
用户输入（股票代码）
  ↓
[研究员 Agent] → 收集市场数据、财务数据、行业信息
  ↓
[分析师 Agent] → 深度分析、估值判断、投资建议
  ↓
[写作 Agent]   → 生成专业研究报告
  ↓
[风险管理 Agent] → 风险评估、合规检查、添加免责声明
  ↓
输出：完整的投资研究报告（Markdown格式）
```

### 核心组件

| 组件 | 职责 | 技术栈 |
|------|------|--------|
| Researcher Agent | 数据收集与研究 | LangChain + 通义千问 |
| Analyst Agent | 深度分析与建议 | LangChain + 通义千问 |
| Writer Agent | 报告撰写 | LangChain + 通义千问 |
| Risk Manager | 风险评估与合规 | LangChain + 通义千问 |
| Orchestrator | 流程编排与协调 | 自定义编排逻辑 |

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n investment-agent python=3.10
conda activate investment-agent

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置本地模型 URL

```powershell
# PowerShell（永久设置，需要管理员权限）
[System.Environment]::SetEnvironmentVariable("LOCAL_LLM_URL", "http://your-server:port/v1", "User")

# 或者临时设置（当前会话有效）
$env:LOCAL_LLM_URL = "http://your-server:port/v1"
```

**配置说明：**
- `LOCAL_LLM_URL` 应指向 OpenAI 兼容的 API 接口
- 例如：`http://localhost:8000/v1` 或 `http://your-server:port/v1`
- 支持任何 OpenAI 兼容的本地模型服务（Ollama、LM Studio、vLLM等）

### 3. 运行示例

```python
from data_agent import InvestmentResearchOrchestrator

# 初始化投研助手
orchestrator = InvestmentResearchOrchestrator()

# 执行完整研究
result = orchestrator.conduct_research(
    stock_code="600519",
    company_name="贵州茅台",
    research_focus="全面分析公司基本面和投资价值",
    save_report=True
)

if result["success"]:
    print(f"✅ 研究完成！")
    print(f"📄 报告路径: {result['report_path']}")
    print(f"⏱️  耗时: {result['duration_seconds']:.2f} 秒")
```

### 4. 查看生成的报告

报告保存在 `reports/` 目录下，格式为 Markdown，可直接阅读或转换为 PDF。

## 📋 使用示例

### 示例 1：完整研究流程

```python
from data_agent import InvestmentResearchOrchestrator

orchestrator = InvestmentResearchOrchestrator()

# 研究贵州茅台
result = orchestrator.conduct_research(
    stock_code="600519",
    company_name="贵州茅台",
    research_focus="全面分析",
    save_report=True
)

# 研究宁德时代
result = orchestrator.conduct_research(
    stock_code="300750",
    company_name="宁德时代",
    research_focus="电池行业竞争格局和技术优势",
    save_report=True
)
```

### 示例 2：快速分析

```python
from data_agent import InvestmentResearchOrchestrator

orchestrator = InvestmentResearchOrchestrator()

# 快速获取核心观点
quick_result = orchestrator.quick_analysis(
    stock_code="600519",
    company_name="贵州茅台"
)

print(quick_result)
```

### 示例 3：单独使用某个 Agent

```python
from data_agent import ResearcherAgent, AnalystAgent

# 单独使用研究员
researcher = ResearcherAgent()
result = researcher.research(
    stock_code="600519",
    company_name="贵州茅台",
    research_focus="财务分析"
)

# 获取财务数据
df = researcher.get_financial_data("600519")
print(df)
```

## 📊 报告结构

生成的研究报告包含以下章节：

1. **核心观点** - 200-300字概括
2. **公司概况** - 主营业务、竞争力、行业地位
3. **财务分析** - 成长性、盈利能力、健康度
4. **投资亮点** - 3-5个核心投资逻辑
5. **风险提示** - 3-5个主要风险点
6. **估值与目标价** - 估值水平、目标价区间
7. **投资建议** - 评级、仓位、操作策略
8. **关键跟踪指标** - 需要持续关注的指标
9. **免责声明** - 法律风险提示

## ⚠️ 重要声明

**本项目仅供学习和研究使用，不构成任何投资建议！**

1. **非投资建议**：AI 生成的报告仅供参考，投资者应独立判断
2. **数据准确性**：数据来源于公开资料，可能存在滞后或不准确
3. **风险提示**：股市有风险，投资需谨慎
4. **责任免除**：不对因使用本报告产生的任何损失承担责任
5. **合规提醒**：未经批准不得从事证券投资咨询业务

**建议：** 在做出投资决策前，请咨询持牌证券顾问或金融专业人士。

## 🔧 技术细节

### 多智能体协作

系统采用**分工协作**的设计模式：

- **Researcher**：专注数据收集，确保信息全面
- **Analyst**：专注深度分析，提供专业见解
- **Writer**：专注报告撰写，保证专业性
- **Risk Manager**：专注风险控制，确保合规

每个 Agent 都有专门的 Prompt 模板和职责边界，通过 Orchestrator 协调工作。

### Prompt 工程

关键的 Prompt 设计原则：

1. **角色定义清晰**：每个 Agent 有明确的专业角色
2. **输出结构化**：使用 JSON 或 Markdown 格式
3. **约束条件明确**：避免幻觉，要求标注"数据不足"
4. **温度控制**：研究和风控用低温（0.2-0.3），写作用中温（0.5）

### 数据源

当前版本使用**模拟数据**进行演示。实际应用中可以接入：

- **Tushare**：免费的金融数据接口
- **聚宽（JoinQuant）**：量化交易平台
- **东方财富 API**：实时行情数据
- **Wind/Choice**：专业金融终端（付费）

## 🛠️ 扩展开发

### 添加新的数据源

修改 `researcher.py` 中的 `get_financial_data()` 方法：

```python
def get_financial_data(self, stock_code: str):
    # 接入 Tushare
    import tushare as ts
    pro = ts.pro_api('your_tushare_token')
    df = pro.income(ts_code=stock_code)
    return df
```

### 自定义 Agent

创建新的 Agent 类，继承统一的接口：

```python
class CustomAgent:
    def __init__(self, api_key=None):
        self.llm = ChatTongyi(model="qwen-plus", api_key=api_key)
    
    def execute(self, input_data):
        # 实现你的逻辑
        pass
```

### 集成到 Web 应用

可以参考 `data-analysis-agent` 项目，使用 FastAPI + Streamlit 构建 Web 界面。

## 📈 性能优化

### 成本控制

- **模型选择**：使用 `qwen-plus` 而非 `qwen-max`，性价比更高
- **缓存机制**：对相同股票的查询结果进行缓存
- **批量处理**：一次性研究多个股票，减少 API 调用次数

### 响应速度

- **并行处理**：研究和财务数据获取可以并行
- **异步调用**：使用 async/await 提高并发能力
- **流式输出**：实时显示研究进度

## 🐛 常见问题

### Q1: API Key 配置后仍然报错？

**A:** 检查以下几点：
1. 确认 API Key 是否正确复制（没有多余空格）
2. 重启终端使环境变量生效
3. 验证 API Key 是否有余额/额度

### Q2: 生成的报告质量不高？

**A:** 可以尝试：
1. 调整 `research_focus` 参数，更具体地描述研究重点
2. 使用更强大的模型（如 `qwen-max`）
3. 手动补充一些关键数据

### Q3: 如何接入真实数据？

**A:** 参考"扩展开发"部分，修改 `researcher.py` 中的数据获取方法。推荐先使用 Tushare 免费接口。

### Q4: 能否用于实盘交易？

**A:** **强烈不建议！** 本系统仅用于学习和研究，存在以下局限：
- 数据可能滞后或不准确
- AI 可能产生幻觉
- 缺乏实时市场监控
- 未考虑个人风险承受能力

## 📚 相关资源

- 💻 **完整代码：** https://github.com/Lee985-cmd/AI-30-Day-Challenge
- 📖 **30天AI挑战教程：** https://blog.csdn.net/m0_67081842
- 📝 **Agent专题文章：** CSDN搜索"Lee的成长日记"
- ❓ **有问题？** 提 [Issue](https://github.com/Lee985-cmd/AI-30-Day-Challenge/issues)

## 🤝 贡献指南

欢迎贡献代码、报告 Bug 或提出建议！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](../../LICENSE) 文件

## 👨‍💻 关于作者

**Lee** - 职场宝爸 / AI 学习者 / Agent 实践者

- CSDN: https://blog.csdn.net/m0_67081842
- GitHub: https://github.com/Lee985-cmd
- 更新频率：每周 2-3 篇技术干货

---

> 💡 **如果这个项目对你有帮助，欢迎 Star ⭐ 支持一下！**
> 
> **你的支持是我持续更新的最大动力！** ❤️
