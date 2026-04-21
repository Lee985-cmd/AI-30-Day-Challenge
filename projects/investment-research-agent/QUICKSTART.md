# 智能投研助手 - 快速开始指南

## ✅ 环境检查完成

你的智能投研助手已经配置完成！

### 当前配置
- ✅ Python 3.14.3
- ✅ LOCAL_LLM_URL: `http://localhost:30001/v1`
- ✅ 所有依赖包已安装
- ✅ 本地模型连接成功

## 🚀 如何使用

### 方式 1: 运行测试脚本

```bash
python test_agent.py
```

然后选择测试模式：
- **1** - 快速分析测试（推荐先试这个）
- **2** - 完整研究流程测试
- **3** - 批量研究测试
- **4** - 运行所有测试

### 方式 2: 在 Python 代码中使用

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

### 方式 3: 快速分析

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

## 📝 注意事项

### 1. 数据源说明

当前版本使用**模拟数据**进行演示。本地模型可能无法获取实时的金融数据，因此会返回"数据不足"。

**改进建议：**
- 接入真实数据源（Tushare、聚宽等）
- 手动补充关键财务数据
- 使用 RAG 技术加载财报文档

### 2. 本地模型要求

确保你的本地模型服务：
- ✅ 支持 OpenAI 兼容接口
- ✅ 模型具有足够的上下文长度（建议 8K+）
- ✅ 模型具备较好的中文理解能力

**推荐的本地模型：**
- Qwen-7B / Qwen-14B（通义千问）
- ChatGLM3-6B
- Llama-3-8B-Chinese

### 3. 性能优化

如果响应较慢，可以：
- 使用更小的模型（7B 而非 14B/70B）
- 降低 temperature 参数（0.2-0.3）
- 简化 Prompt，减少输出长度

## 🔧 常见问题

### Q1: 如何更换本地模型地址？

```powershell
# 修改环境变量
[System.Environment]::SetEnvironmentVariable("LOCAL_LLM_URL", "http://new-server:port/v1", "User")

# 重启终端生效
```

### Q2: 如何提高报告质量？

1. **优化 Prompt**：在 Agent 的提示词中添加更多具体要求
2. **提供数据**：手动传入财务数据作为上下文
3. **调整温度**：研究和风控用 0.2-0.3，写作用 0.5

### Q3: 能否用于实际投资？

**强烈不建议！** 本系统仅用于学习和研究：
- ⚠️ 数据可能不准确
- ⚠️ AI 可能产生幻觉
- ⚠️ 缺乏实时市场监控
- ⚠️ 不构成投资建议

## 📚 下一步

1. **阅读完整文档**：查看 [README.md](README.md)
2. **学习源码**：查看 `data_agent/` 目录下的各个 Agent 实现
3. **自定义开发**：根据需求修改 Prompt 或添加新功能
4. **接入真实数据**：集成 Tushare 或其他金融数据 API

## 💡 示例：接入 Tushare 数据

```python
# 在 researcher.py 中修改 get_financial_data 方法
import tushare as ts

def get_financial_data(self, stock_code: str):
    """获取真实财务数据"""
    pro = ts.pro_api('your_tushare_token')
    
    # 获取基本信息
    df_basic = pro.stock_basic(ts_code=stock_code)
    
    # 获取财务指标
    df_indicator = pro.fina_indicator(ts_code=stock_code)
    
    return df_indicator
```

---

**祝你使用愉快！如有问题，欢迎提 Issue 或在 CSDN 评论区讨论。** 🎉
