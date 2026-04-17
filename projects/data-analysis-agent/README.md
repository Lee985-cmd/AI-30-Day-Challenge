# AI 数据分析 Agent

> 基于 LangChain + 本地大模型的智能数据分析系统，支持自然语言查询、自动数据清洗、智能可视化。

![Web 界面](images/web-1.png)

## 🎯 功能特性

- ✅ **自然语言查询** - 用中文提问即可分析数据
- ✅ **自动数据清洗** - 一键处理缺失值和异常值
- ✅ **智能可视化** - 自动生成合适的图表
- ✅ **多格式支持** - Excel、CSV 文件
- ✅ **Web 界面** - Streamlit 交互式操作
- ✅ **本地模型** - 支持私有化部署的大模型服务

---

## 📸 系统界面

### 1️⃣ 上传数据文件

![上传数据](images/web-1.png)

支持 Excel 和 CSV 格式，上传后自动预览数据结构。

### 2️⃣ 自然语言查询

![数据查询](images/web-2.png)

用中文提问，AI 自动生成分析代码并返回结果。

### 3️⃣ 智能可视化

![数据可视化](images/web-3.png)

AI 自动选择合适的图表类型，生成专业的数据可视化图表。

### 4️⃣ 控制台输出

![控制台](images/console.png)

实时查看 AI 生成的代码和系统日志。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置本地模型地址（可选）

**方案 A：使用系统环境变量（推荐）**

```powershell
# PowerShell 管理员权限运行
[System.Environment]::SetEnvironmentVariable("LOCAL_LLM_URL", "http://your-server:port/v1", "User")
[System.Environment]::SetEnvironmentVariable("LOCAL_LLM_API_KEY", "your-api-key", "User")
```

**方案 B：使用阿里百炼（需付费）**

```powershell
# PowerShell 管理员权限运行
[System.Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", "sk-your-api-key", "User")
```

获取 API Key：https://dashscope.console.aliyun.com/

> 💡 本项目默认使用本地大模型服务（兼容 OpenAI 接口），支持任意部署地址。

### 3. 生成示例数据

```bash
python generate_sample_data.py
```

### 4. 启动服务

```bash
# 启动 API 服务
python data_agent/api.py

# 启动 Web 界面（新终端）
streamlit run data_agent/web_app.py
```

### 5. 访问应用

打开浏览器访问：http://localhost:8501

## 📁 项目结构

```
data-analysis-agent/
├── data_agent/
│   ├── __init__.py
│   ├── data_loader.py          # 数据加载模块
│   ├── pandas_agent.py         # Pandas AI Agent
│   ├── cleaning_agent.py       # 数据清洗 Agent
│   ├── visualization_agent.py  # 可视化 Agent
│   ├── api.py                  # FastAPI 接口
│   └── web_app.py              # Streamlit 界面
├── data/                       # 示例数据目录
├── charts/                     # 生成的图表
├── uploads/                    # 上传的文件
├── generate_sample_data.py     # 数据生成脚本
├── requirements.txt
└── README.md
```

## 💡 使用示例

### 自然语言查询

在 Web 界面输入问题，AI 会自动分析数据：

```python
# 示例 1：查询地区销售排名
问题：哪个地区的销售额最高？
结果：华东

# 示例 2：品类销售对比  
问题：Q3 各品类的销售额对比
结果：电子产品 > 服装 > 食品 > 家居 > 图书

# 示例 3：趋势分析
问题：华东区 Q2 和 Q3 的销售额变化趋势
结果：Q2: 45,231 → Q3: 52,187 (增长 15.4%)
```

### 数据可视化

描述你的需求，AI 会自动生成图表：

```python
# 示例 1：柱状图
需求：绘制各地区销售额对比柱状图
结果：生成横向柱状图，按销售额排序

# 示例 2：折线图
需求：绘制月度销售额趋势折线图  
结果：生成 12 个月的时间序列折线图

# 示例 3：饼图
需求：绘制各品类销售额占比饼图
结果：生成饼图，显示各类别占比
```

---

## 📊 效率提升对比

| 任务 | 传统方式 | Agent 方式 | 提升倍数 |
|------|---------|-----------|---------|
| 数据加载 | 5 分钟 | 1 分钟 | 5x |
| 数据清洗 | 30 分钟 | 5 分钟 | 6x |
| 数据查询 | 20 分钟 | 2 分钟 | 10x |
| 可视化 | 25 分钟 | 3 分钟 | 8x |
| 报告撰写 | 40 分钟 | 4 分钟 | 10x |
| **总计** | **2 小时** | **15 分钟** | **8x** |

## 🔗 相关文章

- [CSDN] 用 Agent 自动化数据处理：从 2 小时到 15 分钟的效率革命

## 👨💻 作者

**Lee** - 职场宝爸 / AI 学习者

- GitHub：https://github.com/Lee985-cmd
- CSDN：https://blog.csdn.net/m0_67081842

## 🏗️ 技术架构

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Streamlit │ ──▶ │   FastAPI    │ ──▶ │ LangChain   │
│   Web 界面  │     │   REST API   │     │ Agent 框架  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │    本地大模型服务     │
                                    │  (OpenAI 兼容接口)    │
                                    └───────────────────────┘
```

### 核心组件

- **LangChain** - Agent 框架，负责任务编排和工具调用
- **本地大模型** - 支持任意兼容 OpenAI 接口的大模型服务
- **FastAPI** - 高性能异步 Web 框架
- **Streamlit** - Python Web 界面框架
- **Pandas** - 数据处理和分析
- **Matplotlib/Seaborn** - 数据可视化

---

## 📝 许可证

MIT License

---

**如果觉得这个项目对你有帮助，欢迎 ⭐ Star 支持！**
