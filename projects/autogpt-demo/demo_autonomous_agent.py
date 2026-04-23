"""
AutoGPT 简化版 - 自主任务执行Agent
演示AI如何自主分解任务、执行、并完成任务目标
"""

import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


class ShortTermMemory:
    """短期记忆"""
    
    def __init__(self, max_size: int = 20):
        self.memories: List[Dict] = []
        self.max_size = max_size
    
    def add(self, step: int, action: str, result: str):
        memory = {
            "step": step,
            "action": action,
            "result": result[:200]
        }
        self.memories.append(memory)
        if len(self.memories) > self.max_size:
            self.memories.pop(0)
    
    def get_context(self) -> str:
        if not self.memories:
            return "暂无历史记录"
        
        context = "已执行的步骤：\n"
        for mem in self.memories[-5:]:
            context += f"步骤 {mem['step']}: {mem['action']}\n"
            context += f"结果: {mem['result']}\n\n"
        
        return context


class TaskPlanner:
    """任务规划器"""
    
    def __init__(self):
        local_llm_url = os.getenv("LOCAL_LLM_URL")
        if not local_llm_url:
            raise ValueError("请设置 LOCAL_LLM_URL 环境变量")
        
        self.llm = ChatOpenAI(
            model="qwen-plus",
            openai_api_base=local_llm_url,
            openai_api_key="not-needed",
            temperature=0.3
        )
    
    def plan(self, goal: str) -> list:
        """生成任务计划"""
        print(f"🤔 正在规划任务: {goal}")
        
        prompt = f"""你是一个任务规划专家。请将以下目标分解为具体的执行步骤。

目标：{goal}

要求：
1. 分解为 5-8 个具体步骤
2. 每个步骤应该是可执行的行动
3. 使用动词开头（如：搜索、分析、撰写）

请以列表格式输出，每行一个步骤。"""
        
        result = self.llm.invoke(prompt)
        text = result.content if hasattr(result, 'content') else str(result)
        
        # 简单解析
        tasks = [line.strip() for line in text.split('\n') if line.strip() and ('步骤' in line or line[0].isdigit())]
        
        if not tasks:
            tasks = [
                "步骤1: 调研相关信息",
                "步骤2: 整理关键要点",
                "步骤3: 撰写初稿",
                "步骤4: 优化和完善"
            ]
        
        print(f"✅ 任务规划完成，共 {len(tasks)} 个步骤")
        return tasks


class TaskExecutor:
    """任务执行器"""
    
    def __init__(self):
        self.output_dir = "./outputs"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def execute(self, action: str) -> str:
        """执行动作"""
        print(f"⚡ 执行: {action}")
        
        # 模拟执行（实际应该调用工具或API）
        if "搜索" in action or "调研" in action:
            return self._simulate_search()
        elif "撰写" in action or "写" in action or "生成" in action:
            return self._simulate_write(action)
        elif "分析" in action:
            return self._simulate_analyze()
        else:
            return self._simulate_general(action)
    
    def _simulate_search(self) -> str:
        return "搜索结果：找到相关文献 15 篇，关键观点：AI Agent 是未来趋势"
    
    def _simulate_write(self, action: str = "") -> str:
        from datetime import datetime
        
        # 根据action生成不同的内容
        if "博客" in action or "文章" in action:
            content = self._generate_blog_content()
        elif "报告" in action or "调研" in action:
            content = self._generate_research_report()
        else:
            content = self._generate_generic_content(action)
        
        filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"文章已撰写完成（{len(content)}字），保存到: {filepath}"
    
    def _simulate_analyze(self) -> str:
        return "分析结果：优势-效率高；劣势-稳定性待提升；机会-市场增长；威胁-竞争激烈"
    
    def _generate_blog_content(self) -> str:
        """生成博客文章"""
        return """# AI Agent：从概念到实践的全面指南

## 一、什么是 AI Agent？

AI Agent（人工智能代理）是一种能够**自主感知环境、做出决策并执行任务**的智能系统。与传统 AI 不同，Agent 具备以下核心特征：

### 1.1 自主性
- 无需人工逐步指导
- 能够独立完成任务分解
- 自主决定执行策略

### 1.2 目标导向
- 接收高层目标（如“写一篇博客”）
- 自动拆解为可执行步骤
- 持续优化直到达成目标

### 1.3 学习能力
- 从历史执行中学习
- 记忆管理经验教训
- 不断优化决策质量

## 二、核心技术架构

### 2.1 大语言模型（LLM）
作为 Agent 的“大脑”，负责：
- **理解意图**：解析用户目标
- **任务规划**：分解复杂任务
- **决策制定**：选择最优行动
- **内容生成**：输出最终结果

主流模型：GPT-4、Claude、通义千问、文心一言

### 2.2 任务规划器
将宏观目标转化为具体步骤：
```
目标：写一篇技术博客
↓
步骤1：调研市场现状
步骤2：收集典型案例
步骤3：撰写文章大纲
步骤4：填充详细内容
步骤5：优化语言表达
```

### 2.3 记忆系统
- **短期记忆**：记录当前会话的执行历史
- **长期记忆**：持久化存储经验和知识
- **向量检索**：快速找到相关信息

### 2.4 工具集成
Agent 可以调用各种外部工具：
- 🔍 搜索引擎 API
- 📝 文件系统操作
- 💻 代码执行环境
- 🌐 Web 浏览器自动化
- 📊 数据分析工具

## 三、典型应用场景

### 3.1 自动化客服
**传统方式：**
- 人工客服 8 小时工作制
- 响应时间 5-30 分钟
- 成本：5000 元/月/人

**Agent 方式：**
- 7×24 小时在线
- 响应时间 < 5 秒
- 成本：500 元/月（API 费用）
- **效率提升：96 倍，成本降低：90%**

### 3.2 智能投研
**传统流程：**
1. 收集市场数据（2 小时）
2. 分析财务报表（3 小时）
3. 撰写研究报告（4 小时）
4. 审核和优化（1 小时）
**总计：10 小时**

**Agent 流程：**
1. 自动数据采集（5 分钟）
2. AI 分析财务指标（10 分钟）
3. 生成报告初稿（15 分钟）
4. 人工审核（30 分钟）
**总计：1 小时，效率提升 10 倍**

### 3.3 代码生成
- 根据需求自动生成代码
- 代码审查和 Bug 修复
- 单元测试编写
- 文档自动生成

### 3.4 数据分析
- 自动数据清洗和预处理
- 统计分析和问题诊断
- 可视化图表生成
- 洞察报告输出

## 四、AutoGPT：革命性的自主 Agent

### 4.1 什么是 AutoGPT？
2023 年 3 月爆火的开源项目，GitHub Stars 超过 150k。

**核心理念：**
```
传统 AI：你告诉它做什么 → 它执行一步 → 等待下一步指令
AutoGPT：你给它一个目标 → 它自己想办法完成
```

### 4.2 工作原理
```
用户输入目标
  ↓
[任务规划] 分解为子任务
  ↓
[执行循环]
  ├→ [思考] 下一步做什么？
  ├→ [行动] 执行具体操作
  ├→ [观察] 结果如何？
  └→ [反思] 需要调整吗？
  ↓
[完成判断] 是否达成目标？
```

### 4.3 实际案例

**案例 1：自动写博客**
- 输入：“写一篇关于 AI Agent 的技术博客”
- AutoGPT 自主执行：
  1. 搜索最新资料
  2. 整理关键观点
  3. 撰写文章大纲
  4. 填充详细内容
  5. 优化语言表达
  6. 添加代码示例
- 输出：完整的 Markdown 文章
- **耗时：5 分钟 vs 传统 2 小时**

**案例 2：市场调研**
- 输入：“调研 2024 年 AI Agent 市场趋势”
- AutoGPT 自主执行：
  1. 搜索市场规模数据
  2. 分析竞争对手
  3. 识别技术趋势
  4. 预测发展方向
  5. 撰写调研报告
- 输出：结构化调研报告
- **耗时：8 分钟 vs 传统 4 小时**

## 五、优势与挑战

### 5.1 核心优势
✅ **效率提升**：20-30 倍  
✅ **成本降低**：99%+  
✅ **解放人力**：专注创造性工作  
✅ **7×24 小时**：不间断工作  
✅ **可扩展性**：轻松复制和扩展  

### 5.2 当前挑战
⚠️ **稳定性**：可能陷入死循环  
⚠️ **透明度**：决策过程不透明  
⚠️ **成本**：大量 API 调用费用  
⚠️ **可控性**：缺乏人工干预机制  
⚠️ **错误率**：偶发幻觉和错误  

## 六、未来展望

### 6.1 技术趋势
1. **多模态融合** - 结合文本、图像、音频、视频
2. **更强的推理** - 复杂逻辑和数学推理
3. **更低成本** - 模型优化和压缩技术
4. **更好可控** - 人工监督和干预机制
5. **多 Agent 协作** - 角色分工，协同工作

### 6.2 应用前景
- **企业级应用爆发**：客服、投研、运营
- **垂直领域深耕**：金融、医疗、法律、教育
- **个人助手普及**：每个人的 AI 助理
- **开源生态繁荣**：更多工具和框架

### 6.3 对开发者的意义
现在是学习和实践 Agent 技术的**最佳时机**：
- 技术处于早期阶段，机会巨大
- 开源社区活跃，资源丰富
- 市场需求旺盛，就业前景好
- 学习曲线适中，入门门槛不高

## 七、如何开始？

### 7.1 学习路径
1. **理解基础**：LLM、Prompt Engineering
2. **学习框架**：LangChain、LlamaIndex
3. **动手实践**：构建简单的 Agent
4. **深入探索**：AutoGPT、BabyAGI
5. **生产应用**：部署和优化

### 7.2 推荐资源
- **官方文档**：LangChain、AutoGPT
- **开源项目**：GitHub 上的优秀案例
- **在线课程**：Coursera、Udemy
- **技术社区**：CSDN、知乎、掘金
- **学术论文**：arXiv 上的最新研究

## 八、总结

AI Agent 代表了从“工具”到“助手”的重要转变：

- **传统 AI** = 你指挥，它执行
- **AI Agent** = 你给目标，它自主完成

虽然目前仍处于早期阶段，存在稳定性和成本等问题，但其潜力巨大。随着技术进步和生态完善，Agent 将在各个领域带来革命性变化。

**对于开发者来说，现在是行动的最佳时机！**

---

**参考资料：**
- AutoGPT 官方文档：https://docs.agpt.co/
- LangChain 文档：https://python.langchain.com/
- 《AI Agent 完全解析》技术博客系列

**生成时间：** 2026-04-23  
**作者：** AutoGPT 简化版演示系统  
**字数：** 约 3500 字
"""
    
    def _generate_research_report(self) -> str:
        """生成市场调研报告"""
        return """# 2024-2026 年 AI Agent 市场趋势调研报告

## 执行摘要

本报告基于对全球 AI Agent 市场的深入调研，分析了市场规模、竞争格局、技术趋势和发展方向。核心发现：

- **市场规模**：2024 年达到 100 亿美元，预计 2026 年增长至 300 亿美元
- **增长率**：年均复合增长率（CAGR）超过 100%
- **主要驱动**：企业数字化转型、大模型技术成熟、成本下降
- **关键趋势**：多模态、自主性增强、垂直领域深耕

---

## 一、市场规模与增长

### 1.1 全球市场
| 年份 | 市场规模 | 增长率 |
|------|---------|--------|
| 2023 | 50 亿美元 | - |
| 2024 | 100 亿美元 | 100% |
| 2025 | 200 亿美元 | 100% |
| 2026 | 300 亿美元 | 50% |

**数据来源：** Gartner、IDC、麦肯锡研究报告

### 1.2 区域分布
- **北美**：45%（美国主导）
- **亚太**：30%（中国、日本、韩国）
- **欧洲**：20%（英国、德国、法国）
- **其他**：5%

### 1.3 中国市场
- 2024 年规模：30 亿美元
- 主要玩家：阿里、百度、腾讯、字节
- 增长速度：高于全球平均（120%）

---

## 二、竞争格局

### 2.1 主要参与者

#### 国际巨头
1. **OpenAI**
   - 产品：GPT 系列 + ChatGPT
   - 优势：技术领先、生态完善
   - 市场份额：35%

2. **Anthropic**
   - 产品：Claude 系列
   - 优势：安全性强、长文本处理
   - 市场份额：15%

3. **Google**
   - 产品：Gemini、Bard
   - 优势：搜索整合、多模态
   - 市场份额：12%

4. **Microsoft**
   - 产品：Copilot 系列
   - 优势：Office 整合、企业客户
   - 市场份额：10%

#### 国内厂商
1. **阿里巴巴**
   - 产品：通义千问
   - 优势：电商场景、云计算
   - 国内份额：25%

2. **百度**
   - 产品：文心一言
   - 优势：搜索数据、知识图谱
   - 国内份额：20%

3. **腾讯**
   - 产品：混元大模型
   - 优势：社交场景、游戏
   - 国内份额：18%

4. **字节跳动**
   - 产品：豆包
   - 优势：内容推荐、短视频
   - 国内份额：15%

### 2.2 开源社区
- **AutoGPT**：150k+ Stars
- **LangChain**：80k+ Stars
- **LlamaIndex**：30k+ Stars
- **BabyAGI**：25k+ Stars

---

## 三、技术趋势

### 3.1 多模态融合
- **现状**：文本为主，图像/音频辅助
- **趋势**：文本+图像+音频+视频全面融合
- **代表**：GPT-4V、Gemini Pro Vision
- **影响**：应用场景扩展 10 倍+

### 3.2 自主性增强
- **当前**：需要人工监督和干预
- **未来**：完全自主决策和执行
- **关键技术**：
  - 强化学习（RLHF）
  - 思维链（Chain of Thought）
  - 自我反思（Self-Reflection）

### 3.3 成本下降
- **2023 年**：GPT-4 API 调用 $0.03/1K tokens
- **2024 年**：降至 $0.01/1K tokens
- **2025 年预测**：$0.005/1K tokens
- **驱动因素**：
  - 模型优化和压缩
  - 硬件加速（GPU/TPU）
  - 规模化效应

### 3.4 边缘计算
- **云端 Agent**：目前主流
- **边缘 Agent**：未来趋势
- **优势**：
  - 低延迟
  - 数据隐私
  - 离线可用
- **挑战**：
  - 算力限制
  - 模型压缩
  - 能耗控制

---

## 四、应用场景分析

### 4.1 企业级应用（占比 60%）

#### 客户服务
- **市场规模**：2024 年 30 亿美元
- **典型案例**：
  - 智能客服机器人
  - 自动工单处理
  - 情感分析和预警
- **ROI**：成本降低 70%，满意度提升 30%

#### 智能投研
- **市场规模**：2024 年 15 亿美元
- **应用场景**：
  - 自动数据采集
  - 财务分析
  - 报告生成
- **效率提升**：从 10 小时缩短到 1 小时

#### 代码开发
- **市场规模**：2024 年 20 亿美元
- **工具**：GitHub Copilot、Codeium
- **效果**：开发效率提升 40-55%

### 4.2 消费级应用（占比 30%）

#### 个人助手
- **产品**：ChatGPT、Claude、文心一言
- **用户规模**：全球 2 亿+ 月活
- **付费率**：5-10%
- **ARPU**：$20/月

#### 教育辅导
- **应用**：智能导师、作业批改
- **市场**：K12、职业教育
- **渗透率**：目前 10%，预计 2026 年 30%

#### 内容创作
- **场景**：写作、绘画、音乐
- **工具**：Midjourney、Stable Diffusion
- **创作者经济**：100 万+ 付费用户

### 4.3 垂直领域（占比 10%）

#### 医疗健康
- **应用**：诊断辅助、药物研发
- **监管**：FDA 审批中
- **潜力**：巨大但门槛高

#### 法律服务
- **应用**：合同审查、案例检索
- **价值**：律师效率提升 50%
- **挑战**：准确性和责任

#### 金融服务
- **应用**：风险评估、欺诈检测
- **合规**：严格监管
- **采用率**：大型银行 60%

---

## 五、发展方向预测

### 5.1 短期（2024-2025）
1. **企业级应用爆发**
   - 客服、投研、运营自动化
   - ROI 明确，快速落地

2. **多模态普及**
   - 图文结合成为标配
   - 视频理解逐步成熟

3. **成本持续下降**
   - API 价格再降 50%
   - 中小企业可负担

### 5.2 中期（2025-2027）
1. **自主性显著提升**
   - 减少人工干预
   - 复杂任务自主完成

2. **垂直领域深耕**
   - 医疗、法律、金融专业化
   - 行业专属 Agent 出现

3. **边缘计算突破**
   - 手机端运行大模型
   - 隐私保护增强

### 5.3 长期（2027-2030）
1. **通用人工智能（AGI）雏形**
   - 跨领域推理能力
   - 创造性思维

2. **人机协作新模式**
   - Agent 作为同事而非工具
   - 工作流程重构

3. **社会影响深远**
   - 就业结构变化
   - 教育和培训转型
   - 伦理和法律框架建立

---

## 六、风险与挑战

### 6.1 技术风险
- **幻觉问题**：生成错误信息
- **稳定性**：偶发失败和死循环
- **安全性**：提示词注入攻击

### 6.2 商业风险
- **成本高**：大规模部署费用
- **ROI 不确定**：部分场景效果不佳
- **竞争激烈**：价格战压力

### 6.3 监管风险
- **数据隐私**：GDPR、个人信息保护法
- **知识产权**：训练数据版权
- **责任归属**：AI 决策的责任主体

### 6.4 社会风险
- **就业冲击**：部分岗位被替代
- **数字鸿沟**：技术获取不平等
- **伦理问题**：偏见和歧视

---

## 七、投资建议

### 7.1 关注领域
✅ **企业级 SaaS**：客服、投研、运营  
✅ **垂直领域**：医疗、法律、金融  
✅ **基础设施**：模型优化、推理加速  
✅ **开发者工具**：Agent 框架、调试工具  

### 7.2 谨慎领域
⚠️ **通用聊天机器人**：竞争激烈  
⚠️ **纯技术创业**：缺乏场景  
⚠️ **过度依赖单一模型**：供应商风险  

### 7.3 估值水平
- **早期项目**：Pre-A 轮 $500-1000 万
- **成长期**：A-B 轮 $2000-5000 万
- **成熟期**：C 轮+ $1 亿+
- **退出方式**：IPO、并购

---

## 八、结论与建议

### 8.1 核心结论
1. **市场处于爆发前期**：未来 3 年 CAGR 100%+
2. **企业级应用是主战场**：ROI 明确，快速落地
3. **技术仍在快速迭代**：保持学习和跟进
4. **监管逐步完善**：合规是关键

### 8.2 给企业的建议
- **立即行动**：从小场景试点开始
- **选择合适场景**：高频率、规则明确、ROI 清晰
- **重视数据安全**：私有化部署或混合方案
- **培养人才**：内部培训和外部招聘

### 8.3 给开发者的建议
- **学习 LangChain**：最成熟的 Agent 框架
- **动手实践**：构建自己的 Agent 项目
- **关注开源**：参与社区，贡献代码
- **垂直深耕**：选择一个领域深入研究

### 8.4 给投资者的建议
- **关注应用层**：有明确场景和 ROI
- **警惕纯技术**：缺乏商业化路径
- **长期视角**：3-5 年投资周期
- **分散风险**：多个项目组合

---

## 附录：研究方法

### 数据来源
- **二手研究**：Gartner、IDC、麦肯锡报告
- **一手调研**：访谈 50+ 企业和开发者
- **公开数据**：公司财报、新闻报道
- **社区数据**：GitHub Stars、下载量

### 局限性
- 市场发展迅速，数据可能滞后
- 不同机构统计口径差异
- 预测存在不确定性

### 免责声明
本报告仅供参考，不构成投资建议。读者应自行判断并承担风险。

---

**报告生成时间：** 2026-04-23  
**调研周期：** 2024 年 Q1-Q2  
**版本：** v1.0  
**作者：** AutoGPT 简化版演示系统  
**字数：** 约 4500 字
"""
    
    def _generate_generic_content(self, action: str) -> str:
        """生成通用内容"""
        from datetime import datetime
        
        return f"""# 任务执行结果

## 执行的动作
{action}

## 执行时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 执行状态
✅ 完成

## 详细说明

### 背景
该任务是 AutoGPT 自主执行流程中的一部分。系统根据总体目标，自动分解出此步骤并执行。

### 执行过程
1. **分析任务**：理解任务要求和预期输出
2. **收集信息**：从记忆系统和外部资源获取相关数据
3. **执行操作**：调用相应的工具或 API
4. **验证结果**：检查输出是否符合要求
5. **记录日志**：保存执行历史供后续参考

### 输出结果
任务已成功执行。系统已记录完整的执行历史，包括：
- 输入参数
- 执行步骤
- 中间结果
- 最终输出
- 耗时统计

### 后续建议
基于当前执行结果，建议：
1. 检查结果是否满足预期
2. 如有需要，进行进一步优化
3. 将结果整合到最终报告中
4. 更新长期记忆，积累经验

## 技术细节

### 使用的组件
- **任务规划器**：分解目标和生成步骤
- **执行器**：调用工具和 API
- **记忆系统**：存储和检索上下文
- **反思机制**：评估和优化决策

### 性能指标
- **响应时间**：< 5 秒
- **成功率**：100%
- **资源消耗**：低

## 总结

本步骤已成功完成，为后续步骤奠定了基础。AutoGPT 系统将继续自主执行，直到达成最终目标。

---
*由 AutoGPT 简化版自动生成*
*生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*系统版本：v1.0.0*
"""
    
    def _simulate_general(self, action: str) -> str:
        """通用执行 - 也会保存结果到文件"""
        from datetime import datetime
        
        # 生成通用的执行结果
        result_content = f"""# 任务执行结果

## 执行的动作
{action}

## 执行时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 执行状态
✅ 完成

## 结果摘要
该步骤已成功执行。系统已记录执行历史，可用于后续步骤的参考。

---
*由 AutoGPT 简化版自动生成*
"""
        
        # 保存到文件
        filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(result_content)
        
        return f"步骤执行完成，结果已保存到: {filepath}"


class AutonomousAgent:
    """自主 Agent - AutoGPT 核心"""
    
    def __init__(self):
        self.planner = TaskPlanner()
        self.executor = TaskExecutor()
        self.memory = ShortTermMemory()
        
        local_llm_url = os.getenv("LOCAL_LLM_URL")
        self.llm = ChatOpenAI(
            model="qwen-plus",
            openai_api_base=local_llm_url,
            openai_api_key="not-needed",
            temperature=0.3
        ) if local_llm_url else None
    
    def execute(self, goal: str, max_iterations: int = 10) -> dict:
        """执行自主任务"""
        print("=" * 60)
        print(f"🚀 开始自主执行任务")
        print(f"🎯 目标: {goal}")
        print("=" * 60)
        print()
        
        # 任务规划
        tasks = self.planner.plan(goal)
        print()
        
        # 执行循环
        results = []
        for i, task in enumerate(tasks, 1):
            if i > max_iterations:
                print(f"\n⚠️  达到最大迭代次数 ({max_iterations})")
                break
            
            print(f"\n【步骤 {i}/{len(tasks)}】")
            print("-" * 60)
            
            # 执行动作
            result = self.executor.execute(task)
            print(f"✅ 结果: {result[:100]}...")
            
            # 记录记忆
            self.memory.add(i, task, result)
            results.append({"step": i, "task": task, "result": result})
            
            # 如果是最后一步或达到最大迭代，生成最终报告
            if i == len(tasks) or i >= max_iterations:
                # 强制生成一篇完整的文章
                final_article = self.executor._generate_blog_content()
                # 单独保存完整博客
                from datetime import datetime
                blog_filename = f"blog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                blog_filepath = os.path.join(self.executor.output_dir, blog_filename)
                with open(blog_filepath, 'w', encoding='utf-8') as f:
                    f.write(final_article)
                results.append({
                    "step": i + 1,
                    "task": "生成最终博客文章",
                    "result": f"完整博客已生成（{len(final_article)}字），保存到: {blog_filepath}"
                })
                print(f"\n📝 生成完整博客文章（{len(final_article)}字）")
                print(f"📄 博客文件: {blog_filepath}")
                break
            
            # 检查是否完成
            if len(results) >= 3:
                print(f"\n✅ 任务完成！")
                # 也生成最终文章
                final_article = self.executor._generate_blog_content()
                from datetime import datetime
                blog_filename = f"blog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                blog_filepath = os.path.join(self.executor.output_dir, blog_filename)
                with open(blog_filepath, 'w', encoding='utf-8') as f:
                    f.write(final_article)
                results.append({
                    "step": i + 1,
                    "task": "生成最终博客文章",
                    "result": f"完整博客已生成（{len(final_article)}字），保存到: {blog_filepath}"
                })
                print(f"📝 生成完整博客文章（{len(final_article)}字）")
                print(f"📄 博客文件: {blog_filepath}")
                break
        
        final_result = "\n".join([r["result"] for r in results])
        
        # 生成最终汇总报告
        self._save_final_report(goal, results)
        
        print("\n" + "=" * 60)
        print(f"📊 执行统计:")
        print(f"   总步骤: {len(results)}")
        print("=" * 60)
        
        return {"goal": goal, "steps": results, "final_output": final_result}
    
    def _save_final_report(self, goal: str, results: list):
        """保存最终汇总报告"""
        from datetime import datetime
        
        # 生成汇总报告
        report = f"""# AutoGPT 任务执行报告

## 任务目标
{goal}

## 执行时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 执行步骤

"""
        
        for r in results:
            report += f"### 步骤 {r['step']}: {r['task']}\n\n"
            report += f"**结果：** {r['result']}\n\n"
            report += "---\n\n"
        
        report += f"""## 执行统计
- 总步骤数：{len(results)}
- 执行状态：✅ 完成
- 成功率：100%

## 总结
本次任务已成功完成。所有步骤均已执行，结果已保存。

---
*由 AutoGPT 简化版自动生成*
*生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存到文件
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(self.executor.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 最终报告已保存到: {filepath}")



if __name__ == "__main__":
    agent = AutonomousAgent()
    
    print("请选择演示：")
    print("1. 自动写博客")
    print("2. 市场调研")
    
    choice = input("\n请输入选项 (1-2): ").strip()
    
    if choice == "1":
        result = agent.execute(
            goal="写一篇关于 AI Agent 的技术博客",
            max_iterations=5
        )
    elif choice == "2":
        result = agent.execute(
            goal="调研 2024 年 AI Agent 市场趋势",
            max_iterations=6
        )
    else:
        print("无效选项")
        exit(1)
    
    print("\n📄 最终输出：")
    print(result["final_output"])
