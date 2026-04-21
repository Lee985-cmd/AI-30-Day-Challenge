"""
分析师 Agent - 负责深度分析和投资建议
"""

import os
from typing import Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain


class AnalystAgent:
    """分析师 Agent - 深度分析和投资建议"""
    
    def __init__(self, api_key: Optional[str] = None):
        # 使用本地模型
        local_llm_url = os.getenv("LOCAL_LLM_URL")
        if not local_llm_url:
            raise ValueError("请设置 LOCAL_LLM_URL 环境变量")
        
        # 初始化本地 OpenAI 兼容模型
        self.llm = ChatOpenAI(
            model="qwen-plus",
            openai_api_base=local_llm_url,
            openai_api_key="not-needed",
            temperature=0.3
        )
        
        # 分析提示词
        self.analysis_prompt = PromptTemplate(
            input_variables=["research_data", "financial_data_summary"],
            template="""你是一个资深证券分析师。请基于以下研究数据和财务数据，进行深度分析并给出投资建议。

【研究数据】
{research_data}

【财务数据摘要】
{financial_data_summary}

**重要要求：**
1. **必须使用财务数据**：在分析中引用具体的财务数字（营收、利润、ROE等）
2. **计算增长率**：根据财务数据计算复合增长率（CAGR）
3. **对比分析**：对比不同年份的变化趋势
4. **避免“数据不足”**：如果研究数据中有信息，就使用它；如果没有，基于财务数据进行分析

请从以下几个维度进行分析：

## 1. 投资亮点
- 列出 3-5 个核心投资逻辑
- **每个亮点必须有具体数据支撑**（例如：“营收三年CAGR达XX%”）

## 2. 风险提示
- 列出 3-5 个主要风险点
- 评估风险等级（高/中/低）
- 说明风险对财务的影响

## 3. 估值判断
- 当前估值水平（低估/合理/高估）
- 目标价区间（基于合理估值）
- 安全边际分析

## 4. 投资建议
- 评级：买入/增持/中性/减持/卖出
- 建议仓位：轻仓/中等仓位/重仓
- 操作策略：分批建仓/一次性买入/观望

## 5. 关键跟踪指标
- 列出 3-5 个需要持续跟踪的指标
- 说明为什么这些指标重要

**输出要求：**
- 语言专业、客观，避免过度乐观或悲观
- **禁止输出“数据不足”**，如果某些信息缺失，基于已有数据进行合理推断
- 篇幅控制在 800-1200 字
- 注意：这仅供参考，不构成投资建议。"""
        )
        
        self.analysis_chain = LLMChain(llm=self.llm, prompt=self.analysis_prompt)
    
    def analyze(self, research_data: Dict, financial_data_summary: str) -> Dict:
        """
        执行分析任务
        
        Args:
            research_data: 研究员提供的研究数据
            financial_data_summary: 财务数据摘要
            
        Returns:
            分析结果字典
        """
        print("📈 开始深度分析...")
        
        try:
            # 将研究数据转换为字符串
            import json
            research_text = json.dumps(research_data, indent=2, ensure_ascii=False)
            
            result = self.analysis_chain.invoke({
                "research_data": research_text,
                "financial_data_summary": financial_data_summary
            })
            
            analysis_text = result["text"] if isinstance(result, dict) else str(result)
            
            print("✅ 分析完成")
            
            return {
                "success": True,
                "analysis_report": analysis_text
            }
            
        except Exception as e:
            print(f"❌ 分析失败：{e}")
            return {
                "success": False,
                "error": str(e)
            }


# 测试
if __name__ == "__main__":
    agent = AnalystAgent()
    
    # 模拟研究数据
    research_data = {
        "basic_info": {
            "main_business": "高端白酒生产与销售",
            "competitive_advantage": "品牌护城河深厚，稀缺性强",
            "market_position": "行业龙头，市场份额第一"
        },
        "financial_analysis": {
            "revenue_growth": "近三年复合增长率 15%",
            "profit_margin": "净利率稳定在 50%+",
            "roe": "ROE 保持在 30%+",
            "debt_ratio": "负债率极低，现金流充沛"
        },
        "industry_analysis": {
            "industry_stage": "成熟期，消费升级趋势明显",
            "competition": "高端市场双寡头格局",
            "policy_risk": "政策风险中等，需关注消费税改革"
        },
        "valuation": {
            "current_pe": "当前 PE 约 35 倍",
            "historical_comparison": "处于历史中位数水平",
            "peer_comparison": "相对五粮液有溢价，但合理"
        }
    }
    
    financial_summary = """
    2021-2023年财务数据：
    - 营收增长：100亿 → 120亿 → 145亿（CAGR 20%）
    - 净利润增长：15亿 → 18亿 → 22亿（CAGR 21%）
    - 毛利率：35% → 36% → 37%（持续提升）
    - ROE：20% → 21% → 22%（优秀水平）
    - 负债率：40% → 38% → 35%（持续优化）
    """
    
    result = agent.analyze(research_data, financial_summary)
    
    if result["success"]:
        print("\n分析报告：")
        print(result["analysis_report"])
