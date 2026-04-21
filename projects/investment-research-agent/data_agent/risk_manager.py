"""
风险管理 Agent - 负责风险评估和合规检查
"""

import os
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain


class RiskManagerAgent:
    """风险管理 Agent - 风险评估和合规检查"""
    
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
            temperature=0.2  # 更低温度，更保守
        )
        
        # 风险评估提示词
        self.risk_prompt = PromptTemplate(
            input_variables=["company_name", "stock_code", "analysis_report"],
            template="""你是一个专业的风险控制专家。请对以下投资分析报告进行严格的风险评估和合规检查。

【公司信息】
- 公司名称：{company_name}
- 股票代码：{stock_code}

【分析报告】
{analysis_report}

请从以下几个维度进行风险评估：

## 1. 数据准确性风险
- 检查报告中是否有明显的数据错误
- 标注可能存在的不准确信息
- 评估数据来源的可靠性

## 2. 逻辑一致性风险
- 检查分析逻辑是否自洽
- 是否存在前后矛盾的结论
- 推理过程是否合理

## 3. 合规性风险
- 是否符合证券法律法规
- 是否有误导性表述
- 免责声明是否充分

## 4. 市场风险
- 系统性风险（宏观经济、政策）
- 行业风险（竞争格局、技术变革）
- 公司特有风险（管理层、财务）

## 5. 操作风险
- 建议的可行性
- 流动性风险
-  timing 风险

请输出风险评估报告，格式如下：

{{
  "overall_risk_level": "低/中/高",
  "risk_score": 0-100,  // 风险分数，越低越好
  "data_accuracy_risk": {{
    "level": "低/中/高",
    "issues": ["问题1", "问题2"]
  }},
  "logic_consistency_risk": {{
    "level": "低/中/高",
    "issues": ["问题1", "问题2"]
  }},
  "compliance_risk": {{
    "level": "低/中/高",
    "issues": ["问题1", "问题2"],
    "recommendations": ["建议1", "建议2"]
  }},
  "market_risks": [
    {{
      "risk_type": "风险类型",
      "level": "低/中/高",
      "description": "详细描述"
    }}
  ],
  "operational_risks": [
    {{
      "risk_type": "风险类型",
      "level": "低/中/高",
      "description": "详细描述"
    }}
  ],
  "final_recommendation": "最终建议（通过/修改后通过/不通过）",
  "required_modifications": ["需要修改的内容1", "需要修改的内容2"]
}}

注意：
1. 保持客观、谨慎的态度
2. 宁可过度警示，不可低估风险
3. 所有判断要有依据"""
        )
        
        self.risk_chain = LLMChain(llm=self.llm, prompt=self.risk_prompt)
    
    def assess_risk(self, company_name: str, stock_code: str, 
                   analysis_report: str) -> Dict:
        """
        执行风险评估
        
        Args:
            company_name: 公司名称
            stock_code: 股票代码
            analysis_report: 分析报告
            
        Returns:
            风险评估结果
        """
        print(f"⚠️ 开始风险评估: {company_name} ({stock_code})...")
        
        try:
            result = self.risk_chain.invoke({
                "company_name": company_name,
                "stock_code": stock_code,
                "analysis_report": analysis_report
            })
            
            risk_text = result["text"] if isinstance(result, dict) else str(result)
            
            # 尝试解析 JSON
            import re
            import json
            json_match = re.search(r'\{[\s\S]*\}', risk_text)
            if json_match:
                risk_data = json.loads(json_match.group())
            else:
                risk_data = {"raw_text": risk_text}
            
            print(f"✅ 风险评估完成")
            print(f"   整体风险等级: {risk_data.get('overall_risk_level', '未知')}")
            print(f"   风险分数: {risk_data.get('risk_score', 'N/A')}/100")
            print(f"   最终建议: {risk_data.get('final_recommendation', '未知')}")
            
            return {
                "success": True,
                "risk_assessment": risk_data
            }
            
        except Exception as e:
            print(f"❌ 风险评估失败：{e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def add_disclaimer(self, report: str) -> str:
        """
        为报告添加标准免责声明
        
        Args:
            report: 原始报告
            
        Returns:
            添加免责声明后的报告
        """
        disclaimer = """

---

## ⚖️ 法律声明与风险提示

**重要声明：**

1. **非投资建议**：本报告由 AI 系统自动生成，仅供学习和研究参考，**不构成任何投资建议**。投资者应基于自身独立判断做出投资决策。

2. **数据准确性**：报告中的数据来源于公开资料整理，可能存在滞后、不完整或不准确的情况。使用者应自行核实关键数据。

3. **风险提示**：股市有风险，投资需谨慎。过往业绩不代表未来表现，任何投资都存在本金损失的风险。

4. **责任免除**：本报告作者及发布平台不对因使用本报告而产生的任何直接或间接损失承担责任。

5. **合规提醒**：根据《证券法》及相关法规，未经批准不得从事证券投资咨询业务。本报告仅为技术分析展示，不涉及具体投资建议。

6. **更新说明**：市场情况瞬息万变，本报告内容可能很快过时。请在做出投资决策前获取最新信息。

**建议：** 在做出任何投资决策前，请咨询持牌证券顾问或金融专业人士。

---

*报告生成时间：AI 投研助手 v1.0*
*最后更新：请参考报告日期*
"""
        
        return report + disclaimer


# 测试
if __name__ == "__main__":
    agent = RiskManagerAgent()
    
    # 模拟分析报告
    analysis_report = """
    ## 投资亮点
    1. 品牌优势明显
    2. 财务表现优秀
    
    ## 风险提示
    1. 政策风险（中）
    
    ## 投资建议
    评级：买入
    目标价：2000元
    """
    
    result = agent.assess_risk(
        company_name="贵州茅台",
        stock_code="600519",
        analysis_report=analysis_report
    )
    
    if result["success"]:
        print("\n风险评估详情：")
        import json
        print(json.dumps(result["risk_assessment"], indent=2, ensure_ascii=False))
    
    # 测试添加免责声明
    sample_report = "这是一份测试报告..."
    report_with_disclaimer = agent.add_disclaimer(sample_report)
    print("\n添加免责声明后的报告长度:", len(report_with_disclaimer))
