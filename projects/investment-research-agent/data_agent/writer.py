"""
写作 Agent - 负责生成专业的投资研究报告
"""

import os
from typing import Dict, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain


class WriterAgent:
    """写作 Agent - 生成专业投资研究报告"""
    
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
            temperature=0.5
        )
        
        # 报告写作提示词
        self.report_prompt = PromptTemplate(
            input_variables=["company_name", "stock_code", "analysis_report", "research_data", "current_date"],
            template="""你是一个专业的证券分析师，擅长撰写高质量的投资研究报告。

请基于以下信息，撰写一份完整的投资研究报告：

【公司信息】
- 公司名称：{company_name}
- 股票代码：{stock_code}
- 报告日期：{current_date}

【分析结论】
{analysis_report}

【研究数据】
{research_data}

**重要要求：**
1. **禁止输出“数据不足”**：如果某些信息缺失，基于已有信息进行合理推断和描述
2. **使用具体数据**：在报告中引用具体的财务数字、增长率等
3. **专业表达**：使用证券行业的专业术语和表达方式
4. **逻辑清晰**：每个章节要有明确的观点和支撑论据

请按照以下结构撰写报告：

---

# {company_name} ({stock_code}) 投资研究报告

**报告日期：** {current_date}  
**分析师：** AI 投研助手  

---

## 📋 核心观点

（用 200-300 字概括核心投资逻辑、评级和目标价）

---

## 🏢 公司概况

### 1.1 主营业务
（介绍公司的主营业务和商业模式）

### 1.2 核心竞争力
（分析公司的护城河和竞争优势）

### 1.3 行业地位
（说明公司在行业中的地位和市场份额）

---

## 📊 财务分析

### 2.1 成长性分析
（营收和利润增长情况）

### 2.2 盈利能力
（毛利率、净利率、ROE 等指标）

### 2.3 财务健康度
（负债率、现金流状况）

---

## 🎯 投资亮点

（列出 3-5 个核心投资逻辑，每个亮点要有数据支撑）

1. **亮点一：** ...
2. **亮点二：** ...
3. **亮点三：** ...

---

## ⚠️ 风险提示

（列出 3-5 个主要风险点，评估风险等级）

1. **风险一（高/中/低）：** ...
2. **风险二（高/中/低）：** ...
3. **风险三（高/中/低）：** ...

---

## 💰 估值与目标价

### 4.1 当前估值水平
（PE、PB 等估值指标，与历史和同行对比）

### 4.2 目标价区间
（基于合理估值给出的目标价）

### 4.3 安全边际
（当前价格相对于目标价的安全边际）

---

## 📈 投资建议

**评级：** 【买入/增持/中性/减持/卖出】  
**建议仓位：** 【轻仓/中等仓位/重仓】  
**操作策略：** 【分批建仓/一次性买入/观望】

（详细说明投资建议的理由和操作建议）

---

## 🔍 关键跟踪指标

（列出 3-5 个需要持续跟踪的指标）

1. **指标一：** ...（为什么重要）
2. **指标二：** ...（为什么重要）
3. **指标三：** ...（为什么重要）

---

## 📝 免责声明

**重要声明：** 本报告由 AI 自动生成，仅供参考，不构成任何投资建议。投资者应独立判断，自行承担投资风险。市场有风险，投资需谨慎。

数据来源：公开资料整理，可能存在滞后或不准确的情况。

---

**报告结束**

要求：
1. 语言专业、客观，避免过度乐观或悲观
2. 数据要准确，如有不确定的地方标注"数据不足"
3. 逻辑清晰，层次分明
4. 篇幅控制在 2000-3000 字
5. 使用 Markdown 格式，便于阅读"""
        )
        
        self.report_chain = LLMChain(llm=self.llm, prompt=self.report_prompt)
    
    def write_report(self, company_name: str, stock_code: str, 
                    analysis_report: str, research_data: Dict) -> Dict:
        """
        撰写投资研究报告
        
        Args:
            company_name: 公司名称
            stock_code: 股票代码
            analysis_report: 分析师的分析报告
            research_data: 研究员的研究数据
            
        Returns:
            完整的研究报告
        """
        print(f"📝 开始撰写 {company_name} 的研究报告...")
        
        try:
            current_date = datetime.now().strftime("%Y年%m月%d日")
            
            import json
            research_text = json.dumps(research_data, indent=2, ensure_ascii=False)
            
            result = self.report_chain.invoke({
                "company_name": company_name,
                "stock_code": stock_code,
                "analysis_report": analysis_report,
                "research_data": research_text,
                "current_date": current_date
            })
            
            report_text = result["text"] if isinstance(result, dict) else str(result)
            
            print("✅ 报告撰写完成")
            
            return {
                "success": True,
                "report": report_text,
                "company_name": company_name,
                "stock_code": stock_code,
                "report_date": current_date
            }
            
        except Exception as e:
            print(f"❌ 报告撰写失败：{e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def save_report(self, report_result: Dict, output_dir: str = "./reports") -> str:
        """
        保存研究报告到文件
        
        Args:
            report_result: 报告结果字典
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        if not report_result["success"]:
            return ""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        company_name = report_result["company_name"]
        stock_code = report_result["stock_code"]
        filename = f"{stock_code}_{company_name}_研究报告.md"
        filepath = os.path.join(output_dir, filename)
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_result["report"])
        
        print(f"💾 报告已保存到: {filepath}")
        
        return filepath


# 测试
if __name__ == "__main__":
    agent = WriterAgent()
    
    # 模拟数据
    research_data = {
        "basic_info": {
            "main_business": "高端白酒生产与销售",
            "competitive_advantage": "品牌护城河深厚",
            "market_position": "行业龙头"
        }
    }
    
    analysis_report = """
    ## 投资亮点
    1. 品牌优势明显，稀缺性强
    2. 财务表现优秀，持续增长
    3. 行业地位稳固
    
    ## 风险提示
    1. 政策风险（中）
    2. 竞争风险（低）
    
    ## 投资建议
    评级：买入
    目标价：2000元
    """
    
    result = agent.write_report(
        company_name="贵州茅台",
        stock_code="600519",
        analysis_report=analysis_report,
        research_data=research_data
    )
    
    if result["success"]:
        print("\n报告预览（前500字）：")
        print(result["report"][:500])
        
        # 保存报告
        filepath = agent.save_report(result)
        print(f"\n完整报告路径: {filepath}")
