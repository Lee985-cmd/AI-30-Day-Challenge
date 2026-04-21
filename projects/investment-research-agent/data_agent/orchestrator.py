"""
投研编排器 - 协调多个 Agent 协作完成投资研究
"""

import os
from typing import Dict, Optional
from datetime import datetime
from .researcher import ResearcherAgent
from .analyst import AnalystAgent
from .writer import WriterAgent
from .risk_manager import RiskManagerAgent


class InvestmentResearchOrchestrator:
    """投研编排器 - 协调多智能体协作"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
        
        # 初始化所有 Agent
        print("🤖 初始化智能投研助手...")
        self.researcher = ResearcherAgent(api_key=self.api_key)
        self.analyst = AnalystAgent(api_key=self.api_key)
        self.writer = WriterAgent(api_key=self.api_key)
        self.risk_manager = RiskManagerAgent(api_key=self.api_key)
        print("✅ 所有 Agent 初始化完成\n")
    
    def conduct_research(self, stock_code: str, company_name: str,
                        research_focus: str = "全面分析",
                        save_report: bool = True,
                        output_dir: str = "./reports") -> Dict:
        """
        执行完整的投资研究流程
        
        Args:
            stock_code: 股票代码
            company_name: 公司名称
            research_focus: 研究重点
            save_report: 是否保存报告
            output_dir: 报告输出目录
            
        Returns:
            完整的研究结果
        """
        print("=" * 60)
        print(f"🚀 开始对 {company_name} ({stock_code}) 进行智能投研")
        print("=" * 60)
        print()
        
        start_time = datetime.now()
        
        # 第 1 步：研究员收集数据
        print("【第 1/4 步】研究员收集数据")
        print("-" * 60)
        research_result = self.researcher.research(
            stock_code=stock_code,
            company_name=company_name,
            research_focus=research_focus
        )
        
        if not research_result["success"]:
            return {
                "success": False,
                "error": f"研究阶段失败: {research_result.get('error')}",
                "stage": "research"
            }
        
        # 获取财务数据
        financial_df = self.researcher.get_financial_data(stock_code)
        financial_summary = ""
        if financial_df is not None:
            financial_summary = financial_df.to_string(index=False)
        
        print()
        
        # 第 2 步：分析师深度分析
        print("【第 2/4 步】分析师深度分析")
        print("-" * 60)
        analysis_result = self.analyst.analyze(
            research_data=research_result["research_data"],
            financial_data_summary=financial_summary
        )
        
        if not analysis_result["success"]:
            return {
                "success": False,
                "error": f"分析阶段失败: {analysis_result.get('error')}",
                "stage": "analysis"
            }
        
        print()
        
        # 第 3 步：写作Agent生成报告
        print("【第 3/4 步】生成投资研究报告")
        print("-" * 60)
        report_result = self.writer.write_report(
            company_name=company_name,
            stock_code=stock_code,
            analysis_report=analysis_result["analysis_report"],
            research_data=research_result["research_data"]
        )
        
        if not report_result["success"]:
            return {
                "success": False,
                "error": f"报告撰写失败: {report_result.get('error')}",
                "stage": "writing"
            }
        
        print()
        
        # 第 4 步：风险管理评估
        print("【第 4/4 步】风险评估与合规检查")
        print("-" * 60)
        risk_result = self.risk_manager.assess_risk(
            company_name=company_name,
            stock_code=stock_code,
            analysis_report=report_result["report"]
        )
        
        if not risk_result["success"]:
            return {
                "success": False,
                "error": f"风险评估失败: {risk_result.get('error')}",
                "stage": "risk_assessment"
            }
        
        # 添加免责声明
        final_report = self.risk_manager.add_disclaimer(report_result["report"])
        
        print()
        
        # 计算耗时
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 保存报告
        report_path = ""
        if save_report:
            report_result["report"] = final_report
            report_path = self.writer.save_report(report_result, output_dir)
        
        print()
        print("=" * 60)
        print(f"✅ 智能投研完成！")
        print(f"   总耗时: {duration:.2f} 秒")
        if report_path:
            print(f"   报告路径: {report_path}")
        print("=" * 60)
        
        # 返回完整结果
        return {
            "success": True,
            "stock_code": stock_code,
            "company_name": company_name,
            "research_data": research_result["research_data"],
            "analysis_report": analysis_result["analysis_report"],
            "final_report": final_report,
            "risk_assessment": risk_result["risk_assessment"],
            "report_path": report_path,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat()
        }
    
    def quick_analysis(self, stock_code: str, company_name: str) -> str:
        """
        快速分析（简化版，仅返回核心观点）
        
        Args:
            stock_code: 股票代码
            company_name: 公司名称
            
        Returns:
            快速分析结果
        """
        print(f"⚡ 快速分析 {company_name} ({stock_code})...\n")
        
        # 简化流程：只研究和简单分析
        research_result = self.researcher.research(
            stock_code=stock_code,
            company_name=company_name,
            research_focus="核心投资逻辑"
        )
        
        if not research_result["success"]:
            return f"分析失败: {research_result.get('error')}"
        
        # 提取核心观点
        import json
        research_data = research_result["research_data"]
        
        quick_summary = f"""
## {company_name} ({stock_code}) 快速分析

### 核心观点
- **主营业务：** {research_data.get('basic_info', {}).get('main_business', '数据不足')}
- **竞争优势：** {research_data.get('basic_info', {}).get('competitive_advantage', '数据不足')}
- **行业地位：** {research_data.get('basic_info', {}).get('market_position', '数据不足')}

### 财务亮点
- **营收增长：** {research_data.get('financial_analysis', {}).get('revenue_growth', '数据不足')}
- **盈利能力：** {research_data.get('financial_analysis', {}).get('profit_margin', '数据不足')}
- **ROE水平：** {research_data.get('financial_analysis', {}).get('roe', '数据不足')}

### 估值情况
- **当前PE：** {research_data.get('valuation', {}).get('current_pe', '数据不足')}
- **历史对比：** {research_data.get('valuation', {}).get('historical_comparison', '数据不足')}

⚠️ 注意：这是快速分析结果，仅供参考。如需详细报告，请使用 conduct_research() 方法。
"""
        
        return quick_summary


# 测试
if __name__ == "__main__":
    # 初始化编排器
    orchestrator = InvestmentResearchOrchestrator()
    
    # 测试完整研究流程
    result = orchestrator.conduct_research(
        stock_code="600519",
        company_name="贵州茅台",
        research_focus="全面分析公司基本面和投资价值",
        save_report=True
    )
    
    if result["success"]:
        print("\n" + "=" * 60)
        print("研究成功！")
        print(f"报告已保存到: {result['report_path']}")
        print("=" * 60)
    else:
        print(f"\n研究失败: {result['error']}")
