"""
智能投研助手测试脚本
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from data_agent import InvestmentResearchOrchestrator


def test_quick_analysis():
    """测试快速分析功能"""
    print("=" * 60)
    print("测试 1: 快速分析")
    print("=" * 60)
    
    orchestrator = InvestmentResearchOrchestrator()
    
    result = orchestrator.quick_analysis(
        stock_code="600519",
        company_name="贵州茅台"
    )
    
    print(result)
    print("\n" + "=" * 60)


def test_full_research():
    """测试完整研究流程"""
    print("=" * 60)
    print("测试 2: 完整研究流程")
    print("=" * 60)
    
    orchestrator = InvestmentResearchOrchestrator()
    
    result = orchestrator.conduct_research(
        stock_code="600519",
        company_name="贵州茅台",
        research_focus="全面分析公司基本面和投资价值",
        save_report=True,
        output_dir="./reports"
    )
    
    if result["success"]:
        print("\n✅ 研究成功！")
        print(f"📄 报告路径: {result['report_path']}")
        print(f"⏱️  耗时: {result['duration_seconds']:.2f} 秒")
        
        # 显示风险评估结果
        risk = result.get('risk_assessment', {})
        print(f"\n⚠️  风险评估:")
        print(f"   整体风险等级: {risk.get('overall_risk_level', 'N/A')}")
        print(f"   风险分数: {risk.get('risk_score', 'N/A')}/100")
        print(f"   最终建议: {risk.get('final_recommendation', 'N/A')}")
    else:
        print(f"\n❌ 研究失败: {result['error']}")
    
    print("\n" + "=" * 60)


def test_multiple_stocks():
    """测试批量研究多只股票"""
    print("=" * 60)
    print("测试 3: 批量研究多只股票")
    print("=" * 60)
    
    orchestrator = InvestmentResearchOrchestrator()
    
    stocks = [
        ("600519", "贵州茅台", "白酒行业龙头"),
        ("300750", "宁德时代", "动力电池龙头"),
        ("000858", "五粮液", "高端白酒第二品牌"),
    ]
    
    for stock_code, company_name, focus in stocks:
        print(f"\n{'=' * 60}")
        print(f"正在研究: {company_name} ({stock_code})")
        print(f"{'=' * 60}")
        
        try:
            result = orchestrator.conduct_research(
                stock_code=stock_code,
                company_name=company_name,
                research_focus=focus,
                save_report=True
            )
            
            if result["success"]:
                print(f"✅ {company_name} 研究完成")
                print(f"   报告: {result['report_path']}")
                print(f"   耗时: {result['duration_seconds']:.2f} 秒")
            else:
                print(f"❌ {company_name} 研究失败: {result['error']}")
        
        except Exception as e:
            print(f"❌ {company_name} 出现异常: {e}")
    
    print("\n" + "=" * 60)
    print("批量研究完成！")
    print("=" * 60)


if __name__ == "__main__":
    print("\n🧪 智能投研助手测试套件\n")
    
    # 选择测试模式
    print("请选择测试模式：")
    print("1. 快速分析测试")
    print("2. 完整研究流程测试")
    print("3. 批量研究测试")
    print("4. 运行所有测试")
    
    choice = input("\n请输入选项 (1-4): ").strip()
    
    if choice == "1":
        test_quick_analysis()
    elif choice == "2":
        test_full_research()
    elif choice == "3":
        test_multiple_stocks()
    elif choice == "4":
        test_quick_analysis()
        test_full_research()
        test_multiple_stocks()
    else:
        print("无效选项，运行默认测试（快速分析）...")
        test_quick_analysis()
    
    print("\n✨ 测试完成！")
