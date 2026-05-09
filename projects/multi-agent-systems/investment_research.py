"""
智能投研系统 - 多Agent协作示例

使用CrewAI框架实现7个Agent协作完成投资研究任务
"""

import argparse
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI


def create_investment_research_team(company: str):
    """创建智能投研团队"""
    
    # 配置LLM（使用本地模型或OpenAI）
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        # 如果使用本地模型：
        # openai_api_base="http://localhost:8000/v1",
        # openai_api_key="local"
    )
    
    # ==================== 创建Agent团队 ====================
    
    # 1. 任务分解专家
    task_decomposer = Agent(
        role='任务分解专家',
        goal='将复杂的投研需求分解为可执行的子任务',
        backstory="""你是一位经验丰富的项目经理，擅长将复杂的大任务
        分解为清晰的、可执行的子任务。你能识别任务之间的依赖关系。""",
        llm=llm,
        verbose=True
    )
    
    # 2. 新闻采集专家
    news_collector = Agent(
        role='新闻采集专家',
        goal='收集目标公司的最新新闻动态和市场舆情',
        backstory="""你是一位资深的新闻记者，擅长从各种渠道收集
        最新的公司新闻和市场信息。你对市场敏感度高。""",
        llm=llm,
        verbose=True
    )
    
    # 3. 财务分析专家
    financial_analyst = Agent(
        role='财务分析专家',
        goal='深入分析公司的财务报表和关键财务指标',
        backstory="""你是一位CFA持证的财务分析师，精通财务报表分析、
        比率分析和估值模型。你的分析严谨、专业。""",
        llm=llm,
        verbose=True
    )
    
    # 4. 竞争对手分析专家
    competitor_analyst = Agent(
        role='竞争对手分析专家',
        goal='分析目标公司的竞争对手和市场竞争格局',
        backstory="""你是一位战略咨询顾问，擅长竞争分析和市场定位。
        你能识别公司的竞争优势和劣势。""",
        llm=llm,
        verbose=True
    )
    
    # 5. 行业研究专家
    industry_researcher = Agent(
        role='行业研究专家',
        goal='研究目标公司所在行业的发展趋势和前景',
        backstory="""你是一位行业研究专家，深入了解各个行业的发展
        动态、政策环境和未来趋势。你的视野宏观、前瞻。""",
        llm=llm,
        verbose=True
    )
    
    # 6. 报告整合专家
    report_integrator = Agent(
        role='报告整合专家',
        goal='整合各方分析结果，形成完整的投资报告',
        backstory="""你是一位资深投行分析师，擅长整合多方信息，
        形成逻辑清晰、结论明确的投资报告。""",
        llm=llm,
        verbose=True
    )
    
    # 7. 报告写作专家
    report_writer = Agent(
        role='报告写作专家',
        goal='将整合的分析结果转化为专业的投资报告文档',
        backstory="""你是一位专业的财经作家，擅长将复杂的分析
        转化为通俗易懂、结构清晰的报告文档。""",
        llm=llm,
        verbose=True
    )
    
    # ==================== 定义任务 ====================
    
    # 任务1：分解投研需求
    decompose_task = Task(
        description=f"""
        将"{company}公司投资分析"这个需求分解为具体的子任务，
        包括需要收集的信息类型和分析维度。
        """,
        agent=task_decomposer,
        expected_output="详细的任务分解清单"
    )
    
    # 任务2：收集新闻
    news_task = Task(
        description=f"""
        收集{company}公司的最新新闻动态，包括：
        1. 最近3个月的重要新闻
        2. 市场舆情和投资者情绪
        3. 重大事件和影响
        """,
        agent=news_collector,
        expected_output="新闻和舆情汇总报告"
    )
    
    # 任务3：财务分析
    financial_task = Task(
        description=f"""
        分析{company}公司的财务状况，包括：
        1. 近3年的营收、利润、现金流趋势
        2. 关键财务比率（ROE、ROA、负债率等）
        3. 盈利能力和成长性评估
        """,
        agent=financial_analyst,
        expected_output="详细的财务分析报告"
    )
    
    # 任务4：竞争对手分析
    competitor_task = Task(
        description=f"""
        分析{company}公司的竞争格局，包括：
        1. 主要竞争对手及其市场份额
        2. 竞争优势和劣势对比
        3. 竞争策略分析
        """,
        agent=competitor_analyst,
        expected_output="竞争对手分析报告"
    )
    
    # 任务5：行业研究
    industry_task = Task(
        description=f"""
        研究{company}公司所在行业，包括：
        1. 行业规模和发展趋势
        2. 政策环境和监管要求
        3. 行业机会和挑战
        """,
        agent=industry_researcher,
        expected_output="行业研究报告"
    )
    
    # 任务6：整合分析
    integrate_task = Task(
        description=f"""
        整合以下分析结果，形成{company}公司的综合分析：
        - 新闻和舆情
        - 财务分析
        - 竞争对手分析
        - 行业研究
        
        提炼关键发现和投资要点。
        """,
        agent=report_integrator,
        context=[news_task, financial_task, competitor_task, industry_task],
        expected_output="综合分析摘要"
    )
    
    # 任务7：撰写报告
    write_task = Task(
        description=f"""
        基于综合分析，撰写{company}公司的投资报告，包括：
        
        ## 1. 执行摘要
        - 核心观点和建议
        
        ## 2. 公司概况
        - 基本信息和业务模式
        
        ## 3. 行业分析
        - 行业现状和趋势
        
        ## 4. 竞争分析
        - 竞争格局和优势
        
        ## 5. 财务分析
        - 财务状况和估值
        
        ## 6. 投资建议
        - 评级和目标价
        
        ## 7. 风险提示
        - 主要风险因素
        
        请使用专业的投行报告格式。
        """,
        agent=report_writer,
        context=[integrate_task],
        expected_output="完整的投资报告（Markdown格式）"
    )
    
    # ==================== 创建团队 ====================
    
    crew = Crew(
        agents=[
            task_decomposer,
            news_collector,
            financial_analyst,
            competitor_analyst,
            industry_researcher,
            report_integrator,
            report_writer
        ],
        tasks=[
            decompose_task,
            news_task,
            financial_task,
            competitor_task,
            industry_task,
            integrate_task,
            write_task
        ],
        process=Process.hierarchical,  # 层级化执行
        verbose=2
    )
    
    return crew


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='智能投研系统')
    parser.add_argument('--company', type=str, required=True, 
                       help='要分析的公司名称')
    
    args = parser.parse_args()
    
    print(f"🚀 开始对{args.company}进行智能投研分析...\n")
    
    # 创建团队
    crew = create_investment_research_team(args.company)
    
    # 执行
    result = crew.kickoff()
    
    print("\n✅ 投研分析完成！")
    print("\n" + "="*60)
    print("投资报告：")
    print("="*60)
    print(result)
    
    # 保存报告
    with open(f"{args.company}_investment_report.md", "w", encoding="utf-8") as f:
        f.write(str(result))
    
    print(f"\n💾 报告已保存到: {args.company}_investment_report.md")


if __name__ == "__main__":
    main()
