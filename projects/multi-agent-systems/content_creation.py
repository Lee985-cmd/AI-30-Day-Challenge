"""
内容创作团队 - 多Agent协作示例

展示4个Agent协作完成内容创作任务
"""

import argparse
from crewai import Agent, Task, Crew


def create_content_team(topic: str):
    """创建内容创作团队"""
    
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.8)
    
    # ==================== 创建Agent ====================
    
    # 1. 选题策划师
    planner = Agent(
        role='选题策划师',
        goal=f'围绕"{topic}"主题，策划有吸引力的内容角度和结构',
        backstory="""你是一位资深的内容策划专家，擅长发现热点话题的
        独特视角，设计引人入胜的内容框架。""",
        llm=llm,
        verbose=True
    )
    
    # 2. 内容研究员
    researcher = Agent(
        role='内容研究员',
        goal=f'深入研究"{topic}"主题，收集详实的资料和案例',
        backstory="""你是一位专业的研究员，擅长从多个渠道收集信息，
        验证事实，整理出有价值的研究资料。""",
        llm=llm,
        verbose=True
    )
    
    # 3. 内容创作者
    writer = Agent(
        role='内容创作者',
        goal=f'基于策划和研究结果，创作高质量的"{topic}"主题文章',
        backstory="""你是一位优秀的作家，文笔流畅，善于用生动的语言
        表达复杂的概念，让读者易于理解。""",
        llm=llm,
        verbose=True
    )
    
    # 4. 编辑审核员
    editor = Agent(
        role='编辑审核员',
        goal='审核和优化文章内容，确保质量和可读性',
        backstory="""你是一位经验丰富的编辑，擅长发现文章中的问题，
        提出改进建议，提升文章的整体质量。""",
        llm=llm,
        verbose=True
    )
    
    # ==================== 定义任务 ====================
    
    planning_task = Task(
        description=f"""
        为"{topic}"主题设计内容策划方案：
        
        1. 目标受众分析
        2. 核心观点提炼（3-5个）
        3. 文章结构设计
        4. 亮点和特色建议
        
        输出详细的策划文档。
        """,
        agent=planner,
        expected_output="内容策划方案"
    )
    
    research_task = Task(
        description=f"""
        围绕"{topic}"主题进行深度研究：
        
        1. 收集相关数据和统计信息
        2. 寻找典型案例和故事
        3. 整理行业观点和专家意见
        4. 识别常见误区和争议点
        
        提供详细的研究笔记。
        """,
        agent=researcher,
        context=[planning_task],
        expected_output="研究报告"
    )
    
    writing_task = Task(
        description=f"""
        基于策划方案和研究资料，撰写"{topic}"主题文章：
        
        要求：
        - 字数：2000-3000字
        - 风格：专业但通俗易懂
        - 结构：引言、正文（3-5个部分）、结论
        - 包含实际案例和数据支撑
        
        使用Markdown格式。
        """,
        agent=writer,
        context=[planning_task, research_task],
        expected_output="完整的文章草稿"
    )
    
    editing_task = Task(
        description="""
        审核和优化文章：
        
        1. 检查逻辑连贯性
        2. 优化语言表达
        3. 补充必要的背景信息
        4. 添加小标题和重点标注
        5. 提供修改建议
        
        输出最终版本的文章。
        """,
        agent=editor,
        context=[writing_task],
        expected_output="优化后的最终文章"
    )
    
    # ==================== 创建团队 ====================
    
    crew = Crew(
        agents=[planner, researcher, writer, editor],
        tasks=[planning_task, research_task, writing_task, editing_task],
        verbose=2
    )
    
    return crew


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='内容创作团队')
    parser.add_argument('--topic', type=str, required=True, 
                       help='创作主题')
    
    args = parser.parse_args()
    
    print(f"🚀 开始创作关于'{args.topic}'的内容...\n")
    
    # 创建团队
    crew = create_content_team(args.topic)
    
    # 执行
    result = crew.kickoff()
    
    print("\n✅ 内容创作完成！")
    print("\n" + "="*60)
    print("最终文章：")
    print("="*60)
    print(result)
    
    # 保存文章
    filename = f"{''.join(c for c in args.topic if c.isalnum())}_article.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(str(result))
    
    print(f"\n💾 文章已保存到: {filename}")


if __name__ == "__main__":
    main()
