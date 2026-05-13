"""
框架选择助手
根据项目需求推荐最适合的Agent开发框架
"""


def ask_questions():
    """询问用户项目需求"""
    print("=" * 60)
    print("AI Agent 框架选择助手")
    print("=" * 60)
    print("\n请回答以下问题，我将为您推荐最合适的框架\n")
    
    answers = {}
    
    # 问题1: 主要用途
    print("1. 您的项目主要用途是什么？")
    print("   [1] RAG知识库问答")
    print("   [2] 复杂Agent工作流")
    print("   [3] 企业级搜索系统")
    print("   [4] 快速原型开发")
    choice = input("请选择 (1-4): ").strip()
    
    usage_map = {
        "1": "rag",
        "2": "agent",
        "3": "search",
        "4": "prototype"
    }
    answers["usage"] = usage_map.get(choice, "agent")
    
    # 问题2: 团队经验
    print("\n2. 团队的Python经验如何？")
    print("   [1] 初学者")
    print("   [2] 中等水平")
    print("   [3] 高级开发者")
    choice = input("请选择 (1-3): ").strip()
    
    experience_map = {
        "1": "beginner",
        "2": "intermediate",
        "3": "advanced"
    }
    answers["experience"] = experience_map.get(choice, "intermediate")
    
    # 问题3: 性能要求
    print("\n3. 对性能的要求？")
    print("   [1] 一般（可以接受秒级响应）")
    print("   [2] 较高（需要亚秒级响应）")
    print("   [3] 极高（需要毫秒级响应）")
    choice = input("请选择 (1-3): ").strip()
    
    perf_map = {
        "1": "normal",
        "2": "high",
        "3": "critical"
    }
    answers["performance"] = perf_map.get(choice, "normal")
    
    # 问题4: 数据规模
    print("\n4. 预计处理的数据规模？")
    print("   [1] 小规模（< 1000文档）")
    print("   [2] 中等规模（1000-10000文档）")
    print("   [3] 大规模（> 10000文档）")
    choice = input("请选择 (1-3): ").strip()
    
    scale_map = {
        "1": "small",
        "2": "medium",
        "3": "large"
    }
    answers["scale"] = scale_map.get(choice, "medium")
    
    # 问题5: 部署环境
    print("\n5. 部署环境？")
    print("   [1] 本地/开发环境")
    print("   [2] 云端服务")
    print("   [3] 企业内网")
    choice = input("请选择 (1-3): ").strip()
    
    deploy_map = {
        "1": "local",
        "2": "cloud",
        "3": "enterprise"
    }
    answers["deployment"] = deploy_map.get(choice, "cloud")
    
    # 问题6: 是否需要多语言支持
    print("\n6. 是否需要多语言支持？")
    print("   [1] 不需要，只需中文/英文")
    print("   [2] 需要支持多种语言")
    choice = input("请选择 (1-2): ").strip()
    answers["multilingual"] = (choice == "2")
    
    # 问题7: 预算考虑
    print("\n7. 预算考虑？")
    print("   [1] 开源免费优先")
    print("   [2] 可以接受付费服务")
    print("   [3] 预算充足，追求最佳方案")
    choice = input("请选择 (1-3): ").strip()
    
    budget_map = {
        "1": "free",
        "2": "moderate",
        "3": "unlimited"
    }
    answers["budget"] = budget_map.get(choice, "moderate")
    
    return answers


def recommend_framework(answers):
    """根据答案推荐框架"""
    scores = {
        "LangChain": 0,
        "LlamaIndex": 0,
        "Haystack": 0
    }
    
    # 根据用途评分
    if answers["usage"] == "rag":
        scores["LlamaIndex"] += 3
        scores["LangChain"] += 2
        scores["Haystack"] += 2
    elif answers["usage"] == "agent":
        scores["LangChain"] += 3
        scores["Haystack"] += 2
        scores["LlamaIndex"] += 1
    elif answers["usage"] == "search":
        scores["Haystack"] += 3
        scores["LangChain"] += 2
        scores["LlamaIndex"] += 1
    elif answers["usage"] == "prototype":
        scores["LangChain"] += 3
        scores["LlamaIndex"] += 2
        scores["Haystack"] += 1
    
    # 根据经验评分
    if answers["experience"] == "beginner":
        scores["LlamaIndex"] += 2
        scores["LangChain"] += 1
        scores["Haystack"] += 0
    elif answers["experience"] == "intermediate":
        scores["LangChain"] += 2
        scores["LlamaIndex"] += 2
        scores["Haystack"] += 1
    elif answers["experience"] == "advanced":
        scores["Haystack"] += 2
        scores["LangChain"] += 2
        scores["LlamaIndex"] += 1
    
    # 根据性能要求评分
    if answers["performance"] == "critical":
        scores["Haystack"] += 3
        scores["LangChain"] += 1
        scores["LlamaIndex"] += 1
    elif answers["performance"] == "high":
        scores["Haystack"] += 2
        scores["LangChain"] += 2
        scores["LlamaIndex"] += 1
    else:
        scores["LangChain"] += 2
        scores["LlamaIndex"] += 2
        scores["Haystack"] += 1
    
    # 根据数据规模评分
    if answers["scale"] == "large":
        scores["Haystack"] += 3
        scores["LlamaIndex"] += 2
        scores["LangChain"] += 1
    elif answers["scale"] == "medium":
        scores["LlamaIndex"] += 2
        scores["Haystack"] += 2
        scores["LangChain"] += 2
    else:
        scores["LangChain"] += 2
        scores["LlamaIndex"] += 2
        scores["Haystack"] += 1
    
    # 根据部署环境评分
    if answers["deployment"] == "enterprise":
        scores["Haystack"] += 3
        scores["LangChain"] += 2
        scores["LlamaIndex"] += 1
    elif answers["deployment"] == "cloud":
        scores["LangChain"] += 2
        scores["Haystack"] += 2
        scores["LlamaIndex"] += 2
    else:
        scores["LangChain"] += 2
        scores["LlamaIndex"] += 2
        scores["Haystack"] += 1
    
    # 多语言支持
    if answers["multilingual"]:
        scores["Haystack"] += 2
        scores["LangChain"] += 1
        scores["LlamaIndex"] += 0
    
    # 预算考虑
    if answers["budget"] == "free":
        scores["LangChain"] += 1
        scores["LlamaIndex"] += 1
        scores["Haystack"] += 1  # Haystack也是开源的
    
    return scores


def display_recommendation(scores, answers):
    """显示推荐结果"""
    print("\n" + "=" * 60)
    print("推荐结果")
    print("=" * 60)
    
    # 排序
    sorted_frameworks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 首选推荐: {sorted_frameworks[0][0]}")
    print(f"   得分: {sorted_frameworks[0][1]}")
    
    if len(sorted_frameworks) > 1:
        print(f"\n🥈 备选方案: {sorted_frameworks[1][0]}")
        print(f"   得分: {sorted_frameworks[1][1]}")
    
    if len(sorted_frameworks) > 2:
        print(f"\n🥉 第三选择: {sorted_frameworks[2][0]}")
        print(f"   得分: {sorted_frameworks[2][1]}")
    
    # 详细分析
    print("\n" + "-" * 60)
    print("详细分析:")
    print("-" * 60)
    
    winner = sorted_frameworks[0][0]
    
    if winner == "LangChain":
        print("\n✅ 为什么选择 LangChain？")
        print("  • 最适合构建复杂的Agent工作流")
        print("  • 拥有丰富的工具和组件生态系统")
        print("  • 社区活跃，文档完善")
        print("  • 灵活性强，适合快速迭代")
        print("\n⚠️ 注意事项:")
        print("  • 学习曲线相对较陡")
        print("  • API变化较快，需要持续关注更新")
        print("  • 生产环境需要仔细优化性能")
        
    elif winner == "LlamaIndex":
        print("\n✅ 为什么选择 LlamaIndex？")
        print("  • RAG应用的专家级框架")
        print("  • 简单易用，上手快")
        print("  • 优秀的数据索引和查询优化")
        print("  • 文档质量高，示例丰富")
        print("\n⚠️ 注意事项:")
        print("  • Agent功能相对有限")
        print("  • 不适合复杂的工作流编排")
        print("  • 自定义能力不如LangChain")
        
    elif winner == "Haystack":
        print("\n✅ 为什么选择 Haystack？")
        print("  • 企业级搜索系统的最佳选择")
        print("  • 生产就绪，稳定性高")
        print("  • 优秀的多语言支持")
        print("  • Pipeline设计清晰，易于维护")
        print("\n⚠️ 注意事项:")
        print("  • 学习曲线较陡")
        print("  • 社区相对较小")
        print("  • 灵活性不如LangChain")
    
    # 基于用户需求的建议
    print("\n" + "-" * 60)
    print("基于您需求的建议:")
    print("-" * 60)
    
    if answers["usage"] == "rag":
        print("  📚 您的主要需求是RAG应用")
        if winner == "LlamaIndex":
            print("  → LlamaIndex是RAG的最佳选择！")
        else:
            print(f"  → {winner}也能胜任，但可以考虑LlamaIndex作为备选")
    
    elif answers["usage"] == "agent":
        print("  🤖 您需要构建复杂Agent")
        if winner == "LangChain":
            print("  → LangChain的Agent能力最强！")
        else:
            print(f"  → {winner}有Agent功能，但LangChain更专业")
    
    if answers["experience"] == "beginner":
        print("\n  👶 您是初学者")
        print("  → 建议从官方教程开始，循序渐进")
        print("  → 加入社区Discord或论坛获取帮助")
    
    if answers["performance"] == "critical":
        print("\n  ⚡ 您对性能要求很高")
        print("  → 务必进行充分的性能测试")
        print("  → 考虑使用缓存、批处理等优化技术")
    
    # 下一步行动
    print("\n" + "=" * 60)
    print("下一步行动")
    print("=" * 60)
    print(f"\n1. 阅读 {winner} 官方文档")
    print("2. 运行本项目的示例代码")
    print("3. 从小项目开始实践")
    print("4. 逐步扩展到完整应用")
    print("\n祝您开发顺利！🚀")


def main():
    """主函数"""
    try:
        # 询问用户
        answers = ask_questions()
        
        # 计算推荐
        scores = recommend_framework(answers)
        
        # 显示结果
        display_recommendation(scores, answers)
        
    except KeyboardInterrupt:
        print("\n\n程序已退出")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        print("请重试或手动选择框架")


if __name__ == "__main__":
    main()
