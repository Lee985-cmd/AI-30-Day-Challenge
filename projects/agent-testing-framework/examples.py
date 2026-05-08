"""
Agent测试框架 - 使用示例

演示如何使用测试框架的各个功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_testing import (
    TestPrompt, 
    TestToolFunction, 
    TestParser,
    QualityEvaluator,
    ABTestFramework
)


# ==================== 1. Prompt模板测试 ====================

def example_prompt_testing():
    """示例：Prompt模板测试"""
    print("=" * 60)
    print("示例1: Prompt模板测试")
    print("=" * 60)
    
    # 定义Prompt模板
    prompt_template = """
    你是一个专业的{role}助手。
    
    用户问题：{question}
    
    请提供详细、准确的回答，包含以下要点：
    1. 核心概念解释
    2. 实际应用案例
    3. 注意事项
    
    回答格式：Markdown
    """
    
    # 创建测试器
    tester = TestPrompt(prompt_template)
    
    # 添加测试用例
    tester.add_test_case(
        input_vars={"role": "技术", "question": "什么是机器学习？"},
        expected_keywords=["核心概念", "实际应用", "注意事项"]
    )
    
    tester.add_test_case(
        input_vars={"role": "业务", "question": "如何提升销售额？"},
        expected_keywords=["核心概念", "实际应用"]
    )
    
    # 运行测试
    results = tester.run_tests()
    
    # 输出结果
    print(f"\n总测试数: {results['total']}")
    print(f"通过: {results['passed']}")
    print(f"失败: {results['failed']}")
    print(f"\n通过率: {results['passed']/results['total']*100:.1f}%")
    
    for detail in results["details"]:
        status = "✅ PASS" if detail["passed"] else "❌ FAIL"
        print(f"\n{status} - 测试用例 {detail['case_id']}")
        print(f"  输入: {detail['input']}")
        if not detail["passed"]:
            print(f"  缺失关键字: {detail.get('missing_keywords', [])}")


# ==================== 2. 工具函数测试 ====================

def calculate_area(length: float, width: float) -> float:
    """示例工具函数：计算面积"""
    if length <= 0 or width <= 0:
        raise ValueError("长度和宽度必须为正数")
    return length * width


def example_tool_function_testing():
    """示例：工具函数测试"""
    print("\n" + "=" * 60)
    print("示例2: 工具函数测试")
    print("=" * 60)
    
    # 创建测试器
    tester = TestToolFunction(calculate_area)
    
    # 添加测试用例
    tester.add_test_case(
        input_args={"length": 5, "width": 3},
        expected_type=float,
        expected_value=15
    )
    
    tester.add_test_case(
        input_args={"length": 10, "width": 2.5},
        expected_type=float,
        expected_value=25.0
    )
    
    tester.add_test_case(
        input_args={"length": -1, "width": 5},
        should_raise=True  # 期望抛出异常
    )
    
    # 运行测试
    results = tester.run_tests()
    
    # 输出结果
    print(f"\n总测试数: {results['total']}")
    print(f"通过: {results['passed']}")
    print(f"失败: {results['failed']}")
    
    for detail in results["details"]:
        status = "✅ PASS" if detail["passed"] else "❌ FAIL"
        print(f"\n{status} - 测试用例 {detail['case_id']}")
        print(f"  输入: {detail['input']}")
        print(f"  执行时间: {detail['execution_time_ms']}ms")
        if not detail["passed"]:
            print(f"  错误: {detail.get('error', 'Unknown')}")


# ==================== 3. 解析器测试 ====================

def parse_json_response(raw_output: str) -> dict:
    """示例解析器：解析JSON响应"""
    import json
    try:
        # 提取JSON部分
        start = raw_output.find('{')
        end = raw_output.rfind('}') + 1
        
        if start == -1 or end == 0:
            raise ValueError("未找到有效的JSON")
        
        json_str = raw_output[start:end]
        return json.loads(json_str)
    except Exception as e:
        raise ValueError(f"解析失败: {str(e)}")


def example_parser_testing():
    """示例：解析器测试"""
    print("\n" + "=" * 60)
    print("示例3: 解析器测试")
    print("=" * 60)
    
    # 创建测试器
    tester = TestParser(parse_json_response)
    
    # 添加测试用例
    tester.add_test_case(
        raw_output='{"name": "张三", "age": 25}',
        expected_result={"name": "张三", "age": 25}
    )
    
    tester.add_test_case(
        raw_output='这是回答\n{"result": "success"}\n结束',
        expected_result={"result": "success"}
    )
    
    tester.add_test_case(
        raw_output='无效的JSON格式',
        should_fail=True  # 期望解析失败
    )
    
    # 运行测试
    results = tester.run_tests()
    
    # 输出结果
    print(f"\n总测试数: {results['total']}")
    print(f"通过: {results['passed']}")
    print(f"失败: {results['failed']}")
    
    for detail in results["details"]:
        status = "✅ PASS" if detail["passed"] else "❌ FAIL"
        print(f"\n{status} - 测试用例 {detail['case_id']}")
        print(f"  原始输出: {detail['raw_output']}")
        print(f"  解析结果: {detail.get('parsed_result', 'N/A')}")


# ==================== 4. 质量评估 ====================

def example_quality_evaluation():
    """示例：质量评估"""
    print("\n" + "=" * 60)
    print("示例4: 质量评估")
    print("=" * 60)
    
    # 创建评估器
    evaluator = QualityEvaluator()
    
    # 测试用例1
    result1 = evaluator.comprehensive_evaluation(
        query="什么是Python？",
        response="Python是一种高级编程语言，由Guido van Rossum于1991年创建。它以其简洁的语法和强大的库支持而闻名。",
        ground_truth="Python是一种高级、通用的编程语言，强调代码可读性。",
        expected_points=["高级编程语言", "Guido van Rossum", "简洁语法"],
        response_time_ms=1500
    )
    
    print("\n测试用例1:")
    print(f"  准确性: {result1['accuracy']}")
    print(f"  相关性: {result1['relevance']}")
    print(f"  完整性: {result1['completeness']}")
    print(f"  时间得分: {result1['response_time_score']}")
    print(f"  总分: {result1['total_score']}")
    print(f"  等级: {result1['grade']}")
    
    # 测试用例2
    result2 = evaluator.comprehensive_evaluation(
        query="如何学习机器学习？",
        response="学习机器学习需要先掌握Python编程和数学基础，然后学习常见的算法如线性回归、决策树等。",
        ground_truth="建议从Python基础开始，学习线性代数、概率论，然后逐步学习监督学习、无监督学习等算法。",
        expected_points=["Python", "数学基础", "常见算法", "线性回归"],
        response_time_ms=2800
    )
    
    print("\n测试用例2:")
    print(f"  准确性: {result2['accuracy']}")
    print(f"  相关性: {result2['relevance']}")
    print(f"  完整性: {result2['completeness']}")
    print(f"  时间得分: {result2['response_time_score']}")
    print(f"  总分: {result2['total_score']}")
    print(f"  等级: {result2['grade']}")
    
    # 平均分数
    avg_scores = evaluator.get_average_scores()
    print("\n平均分数:")
    for metric, score in avg_scores.items():
        print(f"  {metric}: {score}")


# ==================== 5. A/B测试 ====================

def example_ab_testing():
    """示例：A/B测试"""
    print("\n" + "=" * 60)
    print("示例5: A/B测试")
    print("=" * 60)
    
    # 创建A/B测试框架
    ab_test = ABTestFramework()
    
    # 模拟两个不同的Agent响应函数
    def agent_v1(test_case):
        """变体A：基础版本"""
        import random
        # 模拟响应时间和质量
        return {
            "response_time": random.uniform(2000, 3000),
            "quality_score": random.uniform(0.7, 0.85)
        }
    
    def agent_v2(test_case):
        """变体B：优化版本"""
        import random
        # 模拟更快的响应和更高的质量
        return {
            "response_time": random.uniform(1000, 2000),
            "quality_score": random.uniform(0.8, 0.95)
        }
    
    # 创建测试用例
    test_cases = [{"query": f"测试问题{i+1}"} for i in range(20)]
    
    # 创建实验
    ab_test.create_experiment(
        name="Agent响应优化对比",
        variant_a=agent_v1,
        variant_b=agent_v2,
        test_cases=test_cases
    )
    
    # 运行实验
    result = ab_test.run_experiment(0)
    
    # 输出结果
    print(f"\n实验名称: {result['experiment_name']}")
    print(f"样本数量: {result['sample_size']}")
    print(f"\n变体A（基础版）:")
    print(f"  平均响应时间: {result['variant_a']['mean']:.2f}ms")
    print(f"  标准差: {result['variant_a']['std']:.2f}")
    
    print(f"\n变体B（优化版）:")
    print(f"  平均响应时间: {result['variant_b']['mean']:.2f}ms")
    print(f"  标准差: {result['variant_b']['std']:.2f}")
    
    print(f"\n提升: {result['improvement_percent']:.2f}%")
    print(f"P值: {result['p_value']}")
    print(f"统计显著: {'是' if result['statistically_significant'] else '否'}")
    print(f"获胜者: {result['winner']}")
    print(f"\n建议: {result['recommendation']}")


# ==================== 主函数 ====================

if __name__ == "__main__":
    print("\n🧪 Agent测试框架 - 使用示例\n")
    
    # 运行所有示例
    example_prompt_testing()
    example_tool_function_testing()
    example_parser_testing()
    example_quality_evaluation()
    example_ab_testing()
    
    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成！")
    print("=" * 60)
