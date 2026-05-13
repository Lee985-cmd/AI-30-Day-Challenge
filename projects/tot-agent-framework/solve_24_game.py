"""
24点游戏求解器 - Tree of Thoughts实现

使用ToT框架解决经典的24点数学游戏
"""

from basic_tot import TreeOfThoughtsAgent, Node, get_llm_from_env
from typing import List


class Game24Solver:
    """24点游戏专用求解器"""
    
    def __init__(self, llm):
        self.agent = TreeOfThoughtsAgent(
            llm=llm,
            max_depth=4,
            branch_factor=3,
            search_strategy="bfs"
        )
    
    def solve(self, numbers: List[int]) -> dict:
        """
        求解24点
        
        Args:
            numbers: 4个数字的列表
            
        Returns:
            解决方案字典
        """
        problem = f"用数字 {', '.join(map(str, numbers))} 通过加减乘除得到24"
        
        print(f"🎯 问题: {problem}\n")
        print("开始Tree of Thoughts搜索...\n")
        
        result = self.agent.solve(problem)
        
        return result
    
    def format_solution(self, result: dict) -> str:
        """格式化输出解决方案"""
        if not result["solution"]:
            return "❌ 未找到解"
        
        output = []
        output.append("✅ 解决方案:\n")
        
        for i, step in enumerate(result["solution"], 1):
            output.append(f"  Step {i}: {step}")
        
        output.append(f"\n📊 统计信息:")
        output.append(f"  探索节点数: {result['stats']['nodes_explored']}")
        output.append(f"  耗时: {result['time_elapsed']:.2f}秒")
        output.append(f"  最终得分: {result['score']:.1f}/10")
        
        return "\n".join(output)


def example_1():
    """示例1: 经典24点"""
    
    print("=" * 60)
    print("示例1: 经典24点问题")
    print("=" * 60)
    
    # 从环境变量获取LLM
    llm = get_llm_from_env()
    
    # 如果没有配置API，使用Mock LLM
    if llm is None:
        class MockLLM:
            def invoke(self, prompt):
                # 模拟不同的回答
                if "生成" in prompt:
                    return type('obj', (object,), {
                        'content': """思路1: 8 + 8 + 3 + 3 = 22 (接近)
思路2: 8 * 3 = 24, 还剩8和3
思路3: (8 - 3) * (8 - 3) = 25 (接近)"""
                    })()
                elif "评估" in prompt:
                    return type('obj', (object,), {'content': '7.5'})()
                else:
                    return type('obj', (object,), {'content': 'NO'})()
        
        llm = MockLLM()
    solver = Game24Solver(llm)
    
    numbers = [3, 3, 8, 8]
    result = solver.solve(numbers)
    
    print(solver.format_solution(result))


def example_2():
    """示例2: 困难24点"""
    print("\n" + "=" * 60)
    print("示例2: 困难的24点问题")
    print("=" * 60)
    
    llm = get_llm_from_env()
    
    if llm is None:
        class MockLLM:
            def invoke(self, prompt):
                if "生成" in prompt:
                    return type('obj', (object,), {
                        'content': """思路1: 1 + 3 + 4 + 6 = 14 (太小)
思路2: 6 * 4 = 24, 还剩1和3
思路3: (6 - 3) * (4 + 1) = 15 (不对)"""
                    })()
                elif "评估" in prompt:
                    return type('obj', (object,), {'content': '6.0'})()
                else:
                    return type('obj', (object,), {'content': 'NO'})()
        
        llm = MockLLM()
    solver = Game24Solver(llm)
    
    numbers = [1, 3, 4, 6]
    result = solver.solve(numbers)
    
    print(solver.format_solution(result))
    print("\n💡 提示: 这个题目的答案是 6 / (1 - 3/4) = 24")


def example_3():
    """示例3: 批量测试"""
    print("\n" + "=" * 60)
    print("示例3: 批量测试多个24点问题")
    print("=" * 60)
    
    test_cases = [
        [1, 2, 3, 4],
        [5, 5, 5, 1],
        [3, 3, 8, 8],
        [1, 3, 4, 6],
    ]
    
    llm = get_llm_from_env()
    
    if llm is None:
        class MockLLM:
            def invoke(self, prompt):
                return type('obj', (object,), {'content': '思路1: 测试'})()
        
        llm = MockLLM()
    solver = Game24Solver(llm)
    
    for i, numbers in enumerate(test_cases, 1):
        print(f"\n--- 测试 {i}: {numbers} ---")
        result = solver.solve(numbers)
        print(f"  探索节点: {result['stats']['nodes_explored']}")
        print(f"  耗时: {result['time_elapsed']:.2f}秒")


if __name__ == "__main__":
    example_1()
    example_2()
    example_3()
    
    print("\n" + "=" * 60)
    print("🎓 学习要点")
    print("=" * 60)
    print("""
1. ToT通过多路径探索找到最优解
2. BFS适合浅层搜索，保证找到最短路径
3. 评估函数指导搜索方向
4. 本地模型配置：设置LOCAL_LLM_URL环境变量
    
下一步：
- 配置LOCAL_LLM_URL环境变量（如 http://localhost:8000/v1）
- 运行真实的24点求解
- 尝试其他数学问题
    """)
