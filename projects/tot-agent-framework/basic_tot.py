"""
基础Tree of Thoughts实现

展示ToT框架的核心组件和搜索算法
"""
# Windows UTF-8 支持
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import time


@dataclass
class Node:
    """思维树节点"""
    state: str                    # 当前状态
    parent: Optional['Node'] = None  # 父节点
    children: List['Node'] = field(default_factory=list)  # 子节点
    value: float = 0.0           # 评估分数
    depth: int = 0               # 深度
    is_terminal: bool = False    # 是否终止节点
    
    def get_path(self) -> List[str]:
        """获取从根到当前节点的路径"""
        path = []
        node = self
        while node:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))


class TreeOfThoughtsAgent:
    """Tree of Thoughts Agent实现"""
    
    def __init__(
        self, 
        llm,
        max_depth: int = 3,
        branch_factor: int = 3,
        search_strategy: str = "bfs"
    ):
        self.llm = llm
        self.max_depth = max_depth
        self.branch_factor = branch_factor
        self.search_strategy = search_strategy
        
        # 统计信息
        self.stats = {
            "nodes_explored": 0,
            "total_time": 0,
            "best_score": 0
        }
    
    def solve(self, problem: str) -> Dict:
        """
        求解问题
        
        Args:
            problem: 问题描述
            
        Returns:
            包含解决方案和统计信息的字典
        """
        start_time = time.time()
        
        # 创建根节点
        root = Node(state=problem, parent=None, depth=0)
        
        # 执行搜索
        if self.search_strategy == "bfs":
            solution = self.bfs_search(root)
        elif self.search_strategy == "dfs":
            solution = self.dfs_search(root)
        else:
            solution = self.heuristic_search(root)
        
        end_time = time.time()
        
        # 收集结果
        result = {
            "solution": solution.get_path() if solution else None,
            "final_answer": solution.state if solution else "未找到解",
            "score": solution.value if solution else 0,
            "stats": self.stats.copy(),
            "time_elapsed": end_time - start_time
        }
        
        return result
    
    def bfs_search(self, root: Node) -> Optional[Node]:
        """广度优先搜索"""
        from collections import deque
        
        queue = deque([root])
        self.stats["nodes_explored"] += 1
        
        while queue:
            node = queue.popleft()
            
            print(f"🔍 探索节点 (深度={node.depth}, 分数={node.value:.1f})")
            print(f"   状态: {node.state[:80]}...")
            
            # 检查是否达到目标
            if self.is_goal(node):
                print(f"✅ 找到目标！\n")
                return node
            
            # 扩展子节点
            if node.depth < self.max_depth:
                children = self.generate_thoughts(node)
                for child in children:
                    child.value = self.evaluate_state(child)
                    node.children.append(child)
                    queue.append(child)
                    self.stats["nodes_explored"] += 1
        
        return None
    
    def dfs_search(self, node: Node) -> Optional[Node]:
        """深度优先搜索"""
        self.stats["nodes_explored"] += 1
        
        print(f"🔍 探索节点 (深度={node.depth}, 分数={node.value:.1f})")
        print(f"   状态: {node.state[:80]}...")
        
        # 检查终止条件
        if self.is_goal(node):
            print(f"✅ 找到目标！\n")
            return node
        
        if node.depth >= self.max_depth:
            print(f"⚠️  达到最大深度，回溯\n")
            return None
        
        # 递归搜索子节点
        children = self.generate_thoughts(node)
        for child in children:
            child.value = self.evaluate_state(child)
            node.children.append(child)
            
            result = self.dfs_search(child)
            if result:
                return result
        
        print(f"⚠️  此路径无解，回溯\n")
        return None
    
    def heuristic_search(self, root: Node) -> Optional[Node]:
        """启发式搜索（简化版）"""
        import heapq
        
        # 优先队列：(负分数, 深度, 节点)
        priority_queue = [(-root.value, 0, root)]
        self.stats["nodes_explored"] += 1
        
        while priority_queue:
            neg_score, depth, node = heapq.heappop(priority_queue)
            
            print(f"🔍 探索节点 (深度={node.depth}, 分数={node.value:.1f})")
            
            # 检查目标
            if self.is_goal(node):
                print(f"✅ 找到目标！\n")
                return node
            
            # 扩展
            if depth < self.max_depth:
                children = self.generate_thoughts(node)
                for child in children:
                    child.value = self.evaluate_state(child)
                    node.children.append(child)
                    
                    # 计算优先级
                    priority = -child.value
                    heapq.heappush(priority_queue, (priority, child.depth, child))
                    self.stats["nodes_explored"] += 1
        
        return None
    
    def generate_thoughts(self, node: Node) -> List[Node]:
        """生成多个候选思路"""
        prompt = f"""
当前状态：{node.state}

请生成{self.branch_factor}个不同的下一步思路。
每个思路应该：
1. 与当前状态相关
2. 朝着解决问题的方向前进
3. 各不相同（多样性）

格式：
思路1: [内容]
思路2: [内容]
思路3: [内容]
"""
        
        try:
            response = self.llm.invoke(prompt)
            thoughts = self._parse_thoughts(response.content)
        except Exception as e:
            print(f"❌ 生成思路失败: {e}")
            thoughts = ["尝试其他方法"] * self.branch_factor
        
        # 创建子节点
        children = []
        for thought in thoughts:
            child = Node(
                state=thought,
                parent=node,
                depth=node.depth + 1
            )
            children.append(child)
        
        return children
    
    def evaluate_state(self, node: Node) -> float:
        """评估状态质量"""
        path = node.get_path()
        path_text = " → ".join(path[-2:])  # 只看最近两步
        
        prompt = f"""
评估以下推理步骤的质量：

{path_text}

请从以下维度评分（0-10分）：
1. 逻辑正确性
2. 进展程度
3. 可行性

只返回一个数字评分。
"""
        
        try:
            response = self.llm.invoke(prompt)
            score = float(response.content.strip())
            return min(max(score, 0), 10)  # 限制在0-10范围
        except:
            return 5.0  # 默认中等分数
    
    def is_goal(self, node: Node) -> bool:
        """检查是否达到目标"""
        prompt = f"""
判断以下状态是否已经解决了问题：

{node.state}

如果已经完全解决，回答"YES"，否则回答"NO"。
"""
        
        try:
            response = self.llm.invoke(prompt)
            return "YES" in response.content.upper()
        except:
            return False
    
    def _parse_thoughts(self, text: str) -> List[str]:
        """解析LLM输出的思路"""
        thoughts = []
        lines = text.strip().split('\n')
        
        for line in lines:
            if ':' in line and ('思路' in line or 'Thought' in line):
                # 提取冒号后的内容
                thought = line.split(':', 1)[1].strip()
                if thought:
                    thoughts.append(thought)
        
        # 如果解析失败，使用默认值
        if not thoughts:
            thoughts = [text.strip()] * self.branch_factor
        
        return thoughts[:self.branch_factor]


# ==================== 示例用法 ====================

def get_llm_from_env():
    """
    从环境变量获取LLM配置
    
    支持的环境变量：
    - LOCAL_LLM_URL: 本地模型API地址（如 http://localhost:8000/v1）
    - LOCAL_LLM_API_KEY: API密钥（本地模型通常为 not-needed）
    - MODEL_NAME: 模型名称（默认 qwen-plus）
    """
    import os
    from langchain_openai import ChatOpenAI
    
    # 获取本地模型URL
    base_url = os.getenv("LOCAL_LLM_URL", "")
    
    if not base_url:
        print("⚠️  未检测到LOCAL_LLM_URL环境变量")
        print("   使用Mock LLM进行演示\n")
        return None
    
    # 获取API密钥（本地模型通常不需要）
    api_key = os.getenv("LOCAL_LLM_API_KEY", "not-needed")
    
    # 获取模型名称
    model_name = os.getenv("MODEL_NAME", "qwen-plus")
    
    print(f"✅ 使用本地模型: {model_name}")
    print(f"   API地址: {base_url}\n")
    
    # 创建LLM实例
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        openai_api_key=api_key,
        openai_api_base=base_url
    )
    
    return llm


def example_basic_tot():
    """基础ToT示例"""
    
    print("=" * 60)
    print("示例: 基础Tree of Thoughts")
    print("=" * 60)
    
    # 从环境变量获取LLM
    llm = get_llm_from_env()
    
    # 如果没有配置API，使用Mock LLM
    if llm is None:
        class MockLLM:
            def invoke(self, prompt):
                return type('obj', (object,), {'content': '思路1: 测试思路'})()
        
        llm = MockLLM()
    
    agent = TreeOfThoughtsAgent(
        llm=llm,
        max_depth=2,
        branch_factor=2,
        search_strategy="bfs"
    )
    
    problem = "用数字 1, 2, 3, 4 得到10"
    
    print(f"\n🎯 问题: {problem}\n")
    
    # 由于使用Mock LLM，这里只演示流程
    print("⚠️  提示: 配置真实的LLM API密钥后，可以得到完整的ToT求解过程")
    print("\n预期输出:")
    print("  Level 1: 生成3个初始思路")
    print("  Level 2: 每个思路扩展出3个子思路")
    print("  评估: 对9个叶子节点评分")
    print("  选择: 返回最高分的路径")
    

if __name__ == "__main__":
    example_basic_tot()
