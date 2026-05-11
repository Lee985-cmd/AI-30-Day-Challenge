"""
混合记忆系统演示

结合短期记忆、长期记忆和工作记忆
"""

from vector_memory import VectorMemory


class HybridMemorySystem:
    """混合记忆系统
    
    结合短期记忆、工作记忆和长期记忆
    """
    
    def __init__(self, user_id: str, short_term_k: int = 10):
        self.user_id = user_id
        
        # 短期记忆：简单列表（模拟滑动窗口）
        self.short_term_memory = []
        self.short_term_k = short_term_k
        
        # 长期记忆：向量数据库
        self.long_term_memory = VectorMemory(
            collection_name=f"user_{user_id}_memories"
        )
        
        # 工作记忆：临时存储
        self.working_memory = {}
        
        # 记忆统计
        self.stats = {
            "short_term_count": 0,
            "long_term_count": 0,
            "total_interactions": 0
        }
    
    def add_interaction(self, human_input: str, ai_response: str,
                       save_to_long_term: bool = True):
        """添加交互记录"""
        # 添加到短期记忆
        self.short_term_memory.append({
            "human": human_input,
            "ai": ai_response
        })
        
        # 保持滑动窗口大小
        if len(self.short_term_memory) > self.short_term_k:
            self.short_term_memory.pop(0)
        
        self.stats["short_term_count"] += 1
        
        # 提取重要信息保存到长期记忆
        if save_to_long_term:
            important_facts = self._extract_important_facts(human_input)
            for fact in important_facts:
                self.long_term_memory.add_memory(
                    self.user_id,
                    fact["content"],
                    fact["type"]
                )
                self.stats["long_term_count"] += 1
        
        self.stats["total_interactions"] += 1
    
    def get_context(self, query: str) -> str:
        """获取完整的上下文"""
        context_parts = []
        
        # 1. 短期记忆（最近对话）
        if self.short_term_memory:
            context_parts.append("【最近对话】")
            for i, interaction in enumerate(self.short_term_memory[-3:], 1):
                context_parts.append(f"{i}. 用户: {interaction['human']}")
                context_parts.append(f"   AI: {interaction['ai']}")
        
        # 2. 长期记忆（相关信息）
        long_term = self.long_term_memory.search_memories(
            self.user_id,
            query,
            top_k=3
        )
        if long_term:
            context_parts.append("\n【相关记忆】")
            for i, mem in enumerate(long_term, 1):
                context_parts.append(f"{i}. {mem['content']}")
        
        # 3. 工作记忆（临时信息）
        if self.working_memory:
            context_parts.append("\n【当前任务】")
            for key, value in self.working_memory.items():
                context_parts.append(f"- {key}: {value}")
        
        return "\n".join(context_parts)
    
    def set_working_memory(self, key: str, value):
        """设置工作记忆"""
        self.working_memory[key] = value
    
    def clear_working_memory(self):
        """清空工作记忆"""
        self.working_memory.clear()
    
    def _extract_important_facts(self, text: str):
        """从文本中提取重要事实（简化版）"""
        facts = []
        
        # 简单的规则提取
        if "我叫" in text or "名字是" in text:
            facts.append({"content": text, "type": "fact"})
        
        if "我喜欢" in text or "我爱" in text:
            facts.append({"content": text, "type": "preference"})
        
        if "我认为" in text or "我觉得" in text:
            facts.append({"content": text, "type": "opinion"})
        
        return facts
    
    def get_stats(self):
        """获取记忆统计"""
        return self.stats.copy()


# ==================== 使用示例 ====================

def example_hybrid_memory():
    """混合记忆系统示例"""
    
    print("="*60)
    print("混合记忆系统示例")
    print("="*60)
    
    # 创建混合记忆系统
    memory = HybridMemorySystem(user_id="user_001")
    
    # 模拟对话
    interactions = [
        ("我叫张三，今年30岁", "你好张三！很高兴认识你。"),
        ("我喜欢编程和阅读", "编程和阅读都是很好的爱好！"),
        ("我正在学习机器学习", "机器学习是个很有前景的领域！"),
        ("你觉得Python怎么样？", "Python是一门优秀的编程语言。"),
    ]
    
    # 添加交互
    print("\n📝 添加交互:")
    for human_input, ai_response in interactions:
        memory.add_interaction(human_input, ai_response)
    
    # 获取上下文
    print("\n\n🔍 查询'编程'相关上下文:")
    context = memory.get_context("编程")
    print(context)
    
    # 设置工作记忆
    memory.set_working_memory("current_task", "推荐编程书籍")
    
    # 再次获取上下文
    print("\n\n🔍 带工作记忆的上下文:")
    context = memory.get_context("推荐书籍")
    print(context)
    
    # 查看统计
    print("\n\n📊 记忆统计:")
    stats = memory.get_stats()
    print(f"短期记忆交互数: {stats['short_term_count']}")
    print(f"长期记忆保存数: {stats['long_term_count']}")
    print(f"总交互数: {stats['total_interactions']}")


if __name__ == "__main__":
    example_hybrid_memory()
