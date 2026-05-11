"""
知识图谱记忆系统

将记忆组织成图结构，节点是实体，边是关系
"""

from typing import Dict, List, Set
import json
from collections import deque


class KnowledgeGraphMemory:
    """基于知识图谱的记忆系统"""
    
    def __init__(self):
        # 节点：实体 -> 属性
        self.nodes: Dict[str, Dict[str, str]] = {}
        
        # 边：(源实体, 关系, 目标实体)
        self.edges: List[tuple] = []
        
        # 索引：实体 -> 相关边
        self.entity_index: Dict[str, Set[int]] = {}
    
    def add_entity(self, entity: str, attributes: Dict[str, str]):
        """添加实体节点
        
        Args:
            entity: 实体名称
            attributes: 属性字典
        """
        self.nodes[entity] = attributes
        self.entity_index[entity] = set()
        print(f"✅ 添加实体: {entity}")
    
    def add_relation(self, source: str, relation: str, target: str):
        """添加关系边
        
        Args:
            source: 源实体
            relation: 关系类型
            target: 目标实体
        """
        edge_id = len(self.edges)
        self.edges.append((source, relation, target))
        
        # 更新索引
        if source not in self.entity_index:
            self.entity_index[source] = set()
        if target not in self.entity_index:
            self.entity_index[target] = set()
        
        self.entity_index[source].add(edge_id)
        self.entity_index[target].add(edge_id)
        
        print(f"✅ 添加关系: {source} --[{relation}]--> {target}")
    
    def query_entity(self, entity: str) -> Dict:
        """查询实体信息
        
        Args:
            entity: 实体名称
            
        Returns:
            实体信息和相关关系
        """
        if entity not in self.nodes:
            return {"error": f"实体 '{entity}' 不存在"}
        
        # 获取实体属性
        attributes = self.nodes[entity]
        
        # 获取相关关系
        related_edges = []
        if entity in self.entity_index:
            for edge_id in self.entity_index[entity]:
                source, relation, target = self.edges[edge_id]
                related_edges.append({
                    "source": source,
                    "relation": relation,
                    "target": target
                })
        
        return {
            "entity": entity,
            "attributes": attributes,
            "relations": related_edges
        }
    
    def find_path(self, start: str, end: str, max_depth: int = 3) -> List:
        """查找两个实体之间的路径
        
        Args:
            start: 起始实体
            end: 目标实体
            max_depth: 最大搜索深度
            
        Returns:
            路径列表
        """
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current == end:
                return path
            
            # 探索邻居节点
            if current in self.entity_index:
                for edge_id in self.entity_index[current]:
                    source, relation, target = self.edges[edge_id]
                    
                    # 确定下一个节点
                    next_node = target if source == current else source
                    
                    if next_node not in visited:
                        visited.add(next_node)
                        queue.append((next_node, path + [next_node]))
        
        return []
    
    def export_graph(self) -> Dict:
        """导出知识图谱
        
        Returns:
            图谱数据
        """
        return {
            "nodes": self.nodes,
            "edges": [
                {"source": s, "relation": r, "target": t}
                for s, r, t in self.edges
            ]
        }


# ==================== 使用示例 ====================

def example_knowledge_graph():
    """知识图谱记忆示例"""
    
    print("="*60)
    print("知识图谱记忆系统示例")
    print("="*60)
    
    # 创建知识图谱
    kg = KnowledgeGraphMemory()
    
    # 添加实体
    print("\n📝 添加实体:")
    kg.add_entity("张三", {"age": "30", "occupation": "程序员"})
    kg.add_entity("Python", {"type": "编程语言", "paradigm": "多范式"})
    kg.add_entity("机器学习", {"field": "人工智能", "difficulty": "中等"})
    kg.add_entity("李四", {"age": "28", "occupation": "数据科学家"})
    
    # 添加关系
    print("\n🔗 添加关系:")
    kg.add_relation("张三", "喜欢", "Python")
    kg.add_relation("张三", "学习", "机器学习")
    kg.add_relation("张三", "同事", "李四")
    kg.add_relation("李四", "擅长", "Python")
    kg.add_relation("李四", "研究", "机器学习")
    
    # 查询实体
    print("\n🔍 查询张三的信息:")
    info = kg.query_entity("张三")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    
    # 查找路径
    print("\n🔍 查找张三到机器学习的路径:")
    path = kg.find_path("张三", "机器学习")
    print(f"路径: {' -> '.join(path)}")
    
    # 导出图谱
    print("\n📊 知识图谱统计:")
    graph = kg.export_graph()
    print(f"节点数: {len(graph['nodes'])}")
    print(f"边数: {len(graph['edges'])}")


if __name__ == "__main__":
    example_knowledge_graph()
