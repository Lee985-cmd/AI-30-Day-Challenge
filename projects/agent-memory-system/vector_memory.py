"""
向量数据库记忆系统

使用ChromaDB实现长期记忆的存储和检索
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional


class VectorMemory:
    """基于向量数据库的长期记忆系统"""
    
    def __init__(self, collection_name: str = "user_memories"):
        """初始化向量记忆系统
        
        Args:
            collection_name: 集合名称
        """
        try:
            import chromadb
                    
            # 初始化 ChromaDB客户端（新版API）
            self.client = chromadb.PersistentClient(path="./memory_db")
                    
            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name=collection_name
            )
                    
            print(f"✅ 向量记忆系统已初始化: {collection_name}")
            
        except ImportError:
            print("⚠️  ChromaDB未安装，使用模拟模式")
            self._use_mock = True
            self.mock_memories = []
    
    def add_memory(self, user_id: str, content: str, 
                  memory_type: str = "fact") -> str:
        """添加记忆
        
        Args:
            user_id: 用户ID
            content: 记忆内容
            memory_type: 记忆类型（fact/opinion/preference/event）
            
        Returns:
            记忆ID
        """
        memory_id = str(uuid.uuid4())
        
        if hasattr(self, '_use_mock') and self._use_mock:
            # 模拟模式
            self.mock_memories.append({
                "id": memory_id,
                "user_id": user_id,
                "content": content,
                "type": memory_type,
                "timestamp": datetime.now().isoformat()
            })
        else:
            # 真实模式
            self.collection.add(
                documents=[content],
                metadatas=[{
                    "id": memory_id,
                    "user_id": user_id,
                    "type": memory_type,
                    "timestamp": datetime.now().isoformat()
                }],
                ids=[memory_id]
            )
        
        print(f"✅ 记忆已保存: {content[:50]}...")
        
        return memory_id
    
    def search_memories(self, user_id: str, query: str, 
                       top_k: int = 5) -> List[Dict]:
        """搜索相关记忆
        
        Args:
            user_id: 用户ID
            query: 查询内容
            top_k: 返回数量
            
        Returns:
            相关记忆列表
        """
        if hasattr(self, '_use_mock') and self._use_mock:
            # 模拟模式：简单关键词匹配
            results = []
            for mem in self.mock_memories:
                if (mem["user_id"] == user_id and 
                    query.lower() in mem["content"].lower()):
                    results.append({
                        "content": mem["content"],
                        "metadata": {
                            "id": mem["id"],
                            "user_id": mem["user_id"],
                            "type": mem["type"]
                        }
                    })
            return results[:top_k]
        else:
            # 真实模式：向量搜索
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k * 2
            )
            
            # 过滤出指定用户的记忆
            user_memories = []
            for i, metadata in enumerate(results["metadatas"][0]):
                if metadata.get("user_id") == user_id:
                    user_memories.append({
                        "content": results["documents"][0][i],
                        "metadata": metadata
                    })
            
            return user_memories[:top_k]
    
    def get_user_profile(self, user_id: str) -> Dict:
        """获取用户画像
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户画像字典
        """
        # 搜索所有用户记忆
        if hasattr(self, '_use_mock') and self._use_mock:
            all_memories = [
                {"content": m["content"], "metadata": {"type": m["type"]}}
                for m in self.mock_memories
                if m["user_id"] == user_id
            ]
        else:
            all_memories = self.search_memories(user_id, "", top_k=100)
        
        # 分类整理
        profile = {
            "facts": [],
            "preferences": [],
            "opinions": [],
            "events": []
        }
        
        for memory in all_memories:
            mem_type = memory["metadata"].get("type", "fact")
            if mem_type in profile:
                profile[mem_type].append(memory["content"])
        
        return profile
    
    def forget_memory(self, memory_id: str) -> bool:
        """删除记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            是否成功删除
        """
        try:
            if hasattr(self, '_use_mock') and self._use_mock:
                self.mock_memories = [
                    m for m in self.mock_memories
                    if m["id"] != memory_id
                ]
            else:
                self.collection.delete(ids=[memory_id])
            
            print(f"✅ 记忆已删除: {memory_id}")
            return True
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """获取记忆统计"""
        if hasattr(self, '_use_mock') and self._use_mock:
            return {
                "total_memories": len(self.mock_memories),
                "mode": "mock"
            }
        else:
            return {
                "total_memories": self.collection.count(),
                "mode": "chromadb"
            }


# ==================== 使用示例 ====================

def example_vector_memory():
    """向量记忆使用示例"""
    
    print("="*60)
    print("向量记忆系统示例")
    print("="*60)
    
    # 创建记忆系统
    memory = VectorMemory(collection_name="demo_memories")
    
    # 添加记忆
    user_id = "user_001"
    
    print("\n📝 添加记忆:")
    memory.add_memory(user_id, "我叫张三，今年30岁", "fact")
    memory.add_memory(user_id, "我喜欢编程和阅读", "preference")
    memory.add_memory(user_id, "我认为Python是最好的编程语言", "opinion")
    memory.add_memory(user_id, "昨天我去参加了技术大会", "event")
    memory.add_memory(user_id, "我擅长机器学习和深度学习", "fact")
    
    # 搜索记忆
    print("\n🔍 搜索'编程'相关记忆:")
    results = memory.search_memories(user_id, "编程", top_k=3)
    for i, mem in enumerate(results, 1):
        print(f"{i}. [{mem['metadata']['type']}] {mem['content']}")
    
    # 获取用户画像
    print("\n👤 用户画像:")
    profile = memory.get_user_profile(user_id)
    for category, items in profile.items():
        if items:
            print(f"\n{category.upper()}:")
            for item in items:
                print(f"  - {item}")
    
    # 查看统计
    print("\n📊 记忆统计:")
    stats = memory.get_stats()
    print(f"总记忆数: {stats['total_memories']}")
    print(f"模式: {stats['mode']}")


if __name__ == "__main__":
    example_vector_memory()

