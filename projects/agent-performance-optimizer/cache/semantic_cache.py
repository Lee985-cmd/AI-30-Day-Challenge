"""
语义缓存实现
基于向量相似度匹配，能识别语义相似的问题
"""

import hashlib
from typing import Optional, Dict, Any, List

try:
    # 新版 LangChain
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    # 旧版 LangChain
    try:
        from langchain.embeddings import OpenAIEmbeddings
    except ImportError:
        # 如果都没有，提供一个占位类
        class OpenAIEmbeddings:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "请安装 langchain-openai: pip install langchain-openai"
                )

try:
    # 新版 LangChain
    from langchain_chroma import Chroma
except ImportError:
    # 旧版 LangChain
    try:
        from langchain.vectorstores import Chroma
    except ImportError:
        # 如果都没有，提供一个占位类
        class Chroma:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "请安装 langchain-chroma: pip install langchain-chroma"
                )


class SemanticCache:
    """基于语义相似度的智能缓存"""
    
    def __init__(
        self, 
        similarity_threshold: float = 0.95,
        embedding_model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        persist_directory: str = "./semantic_cache_db"
    ):
        """
        初始化语义缓存
        
        Args:
            similarity_threshold: 相似度阈值（0-1），越高越严格
            embedding_model: Embedding模型名称
            api_key: API密钥
            persist_directory: 持久化存储目录
        """
        self.threshold = similarity_threshold
        self.persist_directory = persist_directory
        
        # 初始化Embedding模型
        if api_key:
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=api_key
            )
        else:
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # 初始化向量数据库
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        # 存储原始回答
        self.responses: Dict[str, str] = {}
        
        print(f"✅ 语义缓存初始化完成 (阈值: {similarity_threshold})")
    
    def _generate_id(self, query: str) -> str:
        """生成唯一ID"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()
    
    def search_similar(
        self, 
        query: str, 
        top_k: int = 1
    ) -> Optional[str]:
        """
        搜索语义相似的问题
        
        Args:
            query: 查询问题
            top_k: 返回最相似的K个结果
            
        Returns:
            最相似问题的回答，如果相似度低于阈值则返回None
        """
        try:
            # 向量检索
            results = self.vectorstore.similarity_search_with_score(
                query, 
                k=top_k
            )
            
            if results:
                doc, score = results[0]
                # Chroma返回的是距离，转换为相似度
                similarity = 1 - score
                
                if similarity >= self.threshold:
                    query_id = doc.metadata.get('id')
                    if query_id and query_id in self.responses:
                        print(f"✅ 语义缓存命中 (相似度: {similarity:.3f})")
                        return self.responses[query_id]
            
            return None
        
        except Exception as e:
            print(f"❌ 语义搜索失败: {e}")
            return None
    
    def add_to_cache(self, query: str, response: str) -> bool:
        """
        添加到语义缓存
        
        Args:
            query: 问题
            response: 回答
            
        Returns:
            是否添加成功
        """
        try:
            query_id = self._generate_id(query)
            
            # 存储到向量数据库
            self.vectorstore.add_texts(
                texts=[query],
                metadatas=[{'id': query_id}]
            )
            
            # 存储回答
            self.responses[query_id] = response
            
            # 持久化
            self.vectorstore.persist()
            
            return True
        
        except Exception as e:
            print(f"❌ 添加到缓存失败: {e}")
            return False
    
    def batch_add(self, queries_responses: List[tuple]) -> int:
        """
        批量添加到缓存
        
        Args:
            queries_responses: [(query, response), ...] 列表
            
        Returns:
            成功添加的数量
        """
        success_count = 0
        
        for query, response in queries_responses:
            if self.add_to_cache(query, response):
                success_count += 1
        
        return success_count
    
    def delete(self, query: str) -> bool:
        """
        删除缓存
        
        Args:
            query: 问题
            
        Returns:
            是否删除成功
        """
        try:
            query_id = self._generate_id(query)
            
            # 从向量数据库中删除（Chroma目前不支持直接删除，需要重建）
            # 这里简化处理，只删除回答
            if query_id in self.responses:
                del self.responses[query_id]
                return True
            
            return False
        
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return False
    
    def clear(self) -> None:
        """清空所有缓存"""
        try:
            # 删除向量数据库
            import shutil
            import os
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            
            # 重新初始化
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.responses.clear()
            
            print("✅ 语义缓存已清空")
        
        except Exception as e:
            print(f"❌ 清空失败: {e}")
    
    def stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "type": "Semantic",
            "total_entries": len(self.responses),
            "threshold": self.threshold,
            "persist_directory": self.persist_directory
        }
    
    def update_threshold(self, new_threshold: float) -> None:
        """
        更新相似度阈值
        
        Args:
            new_threshold: 新的阈值（0-1）
        """
        if 0 <= new_threshold <= 1:
            self.threshold = new_threshold
            print(f"✅ 阈值已更新为: {new_threshold}")
        else:
            print("❌ 阈值必须在0-1之间")
    
    def __len__(self) -> int:
        """返回缓存大小"""
        return len(self.responses)
    
    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"SemanticCache(entries={stats['total_entries']}, "
            f"threshold={stats['threshold']})"
        )


# 使用示例
if __name__ == "__main__":
    import os
    
    # 设置API Key（从环境变量或配置文件读取）
    api_key = os.getenv('OPENAI_API_KEY', 'your-api-key')
    
    # 创建语义缓存
    cache = SemanticCache(
        similarity_threshold=0.92,
        api_key=api_key
    )
    
    # 添加缓存
    cache.add_to_cache("公司的报销政策是什么？", "根据员工手册，差旅费报销标准为...")
    cache.add_to_cache("Python有哪些优点？", "Python简单易学，生态丰富...")
    
    # 测试语义搜索
    test_queries = [
        "公司报销规定是怎样的？",  # 应该命中第一个
        "Python的优势是什么？",     # 应该命中第二个
        "天气怎么样？"              # 不应该命中
    ]
    
    print("\n🔍 测试语义搜索:")
    for query in test_queries:
        result = cache.search_similar(query)
        if result:
            print(f"✓ '{query[:20]}...' → 命中")
        else:
            print(f"✗ '{query[:20]}...' → 未命中")
    
    # 查看统计
    print(f"\n📊 缓存统计: {cache.stats()}")
    
    # 批量添加
    batch_data = [
        (f"问题{i}", f"回答{i}")
        for i in range(10)
    ]
    count = cache.batch_add(batch_data)
    print(f"\n✅ 批量添加: {count}条")
    
    print(f"\n最终统计: {cache.stats()}")
