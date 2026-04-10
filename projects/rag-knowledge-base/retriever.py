"""
检索器模块
负责从向量数据库中检索相关文档片段
"""

import logging
from typing import List
from langchain.schema import Document

from config import (
    TOP_K,
    SIMILARITY_THRESHOLD,
    RETRIEVAL_METHOD,
    MMR_LAMBDA
)

logger = logging.getLogger(__name__)


class Retriever:
    """检索器"""
    
    def __init__(self, vectorstore):
        """
        初始化检索器
        
        Args:
            vectorstore: 向量数据库实例
        """
        self.vectorstore = vectorstore
        self.top_k = TOP_K
        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.retrieval_method = RETRIEVAL_METHOD
    
    def retrieve(self, query: str) -> List[Document]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            
        Returns:
            List[Document]: 相关文档列表
        """
        logger.info(f"🔍 检索查询: {query[:50]}...")
        
        # 根据检索方法选择策略
        if self.retrieval_method == 'similarity':
            docs = self._similarity_search(query)
        elif self.retrieval_method == 'mmr':
            docs = self._mmr_search(query)
        else:
            logger.warning(f"⚠️  不支持的检索方法: {self.retrieval_method}，使用默认的 similarity")
            docs = self._similarity_search(query)
        
        logger.info(f"✅ 检索到 {len(docs)} 个相关文档")
        
        return docs
    
    def _similarity_search(self, query: str) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            
        Returns:
            List[Document]: 相关文档列表
        """
        # 使用相似度搜索
        docs = self.vectorstore.similarity_search(
            query=query,
            k=self.top_k
        )
        
        # 过滤低相似度结果（如果有相似度分数）
        if hasattr(docs[0], 'metadata') and 'score' in docs[0].metadata:
            docs = [doc for doc in docs if doc.metadata.get('score', 0) >= self.similarity_threshold]
        
        return docs
    
    def _mmr_search(self, query: str) -> List[Document]:
        """
        MMR (Maximal Marginal Relevance) 搜索
        平衡相关性和多样性
        
        Args:
            query: 查询文本
            
        Returns:
            List[Document]: 相关文档列表
        """
        logger.info(f"🎯 使用 MMR 检索 (lambda={MMR_LAMBDA})")
        
        docs = self.vectorstore.max_marginal_relevance_search(
            query=query,
            k=self.top_k,
            fetch_k=self.top_k * 2,  # 先检索更多，再筛选
            lambda_mult=MMR_LAMBDA
        )
        
        return docs
    
    def retrieve_with_score(self, query: str) -> List[tuple]:
        """
        检索相关文档并返回相似度分数
        
        Args:
            query: 查询文本
            
        Returns:
            List[tuple]: (文档, 分数) 列表
        """
        logger.info(f"🔍 检索查询（带分数）: {query[:50]}...")
        
        # 相似度搜索带分数
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query=query,
            k=self.top_k
        )
        
        # 过滤低分数结果
        filtered_docs = [
            (doc, score) for doc, score in docs_with_scores
            if score >= self.similarity_threshold
        ]
        
        logger.info(f"✅ 检索到 {len(filtered_docs)} 个相关文档")
        
        return filtered_docs


def create_retriever(vectorstore):
    """
    便捷函数：创建检索器
    
    Args:
        vectorstore: 向量数据库实例
        
    Returns:
        Retriever: 检索器实例
    """
    return Retriever(vectorstore)


if __name__ == '__main__':
    # 测试
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from vector_store import load_vector_store
    
    # 加载向量数据库
    manager = load_vector_store()
    
    if manager and manager.vectorstore:
        # 创建检索器
        retriever = create_retriever(manager.vectorstore)
        
        # 测试检索
        query = "如何安装这个软件？"
        docs = retriever.retrieve(query)
        
        print(f"\n📄 检索结果:")
        for i, doc in enumerate(docs, 1):
            print(f"\n--- 文档 {i} ---")
            print(f"来源: {doc.metadata.get('source', '未知')}")
            print(f"内容: {doc.page_content[:200]}...")
