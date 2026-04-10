"""
向量数据库管理模块
支持 ChromaDB 和 FAISS 两种向量数据库
"""

import logging
from pathlib import Path
from typing import List, Optional
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma, FAISS
import logging

from config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    USE_LOCAL_MODEL,
    LOCAL_MODEL_URL,
    VECTOR_DB_TYPE,
    CHROMA_PERSIST_DIR,
    FAISS_INDEX_FILE,
    ENABLE_EMBEDDING_CACHE,
    EMBEDDING_CACHE_DIR
)

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """向量数据库管理器"""
    
    def __init__(self):
        self.vectorstore = None
        self.embeddings = None
        
    def _create_embeddings(self):
        """创建嵌入模型"""
        logger.info(" 创建嵌入模型...")
        
        if USE_LOCAL_MODEL:
            # 使用本地模型（例如通过 API 服务）
            from langchain.embeddings import HuggingFaceEmbeddings
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("✅ 使用本地嵌入模型")
        else:
            # 使用 OpenAI 嵌入模型
            base_embeddings = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                model=OPENAI_EMBEDDING_MODEL
            )
            
            # 是否启用缓存
            if ENABLE_EMBEDDING_CACHE:
                EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                self.embeddings = CacheBackedEmbeddings.from_bytes_store(
                    underlying_embeddings=base_embeddings,
                    document_embedding_cache=EMBEDDING_CACHE_DIR,
                    namespace=OPENAI_EMBEDDING_MODEL
                )
                logger.info("✅ 使用 OpenAI 嵌入模型（已启用缓存）")
            else:
                self.embeddings = base_embeddings
                logger.info("✅ 使用 OpenAI 嵌入模型")
    
    def create_from_documents(self, documents: List[Document], rebuild: bool = False):
        """
        从文档创建向量数据库
        
        Args:
            documents: 文档列表
            rebuild: 是否重建数据库
        """
        if not documents:
            logger.warning("⚠️  没有文档，无法创建向量数据库")
            return
        
        self._create_embeddings()
        
        logger.info(f"📊 文档数量: {len(documents)}")
        logger.info(f"🔧 向量数据库类型: {VECTOR_DB_TYPE}")
        
        if VECTOR_DB_TYPE == 'chroma':
            self._create_chroma_db(documents, rebuild)
        elif VECTOR_DB_TYPE == 'faiss':
            self._create_faiss_db(documents, rebuild)
        else:
            raise ValueError(f"不支持的向量数据库类型: {VECTOR_DB_TYPE}")
        
        logger.info("✅ 向量数据库创建完成")
    
    def _create_chroma_db(self, documents: List[Document], rebuild: bool = False):
        """创建 ChromaDB 向量数据库"""
        
        if rebuild and CHROMA_PERSIST_DIR.exists():
            logger.info("🗑️  删除旧的 ChromaDB 数据库")
            import shutil
            shutil.rmtree(CHROMA_PERSIST_DIR)
        
        logger.info(f"💾 存储路径: {CHROMA_PERSIST_DIR}")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR)
        )
        
        logger.info("✅ ChromaDB 数据库创建成功")
    
    def _create_faiss_db(self, documents: List[Document], rebuild: bool = False):
        """创建 FAISS 向量数据库"""
        
        logger.info(f"💾 存储路径: {FAISS_INDEX_FILE}")
        
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        # 保存到本地
        self.vectorstore.save_local(str(FAISS_INDEX_FILE.parent))
        
        logger.info("✅ FAISS 数据库创建成功")
    
    def load_existing(self):
        """加载已存在的向量数据库"""
        self._create_embeddings()
        
        if VECTOR_DB_TYPE == 'chroma':
            if not CHROMA_PERSIST_DIR.exists():
                logger.error("❌ ChromaDB 数据库不存在，请先构建知识库")
                return False
            
            self.vectorstore = Chroma(
                persist_directory=str(CHROMA_PERSIST_DIR),
                embedding_function=self.embeddings
            )
            logger.info("✅ ChromaDB 数据库加载成功")
            
        elif VECTOR_DB_TYPE == 'faiss':
            if not FAISS_INDEX_FILE.parent.exists():
                logger.error("❌ FAISS 数据库不存在，请先构建知识库")
                return False
            
            self.vectorstore = FAISS.load_local(
                str(FAISS_INDEX_FILE.parent),
                self.embeddings
            )
            logger.info("✅ FAISS 数据库加载成功")
        
        return True
    
    def add_documents(self, documents: List[Document]):
        """
        向现有数据库添加文档
        
        Args:
            documents: 要添加的文档列表
        """
        if self.vectorstore is None:
            logger.error("❌ 向量数据库未初始化")
            return
        
        self.vectorstore.add_documents(documents)
        logger.info(f"✅ 添加了 {len(documents)} 个文档")
        
        # 持久化保存
        if VECTOR_DB_TYPE == 'chroma':
            self.vectorstore.persist()
        elif VECTOR_DB_TYPE == 'faiss':
            self.vectorstore.save_local(str(FAISS_INDEX_FILE.parent))
    
    def get_statistics(self) -> dict:
        """获取数据库统计信息"""
        if self.vectorstore is None:
            return {}
        
        stats = {
            'type': VECTOR_DB_TYPE,
            'embedding_model': OPENAI_EMBEDDING_MODEL if not USE_LOCAL_MODEL else 'local',
        }
        
        if VECTOR_DB_TYPE == 'chroma':
            # ChromaDB 统计
            collection = self.vectorstore._collection
            stats['document_count'] = collection.count()
        
        return stats


def create_vector_store(documents: List[Document], rebuild: bool = False) -> VectorStoreManager:
    """
    便捷函数：创建向量数据库
    
    Args:
        documents: 文档列表
        rebuild: 是否重建
        
    Returns:
        VectorStoreManager: 向量数据库管理器
    """
    manager = VectorStoreManager()
    manager.create_from_documents(documents, rebuild)
    return manager


def load_vector_store() -> Optional[VectorStoreManager]:
    """
    便捷函数：加载已有的向量数据库
    
    Returns:
        VectorStoreManager: 向量数据库管理器，如果加载失败则返回 None
    """
    manager = VectorStoreManager()
    if manager.load_existing():
        return manager
    return None


if __name__ == '__main__':
    # 测试
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from document_loader import load_and_split
    
    # 加载文档
    documents = load_and_split()
    
    if documents:
        # 创建向量数据库
        manager = create_vector_store(documents, rebuild=True)
        
        # 获取统计信息
        stats = manager.get_statistics()
        print(f"\n📊 数据库统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
